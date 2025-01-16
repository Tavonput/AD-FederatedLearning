import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

import threading
import copy
import time
import logging
from typing import Dict, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split, Subset

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import datasets, transforms

import ray

NUM_CLIENTS = 4
NUM_ROUNDS = 3
NUM_EPOCHS = 1
MAX_ROUNDS = 10
TIMEOUT = 20

NUM_CPUS = 1
NUM_GPUS = 0.5

BATCH_SIZE = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)


Parameters = Dict[str, torch.Tensor]


@dataclass
class TrainingConfig:
    num_rounds: int
    num_epochs: int
    num_clients: int
    batch_size: int
    max_rounds: int


@dataclass
class ClientUpdate:
    parameters: Parameters
    client_id: int
    client_round: int


def get_mobile_net_v3_small(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 32 * 32 * 3)
        return self.fc(x)
    

@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class Client:
    def __init__(self, client_id: int, model: nn.Module, train_loader: DataLoader, slowness: float):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        self.log = logging.getLogger(f"CLIENT {client_id}")
        self.log.info(f"Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.client_id = client_id
        self.train_loader = train_loader
        self.round = 0
        self.slowness = slowness
        self.updates: List[ClientUpdate] = []
        self.other_clients = []

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        self.training_thread: threading.Thread = None
        self.stop_flag = threading.Event()

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def add_clients(self, clients: List) -> None:
        self.other_clients += clients
        return

    def receive_update(self, update: ClientUpdate) -> None:
        self.updates.append(update)

    def stop(self) -> None:
        self.log.info(f"Terminating training thread")

        self.stop_flag.set()
        if self.training_thread is not None:
            self.training_thread.join()

        self.log.info(f"Training thread has been cleaned up")

    def train(self, epochs: int) -> None:
        self.log.info(f"Starting training thread")
        self.training_thread = threading.Thread(target=self._train, args=(epochs,))
        self.training_thread.start()

    def _train(self, epochs: int) -> None:
        while not self.stop_flag.is_set():
            self.round += 1
            self.log.info(f"Starting training round {self.round}")

            self._train_round(epochs)
            self._send_updates()
            self._aggregate()

    def _train_round(self, epochs: int) -> None:
        self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

        total_train_time = time.time() - start_time

        # Simulate compute capabilities
        sleep_time = total_train_time * (self.slowness - 1)
        self.log.debug(f"Sleeping for {sleep_time:.2f}s")
        time.sleep(sleep_time)

        self.log.info(f"Finished training")
    
    def _send_updates(self) -> None:
        self.log.info(f"Sending updates")

        self.model.to("cpu")

        update = ClientUpdate(
            parameters=self._get_model_parameters(),
            client_id=self.client_id,
            client_round=self.round,
        )

        [client.receive_update.remote(update) for client in self.other_clients]
    
    def _aggregate(self) -> None:
        self.log.info(f"Aggregating from {len(self.updates)} clients")

        if len(self.updates) == 0:
            return

        # Add our personal model to the list of updates for the subsequent average
        self.updates.append(ClientUpdate(self._get_model_parameters(), self.client_id, self.round))

        with torch.no_grad():
            for name, params in self.model.named_parameters():
                params.data = torch.stack([update.parameters[name] for update in self.updates]).mean(dim=0)

        self.updates = []

    def _get_model_parameters(self) -> Parameters:
        return {name: param for name, param in self.model.named_parameters()}
    
    
class Driver:
    def __init__(self, train_config: TrainingConfig, model_fn: Callable[[], nn.Module]):
        self.log = logging.getLogger("DRIVER")
        self.log.info("Initializing")

        self.train_config = train_config
        self.model_fn = model_fn
        
        self.split_datasets = self._init_datasets()
        self.clients = self._init_clients()

    def run(self, timeout: int) -> None:
        self.log.info(f"Starting training")
        [client.train.remote(epochs=self.train_config.num_epochs) for client in self.clients]

        time.sleep(timeout)

        self.log.info(f"Timeout reached. Stopping clients")
        self._stop_clients()
        self.log.info(f"All clients stopped")
        
    def _init_datasets(self) -> List[Subset]:
        self.log.info("Creating datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        full_dataset = datasets.CIFAR10(root="../Data", train=True, transform=transform, download=True)
        split_datasets = random_split(
            full_dataset, [len(full_dataset) // self.train_config.num_clients] * self.train_config.num_clients
        )

        self.log.info(f"Dataset size: {len(split_datasets[0])}/{len(full_dataset)}")
        return split_datasets
        
    def _init_clients(self) -> List[Client]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            Client.remote(
                client_id=i, 
                model=self.model_fn(), 
                train_loader=DataLoader(self.split_datasets[i], batch_size=self.train_config.batch_size, shuffle=True),
                slowness=(i * 0.25 + 1),
            ) 
            for i in range(self.train_config.num_clients)
        ]

        # Make sure that all of the clients are initialized
        ray.get([client.initialize.remote() for client in clients])
        
        for id, client in enumerate(clients):
            clients_to_add = [clients[c] for c, _ in enumerate(clients) if c != id]
            ray.get(client.add_clients.remote(clients_to_add))
        
        return clients
    
    def _stop_clients(self) -> None:
        ray.get([client.stop.remote() for client in self.clients])

def test():
    return

def main():
    ray.init(_temp_dir="/data/tavonputl/tmp/ray")

    train_config = TrainingConfig(
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        max_rounds=MAX_ROUNDS,
    )

    def generate_model() -> nn.Module:
        return get_mobile_net_v3_small(num_classes=10)

    driver = Driver(train_config, generate_model)
    driver.run(TIMEOUT)
    
    ray.timeline(filename="timeline.json")
    ray.shutdown()

if __name__ == "__main__":
    main()