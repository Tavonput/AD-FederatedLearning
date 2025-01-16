import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

import copy
import time
import logging
from typing import Dict, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import datasets, transforms

import ray

NUM_CLIENTS = 4
NUM_ROUNDS = 3
NUM_EPOCHS = 2
MAX_ROUNDS = 3
TIMEOUT = 30

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
    def __init__(self, client_id: int, model: nn.Module, train_loader: DataLoader, slowness: float, server):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        self.log = logging.getLogger(f"CLIENT {client_id}")
        self.log.info(f"Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.client_id = client_id
        self.train_loader = train_loader
        self.slowness = slowness
        self.server = server
        self.round = 0

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def stop(self) -> None:
        self.log.info(f"Stopping")

    def train(self, parameters: Parameters, epochs: int = 1) -> None:
        self.round += 1
        self.log.info(f"Starting training round {self.round}")

        self._update_model(parameters)

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

        self.log.info(f"Finished training. Sending updated model to server")
        self._send_model_to_server()
    
    def _send_model_to_server(self) -> None:
        self.model.to("cpu")

        update = ClientUpdate(
            parameters=self._get_model_parameters(),
            client_id=self.client_id,
            client_round=self.round,
        )

        self.server.update_model.remote(update)

    def _update_model(self, parameters: Parameters) -> None:
        for name, params in self.model.named_parameters():
            params.data = parameters[name]

    def _get_model_parameters(self) -> Parameters:
        return {name: param for name, param in self.model.named_parameters()}


@ray.remote
class Server:
    def __init__(self, model_fn: Callable[[], nn.Module], train_config: TrainingConfig):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )

        self.log = logging.getLogger("SERVER")
        self.log.info(f"Initializing")

        self.train_config = train_config
        self.global_model = model_fn()
        self.clients: List[Client] = []
        self.finished_clients = 0

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def add_clients(self, clients: List[Client]) -> None:
        self.clients += clients

    def get_model(self) -> nn.Module:
        return copy.deepcopy(self.global_model)

    def run(self) -> None:
        self.log.info(f"Starting training")
        
        for i, _ in enumerate(self.clients):
            self._train_client(i, round=1)

    def stop(self) -> None:
        self.log.info(f"Stopping training. Waiting for all clients to finish")
        ray.get([client.stop.remote() for client in self.clients])
        self.log.info(f"All clients have stopped")

    def update_model(self, client_update: ClientUpdate) -> None:
        self.log.info(f"Aggregating updates from Client {client_update.client_id}")

        with torch.no_grad():
            for name, params in self.global_model.named_parameters():
                params.data = (params + client_update.parameters[name]) / 2

        self.log.info(f"Finished update aggregation")

        if client_update.client_round < self.train_config.max_rounds:
            self._train_client(client_update.client_id, client_update.client_round + 1)
        else:
            self.finished_clients += 1

    def _train_client(self, client_id: int, round: int) -> None:
        self.log.info(f"Sending training job to Client {client_id}: Round {round}")
        self.clients[client_id].train.remote(
            parameters=self._get_model_parameters(), epochs=self.train_config.num_epochs
        )

    def _get_model_parameters(self) -> Parameters:
        return {name: param for name, param in self.global_model.named_parameters()}


def main():
    ray.init(_temp_dir="/data/tavonputl/tmp/ray")
    log = logging.getLogger("MAIN")

    train_config = TrainingConfig(
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        max_rounds=MAX_ROUNDS,
    )

    def generate_model() -> nn.Module:
        return get_mobile_net_v3_small(num_classes=10)

    server = Server.remote(model_fn=generate_model, train_config=train_config)
    ray.get(server.initialize.remote())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    log.info("Creating datasets")
    full_dataset = datasets.CIFAR10(root="../Data", train=True, transform=transform, download=True)
    dataset_splits = random_split(
        full_dataset, [len(full_dataset) // train_config.num_clients] * train_config.num_clients
    )
    log.info(f"Dataset size: {len(dataset_splits[0])}/{len(full_dataset)}")

    log.info(f"Initializing {train_config.num_clients} clients")
    clients = [
        Client.remote(
            client_id=i, 
            model=generate_model(), 
            train_loader=DataLoader(dataset_splits[i], batch_size=train_config.batch_size, shuffle=True),
            slowness=(i * 0.25 + 1),
            server=server,
        ) 
        for i in range(train_config.num_clients)
    ]

    # Make sure that all of the clients are initialized before starting
    ray.get([client.initialize.remote() for client in clients])
    ray.get(server.add_clients.remote(clients))
    log.info("Finished initializing clients")
    
    server.run.remote()
    
    time.sleep(TIMEOUT)

    ray.get(server.stop.remote())
    
    ray.timeline(filename="timeline.json")
    ray.shutdown()

if __name__ == "__main__":
    main()