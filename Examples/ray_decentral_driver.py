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
NUM_ROUNDS = 5
NUM_EPOCHS = 2

NUM_CPUS = 1
NUM_GPUS = 0.5

BATCH_SIZE = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)


@dataclass
class TrainingConfig:
    num_rounds: int
    num_epochs: int
    num_clients: int
    batch_size: int


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
    def __init__(self, client_id: int, model_fn: Callable[[], nn.Module], train_loader: DataLoader, other_clients):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        self.log = logging.getLogger(f"CLIENT {client_id}")
        self.log.info(f"Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.client_id = client_id
        self.model = model_fn().to(self.device)
        self.train_loader = train_loader
        self.other_clients = other_clients

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def get_model_state_dict(self) -> Dict:
        return self.model.state_dict()
        
    def train(self, epochs: int = 1) -> None:
        self.log.info(f"Starting training")

        self.model.to(self.device)
        self.model.train()

        optimizer = SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.log.info(f"Epoch {epoch + 1}/{epochs}")

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

        self.log.info(f"Finished training")
    
    def aggregate(self) -> None:
        return
    

class Driver:
    def __init__(self, clients: List[Client], train_config: TrainingConfig):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        self.log = logging.getLogger("DRIVER")
        self.log.info(f"Initializing")

        self.clients = clients
        self.train_config = train_config

    def train_round(self) -> None:
        """Waits until all clients are finished."""
        ray.get([client.train.remote() for client in self.clients]) 

    def initiate_aggregation(self) -> None:
        """Wait until all clients are finished."""
        ray.get([client.aggregate.remote() for client in self.clients])

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

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def add_clients(self, clients: List[Client]) -> None:
        self.clients += clients

    def run(self) -> None:
        self.log.info(f"Training with {self.train_config.num_clients} clients")
       
        for round_num in range(self.train_config.num_rounds):
            self.log.info(f"Round {round_num + 1}/{self.train_config.num_rounds}")
            
            # Waits until all clients are finished
            client_updates = ray.get([
                client.train.remote(epochs=self.train_config.num_epochs) for client in self.clients
            ])

            self.aggregate(client_updates)

            self.log.info("Sending updated model to clients")
            global_state_dict = self.global_model.state_dict()
            ray.get([
                client.update_model.remote(global_state_dict) for client in self.clients
            ])

        self.log.info("Finished training")

    def aggregate(self, client_updates: List[Dict]):
        self.log.info(f"Aggregating updates")

        new_state_dict = copy.deepcopy(self.global_model.state_dict())
        for key in new_state_dict.keys():
            new_state_dict[key] = sum([client_update[key] for client_update in client_updates]) / len(client_updates)
        self.global_model.load_state_dict(new_state_dict)

        self.log.info(f"Finished update aggregation")


def main():
    ray.init(_temp_dir="/data/tavonputl/tmp/ray")
    log = logging.getLogger("MAIN")

    train_config = TrainingConfig(
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
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
            server=server,
        ) 
        for i in range(train_config.num_clients)
    ]

    # Make sure that all of the clients are initialized before starting
    ray.get([client.initialize.remote() for client in clients])
    log.info("Finished initializing clients")

    ray.get(server.add_clients.remote(clients))
    ray.get(server.run.remote())
    
    ray.timeline(filename="timeline.json")
    ray.shutdown()

if __name__ == "__main__":
    main()