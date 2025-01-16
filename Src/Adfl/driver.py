from typing import List

import ray

import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from . import my_logging
from .types import TrainingConfig
from .model import get_mobile_net_v3_small
from .server import SyncServer
from .client import SyncClient

class SyncDriver:
    def __init__(self, timeline_path: str = None, tmp_path: str = None):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path

        self.train_config: TrainingConfig = None

        self.server: SyncServer = None
        self.clients: List[SyncClient] = []
        self.dataset_splits: List[Subset] = []

    def init_backend(self) -> None:
        self.log.info("Initializing ray backend")

        if self.tmp_path is not None:
            ray.init(_temp_dir=self.tmp_path)
        else:
            ray.init()
    
    def init_training(self, train_config: TrainingConfig) -> None:
        self.log.info(f"Initialing training with config: {train_config}")
        self.train_config = train_config

        self.dataset_splits = self._init_datasets()
        self.clients = self._init_clients()
        self.server = self._init_server()

    def run(self) -> None: 
        self.log.info("Initiating training")
        ray.get(self.server.run.remote())

        if self.timeline_path is not None:
            ray.timeline(filename=self.timeline_path)
            self.log.info(f"Timeline saved to {self.timeline_path}")

        self.log.info("Training complete")
    
    def shutdown(self) -> None:
        self.log.info("Shutting down driver")
        ray.shutdown()

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
        
    def _init_clients(self) -> List[SyncClient]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            SyncClient.remote(
                client_id=i, 
                model=_generate_model(), 
                train_loader=DataLoader(self.dataset_splits[i], batch_size=self.train_config.batch_size, shuffle=True),
                slowness=(i * 0.25 + 1),
            ) 
            for i in range(self.train_config.num_clients)
        ]

        # Make sure that all of the clients are initialized
        ray.get([client.initialize.remote() for client in clients])
        
        return clients

    def _init_server(self) -> SyncServer:
        self.log.info("Initializing server")

        server = SyncServer.remote(model_fn=_generate_model, train_config=self.train_config)
        ray.get(server.initialize.remote())
        ray.get(server.add_clients.remote(self.clients))

        return server


def _generate_model() -> nn.Module:
    return get_mobile_net_v3_small(num_classes=10)
