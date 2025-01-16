from typing import Callable, List

import torch
import torch.nn as nn

import ray

from . import my_logging
from .types import Parameters, TrainingConfig
from .messages import ClientUpdate
from .client import SyncClient


@ray.remote
class SyncServer:
    def __init__(self, model_fn: Callable[[], nn.Module], train_config: TrainingConfig):
        self.log = my_logging.get_logger("SERVER")
        self.log.info(f"Initializing")

        self.train_config = train_config
        self.global_model = model_fn()
        self.clients: List[SyncClient] = []

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def add_clients(self, clients: List[SyncClient]) -> None:
        self.clients += clients

    def run(self) -> None:
        self.log.info(f"Training with {self.train_config.num_clients} clients")
       
        for round_num in range(self.train_config.num_rounds):
            self.log.info(f"Round {round_num + 1}/{self.train_config.num_rounds}")
            
            # Waits until all clients are finished
            parameters = self._get_model_parameters()
            client_updates = ray.get([
                client.train.remote(parameters, epochs=self.train_config.num_epochs) for client in self.clients
            ])

            self.aggregate(client_updates)

        self.log.info("Finished training")

    def aggregate(self, client_updates: List[ClientUpdate]):
        self.log.info(f"Aggregating updates")   

        with torch.no_grad():
            for name, params in self.global_model.named_parameters():
                params.data = sum([update.parameters[name] for update in client_updates]) / len(client_updates)

        self.log.info(f"Finished update aggregation")

    def _get_model_parameters(self) -> Parameters:
        return {name: param for name, param in self.global_model.named_parameters()}
