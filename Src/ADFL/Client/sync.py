import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import ray

from ADFL import my_logging
from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import RoundResults
from ADFL.messages import ClientUpdate
from ADFL.model import Parameters, get_model_parameters, set_model_parameters

from .common import LR, _train_epoch


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class SyncClient:
    def __init__(
        self, 
        client_id: int, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        slowness: float
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.client_id = client_id
        self.round = 0
        self.slowness = slowness

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def train(self, parameters: Parameters, epochs: int = 1) -> ClientUpdate:
        self.round += 1
        self.log.info(f"Starting training round {self.round}")
        start_time = time.time()

        set_model_parameters(self.model, parameters)

        round_results = RoundResults()
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            results = _train_epoch(
                self.model, self.optimizer, self.criterion, self.train_loader, self.device, self.slowness
            )
            round_results.train_results.append(results)

        round_results.round_time = time.time() - start_time
        round_results.g_start_time = start_time
        round_results.epochs = epochs
        round_results.train_round = self.round

        self.log.info("Finished training")
        return self._create_update(round_results)

    def _create_update(self, round_results: RoundResults) -> ClientUpdate:
        self.model.to("cpu")

        return ClientUpdate(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            client_round=self.round,
            round_results=round_results,
        )


