import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import ray

from . import my_logging
from .types import Parameters
from .messages import ClientUpdate

NUM_CPUS = 1
NUM_GPUS = 0.5


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class SyncClient:
    def __init__(self, client_id: int, model: nn.Module, train_loader: DataLoader, slowness: float):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info(f"Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.client_id = client_id
        self.round = 0
        self.slowness = slowness

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def train(self, parameters: Parameters, epochs: int = 1):
        self.log.info(f"Starting training")

        self.round += 1
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

        self.log.info(f"Finished training")
        return self._create_update()
    
    def _create_update(self) -> ClientUpdate:
        self.model.to("cpu")    

        return ClientUpdate(
            parameters=self._get_model_parameters(),
            client_id=self.client_id,
            client_round=self.round,
        )

    def _update_model(self, parameters: Parameters) -> None:
        for name, params in self.model.named_parameters():
            params.data = parameters[name]

    def _get_model_parameters(self) -> Parameters:
        return {name: param for name, param in self.model.named_parameters()}
