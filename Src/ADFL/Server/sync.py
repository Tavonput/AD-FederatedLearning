import time
from typing import Callable, List

import torch
import torch.nn as nn

import ray

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, ClientResults
from ADFL.messages import ClientUpdate
from ADFL.model import get_model_parameters

from ADFL.Client import SyncClient


@ray.remote
class SyncServer:
    """Synchronous Server.

    Implementation of a synchronous federated learning server. This server interacts with SyncClient, which will return 
    a response upon requesting a train task. Thus, this server will block until all responses are received.
    """
    def __init__(self, model_fn: Callable[[], nn.Module], train_config: TrainingConfig):
        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")
        self.ready = False

        self.train_config = train_config
        self.global_model = model_fn()
        self.clients: List[SyncClient] = []

        self.federated_results = FederatedResults()
        self.federated_results.paradigm = "SyncServerClient"
        self.federated_results.train_config = train_config
        self.federated_results.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[SyncClient]) -> None:
        self.clients += clients


    def run(self) -> FederatedResults:
        self.log.info(f"Training with {self.train_config.num_clients} clients")

        start_time = time.time()
        self.federated_results.g_start_time = start_time

        for round_num in range(self.train_config.num_rounds):
            self.log.info(f"Round {round_num + 1}/{self.train_config.num_rounds}")

            # Waits until all clients are finished
            parameters = get_model_parameters(self.global_model)
            client_updates: List[ClientUpdate] = ray.get([
                client.train.remote(parameters, epochs=self.train_config.num_epochs) for client in self.clients
            ])

            self._save_updates(client_updates)
            self._aggregate(client_updates)

        self.log.info("Finished training")
        return self.federated_results


    def _aggregate(self, client_updates: List[ClientUpdate]) -> None:
        self.log.info("Aggregating updates")   

        with torch.no_grad():
            for name, params in self.global_model.named_parameters():
                params.data = sum([update.parameters[name] for update in client_updates]) / len(client_updates)


    def _save_updates(self, client_updates: List[ClientUpdate]) -> None:
        for update in client_updates:
            self.federated_results.client_results[update.client_id].rounds.append(update.round_results)

