from typing import Callable, List

import torch.nn as nn

import ray

from ADFL import my_logging
from ADFL.types import Parameters, TrainingConfig, FederatedResults, ClientResults, Accuracy
from ADFL.messages import ClientUpdate
from ADFL.model import get_model_parameters, set_model_parameters,simple_aggregate

from ADFL.Client import AsyncClientProxy


@ray.remote
class AsyncServer:
    """Asynchronous Server.

    Implementation of an asynchronous federated learning server.
    """
    def __init__(self, model_fn: Callable[[], nn.Module], train_config: TrainingConfig):
        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")

        self.train_config = train_config
        self.global_model = model_fn()

        self.clients: List[AsyncClientProxy] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]
        self.finished_clients = 0

        self.ready = False


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClientProxy]) -> None:
        self.clients += clients


    def get_model(self) -> Parameters:
        return get_model_parameters(self.global_model)


    def get_client_results(self) -> List[ClientResults]:
        for i, client in enumerate(self.clients):
            accuracies: List[Accuracy] = client.get_accuracies()
            self.client_results[i].accuracies = accuracies

        return self.client_results


    def run(self) -> None:
        self.log.info("Starting training")

        for i, _ in enumerate(self.clients):
            self._train_client(i, round=1)


    def stop(self) -> None:
        self.log.info("Stopping training. Waiting for all clients to finish")
        [client.stop() for client in self.clients]
        self.log.info("All clients have stopped")


    def client_update(self, client_update: ClientUpdate) -> None:
        """Message: Process a client update."""
        self.log.info(f"Aggregating updates from Client {client_update.client_id}")
        self._save_update(client_update)

        # Aggregate the client update with the global model
        parameters = [get_model_parameters(self.global_model), client_update.parameters]
        parameters_prime = simple_aggregate(parameters)
        set_model_parameters(self.global_model, parameters_prime)

        if client_update.client_round < self.train_config.max_rounds:
            self._train_client(client_update.client_id, client_update.client_round + 1)
        else:
            self.finished_clients += 1


    def _train_client(self, client_id: int, round: int) -> None:
        """Send a training job to a client."""
        self.log.info(f"Sending training job to Client {client_id}: Round {round}")
        self.clients[client_id].train(get_model_parameters(self.global_model), self.train_config.num_epochs)


    def _save_update(self, update: ClientUpdate) -> None:
        """Save a client update."""
        self.client_results[update.client_id].rounds.append(update.round_results)


@ray.remote
class TraditionalServer:
    """Traditional Server.

    Implementation of an actor-based synchronous federated learning server.
    """
    def __init__(self, model_fn: Callable[[], nn.Module], train_config: TrainingConfig):
        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")

        self.train_config = train_config
        self.global_model = model_fn()

        self.train_counter = 0
        self.round = 0
        self.updates: List[ClientUpdate] = []

        self.clients: List[AsyncClientProxy] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]

        self.ready = False


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClientProxy]) -> None:
        self.clients += clients


    def get_model(self) -> Parameters:
        return get_model_parameters(self.global_model)


    def get_client_results(self) -> FederatedResults:
        for i, client in enumerate(self.clients):
            accuracies: List[Accuracy] = client.get_accuracies()
            self.client_results[i].accuracies = accuracies

        return self.client_results


    def run(self) -> None:
        self.log.info("Starting training")
        self.round = 1
        self._train_round(self.round)


    def stop(self) -> None:
        self.log.info("Stopping training. Waiting for all clients to finish")
        [client.stop() for client in self.clients]
        self.log.info("All clients have stopped")


    def client_update(self, client_update: ClientUpdate) -> None:
        """Message: Process a client update."""
        self.log.info(f"Received update from Client {client_update.client_id}")
        self._save_update(client_update)

        self.train_counter -= 1
        self.updates.append(client_update)
        if self.train_counter > 0:
            self.log.info(f"Waiting for {self.train_counter} more responses")
            return

        self.log.info("Received all client responses. Proceeding to aggregation")
        self._aggregate()
        self.updates.clear()

        self.round += 1
        self._train_round(self.round)


    def _aggregate(self) -> None:
        """Aggregate client updates into global model."""
        parameters = [update.parameters for update in self.updates]
        parameters_prime = simple_aggregate(parameters)
        set_model_parameters(self.global_model, parameters_prime)


    def _train_round(self, train_round: int) -> None:
        """Do a train round."""
        if train_round > self.train_config.num_rounds:
            self.stop()
            return

        self.log.info(f"Training Round {train_round}/{self.train_config.num_rounds}")

        self.log.info(f"Sending out {len(self.clients)} training jobs")
        for id, _ in enumerate(self.clients):
            self._train_client(id)
            self.train_counter += 1


    def _train_client(self, client_id: int) -> None:
        """Send a train job to a client."""
        self.clients[client_id].train(get_model_parameters(self.global_model), self.train_config.num_epochs)


    def _save_update(self, update: ClientUpdate) -> None:
        self.client_results[update.client_id].rounds.append(update.round_results)

