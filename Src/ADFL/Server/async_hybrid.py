import time
from typing import Callable, List, Dict

import torch.nn as nn

import ray

from ADFL import my_logging
from ADFL.types import TrainingConfig, ClientResults
from ADFL.messages import ClientUpdate, ServerUpdate
from ADFL.model import Parameters, get_model_parameters, set_model_parameters,simple_aggregate
from ADFL.compression import decompress_params

from ADFL.Client import AsyncClientProxy

from .common import compress_model, train_client


@ray.remote
class AsyncHybridServer:
    """Asynchronous Hybrid Server.

    Implementation of an asynchronous hybrid federated learning server.
    """
    def __init__(
        self, 
        server_id: int, 
        model_fn: Callable[[], nn.Module], 
        train_config: TrainingConfig
    ):
        self.log = my_logging.get_logger(f"SERVER {server_id}")
        self.log.info("Initializing")

        self.server_id = server_id
        self.train_config = train_config
        self.global_model = model_fn()

        self.servers: List[AsyncHybridServer] = []  # type: ignore
        self.clients: Dict[int, AsyncClientProxy] = {}
        self.client_results: Dict[int, ClientResults] = {}
        self.finished_clients = 0

        self.message_log: List[str] = []

        self.ready = False


    def initialize(self) -> bool:
        """Set the ready status to True."""
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClientProxy]) -> None:
        """Add AsyncClients."""
        for client in clients:
            self.clients[client.client_id] = client
            self.client_results[client.client_id] = ClientResults(client.client_id)


    def add_servers(self, servers: List) -> None:
        "Add external AsyncHybridServer."
        self.servers += servers


    def get_model(self) -> Parameters:
        """Get the model parameters."""
        return get_model_parameters(self.global_model)


    def get_client_results(self) -> List[ClientResults]:
        """Get the ClientResults."""
        results = []
        for c_id, client in self.clients.items():
            accuracies = client.get_accuracies(block=True)
            self.client_results[c_id].accuracies = accuracies  # type: ignore
            results.append(self.client_results[c_id])

        return results


    def get_message_log(self) -> List[str]:
        """Get the message logs."""
        return self.message_log


    def log_node_setup(self) -> None:
        """Log the server and client ids."""
        client_ids = [client.client_id for client in self.clients.values()]
        self.log.info(f"Server: {self.server_id}, Clients: {client_ids}")


    def run(self) -> None:
        """Run the server."""
        self.log.info("Starting training")

        for i in self.clients.keys():
            self._train_client(i, round=1)


    def stop(self) -> None:
        """Stop all clients."""
        self.log.info("Stopping training. Waiting for all clients to finish")
        ray.get([client.stop(block=False) for client in self.clients.values()])
        self.log.info("All clients have stopped")


    def client_update(self, client_update: ClientUpdate) -> None:
        """Message: Received update from client, thus aggregate."""
        self.message_log.append(f"C_{client_update.client_id}")

        self.log.debug(f"Decompressing parameters from Client {client_update.client_id}")
        client_params, d_time = decompress_params(client_update.parameters)

        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._aggregate(client_params)

        self.log.debug("Sending updates to external servers")
        self._send_external_update()

        # Send out the next train message
        if client_update.client_round < self.train_config.max_rounds:
            self._train_client(client_update.client_id, client_update.client_round + 1)
        else:
            self.finished_clients += 1

        self._save_client_update(client_update)


    def external_update(self, server_update: ServerUpdate) -> None:
        """Message: Received update from external server, thus aggregate."""
        self.message_log.append(f"E_{server_update.server_id}")

        self.log.debug(f"Decompressing updates from Server {server_update.server_id}")
        params, d_time = decompress_params(server_update.parameters)

        self.log.debug(f"Aggregating updates from Server {server_update.server_id}")
        self._aggregate(params)


    def _aggregate(self, parameters: Parameters) -> None:
        """Aggregate parameters into local model."""
        all_parameters = [get_model_parameters(self.global_model), parameters]
        parameters_prime = simple_aggregate(all_parameters)
        set_model_parameters(self.global_model, parameters_prime)


    def _train_client(self, client_id: int, round: int) -> None:
        """Send a train message to a client."""
        self.log.debug(f"Sending training job to Client {client_id}: Round {round}")
        _, _ = train_client(
            client   = self.clients[client_id],
            model    = self.global_model,
            epochs   = self.train_config.num_epochs,
            method   = self.train_config.compress,
            bits     = self.train_config.quant_lvl_1
        )

    def _send_external_update(self) -> None:
        """Send current local model to all external servers."""
        c_params, c_time = compress_model(self.global_model, self.train_config.compress, self.train_config.quant_lvl_2)

        update = ServerUpdate(
            parameters = c_params,
            server_id  = self.server_id
        )

        start_time = time.time()
        [server.external_update.remote(update) for server in self.servers]
        b_time = time.time() - start_time


    def _save_client_update(self, update: ClientUpdate) -> None:
        """Save a training round update from a client."""
        self.client_results[update.client_id].rounds.append(update.round_results)

