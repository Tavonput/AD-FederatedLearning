import time
from typing import List, Union

import ray

from torch.utils.data import DataLoader

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, ClientResults, EvalConfig
from ADFL.eval import EvalActorProxy
from ADFL.Client import AsyncPeerClient, AsyncPeerClientV2

from .common import (
    Driver, DataSetSplit, _init_ray, check_eval_config, _check_slowness_map, _create_datasets, _generate_model,
    _federated_results_to_json
)


AsyncPeerClientU = Union[AsyncPeerClient, AsyncPeerClientV2]


class AsyncPeerDriver(Driver):
    """ Asynchronous Peer-to-Peer Driver.
    
    TODO: Explanation of how this thing works.
    """
    def __init__(self, timeline_path: str = None, tmp_path: str = None, results_path: str = "./results.json"):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path
        self.results_path = results_path

        self.train_config: TrainingConfig = None
        self.eval_config: EvalConfig = None

        self.clients: List[AsyncPeerClientU] = [] # type: ignore
        self.evaluators: List[EvalActorProxy] = []
        self.dataset: DataSetSplit = None

    def init_backend(self) -> None:
        self.log.info("Initializing ray backend")
        _init_ray(self.tmp_path)
    
    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        self.log.info(f"Initialing training with config: {train_config}")
        self.train_config = train_config
        self.eval_config = eval_config

        _check_slowness_map(train_config)
        check_eval_config(eval_config, train_config)

        self.dataset = self._init_datasets()
        self.evaluators = self._init_evaluators()
        self.clients = self._init_clients()

    def run(self) -> None: 
        self.log.info("Initiating training")
        start_time = time.time()

        # Start training all of the clients
        [client.train.remote(train_round=1, epochs=self.train_config.num_epochs) for client in self.clients]
        time.sleep(self.train_config.timeout)

        self.log.info("Timeout reached. Stopping clients and evaluators")
        self._stop()

        # Retrieve and save results
        client_results: List[ClientResults] = ray.get([
            client.get_client_results.remote() for client in self.clients
        ])
        self._build_and_save_federated_results(client_results, start_time)

        if self.timeline_path is not None:
            ray.timeline(filename=self.timeline_path)
            self.log.info(f"Timeline saved to {self.timeline_path}")

        self.log.info("Training complete")

    def shutdown(self) -> None:
        """Shutdown the driver."""
        self.log.info("Shutting down driver")
        ray.shutdown()

    def _init_datasets(self) -> DataSetSplit:
        """Initialize the client datasets."""
        self.log.info("Creating datasets")
        dataset_split = _create_datasets(data_path="../Data", num_splits=self.train_config.num_clients)
        self.log.info(f"Dataset size: {dataset_split.split_size}/{dataset_split.full_size}")
        return dataset_split

    def _init_evaluators(self) -> List[EvalActorProxy]:
        self.log.info(f"Initializing {self.eval_config.num_actors} evaluators")

        evaluators = [
            EvalActorProxy(
                eval_id     = i,
                model       = _generate_model(),
                test_loader = self._create_eval_loader(),
            )
            for i in range(self.eval_config.num_actors)
        ]
        [evaluator.initialize(block=True) for evaluator in evaluators]

        return evaluators

    def _init_clients(self) -> List[AsyncPeerClient]:
        """Create and initialize AsyncPeerClients."""
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            AsyncPeerClient.remote(
                client_id=i, 
                model=_generate_model(), 
                train_loader=DataLoader(self.dataset.sets[i], batch_size=self.train_config.batch_size, shuffle=True),
                test_loader=DataLoader(self.dataset.test, batch_size=self.train_config.batch_size, shuffle=False),
                slowness=(self.train_config.slowness_map[i] if self.train_config.slowness_map is not None else 1.0),
                train_config=self.train_config,
            ) 
            for i in range(self.train_config.num_clients)
        ]

        # Make sure that all of the clients are initialized
        ray.get([client.initialize.remote() for client in clients])
        
        # Establish client connections
        for id, client in enumerate(clients):
            clients_to_add = [clients[c] for c, _ in enumerate(clients) if c != id]
            ray.get(client.add_clients.remote(clients_to_add))

        return clients
    
    def _create_train_loader(self, i: int) -> DataLoader:
        """Get a train DataLoader."""
        return DataLoader(self.dataset.sets[i], batch_size=self.train_config.batch_size, shuffle=True)
    
    def _create_eval_loader(self) -> DataLoader:
        """Get a test DataLoader."""
        return DataLoader(self.dataset.test, batch_size=self.train_config.batch_size, shuffle=False)

    def _get_evaluator_for_client(self, client_id: int) -> EvalActorProxy:
        """Get the corresponding evaluator for a client."""
        if self.eval_config.num_actors == 0:
            return None

        assert self.eval_config.client_map is not None

        for e_id, client_list in enumerate(self.eval_config.client_map):
            for c_id in client_list:
                if client_id == c_id:
                    return self.evaluators[e_id]

    def _stop(self) -> None:
        """Wait for all clients and evaluators to stop."""
        ray.get([client.stop.remote() for client in self.clients])
        [e.stop(block=True) for e in self.evaluators]

    def _build_and_save_federated_results(self, client_results: List[ClientResults], start_time: float) -> None:
        """Build and save the FederatedResults from ClientResults."""
        # Ensure that the ClientResults are ordered by client id.
        client_results.sort(key=lambda x: x.client_id)

        federated_results = FederatedResults(
            paradigm       = "AsyncPeerToPeer",
            train_config   = self.train_config,
            eval_config    = self.eval_config,
            g_start_time   = start_time,
            client_results = client_results,
        )

        _federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")


class AsyncPeerDriverV2(AsyncPeerDriver):
    """Asynchronous Peer-to-Peer Driver V2.
    
    Just creates AsyncPeerClientV2 instead of AsyncPeerClient and adds self-reference to the clients.
    """
    def _init_clients(self) -> List[AsyncPeerClientV2]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            AsyncPeerClientV2.remote(
                client_id    = i, 
                model        = _generate_model(), 
                train_loader = self._create_train_loader(i),
                test_loader  = self._create_eval_loader() if self.eval_config.num_actors == 0 else None,
                slowness     = (self.train_config.slowness_map[i] if self.train_config.slowness_map is not None
                                else 1.0),
                train_config = self.train_config,
                eval_config  = self.eval_config, 
                evaluator    = self._get_evaluator_for_client(i)
            ) 
            for i in range(self.train_config.num_clients)
        ]

        # Make sure that all of the clients are initialized
        ray.get([client.initialize.remote() for client in clients])
        
        # Establish client connections
        for id, client in enumerate(clients):
            clients_to_add = [clients[c] for c, _ in enumerate(clients) if c != id]
            ray.get(client.add_clients.remote(clients_to_add))

        # Add self reference
        ray.get([client.add_self_ref.remote(client) for client in clients])

        return clients
