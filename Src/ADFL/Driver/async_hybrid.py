import time
from typing import List
from itertools import chain

import ray

from torch.utils.data import DataLoader

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, ClientResults, EvalConfig
from ADFL.eval import EvalActorProxy
from ADFL.Client import AsyncClientProxy
from ADFL.Server import AsyncHybridServer

from .common import (
    Driver, DataSetSplit, 
    _init_ray, check_eval_config, _check_slowness_map, _create_datasets, _generate_model, _check_sc_map,
    _federated_results_to_json
)


class AsyncHybridDriver(Driver):
    """ Asynchronous Hybrid Driver.

    TODO: Explanation of how this thing works.
    """
    def __init__(self, timeline_path: str = None, tmp_path: str = None, results_path: str = "./results.json"):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path
        self.results_path = results_path

        self.train_config: TrainingConfig = None
        self.eval_config: EvalConfig = None

        self.clients: List[AsyncClientProxy] = []
        self.servers: List[AsyncHybridServer] = []
        self.evaluators: List[EvalActorProxy] = []
        self.dataset: DataSetSplit = None

        self.server_client_pairings: List[List[int]] = []


    def init_backend(self) -> None:
        """Initialize the backend."""
        self.log.info("Initializing ray backend")
        _init_ray(self.tmp_path)


    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        """Initialize the training setup."""
        self.log.info(f"Initialing training with config: {train_config}")

        self.train_config = train_config
        self.eval_config = eval_config

        _check_slowness_map(train_config)
        _check_sc_map(train_config)
        check_eval_config(eval_config, train_config)

        self.server_client_pairings = self._get_server_client_pairings()

        self.dataset = self._init_datasets()
        self.evaluators = self._init_evaluators()
        self.servers = self._init_servers()
        self.clients = self._init_clients()

        self._add_clients_to_servers()

        ray.get([server.log_node_setup.remote() for server in self.servers])


    def run(self) -> None: 
        """Run federated learning."""
        self.log.info("Initiating training")
        start_time = time.time()

        [server.run.remote() for server in self.servers]
        time.sleep(self.train_config.timeout)

        # Note that the servers will stop their clients
        ray.get([server.stop.remote() for server in self.servers])
        [e.stop(block=True) for e in self.evaluators]

        # Get the training results
        client_results: List[List[ClientResults]] = ray.get([
            server.get_client_results.remote() for server in self.servers
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
        """Create the datasets."""
        self.log.info("Creating datasets")
        dataset_split = _create_datasets(data_path="../Data", num_splits=self.train_config.num_clients)
        self.log.info(f"Dataset size: {dataset_split.split_size}/{dataset_split.full_size}")
        return dataset_split


    def _init_evaluators(self) -> List[EvalActorProxy]:
        """Create and initialize the evaluators."""
        self.log.info(f"Initializing {self.eval_config.num_actors} evaluators")

        evaluators = [
            EvalActorProxy(
                eval_id=i,
                model=_generate_model(),
                test_loader=self._create_eval_loader(),
            )
            for i in range(self.eval_config.num_actors)
        ]
        [evaluator.initialize(block=True) for evaluator in evaluators]

        return evaluators


    def _init_clients(self) -> List[AsyncClientProxy]:
        """Create and initialize the clients."""
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients: List[AsyncClientProxy] = []
        for i in range(self.train_config.num_clients):
            clients.append(
                AsyncClientProxy(
                    client_id    = i, 
                    model        = _generate_model(), 
                    train_loader = self._create_train_loader(i),
                    test_loader  = self._create_eval_loader() if self.eval_config.num_actors == 0 else None,
                    slowness     = (self.train_config.slowness_map[i] if self.train_config.slowness_map is not None \
                                    else 1.0),
                    train_config = self.train_config,
                    eval_config  = self.eval_config, 
                    server       = self._get_server_for_client(i),
                    evaluator    = self._get_evaluator_for_client(i),
                )
            )

        # Make sure that all of the clients are initialized
        ray.get([client.initialize(block=False) for client in clients])

        return clients


    def _init_servers(self) -> List[AsyncHybridServer]: 
        """Create and initialize the servers."""
        self.log.info("Initializing servers")

        servers = [
            AsyncHybridServer.remote(
                server_id=i, 
                model_fn=_generate_model, 
                train_config=self.train_config
            )
            for i in range(self.train_config.num_servers)  
        ]

        # Make sure that all of the servers are initialized 
        ray.get([server.initialize.remote() for server in servers])

        # Establish server connections
        for id, server in enumerate(servers):
            servers_to_add = [servers[s] for s, _ in enumerate(servers) if s != id]
            ray.get(server.add_servers.remote(servers_to_add))

        return servers


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


    def _get_server_for_client(self, client_id: int) -> AsyncHybridServer:
        """Get the corresponding server for a client.
        
        The current implementation is just a linear search, but you could probably make it constant.
        """
        assert len(self.servers) == self.train_config.num_servers

        for s_id, clients in enumerate(self.server_client_pairings):
            for c_id in clients:
                if client_id == c_id:
                    return self.servers[s_id]
            

    def _get_server_client_pairings(self) -> List[List[int]]:
        """Get the server-client pairings.
        
        Example: If num_servers=3 and num_clients=7, then pairings will be [ [0, 1, 2], [3, 4], [5, 6] ].
        """
        if self.train_config.sc_map is not None:
            return self.train_config.sc_map

        pairings = []
        clients = list(range(self.train_config.num_clients))
        base_size = self.train_config.num_clients // self.train_config.num_servers
        extra_clients = self.train_config.num_clients % self.train_config.num_servers

        start = 0
        for i in range(self.train_config.num_servers):
            size = base_size + (1 if i < extra_clients else 0)
            pairings.append(clients[start:start + size])
            start += size

        return pairings


    def _add_clients_to_servers(self) -> None:
        """Add the clients to the servers."""
        for s_id, server in enumerate(self.servers):
            clients_to_add = [self.clients[c_id] for c_id in self.server_client_pairings[s_id]]
            ray.get(server.add_clients.remote(clients_to_add))


    def _build_and_save_federated_results(self, client_results: List[List[ClientResults]], start_time: float) -> None:
        """Build and save the FederatedResults from ClientResults."""
        all_client_results = list(chain.from_iterable(client_results))

        # Ensure that the ClientResults are ordered by client id.
        all_client_results.sort(key=lambda x: x.client_id)

        federated_results = FederatedResults(
            paradigm       = "AsyncHybrid",
            train_config   = self.train_config,
            eval_config    = self.eval_config,
            g_start_time   = start_time,
            client_results = all_client_results,
        )

        _federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")
