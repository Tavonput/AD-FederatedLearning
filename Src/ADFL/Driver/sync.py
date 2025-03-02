from typing import List

import ray

from torch.utils.data import DataLoader, Subset

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults
from ADFL.Client import SyncClient
from ADFL.Server import SyncServer

from .common import (
    Driver, _init_ray, _check_slowness_map, _create_datasets, _generate_model, _federated_results_to_json
)

class SyncDriver(Driver):
    """ Synchronous Server-Client Driver.
    
    TODO: Explanation of how this thing works.
    """
    def __init__(self, timeline_path: str = None, tmp_path: str = None, results_path: str = "./results.json"):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path
        self.results_path = results_path

        self.train_config: TrainingConfig = None

        self.server: SyncServer = None
        self.clients: List[SyncClient] = []
        self.dataset_splits: List[Subset] = []

    def init_backend(self) -> None:
        self.log.info("Initializing ray backend")
        _init_ray(self.tmp_path)
    
    def init_training(self, train_config: TrainingConfig) -> None:
        self.log.info(f"Initialing training with config: {train_config}")
        self.train_config = train_config
        _check_slowness_map(train_config)

        self.dataset_splits = self._init_datasets()
        self.clients = self._init_clients()
        self.server = self._init_server()

    def run(self) -> None: 
        self.log.info("Initiating training")
        federated_results: FederatedResults = ray.get(self.server.run.remote())

        _federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")

        if self.timeline_path is not None:
            ray.timeline(filename=self.timeline_path)
            self.log.info(f"Timeline saved to {self.timeline_path}")

        self.log.info("Training complete")
    
    def shutdown(self) -> None:
        self.log.info("Shutting down driver")
        ray.shutdown()

    def _init_datasets(self) -> List[Subset]:
        self.log.info("Creating datasets")
        dataset_split = _create_datasets(data_path="../Data", num_splits=self.train_config.num_clients)
        self.log.info(f"Dataset size: {dataset_split.split_size}/{dataset_split.full_size}")
        return dataset_split.sets

    def _init_clients(self) -> List[SyncClient]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            SyncClient.remote(
                client_id=i, 
                model=_generate_model(),
                train_loader=DataLoader(self.dataset_splits[i], batch_size=self.train_config.batch_size, shuffle=True),
                slowness=(self.train_config.slowness_map[i] if self.train_config.slowness_map is not None else 1.0),
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
