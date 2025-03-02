import time
from typing import List, Union

import ray

from torch.utils.data import DataLoader

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, ClientResults, EvalConfig
from ADFL.eval import EvalActorProxy
from ADFL.Client import AsyncClientProxy
from ADFL.Server import AsyncServer, TraditionalServer

from .common import (
    Driver, DataSetSplit, _init_ray, _check_slowness_map, _check_eval_client_map, _create_datasets, _generate_model,
    _federated_results_to_json
)

AsyncServerU = Union[AsyncServer, TraditionalServer]


class AsyncDriver(Driver):
    """Asynchronous Server-Client Driver.

    TODO: Explanation of how this thing works.
    """
    def __init__(
        self, 
        timeline_path: str = None, 
        tmp_path: str = None, 
        results_path: str = "./results.json", 
        traditional: bool = False
    ):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path
        self.results_path = results_path
        self.use_traditional = traditional

        self.train_config: TrainingConfig = None
        self.eval_config: EvalConfig = None

        self.server: AsyncServerU = None # type: ignore
        self.clients: List[AsyncClientProxy] = []
        self.evaluators: List[EvalActorProxy] = []

        self.dataset: DataSetSplit = None

    def init_backend(self) -> None:
        self.log.info("Initializing ray backend")
        _init_ray(self.tmp_path)

    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        self.log.info("Initialing training")
        self.log.info(f"  Train config: {train_config}")
        self.log.info(f"  Eval config: {eval_config}")

        self.train_config = train_config
        self.eval_config = eval_config

        _check_slowness_map(train_config)
        _check_eval_client_map(eval_config, train_config)

        self.dataset = self._init_datasets()
        self.evaluators = self._init_evaluators()
        self.server = self._init_server()
        self.clients = self._init_clients()

        ray.get(self.server.add_clients.remote(self.clients))

    def run(self) -> None: 
        self.log.info("Initiating training")

        start_time = time.time()
        self.server.run.remote()

        time.sleep(self.train_config.timeout)

        ray.get(self.server.stop.remote())
        [e.stop(block=True) for e in self.evaluators]

        client_results: List[ClientResults] = ray.get(self.server.get_client_results.remote())
        self._build_and_save_federated_results(client_results, start_time)

        if self.timeline_path is not None:
            ray.timeline(filename=self.timeline_path)
            self.log.info(f"Timeline saved to {self.timeline_path}")

        self.log.info("Training complete")

    def shutdown(self) -> None:
        self.log.info("Shutting down driver")
        ray.shutdown()

    def _init_datasets(self) -> DataSetSplit:
        self.log.info("Creating datasets")
        dataset_split = _create_datasets(data_path="../Data", num_splits=self.train_config.num_clients)
        self.log.info(f"Dataset size: {dataset_split.split_size}/{dataset_split.full_size}")
        return dataset_split

    def _init_clients(self) -> List[AsyncClientProxy]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")

        clients = [
            AsyncClientProxy(
                client_id    = i, 
                model        = _generate_model(), 
                train_loader = self._create_train_loader(i),
                test_loader  = self._create_eval_loader() if self.eval_config.num_actors == 0 else None,
                slowness     = (self.train_config.slowness_map[i] if self.train_config.slowness_map is not None \
                                else 1.0),
                eval_config  = self.eval_config,
                server       = self.server,
                evaluator    = self._get_evaluator_for_client(i)
            )
            for i in range(self.train_config.num_clients)
        ]

        # Make sure that all of the clients are initialized
        [client.initialize() for client in clients]

        return clients

    def _init_server(self) -> AsyncServerU: # type: ignore
        self.log.info("Initializing server")

        if self.use_traditional:
            server = TraditionalServer.remote(model_fn=_generate_model, train_config=self.train_config)
        else:
            server = AsyncServer.remote(model_fn=_generate_model, train_config=self.train_config)
        ray.get(server.initialize.remote())

        return server

    def _init_evaluators(self) -> List[EvalActorProxy]:
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

    def _build_and_save_federated_results(self, client_results: List[ClientResults], start_time: float) -> None:
        """Build and save a FederatedResults."""
        federated_results = FederatedResults(
            paradigm       = ("TraditionalServerClient" if self.use_traditional else "AsyncServerClient"),
            train_config   = self.train_config,
            eval_config    = self.eval_config,
            g_start_time   = start_time,
            client_results = client_results,
        )

        _federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")        

