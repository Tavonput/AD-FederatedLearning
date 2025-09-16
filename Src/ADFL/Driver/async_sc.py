import time
from typing import List, Union, Callable

import ray

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, ClientResults, EvalConfig, Accuracy
from ADFL.model import Parameters
from ADFL.eval import EvalActorProxy
from ADFL.flag import RemoteFlag
from ADFL.dataset import DatasetSplit
from ADFL.Client import AsyncClient, AsyncClientWorkerProxy
from ADFL.Server import AsyncServer, TraditionalServer

from .common import (
    Driver, check_strategy, init_ray, get_delay_map, check_eval_config, create_datasets, generate_model,
    federated_results_to_json
)

AsyncServerU = Union[AsyncServer, TraditionalServer]

POLL_INT = 1


class AsyncDriver(Driver):
    """Asynchronous Server-Client Driver.

    TODO: Explanation of how this thing works.
    """
    def __init__(
        self, 
        timeline_path: str = None,  # type: ignore
        tmp_path: str = None,  # type: ignore
        results_path: str = "./results.json", 
        traditional: bool = False
    ):
        self.log = my_logging.get_logger("DRIVER")

        self.timeline_path = timeline_path
        self.tmp_path = tmp_path
        self.results_path = results_path
        self.use_traditional = traditional

        self.train_config: TrainingConfig = None  # type: ignore
        self.eval_config: EvalConfig = None  # type: ignore

        self.server: AsyncServerU = None  # type: ignore
        self.clients: List[AsyncClient] = []
        self.workers: List[AsyncClientWorkerProxy] = []
        self.evaluators: List[EvalActorProxy] = []

        self.dataset: DatasetSplit = None  # type: ignore

        self.stop_flag: RemoteFlag = None  # type: ignore

    def init_backend(self) -> None:
        self.log.info("Initializing ray backend")
        init_ray(self.tmp_path)


    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        self.log.info("Initialing training")

        self.train_config = train_config
        self.eval_config = eval_config

        assert train_config.num_cur_clients <= train_config.num_clients, "Not supported yet."

        # Get/check the delay map
        self.train_config.delay.delay_map = get_delay_map(train_config.delay, train_config.num_clients)

        check_eval_config(eval_config, train_config)

        assert train_config.strategy is not None
        check_strategy(self.use_traditional, train_config.strategy)

        self.log.info(f"  Train config: {train_config}")
        self.log.info(f"  Eval config: {eval_config}")

        self.log.info("Initializing remote flag")
        self.stop_flag = RemoteFlag()

        self.dataset = self._init_datasets()
        self.evaluators = self._init_evaluators()
        self.server = self._init_server()
        self.clients = self._init_clients()
        self.workers = self._init_workers()

        ray.get(self.server.add_clients.remote(self.clients))
        ray.get(self.server.add_workers.remote(self.workers))


    def run(self) -> None:
        self.log.info("Initiating training")

        start_time = time.time()
        self.server.run.remote()

        server_finished = self.stop_flag.poll(self.train_config.timeout, POLL_INT)
        if server_finished:
            self.log.info("Server has finished all jobs")
        else:
            self.log.info("Timeout reached. Stopping server")
            ray.get(self.server.stop.remote())

        self.log.info("Stopping evaluators")
        [e.stop(block=True) for e in self.evaluators]

        server_accuracies = self._get_server_accuracies()
        client_results: List[ClientResults] = ray.get(self.server.get_client_results.remote())
        self._build_and_savefederated_results(client_results, server_accuracies, start_time)

        finished_model = ray.get(self.server.get_model.remote())
        if self.train_config.model_save is not None:
            torch.save(finished_model, self.train_config.model_save)
            self.log.info(f"Model saved to {self.train_config.model_save}")

        if self.timeline_path is not None:
            ray.timeline(filename=self.timeline_path)
            self.log.info(f"Timeline saved to {self.timeline_path}")

        self.log.info("Training complete")


    def shutdown(self) -> None:
        self.log.info("Shutting down driver")
        ray.shutdown()


    def _init_datasets(self) -> DatasetSplit:
        self.log.info("Creating datasets")
        dataset_split = create_datasets(
            dataset    = self.train_config.dataset,
            data_path  = self.train_config.data_dir,
            train_path = self.train_config.train_file,
            test_path  = self.train_config.test_file,
            num_splits = self.train_config.num_clients,
            iid        = self.train_config.iid,
            alpha      = self.train_config.dirichlet_a
        )
        self.log.info(f"Dataset size: {dataset_split.split_size}/{dataset_split.full_size}")
        return dataset_split


    def _init_clients(self) -> List[AsyncClient]:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")
        assert self.train_config.delay.delay_map is not None

        return [
            AsyncClient(
                client_id     = i,
                model_fn      = lambda: generate_model(self.train_config.dataset, self.train_config.model),
                train_loader  = self._create_train_loader(i),
                test_loader   = self._create_eval_loader() if self.eval_config.num_actors == 0 else None,  # type: ignore
                compute_delay = self.train_config.delay.delay_map[i][0],  # 0 is compute delay
                network_delay = self.train_config.delay.delay_map[i][1],  # 1 is network delay
                train_config  = self.train_config,
                eval_config   = self.eval_config,
                server        = self.server,
                evaluator     = self._get_evaluator_for_client(i)  # type: ignore
            )
            for i in range(self.train_config.num_clients)
        ]


    def _init_server(self) -> AsyncServerU: # type: ignore
        self.log.info("Initializing server")

        evaluator = self.evaluators[0] if self.eval_config.central else None

        if self.use_traditional:
            server = TraditionalServer.remote(
                self._get_model_fn(),
                self.train_config,
                self.eval_config,
                evaluator,
                self.stop_flag,
            )
        else:
            server = AsyncServer.remote(
                self._get_model_fn(),
                self.train_config,
                self.eval_config,
                evaluator,
                self.stop_flag,
            )

        ray.get(server.initialize.remote())  # type: ignore
        return server


    def _init_evaluators(self) -> List[EvalActorProxy]:
        self.log.info(f"Initializing {self.eval_config.num_actors} evaluators")

        evaluators = [
            EvalActorProxy(
                eval_id     = i,
                model_fn    = lambda: generate_model(self.train_config.dataset, self.train_config.model),
                test_loader = self._create_eval_loader(),
            )
            for i in range(self.eval_config.num_actors)
        ]

        [evaluator.initialize(block=True) for evaluator in evaluators]
        return evaluators


    def _init_workers(self) -> List[AsyncClientWorkerProxy]:
        self.log.info(f"Initializing {self.train_config.num_cur_clients} workers")

        workers = [
            AsyncClientWorkerProxy(
                worker_id    = i,
                train_config = self.train_config,
                eval_config  = self.eval_config,
            )
            for i in range(self.train_config.num_cur_clients)
        ]

        [worker.initialize() for worker in workers]
        return workers


    def _create_train_loader(self, i: int) -> DataLoader:
        """Get a train DataLoader."""
        return DataLoader(self.dataset.sets[i], batch_size=self.train_config.batch_size, shuffle=True)


    def _create_eval_loader(self) -> DataLoader:
        """Get a test DataLoader."""
        return DataLoader(self.dataset.test, batch_size=self.train_config.batch_size, shuffle=False)


    def _get_evaluator_for_client(self, client_id: int) -> Union[EvalActorProxy, None]:
        """Get the corresponding evaluator for a client."""
        if self.eval_config.num_actors == 0 or self.eval_config.central == True:
            return None

        assert self.eval_config.client_map is not None

        for e_id, client_list in enumerate(self.eval_config.client_map):
            for c_id in client_list:
                if client_id == c_id:
                    return self.evaluators[e_id]


    def _build_and_savefederated_results(
        self, client_results: List[ClientResults],server_accuracies: List[Accuracy], start_time: float
    ) -> None:
        """Build and save a FederatedResults."""
        federated_results = FederatedResults(
            paradigm       = ("TraditionalServerClient" if self.use_traditional else "AsyncServerClient"),
            train_config   = self.train_config,
            eval_config    = self.eval_config,
            g_start_time   = start_time,
            client_results = client_results,
            c_accuracies   = server_accuracies,
            total_g_rounds = ray.get(self.server.get_g_rounds.remote()),
            q_errors_mse   = ray.get(self.server.get_q_errors_mse.remote()),
            q_errors_cos   = ray.get(self.server.get_q_errors_cos.remote()),
            model_dists    = ray.get(self.server.get_model_dist.remote()),
        )

        federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")


    def _get_server_accuracies(self) -> List[Accuracy]:
        """Get the server accuracies if possible."""
        if self.eval_config.central == True:
            return ray.get(self.server.get_accuracies.remote())
        return []


    def _check_client_models(self) -> None:
        """Check if all clients have the same models."""
        all_params: List[Parameters] = [client.get_model() for client in self.clients]  # type: ignore
        base = all_params[0]

        for other in all_params[1:]:
            assert base.keys() == other.keys()

            for key in base:
                assert torch.equal(base[key], other[key])


    def _get_model_fn(self) -> Callable[[], nn.Module]:
        if self.train_config.model_load is None:
            return lambda: generate_model(self.train_config.dataset, self.train_config.model)
        else:
            self.log.info(f"Setting server model initialization to {self.train_config.model_load}")
            load_path = self.train_config.model_load

            def get_model():
                model = generate_model(self.train_config.dataset, self.train_config.model)
                model.load_state_dict(torch.load(load_path, weights_only=True))
                return model

            return get_model

