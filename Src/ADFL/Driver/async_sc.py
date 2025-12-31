import time
from typing import List, Union, Callable

import ray

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ADFL import my_logging
from ADFL.types import TrainingConfig, FederatedResults, EvalConfig, AsyncServerResults
from ADFL.model import Parameters
from ADFL.eval import EvalActorProxy
from ADFL.flag import RemoteFlag
from ADFL.dataset import DatasetSplit
from ADFL.Client import AsyncClientV2, AsyncClientWorkerProxy, DistributedClientPool
from ADFL.Server import ServerType, ServerProxy

from .common import (
    Driver, check_strategy, init_ray, get_delay_map, check_eval_config, create_datasets, generate_model,
    federated_results_to_json
)


POLL_INT = 1


class AsyncDriver(Driver):
    """Asynchronous Server-Client Driver.

    Initializes all resources and actors. Makes the call to start the server. Establishes a stop flag between the server
    and periodically polls it to see if the server is done, otherwise the driver the send a stop request to the server
    upon timeout.

    The driver does not own the clients. The driver will create the clients and push them to client pools actors. The
    driver does maintain all actor references.
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
        if self.use_traditional:
            self.server_type = ServerType.TRADITIONAL
        else:
            self.server_type = ServerType.ASYNC

        self.train_config: TrainingConfig = None  # type: ignore
        self.eval_config: EvalConfig = None  # type: ignore

        self.server: ServerProxy = None  # type: ignore
        self.client_pool: DistributedClientPool = None  # type: ignore
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
        self.client_pool = self._init_clients()
        self.workers = self._init_workers()

        self.server.add_workers(self.workers, block=True)
        self.server.attach_client_pool(self.client_pool)


    def run(self) -> None:
        self.log.info("Initiating training")

        start_time = time.time()
        self.server.run()

        server_finished = self.stop_flag.poll(self.train_config.timeout, POLL_INT)
        if server_finished:
            self.log.info("Server has finished all jobs")
        else:
            self.log.info("Timeout reached. Stopping server")
            self.server.stop(block=True)

        self._stop_evaluators()

        server_results: AsyncServerResults = self.server.get_results(block=True)
        self._build_and_savefederated_results(server_results, start_time)

        if self.train_config.model_save is not None:
            torch.save(server_results.model, self.train_config.model_save)
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


    def _init_clients(self) -> DistributedClientPool:
        self.log.info(f"Initializing {self.train_config.num_clients} clients")
        assert self.train_config.delay.delay_map is not None

        clients = [
            AsyncClientV2(
                client_id     = i,
                train_loader  = self._create_train_loader(i),
                test_loader   = self._create_eval_loader() if self.eval_config.num_actors == 0 else None,  # type: ignore
                compute_delay = self.train_config.delay.delay_map[i][0],  # 0 is compute delay
                network_delay = self.train_config.delay.delay_map[i][1],  # 1 is network delay
                server        = self.server,  # type: ignore
                evaluator     = self._get_evaluator_for_client(i)  # type: ignore
            )
            for i in range(self.train_config.num_clients)
        ]

        client_pool = DistributedClientPool(
            self.train_config.num_clients, self.train_config.num_client_pools, self.log
        )
        client_pool.init_pools(clients)

        return client_pool


    def _init_server(self) -> ServerProxy:
        self.log.info("Initializing server")

        evaluator = self.evaluators[0] if self.eval_config.central else None

        server = ServerProxy(
            self.server_type,
            self._get_model_fn(),
            self.train_config,
            self.eval_config,
            evaluator,
            self.stop_flag,
        )

        server.initialize()
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
                model_fn     = self._get_model_fn(),
            )
            for i in range(self.train_config.num_cur_clients)
        ]

        for worker in workers:
            worker.initialize()
            worker.attach_client_pool(self.client_pool)

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
        self, server_results: AsyncServerResults, start_time: float
    ) -> None:
        """Build and save a FederatedResults."""
        worker_results = [worker.get_metrics() for worker in self.workers]

        federated_results = FederatedResults(
            paradigm        = (ServerType.TRADITIONAL.value if self.use_traditional else ServerType.ASYNC.value),
            train_config    = self.train_config,
            eval_config     = self.eval_config,
            g_start_time    = start_time,
            g_end_time      = server_results.g_end_time,
            client_results  = server_results.client_results,
            c_accuracies    = server_results.accuracies,
            total_g_rounds  = server_results.g_rounds,
            q_errors_mse    = server_results.q_errors_mse,
            q_errors_cos    = server_results.q_errors_cos,
            model_dists     = server_results.model_dist,
            trainer_results = worker_results,
            staleness       = server_results.staleness,
        )

        federated_results_to_json(self.results_path, federated_results)
        self.log.info(f"FederatedResults saved to {self.results_path}")


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


    def _stop_evaluators(self) -> None:
        """Pool evaluators until they have completed all evaluation requests."""
        worker_results = [w.get_metrics() for w in self.workers]
        num_evals = sum([wr.num_eval_req for wr in worker_results])

        self.log.info("Stopping evaluators...")

        # Poll the evaluators until they are done
        while True:
            num_req_completed = sum(
                [e.get_num_evals_completed() for e in self.evaluators]
            )
            self.log.debug(f"Polled evaluators. {num_req_completed}/{num_evals} completed")

            if num_req_completed >= num_evals:
                break

            time.sleep(POLL_INT)

        [e.stop(block=True) for e in self.evaluators]
