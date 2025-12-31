from typing import Callable, List, Union, Dict, Optional
import random

import torch.nn as nn

import ray
import memray

from ADFL import my_logging
from ADFL.types import EvalConfig, TrainingConfig, AsyncServerResults
from ADFL.messages import AsyncClientTrainMessage, ClientUpdateMessage
from ADFL.model import Parameters, get_model_parameters
from ADFL.flag import RemoteFlag
from ADFL.eval import EvalActorProxy
from ADFL.resources import NUM_CPUS
from ADFL.memory import MEMRAY_PATH, MEMRAY_RECORD

from ADFL.Client import AsyncClientV2, AsyncClientWorkerProxy, DistributedClientPool

from .common import Server, ServerCore


@ray.remote(num_cpus=NUM_CPUS)
class AsyncServer(Server):
    """Asynchronous Server.

    Implementation of an asynchronous federated learning server.
    """
    def __init__(
        self,
        model_fn:     Callable[[], nn.Module],
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    Optional[EvalActorProxy],
        stop_flag:    RemoteFlag,
    ):
        if MEMRAY_RECORD:
            memray.Tracker(f"{MEMRAY_PATH}{self.__class__.__name__}_mem_profile.bin").__enter__()

        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")

        self.core = ServerCore(model_fn, train_config, eval_config, evaluator, stop_flag, self.log)

        # {client_id: worker_id}
        self.client_worker_map: Dict[int, int] = {}


    def initialize(self) -> bool:
        return True


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        self.core.add_clients(clients)


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.core.add_workers(workers)


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        self.core.attach_client_pool(pool)


    def get_results(self) -> AsyncServerResults:
        return self.core.get_results()


    def get_model(self) -> Parameters:
        return self.core.get_model()


    def stop(self) -> None:
        self.core.stop()


    def run(self) -> None:
        self.log.info("Starting training")

        for i, _ in enumerate(self.core.workers):
            client_idx = self.core.select_client()
            self._train_client(i, client_idx, round=1)


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        if self.core.stop_flag.state() == True:
            # This will happen if the driver calls for a stop in the event of a timeout.
            return

        # Simulate receiving the parameters over the communication channel (performs decompression).
        client_params, _ = self.core.channel.on_server_receive(client_update.parameters)

        # Notify the strategy that this client has finished.
        self.core.strategy.on_client_finish(client_update.client_id)

        # Get the worker that was working on this client.
        worker_idx = self.client_worker_map[client_update.client_id]
        self.client_worker_map[client_update.client_id] = -1

        # Stop the worker if we hit the maximum number of global rounds.
        if self.core.g_round > self.core.train_config.num_rounds:
            # Stop the worker without blocking. At this point the worker should be idle and we won't be sending any
            # more requests to it.
            self.core.workers[worker_idx].stop(block=False)
            self.core.end_of_training()

            self.core.finished_workers += 1
            if self.core.finished_workers >= self.core.train_config.num_cur_clients:
                self.log.info("All workers have finished")
                self.stop()

            return

        self.core.g_round += 1
        self.core.save_update(client_update)

        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._aggregate(client_params, client_update.g_round)
        self.core.possibly_eval()

        client_idx = self.core.select_client()
        self._train_client(worker_idx, client_idx, client_update.client_round + 1)


    def _aggregate(self, client_params: Parameters, client_model_step: int) -> None:
        """Aggregate client updates into global model."""
        staleness = self.core.strategy.get_round() - client_model_step
        if self.core.train_config.metrics.staleness:
            self.core.staleness.append(staleness)
        self.core.aggregate([client_params], staleness)


    def _train_client(self, worker_id: int, client_id: int, round: int) -> None:
        """Send a client training job to a worker."""
        self.log.debug(f"Sending training job: client={client_id} worker={worker_id} round={round}")
        self.client_worker_map[client_id] = worker_id

        self.core.train_client(client_id, worker_id)


@ray.remote
class TraditionalServer:
    """Traditional Server.

    Implementation of an actor-based synchronous federated learning server.
    """
    def __init__(
        self,
        model_fn:     Callable[[], nn.Module],
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    Union[EvalActorProxy, None],
        stop_flag:    RemoteFlag,
    ):
        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")

        self.core = ServerCore(model_fn, train_config, eval_config, evaluator, stop_flag, self.log)
        self.updates: List[Parameters] = []

        self.random_client_order = False

        self.train_counter = 0


    def initialize(self) -> bool:
        return True


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        self.core.add_clients(clients)


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.core.add_workers(workers)


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        self.core.attach_client_pool(pool)


    def get_results(self) -> AsyncServerResults:
        return self.core.get_results()


    def get_model(self) -> Parameters:
        return self.core.get_model()


    def stop(self) -> None:
        self.core.stop()


    def run(self) -> None:
        self.log.info("Starting training")
        self._train_round(self.core.strategy.get_round())


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        self.log.debug(f"Received update from Client {client_update.client_id}")
        self.core.save_update(client_update)

        # Notify the strategy that this client has finished
        self.core.strategy.on_client_finish(client_update.client_id)

        # Simulate receiving the parameters over the communication channel (performs decompression)
        params, _ = self.core.channel.on_server_receive(client_update.parameters)

        self.core.g_round += 1
        self.train_counter -= 1
        self.updates.append(params)

        if self.train_counter > 0:
            self.log.debug(f"Waiting for {self.train_counter} more responses")
            return

        self.log.info("Received all client responses. Proceeding to aggregation")
        self.core.aggregate(self.updates, 0)
        self.updates.clear()

        self.core.possibly_eval()

        self._train_round(self.core.strategy.get_round())


    def _train_round(self, train_round: int) -> None:
        """Do a train round."""
        if train_round > self.core.train_config.num_rounds or self.core.stop_flag.state() == True:
            self.stop()
            return

        self.log.info(f"Training Round {train_round}/{self.core.train_config.num_rounds}")

        params = get_model_parameters(self.core.global_model)
        c_params, _ = self.core.channel.on_server_send(params)
        c_params_ref = ray.put(c_params)
        msg = AsyncClientTrainMessage(c_params_ref, self.core.train_config.num_epochs, self.core.strategy.get_round())

        worker_ids = list(range(len(self.core.workers)))
        if self.random_client_order:
            random.shuffle(worker_ids)

        for id in worker_ids:
            client_id = self.core.select_client()
            self._train_client(id, client_id, msg)
            self.train_counter += 1


    def _train_client(self, worker_id: int,  client_id: int, msg: AsyncClientTrainMessage) -> None:
        """Send a train job to a client."""
        self.log.debug(f"Sending out client training job: client={client_id} worker={worker_id}")
        self.core.workers[worker_id].train_client(client_id, msg)
