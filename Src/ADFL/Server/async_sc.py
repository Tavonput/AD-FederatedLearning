from typing import Callable, List, Union, Dict, Optional

import torch.nn as nn

import ray
import memray

from ADFL import my_logging
from ADFL.types import Accuracy, EvalConfig, TrainingConfig, ClientResults, Accuracy, ScalarPair
from ADFL.messages import ClientUpdateMessage
from ADFL.model import Parameters, get_model_parameters, set_model_parameters
from ADFL.flag import RemoteFlag
from ADFL.eval import EvalActorProxy
from ADFL.resources import NUM_CPUS
from ADFL.memory import MEMRAY_PATH, MEMRAY_RECORD

from ADFL.Channel import Channel
from ADFL.Strategy.base import Strategy, AggregationInfo
from ADFL.Client import AsyncClient, AsyncClientWorkerProxy
from ADFL.Client.async_sc import MEASURE_QERROR, MEASURE_MODEL_DIST

from .common import train_client, need_to_eval, send_eval_message


@ray.remote(num_cpus=NUM_CPUS)
class AsyncServer:
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

        self.train_config = train_config
        self.eval_config = eval_config
        self.global_model = model_fn()
        self.evaluator = evaluator

        self.clients: List[AsyncClient] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]

        self.workers: List[AsyncClientWorkerProxy] = []
        self.client_worker_map: Dict[int, int] = {}
        self.finished_workers = 0
        self.g_round = 0

        self.ready = False
        self.stop_flag = stop_flag

        assert isinstance(train_config.strategy, Strategy)
        self.strategy: Strategy = train_config.strategy

        self.channel: Channel = train_config.channel
        self.bandwidth = train_config.delay.server_mbps

        # {client_id: initial_round}
        self.client_delay_map: Dict[int, int] = {}

        # Needed to ensure that only one eval happens per round (caused by buffered aggregations)
        self.last_eval_round = -1

        self.q_errors_mse: List[float] = []
        self.q_errors_cos: List[float] = []
        self.model_dist: List[ScalarPair] = []


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClient]) -> None:
        self.clients += clients


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.workers += workers


    def get_model(self) -> Parameters:
        return get_model_parameters(self.global_model)


    def get_g_rounds(self) -> int:
        return self.g_round


    def get_q_errors_mse(self) -> List[float]:
        return self.q_errors_mse


    def get_q_errors_cos(self) -> List[float]:
        return self.q_errors_cos


    def get_model_dist(self) -> List[ScalarPair]:
        return self.model_dist


    def get_client_results(self) -> List[ClientResults]:
        for i, client in enumerate(self.clients):
            accuracies = client.get_accuracies()
            self.client_results[i].accuracies = accuracies  # type: ignore

        return self.client_results


    def get_accuracies(self) -> List[Accuracy]:
        """Get the global accuracies if possible."""
        if self.evaluator is not None:
            return self.evaluator.get_client_accuracies(0, block=True)  # type: ignore
        return []


    def run(self) -> None:
        self.log.info("Starting training")

        for i, _ in enumerate(self.workers):
            client_idx = self.strategy.select_client(self.train_config.num_clients)
            self._train_client(i, client_idx, round=1)


    def stop(self) -> None:
        if self.stop_flag.state() == True:
            # Already stopped
            return

        self.log.info("Stopping training. Waiting for all workers to finish")
        [worker.stop() for worker in self.workers]
        self.log.info("All workers have stopped")

        self.stop_flag.set()


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        if self.stop_flag.state() == True:
            # This will happen if the driver calls for a stop in the event of a timeout.
            return

        self._save_update(client_update)

        # Simulate receiving the parameters over the communication channel (performs decompression).
        client_params, _ = self.channel.on_server_receive(client_update.parameters)

        # Notify the strategy that this client has finished.
        self.strategy.on_client_finish(client_update.client_id)

        # Get the worker that was working on this client.
        worker_idx = self.client_worker_map[client_update.client_id]
        self.client_worker_map[client_update.client_id] = -1

        # Stop the worker if we hit the maximum number of global rounds.
        self.g_round += 1
        if self.g_round > self.train_config.num_rounds:
            # Stop the worker without blocking. At this point the worker should be idle and we won't be sending any
            # more requests to it.
            self.workers[worker_idx].stop(block=False)

            self.finished_workers += 1
            if self.finished_workers >= self.train_config.num_cur_clients:
                self.log.info("All workers have finished")
                self.stop()

            return

        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._aggregate(client_params, client_update.client_id)
        self._possibly_eval()

        client_idx = self.strategy.select_client(self.train_config.num_clients)
        self._train_client(worker_idx, client_idx, client_update.client_round + 1)


    def _aggregate(self, client_params: Parameters, client_id: int) -> None:
        """Aggregate client updates into global model."""
        agg_info = AggregationInfo(
            g_params     = get_model_parameters(self.global_model),
            all_c_params = [client_params],
            staleness    = self.strategy.get_round() - self.client_delay_map[client_id],
        )
        parameters_prime = self.strategy.produce_update(agg_info)
        set_model_parameters(self.global_model, parameters_prime)
        return


    def _train_client(self, worker_id: int, client_id: int, round: int) -> None:
        """Send a client training job to a worker."""
        self.log.debug(f"Sending training job: client={client_id} worker={worker_id} round={round}")
        self.client_delay_map[client_id] = self.strategy.get_round()
        self.client_worker_map[client_id] = worker_id

        _, _ = train_client(
            client    = self.clients[client_id],
            worker    = self.workers[worker_id],
            params    = get_model_parameters(self.global_model),
            g_round   = self.strategy.get_round(),
            epochs    = self.train_config.num_epochs,
            channel   = self.channel,
            bandwidth = self.bandwidth,
        )


    def _possibly_eval(self) -> None:
        """Possibly send a message to the Evaluator."""
        if self.evaluator is None:
            return

        if (
            need_to_eval(self.eval_config.method, self.strategy.get_round(), self.eval_config.threshold) and
            self.strategy.get_round() != self.last_eval_round
        ):
            self.log.info("Sending evaluation request.")
            send_eval_message(get_model_parameters(self.global_model), 0, self.evaluator)
            self.last_eval_round = self.strategy.get_round()


    def _save_update(self, update: ClientUpdateMessage) -> None:
        """Save a client update."""
        if MEASURE_QERROR:
            self.q_errors_mse.append(update.round_results.q_error_mse)
            self.q_errors_cos.append(update.round_results.q_error_cos)
        if MEASURE_MODEL_DIST:
            self.model_dist.append(update.round_results.model_dist)

        self.client_results[update.client_id].rounds.append(update.round_results)


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

        self.train_config = train_config
        self.eval_config = eval_config
        self.global_model = model_fn()
        self.evaluator = evaluator

        self.g_round = 0
        self.train_counter = 0
        self.updates: List[Parameters] = []

        self.clients: List[AsyncClient] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]

        self.workers: List[AsyncClientWorkerProxy] = []

        self.ready = False
        self.stop_flag = stop_flag

        assert isinstance(train_config.strategy, Strategy)
        self.strategy: Strategy = train_config.strategy

        self.channel: Channel = train_config.channel
        self.bandwidth = train_config.delay.server_mbps

        self.q_errors_mse: List[float] = []
        self.q_errors_cos: List[float] = []
        self.model_dist: List[ScalarPair] = []


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClient]) -> None:
        self.clients += clients


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.workers += workers


    def get_g_rounds(self) -> int:
        return self.g_round


    def get_q_errors_mse(self) -> List[float]:
        return self.q_errors_mse


    def get_q_errors_cos(self) -> List[float]:
        return self.q_errors_cos


    def get_model_dist(self) -> List[ScalarPair]:
        return self.model_dist


    def get_model(self) -> Parameters:
        return get_model_parameters(self.global_model)


    def get_client_results(self) -> List[ClientResults]:
        for i, client in enumerate(self.clients):
            accuracies = client.get_accuracies()
            self.client_results[i].accuracies = accuracies  # type: ignore

        return self.client_results


    def get_accuracies(self) -> List[Accuracy]:
        """Get the global accuracies if possible."""
        if self.evaluator is not None:
            return self.evaluator.get_client_accuracies(0, block=True)  # type: ignore
        return []


    def run(self) -> None:
        self.log.info("Starting training")
        self._train_round(self.strategy.get_round())


    def stop(self) -> None:
        if self.stop_flag.state() == True:
            # Already stopped
            return

        self.log.info("Stopping training. Waiting for all workers to finish")
        [worker.stop() for worker in self.workers]
        self.log.info("All clients have stopped")

        self.stop_flag.set()


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        self.log.debug(f"Received update from Client {client_update.client_id}")
        self._save_update(client_update)

        # Notify the strategy that this client has finished
        self.strategy.on_client_finish(client_update.client_id)

        # Simulate receiving the parameters over the communication channel (performs decompression)
        params, _ = self.channel.on_server_receive(client_update.parameters)

        self.g_round += 1
        self.train_counter -= 1
        self.updates.append(params)

        if self.train_counter > 0:
            self.log.debug(f"Waiting for {self.train_counter} more responses")
            return

        self.log.info("Received all client responses. Proceeding to aggregation")
        self._aggregate()
        self.updates.clear()

        self._possibly_eval()

        self._train_round(self.strategy.get_round())


    def _aggregate(self) -> None:
        """Aggregate client updates into global model."""
        agg_info = AggregationInfo(
            g_params     = get_model_parameters(self.global_model),
            all_c_params = self.updates,
            staleness    = 0
        )

        parameters_prime = self.strategy.produce_update(agg_info)
        set_model_parameters(self.global_model, parameters_prime)


    def _train_round(self, train_round: int) -> None:
        """Do a train round."""
        if train_round > self.train_config.num_rounds or self.stop_flag.state() == True:
            self.stop()
            return

        self.log.info(f"Training Round {train_round}/{self.train_config.num_rounds}")

        for id, _ in enumerate(self.workers):
            client_id = self.strategy.select_client(len(self.clients))
            self._train_client(id, client_id)
            self.train_counter += 1


    def _train_client(self, worker_id: int,  client_id: int) -> None:
        """Send a train job to a client."""
        self.log.debug(f"Sending out client training job: client={client_id} worker={worker_id}")
        _, _ = train_client(
            client    = self.clients[client_id],
            worker    = self.workers[worker_id],
            params    = get_model_parameters(self.global_model),
            g_round   = self.strategy.get_round(),
            epochs    = self.train_config.num_epochs,
            channel   = self.channel,
            bandwidth = self.bandwidth,
        )


    def _possibly_eval(self) -> None:
        """Possibly send a message to the Evaluator."""
        if self.evaluator is None:
            return

        if need_to_eval(self.eval_config.method, self.strategy.get_round(), self.eval_config.threshold):
            self.log.info("Sending evaluation request.")
            send_eval_message(get_model_parameters(self.global_model), 0, self.evaluator)


    def _save_update(self, update: ClientUpdateMessage) -> None:
        if MEASURE_QERROR:
            self.q_errors_mse.append(update.round_results.q_error_mse)
            self.q_errors_cos.append(update.round_results.q_error_cos)
        if MEASURE_MODEL_DIST:
            self.model_dist.append(update.round_results.model_dist)

        self.client_results[update.client_id].rounds.append(update.round_results)

