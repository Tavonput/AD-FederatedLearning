import time
import logging
from typing import Tuple, Optional, List, Callable
from abc import ABC, abstractmethod

import torch.nn as nn

from ADFL.messages import AsyncClientTrainMessage, ClientUpdateMessage, EvalMessage
from ADFL.types import Scalar, AsyncServerResults, TrainingConfig, EvalConfig, ClientResults, ScalarPair
from ADFL.model import Parameters, get_model_parameters, set_model_parameters
from ADFL.eval import EvalActorProxy
from ADFL.flag import RemoteFlag
from ADFL.Strategy import Strategy, AggregationInfo

from ADFL.Channel import Channel
from ADFL.Client import AsyncClientV2, AsyncClientWorkerProxy, DistributedClientPool


class Server(ABC):
    """Base Server Interface.

    Notice that clients and workers are passed to the Server by function calls rather during initialization. This is to
    achieve circular references between ray actor handles.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass


    @abstractmethod
    def initialize(self) -> bool:
        """Different from __init__. Used to make sure the server actor has been initialized. Usually does nothing."""
        pass


    @abstractmethod
    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        """Attach clients to the server."""
        pass


    @abstractmethod
    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        """Attach client workers to the server."""
        pass


    @abstractmethod
    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        """Attach a distributed client pool."""
        pass


    @abstractmethod
    def get_results(self) -> AsyncServerResults:
        """Get the server results."""
        pass


    @abstractmethod
    def get_model(self) -> Parameters:
        """Get the model parameters."""
        pass


    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""
        pass


    @abstractmethod
    def run(self) -> None:
        """Run the server asynchronously."""
        pass


    @abstractmethod
    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        pass


class ServerCore:
    """Server Core.

    Common server fields and functionality.
    """
    def __init__(
        self,
        model_fn:     Callable[[], nn.Module],
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    Optional[EvalActorProxy],
        stop_flag:    RemoteFlag,
        logger:       logging.Logger,
    ) -> None:
        self.log = logger

        self.train_config = train_config
        self.eval_config = eval_config
        self.global_model = model_fn()
        self.evaluator = evaluator

        self.clients: List[AsyncClientV2] = []  # Not used anymore
        self.client_pool: Optional[DistributedClientPool] = None
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]

        self.workers: List[AsyncClientWorkerProxy] = []
        self.finished_workers = 0
        self.g_round = 0

        self.stop_flag = stop_flag

        assert isinstance(train_config.strategy, Strategy)
        self.strategy: Strategy = train_config.strategy

        self.channel: Channel = train_config.channel
        self.bandwidth = train_config.delay.server_mbps

        # Needed to ensure that only one eval happens per round (caused by buffered aggregations)
        self.last_eval_round = -1

        self.q_errors_mse: List[float] = []
        self.q_errors_cos: List[float] = []
        self.model_dist: List[ScalarPair] = []
        self.staleness: List[int] = []

        self.g_end_time = float("inf")


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        """Deprecated."""
        assert False, (
            "ServerCore.add_clients is deprecated. The server should not own the clients." +
            "Use ServerCore.attach_client_pool instead."
        )
        self.clients += clients


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.workers += workers


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        self.client_pool = pool
        self.client_pool.set_logger(self.log)


    def stop(self) -> None:
        if self.stop_flag.state() == True:
            # Already stopped
            return

        self.end_of_training()

        self.log.info("Stopping training. Waiting for all workers to finish")
        [worker.stop(block=True) for worker in self.workers]
        self.log.info("All workers have stopped")

        self.stop_flag.set()


    def end_of_training(self) -> None:
        self.g_end_time = min(self.g_end_time, time.time())


    def get_results(self) -> AsyncServerResults:
        results = AsyncServerResults()
        results.model = get_model_parameters(self.global_model)
        results.model_dist = self.model_dist
        results.g_rounds = self.g_round
        results.q_errors_cos = self.q_errors_cos
        results.q_errors_mse = self.q_errors_mse
        results.staleness = self.staleness
        results.g_end_time = self.g_end_time

        # Retrieve local client accuracy.
        assert self.client_pool is not None
        clients = self.client_pool.get_all_clients_meta()
        for i, client in enumerate(clients):
            accuracies = client.get_accuracies()
            self.client_results[i].accuracies = accuracies  # type: ignore

        results.client_results = self.client_results

        if self.evaluator is not None:
            results.accuracies = self.evaluator.get_client_accuracies(0, block=True)  # type: ignore

        return results


    def get_model(self) -> Parameters:
        return get_model_parameters(self.global_model)


    def possibly_eval(self) -> None:
        """Possibly send a message to the Evaluator."""
        if self.evaluator is None:
            return

        if (
            _need_to_eval(self.eval_config.method, self.strategy.get_round(), self.eval_config.threshold) and
            self.strategy.get_round() != self.last_eval_round
        ):
            self.log.info("Sending evaluation request.")
            _send_eval_message(get_model_parameters(self.global_model), 0, self.evaluator)
            self.last_eval_round = self.strategy.get_round()


    def save_update(self, update: ClientUpdateMessage) -> None:
        """Save a client update."""
        if self.train_config.metrics.q_error:
            self.q_errors_mse.append(update.round_results.q_error_mse)
            self.q_errors_cos.append(update.round_results.q_error_cos)
        if self.train_config.metrics.model_dist:
            self.model_dist.append(update.round_results.model_dist)

        self.client_results[update.client_id].rounds.append(update.round_results)


    def select_client(self) -> int:
        """Select a client idx from the strategy."""
        return self.strategy.select_client(self.train_config.num_clients)


    def aggregate(self, updates: List[Parameters], staleness: int) -> None:
        """Aggregate and update the global model."""
        agg_info = AggregationInfo(
            g_params     = get_model_parameters(self.global_model),
            all_c_params = updates,
            staleness    = staleness,
        )
        parameters_prime = self.strategy.produce_update(agg_info)
        set_model_parameters(self.global_model, parameters_prime)


    def train_client(
        self,
        client_id: int,
        worker_id: int,
        activate_bandwidth: bool = False,
    ) -> Tuple[float, float]:
        """Train an AsyncClient and get the compression and bandwidth times."""
        params = get_model_parameters(self.global_model)

        c_params, c_time = self.channel.on_server_send(params)
        msg = AsyncClientTrainMessage(c_params, self.train_config.num_epochs, self.strategy.get_round())

        start_time = time.time()

        if activate_bandwidth:
            assert self.bandwidth is not None
            _ = self.channel.simulate_bandwidth(params, self.bandwidth)

        self.workers[worker_id].train_client(client_id, msg)

        b_time = time.time() - start_time
        return c_time, b_time


def _need_to_eval(method: str, g_round: int, threshold: Scalar) -> bool:
    """Check if we need to eval."""
    if method == "time":
        assert False, "Central time eval is not supported"

    elif method == "round":
        return (g_round % threshold == 0)

    else:
        assert False, "Invalid eval config method"


def _send_eval_message(params: Parameters, s_id: int, evaluator: EvalActorProxy) -> None:
    """Send an evaluation message to the evaluator."""
    message = EvalMessage(
        parameters = params,
        client_id  = s_id,
        g_time     = time.time(),
    )
    evaluator.evaluate(message)
