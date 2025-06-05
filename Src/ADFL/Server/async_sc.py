from typing import Callable, List, Union, Dict

import torch.nn as nn

import ray

from ADFL import my_logging
from ADFL.types import Accuracy, EvalConfig, TrainingConfig, ClientResults, Accuracy
from ADFL.messages import ClientUpdate
from ADFL.model import Parameters, get_model_parameters, set_model_parameters
from ADFL.compression import decompress_params
from ADFL.flag import RemoteFlag
from ADFL.eval import EvalActorProxy

from ADFL.Strategy.base import Strategy, AggregationInfo
from ADFL.Client import AsyncClientProxy

from .common import train_client, need_to_eval, send_eval_message


@ray.remote
class AsyncServer:
    """Asynchronous Server.

    Implementation of an asynchronous federated learning server.
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

        self.clients: List[AsyncClientProxy] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]
        self.finished_clients = 0

        self.ready = False
        self.stop_flag = stop_flag

        assert isinstance(train_config.strategy, Strategy)
        self.strategy: Strategy = train_config.strategy

        # {client_id: initial_round}
        self.client_delay_map: Dict[int, int] = {}

        # Needed to ensure that only on eval happens per round (caused by buffered aggregations)
        self.last_eval_round = -1


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClientProxy]) -> None:
        self.clients += clients


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

        for i, _ in enumerate(self.clients):
            self._train_client(i, round=1)


    def stop(self) -> None:
        if self.stop_flag.state() == True:
            # Already stopped
            return

        self.log.info("Stopping training. Waiting for all clients to finish")
        [client.stop() for client in self.clients]
        self.log.info("All clients have stopped")

        self.stop_flag.set()


    def client_update(self, client_update: ClientUpdate) -> None:
        """Message: Process a client update."""
        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._save_update(client_update)

        self.log.debug(f"Decompressing updates from Client {client_update.client_id}")
        client_params, _ = decompress_params(client_update.parameters)

        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._aggregate(client_params, client_update.client_id)

        self._possibly_eval()

        if self.stop_flag.state() == True:
            return

        if client_update.client_round < self.train_config.num_rounds:
            self._train_client(client_update.client_id, client_update.client_round + 1)
        else:
            self.log.info(f"Client {client_update.client_id} has finished. Stopping it")
            self.clients[client_update.client_id].stop(block=False)

            self.finished_clients += 1
            if self.finished_clients >= self.train_config.num_clients:
                self.log.info("All clients have finished")
                self.stop()


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


    def _train_client(self, client_id: int, round: int) -> None:
        """Send a training job to a client."""
        self.log.debug(f"Sending training job to Client {client_id}: Round {round}")
        self.client_delay_map[client_id] = self.strategy.get_round()
        _, _ = train_client(
            client   = self.clients[client_id],
            model    = self.global_model,
            g_round  = self.strategy.get_round(),
            epochs   = self.train_config.num_epochs,
            method   = self.train_config.compress,
            bits     = self.train_config.quant_lvl_1
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


    def _save_update(self, update: ClientUpdate) -> None:
        """Save a client update."""
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

        self.train_counter = 0
        self.updates: List[ClientUpdate] = []

        self.clients: List[AsyncClientProxy] = []
        self.client_results = [ClientResults(client_id=i) for i in range(train_config.num_clients)]

        self.ready = False
        self.stop_flag = stop_flag

        assert isinstance(train_config.strategy, Strategy)
        self.strategy: Strategy = train_config.strategy


    def initialize(self) -> bool:
        self.ready = True
        return True


    def add_clients(self, clients: List[AsyncClientProxy]) -> None:
        self.clients += clients


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

        self.log.info("Stopping training. Waiting for all clients to finish")
        [client.stop() for client in self.clients]
        self.log.info("All clients have stopped")

        self.stop_flag.set()


    def client_update(self, client_update: ClientUpdate) -> None:
        """Message: Process a client update."""
        self.log.debug(f"Received update from Client {client_update.client_id}")
        self._save_update(client_update)

        self.train_counter -= 1
        self.updates.append(client_update)
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
        self.log.debug(f"Decompressing {len(self.updates)} updates")
        parameters: List[Parameters] = []
        for update in self.updates:
            param, _ = decompress_params(update.parameters)
            parameters.append(param)

        agg_info = AggregationInfo(
            g_params     = get_model_parameters(self.global_model),
            all_c_params = parameters,
            staleness    = 0
        )

        parameters_prime = self.strategy.produce_update(agg_info)
        set_model_parameters(self.global_model, parameters_prime)
        return


    def _train_round(self, train_round: int) -> None:
        """Do a train round."""
        if train_round > self.train_config.num_rounds or self.stop_flag.state() == True:
            self.stop()
            return

        self.log.info(f"Training Round {train_round}/{self.train_config.num_rounds}")

        for id, _ in enumerate(self.clients):
            self._train_client(id)
            self.train_counter += 1


    def _train_client(self, client_id: int) -> None:
        """Send a train job to a client."""
        self.log.debug(f"Sending out traing job to Client {client_id}")
        _, _ = train_client(
            client   = self.clients[client_id],
            model    = self.global_model,
            g_round  = self.strategy.get_round(),
            epochs   = self.train_config.num_epochs,
            method   = self.train_config.compress,
            bits     = self.train_config.quant_lvl_1
        )


    def _possibly_eval(self) -> None:
        """Possibly send a message to the Evaluator."""
        if self.evaluator is None:
            return

        if need_to_eval(self.eval_config.method, self.strategy.get_round(), self.eval_config.threshold):
            self.log.info("Sending evaluation request.")
            send_eval_message(get_model_parameters(self.global_model), 0, self.evaluator)


    def _save_update(self, update: ClientUpdate) -> None:
        self.client_results[update.client_id].rounds.append(update.round_results)

