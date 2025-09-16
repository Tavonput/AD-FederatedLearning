import time
from copy import deepcopy
from typing import List, Union, Tuple, Callable, Optional, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, AdamW

import ray

from ADFL import my_logging
from ADFL.types import RoundResults, Accuracy, EvalConfig, TrainingConfig, TrainResults
from ADFL.messages import AsyncClientTrainMessage, ClientUpdateMessage, EvalMessage
from ADFL.eval import EvalActorProxy
from ADFL.model import (
    Parameters, get_model_parameters, parameter_mean_var, set_model_parameters, parameter_relative_mse,
    parameter_mse, parameter_cosine_similarity
)
from ADFL.Channel import Channel
from ADFL.Strategy.base import Strategy, CommType

from .common import LR, train_epoch, evaluate, AsyncServer, diff_parameters


MEASURE_QERROR = False
MEASURE_MODEL_DIST = False


class AsyncClient:
    def __init__(
        self,
        client_id:     int,
        model_fn:      Callable[[], nn.Module],
        train_loader:  DataLoader,
        test_loader:   DataLoader,
        compute_delay: float,
        network_delay: float,
        train_config:  TrainingConfig,
        eval_config:   EvalConfig,
        server:        "AsyncServer",
        evaluator:     EvalActorProxy,
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.debug("Initializing")

        self.device: torch._C.device = torch.device("cpu")

        self.client_id = client_id
        self.server = server
        self.evaluator = evaluator

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.compute_delay = compute_delay
        self.network_delay = network_delay
        self.round = 0

        self.train_config = train_config
        self.eval_config = eval_config
        self.global_start = time.time()
        self.last_eval_interval = -1

        self.model_fn = model_fn
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

        self.accuracies: List[Accuracy] = []

        self.stop_flag = False
        self.ready = True

        assert isinstance(train_config.strategy, Strategy)
        self.comm_type = train_config.strategy.get_comm_type()

        self.channel: Channel = train_config.channel


    def init_model(self, model_fn: Callable[[], nn.Module], params: Parameters) -> None:
        """Initialize the model with a set of parameters."""
        self.model = model_fn()
        set_model_parameters(self.model, params)
        self.model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.ready = True


    def get_id(self) -> int:
        """Get the client id."""
        return self.client_id


    def get_accuracies(self) -> List[Accuracy]:
        """Get the accuracies."""
        if self.evaluator is None:
            return self.accuracies
        return self.evaluator.get_client_accuracies(self.client_id, block=True)  # type: ignore


    def get_model(self) -> Parameters:
        """Get the model parameters."""
        if self.model is None:
            return {}

        return get_model_parameters(self.model)


    def stop(self) -> None:
        """Stop working."""
        if self.stop_flag == True:
            return

        self.log.info("Stopping")
        self.stop_flag = True


    def train(self, msg: AsyncClientTrainMessage, device: torch._C.device) -> None:
        """Message: Train a federated learning round."""
        if self.stop_flag:
            return

        start_time = time.time()
        self.device = device

        # Initialize the model structure and optimizer
        self.model = self.model_fn()
        self.optimizer = SGD(self.model.parameters(), lr=LR, weight_decay=0.001)
        # self.optimizer = AdamW(self.model.parameters(), lr=LR)

        # Simulate parameter transfer across the communication channel
        params, d_time = self.channel.on_client_receive(msg.parameters)
        set_model_parameters(self.model, params)

        self._possibly_eval()

        self.round += 1
        self.log.debug(f"Starting training round {self.round}")

        round_results = RoundResults()
        round_results.g_start_time = start_time
        round_results.epochs = msg.epochs
        round_results.train_round = self.round

        if self.comm_type == CommType.NORMAL:
            round_results.train_results, parameters_prime = self._train_normal(msg.epochs)

        elif self.comm_type == CommType.DELTA:
            params_prev = get_model_parameters(self.model)

            round_results.train_results, parameters_prime = self._train_normal(msg.epochs)
            round_results.mse = parameter_mse(params_prev, parameters_prime, exclude_bias=False)

            parameters_prime = diff_parameters(params_prev, parameters_prime)
        else:
            assert False, f"Encountered an invalid CommType"

        self.log.debug("Finished training. Sending updated model to server")
        self._send_update_to_server(parameters_prime, round_results, d_time)

        self.optimizer = None
        self.model = None


    def _train_normal(self, epochs: int) -> Tuple[List[TrainResults], Parameters]:
        assert self.model is not None
        assert self.optimizer is not None

        results: List[TrainResults] = []
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            epoch_results = train_epoch(
                self.model, self.optimizer, self.criterion, self.train_loader, self.device
            )
            results.append(epoch_results)

        return results, get_model_parameters(self.model)


    def _train_diff(self, epochs: int) -> Tuple[List[TrainResults], Parameters]:
        assert self.model is not None
        assert self.optimizer is not None

        model_clone = deepcopy(self.model)
        self.optimizer = SGD(model_clone.parameters(), lr=LR, weight_decay=0.001)

        results: List[TrainResults] = []
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            epoch_results = train_epoch(
                model_clone, self.optimizer, self.criterion, self.train_loader, self.device
            )
            results.append(epoch_results)

        self.model.to("cpu")
        model_clone.to("cpu")
        diff = diff_parameters(get_model_parameters(self.model), get_model_parameters(model_clone))

        return results, diff


    def _send_update_to_server(
        self, parameters: Parameters, round_results: RoundResults, d_time: float
    ) -> None:
        """Send a round update to the server."""
        assert self.model is not None
        self.model.to("cpu")

        # Simulate parameter transfer across the communication channel (performs compression)
        c_params, c_time = self.channel.on_client_send(parameters)

        # Simulate delays
        compute_delay, compute_time = self._simulate_compute_delay(d_time, c_time, round_results)
        network_delay = self._simulate_network_delay()

        round_results.compute_time = compute_delay + compute_time
        round_results.network_time = network_delay
        round_results.round_time = time.time() - round_results.g_start_time

        if MEASURE_QERROR:
            d_params, _ = self.channel.on_server_receive(c_params)
            round_results.q_error_mse = parameter_relative_mse(parameters, d_params, exclude_bias=True)
            round_results.q_error_cos = parameter_cosine_similarity(parameters, d_params, exclude_bias=True)

        if MEASURE_MODEL_DIST:
            round_results.model_dist = parameter_mean_var(parameters, exclude_bias=True)

        update = ClientUpdateMessage(
            parameters    = c_params,
            client_id     = self.client_id,
            client_round  = self.round,
            round_results = round_results,
            num_examples  = len(self.train_loader.dataset),  # type: ignore
        )

        self.server.client_update.remote(update)  # type: ignore


    def _possibly_eval(self) -> None:
        """Evaluate if needed or send the job to an evaluator."""
        if self._need_to_eval() is False or self.eval_config.central is True:
            return

        if self.evaluator is None:
            self._self_eval()
        else:
            self._send_eval_message()


    def _need_to_eval(self) -> bool:
        """Check if we need to eval."""
        if self.eval_config.method == "time":
            current_eval_interval = (time.time() - self.global_start) // self.eval_config.threshold
            return (current_eval_interval > self.last_eval_interval)

        elif self.eval_config.method == "round":
            return (self.round % self.eval_config.threshold == 0)

        else:
            assert False, "This should not happen"


    def _self_eval(self) -> None:
        """Perform a self evaluation if needed."""
        assert self.model is not None

        accuracy = 0

        if self.eval_config.method == "time":
            current_eval_interval = (time.time() - self.global_start) // self.eval_config.threshold
            self.log.debug(f"Evaluating interval: {current_eval_interval}")
            self.last_eval_interval = current_eval_interval

        elif self.eval_config.method == "round":
            self.log.debug(f"Evaluating round: {self.round}")

        accuracy = evaluate(self.model, self.test_loader, self.device)
        self.accuracies.append(Accuracy(accuracy, time.time()))

        self.log.warning(accuracy)


    def _send_eval_message(self) -> None:
        """Send an evaluation message to the evaluator if needed."""
        assert self.model is not None

        self.log.debug("Sending evaluation request.")

        message = EvalMessage(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            g_time=time.time(),
        )
        self.evaluator.evaluate(message)


    def _simulate_compute_delay(
        self, decompress_time: float, compress_time: float, round_results: RoundResults
    ) -> Tuple[float, float]:
        """Simulate compute delay."""
        train_time = 0
        for train_result in round_results.train_results:
            train_time += train_result.elapsed_time

        total_compute_time = train_time + decompress_time + compress_time
        delay = total_compute_time * self.compute_delay
        time.sleep(delay)

        return delay, total_compute_time


    def _simulate_network_delay(self) -> float:
        """Simulate network delay."""
        assert self.model is not None
        if self.train_config.delay.network_sigma is not None:
            return self.channel.simulate_bandwidth(get_model_parameters(self.model), self.network_delay)
        else:
            return 0


class AsyncClientProxy:
    """Proxy for interacting with an AsyncClient ray actor.

    Deprecated. The AsyncClient is no longer a ray actor.
    """
    def __init__(
        self, 
        client_id:    int,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        slowness:     float,
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        server:       "AsyncServer",
        evaluator:    EvalActorProxy,
    ):
        self.actor: Any = None
        assert False, "AsyncClientProxy is deprecated"

        self.client_id = client_id
        self.actor = AsyncClient.remote(
            client_id=client_id,  # type: ignore
            train_loader=train_loader,
            test_loader=test_loader,
            slowness=slowness,
            train_config=train_config,
            eval_config=eval_config,
            server=server,
            evaluator=evaluator,
        )


    def init_model(self, model_fn: Callable[[], nn.Module], params: Parameters) -> None:
        ray.get(self.actor.init_model.remote(model_fn, params))


    def get_id(self, block: bool = True) -> Union[int, ray.ObjectRef]:
        if block:
            return ray.get(self.actor.get_id.remote())
        else:
            return self.actor.get_id.remote()


    def get_accuracies(self, block: bool = True) -> Union[List[Accuracy], ray.ObjectRef]:
        if block:
            return ray.get(self.actor.get_accuracies.remote())
        else:
            return self.actor.get_accuracies.remote()


    def get_model(self, block: bool = True) -> Union[Parameters, ray.ObjectRef]:
        if block:
            return ray.get(self.actor.get_model.remote())
        else:
            return self.actor.get_model.remote()


    def stop(self, block: bool = True) -> ray.ObjectRef:
        if block:
            return ray.get(self.actor.stop.remote())
        else:
            return self.actor.stop.remote()


    def train(self, msg: AsyncClientTrainMessage) -> ray.ObjectRef:
        return self.actor.train.remote(msg)

