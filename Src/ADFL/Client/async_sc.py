import time
from copy import deepcopy
from typing import List, Union, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer

import ray

from ADFL import my_logging
from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import RoundResults, Accuracy, EvalConfig, TrainingConfig
from ADFL.messages import AsyncClientTrainMessage, ClientUpdate, EvalMessage
from ADFL.eval import EvalActorProxy
from ADFL.model import Parameters, get_model_parameters, set_model_parameters, parameter_mse
from ADFL.compression import decompress_params, compress_params

from ADFL.Strategy.base import Strategy, CommType

from .common import LR, _train_epoch, _evaluate, AsyncServer, diff_parameters


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncClient:
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
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

        self.client_id = client_id
        self.server = server
        self.evaluator = evaluator

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.slowness = slowness
        self.round = 0

        self.train_config = train_config
        self.eval_config = eval_config
        self.global_start = time.time()
        self.last_eval_interval = -1

        self.model: nn.Module = None  # type: ignore
        self.optimizer: Optimizer = None  # type: ignore
        self.criterion = nn.CrossEntropyLoss()

        self.accuracies: List[Accuracy] = []

        self.stop_flag = False
        self.ready = False

        assert isinstance(train_config.strategy, Strategy)
        self.comm_type = train_config.strategy.get_comm_type()


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
        return get_model_parameters(self.model)


    def stop(self) -> None:
        """Stop working."""
        if self.stop_flag == True:
            return

        self.log.info("Stopping")
        self.stop_flag = True


    def train(self, msg: AsyncClientTrainMessage) -> None:
        """Message: Train a federated learning round."""
        assert self.ready == True
        if self.stop_flag:
            return

        self.log.debug(f"Decompressing model")
        params, _ = decompress_params(msg.parameters)
        set_model_parameters(self.model, params)

        self._possibly_eval()

        self.round += 1
        self.log.debug(f"Starting training round {self.round}")
        start_time = time.time()

        if self.comm_type == CommType.NORMAL:
            round_results, parameters_prime = self._train_normal(msg.epochs)

        elif self.comm_type == CommType.DELTA:
            params_prev = get_model_parameters(self.model)

            round_results, parameters_prime = self._train_normal(msg.epochs)
            round_results.mse = parameter_mse(params_prev, parameters_prime)

            parameters_prime = diff_parameters(params_prev, parameters_prime)
        else:
            assert False, f"Encountered an invalid CommType"

        round_results.round_time = time.time() - start_time
        round_results.g_start_time = start_time
        round_results.epochs = msg.epochs
        round_results.train_round = self.round

        self.log.debug("Finished training. Sending updated model to server")
        self._send_update_to_server(parameters_prime, round_results)


    def _train_normal(self, epochs: int) -> Tuple[RoundResults, Parameters]:
        round_results = RoundResults()
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            results = _train_epoch(
                self.model, self.optimizer, self.criterion, self.train_loader, self.device, self.slowness
            )
            round_results.train_results.append(results)
        return round_results, get_model_parameters(self.model)


    def _train_diff(self, epochs: int) -> Tuple[RoundResults, Parameters]:
        model_clone = deepcopy(self.model)
        self.optimizer = SGD(model_clone.parameters(), lr=LR)

        round_results = RoundResults()
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            results = _train_epoch(
                model_clone, self.optimizer, self.criterion, self.train_loader, self.device, self.slowness
            )
            round_results.train_results.append(results)

        self.model.to("cpu")
        model_clone.to("cpu")
        diff = diff_parameters(get_model_parameters(self.model), get_model_parameters(model_clone))

        return round_results, diff


    def _send_update_to_server(self, parameters: Parameters, round_results: RoundResults) -> None:
        """Send a round update to the server."""
        self.model.to("cpu")

        self.log.debug("Compressing model")
        c_params, c_time = compress_params(parameters, self.train_config.compress, self.train_config.quant_lvl_1)

        update = ClientUpdate(
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
        accuracy = 0

        if self.eval_config.method == "time":
            current_eval_interval = (time.time() - self.global_start) // self.eval_config.threshold
            self.log.debug(f"Evaluating interval: {current_eval_interval}")
            self.last_eval_interval = current_eval_interval

        elif self.eval_config.method == "round":
            self.log.debug(f"Evaluating round: {self.round}")

        accuracy = _evaluate(self.model, self.test_loader, self.device)
        self.accuracies.append(Accuracy(accuracy, time.time()))

        self.log.warning(accuracy)


    def _send_eval_message(self) -> None:
        """Send an evaluation message to the evaluator if needed."""
        self.log.debug("Sending evaluation request.")

        message = EvalMessage(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            g_time=time.time(),
        )
        self.evaluator.evaluate(message)


class AsyncClientProxy:
    """Proxy for interacting with an AsyncClient ray actor."""
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

