import time
from typing import List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import ray

from ADFL import my_logging
from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import Parameters, RoundResults, Accuracy, EvalConfig
from ADFL.messages import ClientUpdate, EvalMessage
from ADFL.eval import EvalActorProxy
from ADFL.model import get_model_parameters, set_model_parameters

from .common import LR, _train_epoch, _evaluate, AsyncServer


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncClient:
    def __init__(
        self, 
        client_id:    int, 
        model:        nn.Module, 
        train_loader: DataLoader, 
        test_loader:  DataLoader,
        slowness:     float, 
        eval_config:  EvalConfig,
        server:       "AsyncServer",
        evaluator:    EvalActorProxy,
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.client_id = client_id
        self.server = server
        self.evaluator = evaluator

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.slowness = slowness
        self.round = 0

        self.eval_config = eval_config
        self.global_start = time.time()
        self.last_eval_interval = -1

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

        self.accuracies: List[Accuracy] = []

        self.stop_flag = False
        self.ready = False

    def initialize(self) -> bool:
        """Set the ready status to True."""
        self.ready = True
        return True

    def get_id(self) -> int:
        """Get the client id."""
        return self.client_id

    def get_accuracies(self) -> List[Accuracy]:
        """Get the accuracies."""
        if self.evaluator is None:
            return self.accuracies
        return self.evaluator.get_client_accuracies(self.client_id, block=True)

    def stop(self) -> None:
        """Stop working."""
        self.log.info("Stopping")
        self.stop_flag = True

    def train(self, parameters: Parameters, epochs: int = 1) -> None:
        """Message: Train a federated learning round."""
        if self.stop_flag:
            return

        self.round += 1
        self.log.info(f"Starting training round {self.round}")
        start_time = time.time()

        set_model_parameters(self.model, parameters)

        self._possibly_eval()

        round_results = RoundResults()
        for epoch in range(epochs):
            self.log.debug(f"Epoch {epoch + 1}/{epochs}")
            results = _train_epoch(
                self.model, self.optimizer, self.criterion, self.train_loader, self.device, self.slowness
            )
            round_results.train_results.append(results)

        round_results.round_time = time.time() - start_time
        round_results.g_start_time = start_time
        round_results.epochs = epochs
        round_results.train_round = self.round

        self.log.info("Finished training. Sending updated model to server")
        self._send_update_to_server(round_results)

    def _send_update_to_server(self, round_results: RoundResults) -> None:
        """Send a round update to the server."""
        self.model.to("cpu")

        update = ClientUpdate(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            client_round=self.round,
            round_results=round_results,
            num_examples=len(self.train_loader.dataset),
        )

        self.server.client_update.remote(update)

    def _possibly_eval(self) -> None:
        """Evaluate if needed or send the job to an evaluator."""
        if self._need_to_eval() is False:
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
                
    def _self_eval(self) -> None:
        """Perform a self evaluation if needed."""
        accuracy = 0

        if self.eval_config.method == "time":
            current_eval_interval = (time.time() - self.global_start) // self.eval_config.threshold
            self.log.info(f"Evaluating interval: {current_eval_interval}")
            self.last_eval_interval = current_eval_interval

        elif self.eval_config.method == "round":
            self.log.info(f"Evaluating round: {self.round}")

        accuracy = _evaluate(self.model, self.test_loader, self.device)
        self.accuracies.append(Accuracy(accuracy, time.time()))

        self.log.warning(accuracy)

    def _send_eval_message(self) -> None:
        """Send an evaluation message to the evaluator if needed."""
        self.log.info("Sending evaluation request.")

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
        model:        nn.Module,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        slowness:     float, 
        eval_config:  float,
        server:       "AsyncServer",
        evaluator:    EvalActorProxy,
    ):
        self.client_id = client_id
        self.actor = AsyncClient.remote(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            slowness=slowness,
            eval_config=eval_config,
            server=server,
            evaluator=evaluator,
        )

    def initialize(self, block: bool = True) -> Union[bool, ray.ObjectRef]:
        if block:
            return ray.get(self.actor.initialize.remote())
        else:
            return self.actor.initialize.remote()

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

    def stop(self, block: bool = True) -> ray.ObjectRef:
        if block:
            return ray.get(self.actor.stop.remote())
        else:
            return self.actor.stop.remote()

    def train(self, parameters: Parameters, epochs: int = 1) -> ray.ObjectRef:
        return self.actor.train.remote(parameters, epochs)

