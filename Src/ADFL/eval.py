from typing import Dict, List, Union, Callable
from collections import defaultdict

import ray
import memray

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import my_logging
from .resources import NUM_GPUS, NUM_CPUS
from .model import Parameters, model_forward
from .types import Accuracy
from .messages import EvalMessage
from .memory import MEMRAY_PATH, MEMRAY_RECORD


AccuracyDict = Dict[int, List[Accuracy]]


# @ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
@ray.remote(num_cpus=NUM_CPUS)
class EvalActor:
    """Eval Actor.

    Performs accuracy evaluations for client models.

    TODO: Make the functions not be specific to clients since the servers can use this now.
    """
    def __init__(self, eval_id: int, model_fn: Callable[[], nn.Module], test_loader: DataLoader):
        if MEMRAY_RECORD:
            memray.Tracker(f"{MEMRAY_PATH}{self.__class__.__name__}_{eval_id}_mem_profile.bin").__enter__()

        self.log = my_logging.get_logger(f"EVAL {eval_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.log.info(f"Using device: {self.device}:{torch.cuda.current_device()}")
        else:
            self.log.info(f"Using device: {self.device}")

        self.model = model_fn()
        self.test_loader = test_loader

        self.accuracies: AccuracyDict = defaultdict(list)

        self.ready = False
        self.stop_flag = False


    def initialize(self) -> None:
        """Set the ready state to true."""
        self.ready = True


    def stop(self) -> None:
        """Stop."""
        self.log.info("Stopping")
        self.stop_flag = True


    def get_accuracies(self) -> AccuracyDict:
        """Get the accuracy record."""
        return self.accuracies


    def get_client_accuracies(self, client_id: int) -> List[Accuracy]:
        """Get the accuracy record for a given client."""
        return self.accuracies[client_id]


    def evaluate(self, message: EvalMessage) -> None:
        """Message: Evaluate a model."""
        if self.stop_flag:
            return

        self.log.debug(f"Processing eval request from {message.client_id}")
        self._set_model(message.parameters)

        self.model.eval()
        self.model.to(self.device)

        num_samples = 0
        num_correct = 0

        with torch.no_grad():
            for batch in self.test_loader:
                forward_results = model_forward(self.model, batch, self.device)

                num_samples += forward_results.n_samples
                num_correct += forward_results.correct

        accuracy = num_correct / num_samples * 100
        self._add_accuracy(message.client_id, accuracy, message.g_time)

        self.log.debug(f"Finished eval for client {message.client_id}: {accuracy:.2f}")


    def _add_accuracy(self, c_id: int, accuracy: float, g_time: float) -> None:
        """Add an accuracy entry for a client."""
        self.accuracies[c_id].append(Accuracy(accuracy, g_time))


    def _set_model(self, parameters: Parameters) -> None:
        """Set new model parameters."""
        self.model.to("cpu")
        self.model.load_state_dict(parameters)


class EvalActorProxy:
    """ Eval Actor Proxy.

    Wrapper for interacting with a ray EvalActor.
    """
    def __init__(self, eval_id: int, model_fn: Callable[[], nn.Module], test_loader: DataLoader):
        self.eval_id = eval_id
        self._actor = EvalActor.remote(eval_id, model_fn, test_loader)


    def initialize(self, block: bool = True) -> ray.ObjectRef:
        """Set the ready state to true."""
        if block:
            return ray.get(self._actor.initialize.remote())  # type: ignore
        else:
            return self._actor.initialize.remote()  # type: ignore


    def stop(self, block: bool = True) -> ray.ObjectRef:
        """Stop."""
        if block:
            return ray.get(self._actor.stop.remote())  # type: ignore
        else:
            return self._actor.stop.remote()  # type: ignore


    def get_accuracies(self, block: bool = True) -> Union[AccuracyDict, ray.ObjectRef]:
        """Get the accuracy record."""
        if block:
            return ray.get(self._actor.get_accuracies.remote())  # type: ignore
        else:
            return self._actor.get_accuracies.remote()  # type: ignore


    def get_client_accuracies(self, client_id: int, block: bool = True) -> Union[List[Accuracy], ray.ObjectRef]:
        """Get the accuracy record for a given client."""
        if block:
            return ray.get(self._actor.get_client_accuracies.remote(client_id))  # type: ignore
        else:
            return self._actor.get_client_accuracies.remote(client_id)  # type: ignore


    def evaluate(self, message: EvalMessage) -> ray.ObjectRef:
        """Message: Evaluate a model."""
        return self._actor.evaluate.remote(message)  # type: ignore
