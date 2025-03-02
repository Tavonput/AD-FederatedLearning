import time
import threading
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import ray

from ADFL import my_logging
from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import (
    TrainingConfig, RoundResults, ClientResults, Accuracy, EvalConfig
) 
from ADFL.messages import ClientUpdate, EvalMessage
from ADFL.eval import EvalActorProxy
from ADFL.model import get_model_parameters, set_model_parameters, simple_aggregate

from .common import LR, _train_epoch, _evaluate


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncPeerClient:
    def __init__(
        self, 
        client_id: int, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader,
        slowness: float, 
        train_config: TrainingConfig
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.train_config = train_config
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.round = 0
        self.slowness = slowness
        self.updates: List[ClientUpdate] = []
        self.other_clients = []
        self.results = ClientResults(client_id)

        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

        self.message_log: List[str] = []

        self.training_thread: threading.Thread = None
        self.stop_flag = threading.Event()
        self.update_lock = threading.Lock()

        self.ready = False

    def initialize(self) -> bool:
        self.ready = True
        return True

    def add_clients(self, clients: List) -> None:
        self.other_clients += clients
        return

    def get_client_results(self) -> ClientResults:
        """Get the ClientResults."""
        return self.results

    def get_message_log(self) -> List[str]:
        """Get the message logs."""
        return self.message_log

    def receive_update(self, update: ClientUpdate) -> None:
        with self.update_lock:
            self.updates.append(update)

    def stop(self) -> None:
        self.log.info("Terminating training thread")

        self.stop_flag.set()
        if self.training_thread is not None:
            self.training_thread.join()

        self.log.info("Training thread has been cleaned up")

    def train(self, train_round: int, epochs: int) -> None:
        self.log.info("Starting training thread")
        
        if self.training_thread is None:
            self.training_thread = threading.Thread(target=self._train, args=(train_round, epochs))
            self.training_thread.start()
        else:
            self.log.error("Tried to start training, but already doing so")

    def _train(self, train_round: int, epochs: int) -> None:
        self.round = train_round

        while not self.stop_flag.is_set():
            if self.round > self.train_config.max_rounds:
                break

            self.log.info(f"Starting training round {self.round}")

            self._train_round(epochs)
            self._send_updates()
            self._aggregate()

            self.round += 1

    def _train_round(self, epochs: int) -> None:
        """Train for a round."""
        self.message_log.append(f"T_{self.round}")
        start_time = time.time()

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

        self.results.rounds.append(round_results)

        self.log.info("Finished training")

    def _send_updates(self) -> None:
        self.log.info("Sending updates")

        self.model.to("cpu")

        update = ClientUpdate(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            client_round=self.round,
            round_results=None,
        )

        # Send all of the other clients updates
        [client.receive_update.remote(update) for client in self.other_clients]
    
    def _aggregate(self) -> None:
        with self.update_lock:
            self.log.info(f"Aggregating from {len(self.updates)} clients")

            if len(self.updates) == 0:
                return

            for update in self.updates:
                self.message_log.append(f"U_{update.client_id}")

            self.model.to("cpu")

            # with torch.no_grad():
            #     for update in self.updates:
            #         for name, params in self.model.named_parameters():
            #             params.data = (params + update.parameters[name]) / 2

            # Add our personal model to the list of updates for the subsequent average
            self.updates.append(ClientUpdate(get_model_parameters(self.model), self.client_id, self.round, None))

            with torch.no_grad():
                for name, params in self.model.named_parameters():
                    params.data = torch.stack([update.parameters[name] for update in self.updates]).mean(dim=0)

            self.updates.clear()


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncPeerClientV2:
    """ Asynchronous Peer-to-Peer Client V2.

    This version uses self-messages instead of shared memory as done by AsyncPeerClient.
    """
    def __init__(
        self, 
        client_id:    int, 
        model:        nn.Module, 
        train_loader: DataLoader,
        test_loader:  DataLoader, 
        slowness:     float, 
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    EvalActorProxy,
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.train_config = train_config
        self.eval_config = eval_config

        self.client_id = client_id
        self.slowness = slowness
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.global_start = time.time()
        self.last_eval_interval = -1

        self.results = ClientResults(client_id)
        self.message_log: List[str] = []

        self.evaluator = evaluator
        self.other_clients = []
        self.ref = None

        self.model = model
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

        self.round = 0
        self.stop_flag = False
        self.ready = False

    def initialize(self) -> bool:
        """Set the ready status to True."""
        self.ready = True
        return True
    
    def stop(self) -> None:
        """Stop working."""
        self.log.info("Stopping")
        self.stop_flag = True

    def add_self_ref(self, ref) -> None:
        """Set the self reference."""
        self.ref = ref 

    def add_clients(self, clients: List) -> None:
        """Add other clients."""
        self.other_clients += clients

    def get_client_results(self) -> ClientResults:
        """Get the ClientResults."""
        if self.evaluator is not None:
            self.results.accuracies = self.evaluator.get_client_accuracies(self.client_id, block=True)
        return self.results
        
    def get_message_log(self) -> List[str]:
        """Get the operations."""
        return self.message_log

    def train(self, train_round: int, epochs: int) -> None:
        """Message: Train a federated learning round."""
        self.message_log.append(f"T_{train_round}")

        if self.stop_flag:
            return
        
        if train_round > self.train_config.max_rounds:
            self.stop()
            return

        self.log.info(f"Starting training round {train_round}")
        self.round = train_round

        self._possibly_eval()

        self._train_round(train_round, epochs)
        self.ref.train.remote(train_round + 1, epochs)

        self._send_updates(train_round)

    def receive_update(self, client_update: ClientUpdate) -> None:
        """Message: Received update from another client, thus aggregate."""
        self.message_log.append(f"U_{client_update.client_id}")

        if self.stop_flag:
            return

        self.log.debug(f"Received and processing update from Client {client_update.client_id}")
        parameters = [get_model_parameters(self.model), client_update.parameters]
        parameters_prime = simple_aggregate(parameters)
        set_model_parameters(self.model, parameters_prime)

    def _train_round(self, train_round: int, epochs: int) -> None:
        """Train for a round."""
        start_time = time.time()

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
        round_results.train_round = train_round

        self.results.rounds.append(round_results)

        self.log.info("Finished training")

    def _send_updates(self, train_round: int) -> None:
        """Send a ClientUpdate to the other clients."""
        self.log.info("Sending updates")
        self.model.to("cpu")

        update = ClientUpdate(
            parameters    = get_model_parameters(self.model),
            client_id     = self.client_id,
            client_round  = train_round,
            round_results = None, # The other clients don't need this
        )

        # Send all of the other clients updates
        [client.receive_update.remote(update) for client in self.other_clients]

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
        self.results.accuracies.append(Accuracy(accuracy, time.time()))

        self.log.warning(accuracy)

    def _send_eval_message(self) -> None:
        """Send an evaluation message to the evaluator."""
        self.log.info("Sending evaluation request.")

        message = EvalMessage(
            parameters = get_model_parameters(self.model),
            client_id  = self.client_id,
            g_time     = time.time(),
        )
        self.evaluator.evaluate(message)


@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncPeerClientV3:
    """ Asynchronous Peer-to-Peer Client V3.

    This version uses self-messages instead of shared memory as done by AsyncPeerClient.
    """
    def __init__(
        self, 
        client_id: int, 
        model: nn.Module, 
        train_loader: DataLoader,
        test_loader: DataLoader, 
        slowness: float, 
        train_config: TrainingConfig
    ):
        self.log = my_logging.get_logger(f"CLIENT {client_id}")
        self.log.info("Initializing")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        self.train_config = train_config
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.slowness = slowness

        self.global_start = time.time()
        self.last_eval_interval = -1

        self.results = ClientResults(client_id)
        self.message_log: List[str] = []

        self.other_clients = []
        self.ref = None

        self.model = model
        self.old_model = get_model_parameters(self.model)
        self.optimizer = SGD(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)

        self.stop_flag = False
        self.ready = False

    def initialize(self) -> bool:
        """Set the ready status to True."""
        self.ready = True
        return True
    
    def stop(self) -> None:
        """Stop working."""
        self.log.info("Stopping")
        self.stop_flag = True

    def add_self_ref(self, ref) -> None:
        """Set the self reference."""
        self.ref = ref 

    def add_clients(self, clients: List) -> None:
        """Add other clients."""
        self.other_clients += clients

    def get_client_results(self) -> ClientResults:
        """Get the ClientResults."""
        return self.results

    def get_message_log(self) -> List[str]:
        """Get the operations."""
        return self.message_log

    def train(self, train_round: int, epochs: int) -> None:
        """Message: Train a federated learning round."""
        self.message_log.append(f"T_{train_round}")

        if self.stop_flag:
            return

        if train_round > self.train_config.max_rounds:
            self.stop()
            return

        self.log.info(f"Starting training round {train_round}")

        if train_round != 1:
            self.model.to("cpu")
            with torch.no_grad():
                for name, params in self.model.named_parameters():
                    params.data = (params + self.old_model[name]) / 2

            self._send_updates(train_round - 1)

        self._possibly_eval()

        self.model.to("cpu")
        self.old_model = get_model_parameters(self.model)

        self._train_round(train_round, epochs)
        self.ref.train.remote(train_round + 1, epochs)

    def receive_update(self, client_update: ClientUpdate) -> None:
        """Message: Received update from another client, thus aggregate."""
        self.message_log.append(f"U_{client_update.client_id}")

        if self.stop_flag:
            return

        self.log.debug(f"Received and processing update from Client {client_update.client_id}")

        # with torch.no_grad():
        #     for name, params in self.model.named_parameters():
        #         params.data = (params + client_update.parameters[name]) / 2

        with torch.no_grad():
            for name, params in self.old_model.items():
                self.old_model[name] = (params + client_update.parameters[name]) / 2

    def _train_round(self, train_round: int, epochs: int) -> None:
        """Train for a round."""
        start_time = time.time()

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
        round_results.train_round = train_round

        self.results.rounds.append(round_results)

        self.log.info("Finished training")

    def _send_updates(self, train_round: int) -> None:
        """Send a ClientUpdate to the other clients."""
        self.log.info("Sending updates")

        self.model.to("cpu")

        update = ClientUpdate(
            parameters=get_model_parameters(self.model),
            client_id=self.client_id,
            client_round=train_round,
            round_results=None, # The other clients don't need this
        )

        # Send all of the other clients updates
        [client.receive_update.remote(update) for client in self.other_clients]

    def _possibly_eval(self) -> None:
        """Evaluate if needed."""
        current_eval_interval = (time.time() - self.global_start) // self.train_config.eval_time
        if current_eval_interval > self.last_eval_interval:
            self.log.info(f"Evaluating Interval {current_eval_interval}")

            accuracy = _evaluate(self.model, self.test_loader, self.device)
            self.results.accuracies.append(Accuracy(accuracy, time.time()))

            self.last_eval_interval = current_eval_interval


