from enum import Enum
from typing import Callable, Optional, List

import torch.nn as nn
import ray

from ADFL.types import AsyncServerResults, TrainingConfig, EvalConfig
from ADFL.messages import ClientUpdateMessage
from ADFL.eval import EvalActorProxy
from ADFL.flag import RemoteFlag
from ADFL.Client import AsyncClientV2, AsyncClientWorkerProxy, DistributedClientPool
from ADFL.model import Parameters

from .common import Server
from .async_sc import AsyncServer, TraditionalServer
from .qafel import QAFeLServer


class ServerType(Enum):
    ASYNC        = "Async"
    TRADITIONAL  = "Traditional"
    QAFEL        = "QAFeL"


class ServerProxy(Server):
    """Proxy class for interacting with a Server actor."""
    def __init__(
        self,
        server_type:  ServerType,
        model_fn:     Callable[[], nn.Module],
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    Optional[EvalActorProxy],
        stop_flag:    RemoteFlag,
    ):
        self.server_type = server_type
        if server_type == ServerType.ASYNC:
            actor_func = AsyncServer
        elif server_type == ServerType.TRADITIONAL:
            actor_func = TraditionalServer
        elif server_type == ServerType.QAFEL:
            actor_func = QAFeLServer

        self.actor = actor_func.remote(model_fn, train_config, eval_config, evaluator, stop_flag)


    def initialize(self) -> bool:
        return ray.get(self.actor.initialize.remote())  # type: ignore


    def add_clients(self, clients: List[AsyncClientV2], block: bool = True) -> None:
        """Deprecated."""
        assert False, "ServerProxy.add_clients is deprecated. Please do not use"
        if block:
            ray.get(self.actor.add_clients.remote(clients))  # type: ignore
        else:
            self.actor.add_clients.remote(clients)  # type: ignore


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        ray.get(self.actor.attach_client_pool.remote(pool))  # type: ignore


    def add_workers(self, workers: List[AsyncClientWorkerProxy], block: bool = True) -> None:
        if block:
            ray.get(self.actor.add_workers.remote(workers))  # type: ignore
        else:
            self.actor.add_workers.remote(workers)  # type: ignore


    def stop(self, block: bool = True) -> None:
        if block:
            ray.get(self.actor.stop.remote())  # type: ignore
        else:
            self.actor.stop.remote()  # type: ignore


    def get_results(self, block: bool = True) -> AsyncServerResults:
        if block:
            return ray.get(self.actor.get_results.remote())  # type: ignore
        else:
            return self.actor.get_results.remote()  # type: ignore


    def get_model(self, block: bool = True) -> Parameters:
        if block:
            return ray.get(self.actor.get_model.remote())  # type: ignore
        else:
            return self.actor.get_model.remote()  # type: ignore


    def run(self) -> None:
        self.actor.run.remote()  # type: ignore


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        self.actor.client_update.remote(client_update)  # type: ignore

