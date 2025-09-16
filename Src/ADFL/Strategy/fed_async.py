from typing import Dict, List
from enum import Enum
import random

from ADFL.model import Parameters, add_parameters

from .base import AggregationInfo, Strategy, CommType


class FedAsync(Strategy):
    """FedAsync.

    FedAsync with poly aggregation.
    """
    class Method(Enum):
        CONSTANT = "constant"
        POLY     = "poly"
        HINGE    = "hinge"


    def __init__(self, method: Method, alpha: float = 0.6, a: float = 0.5, b: int = 10) -> None:
        assert alpha >= 0 and alpha <= 1

        self.method = method
        self.alpha = alpha
        self.a = a
        self.b = b

        self.client_working_status: List[bool] = []
        self.free_clients: List[int] = []

        self.round = 1


    def get_comm_type(self) -> CommType:
        return CommType.NORMAL


    def get_round(self) -> int:
        return self.round


    def select_client(self, num_clients: int) -> int:
        assert num_clients > 0

        if len(self.free_clients) == 0:
            self.free_clients = list(range(num_clients))
            self.client_working_status = [False] * num_clients

        idx = random.randint(0, len(self.free_clients) - 1)
        client = self.free_clients[idx]
        self.client_working_status[client] = True

        self.free_clients[idx], self.free_clients[-1] = self.free_clients[-1], self.free_clients[idx]
        self.free_clients.pop()

        return client


    def on_client_finish(self, client_id: int) -> None:
        assert self.client_working_status[client_id] is True, "Detected client finished but was never working?"
        self.client_working_status[client_id] = False
        self.free_clients.append(client_id)


    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        self.round += 1

        assert len(agg_info.all_c_params) == 1
        c_update = agg_info.all_c_params[0]

        staleness_factor = 1
        if self.method == FedAsync.Method.CONSTANT:
            staleness_factor = 1
        elif self.method == FedAsync.Method.POLY:
            staleness_factor = self._poly(agg_info.staleness)
        elif self.method == FedAsync.Method.HINGE:
            staleness_factor = self._hinge(agg_info.staleness)

        alpha_t = self.alpha * staleness_factor
        return add_parameters(agg_info.g_params, c_update, (1 - alpha_t), alpha_t)


    def to_json(self) -> Dict:
        return {
            "name": "FedAsync",
            "method": self.method.value,
            "alpha": self.alpha,
            "a": self.a,
            "b": self.b,
        }


    def _poly(self, staleness: int) -> float:
        return (staleness + 1) ** (-self.a)


    def _hinge(self, staleness: int) -> float:
        if staleness <= self.b:
            return 1

        den = self.a * (staleness - self.b) + 1
        return 1 / den

