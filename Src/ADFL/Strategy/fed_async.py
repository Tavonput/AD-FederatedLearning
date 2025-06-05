from typing import Dict
from enum import Enum

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

        self.round = 1


    def get_comm_type(self) -> CommType:
        return CommType.NORMAL


    def get_round(self) -> int:
        return self.round


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

