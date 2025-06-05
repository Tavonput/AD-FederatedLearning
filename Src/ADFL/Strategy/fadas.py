from typing import Dict
import math

import torch

from ADFL.model import Parameters, add_parameters_inpace, parameter_mse

from .base import AggregationInfo, Strategy, CommType


class FADAS(Strategy):
    """FADAS."""
    def __init__(
        self,
        max_buffer_size: int,
        lr:              float,
        beta_1:          float = 0.9,
        beta_2:          float = 0.999,
        eps:             float = 1e-8,
        delay_adaptive:  bool  = True,
        max_delay:       int   = 10
    ) -> None:
        self.lr = lr

        self.buffer: Parameters = {}
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0

        self.m: Parameters = {}
        self.v: Parameters = {}
        self.v_hat: Parameters = {}

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.delay_adaptive = delay_adaptive
        self.max_delay = max_delay
        self.max_staleness = 0

        self.round = 1


    def get_comm_type(self) -> CommType:
        return CommType.DELTA


    def get_round(self) -> int:
        return self.round


    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        assert len(agg_info.all_c_params) == 1
        c_update = agg_info.all_c_params[0]

        if self.buffer_size == 0:
            self.buffer = c_update
        else:
            add_parameters_inpace(self.buffer, c_update, 1.0, 1.0, to_float=True)

        self.buffer_size += 1
        self.max_staleness = max(self.max_staleness, agg_info.staleness)

        if self.buffer_size >= self.max_buffer_size:
            for tensor in self.buffer.values():
                tensor.float().div_(self.max_buffer_size)

            self._update_amsgrad(agg_info)
            params_prime = self._compute_model_step(agg_info.g_params)

            self.buffer.clear()
            self.buffer_size = 0
            self.max_staleness = 0
            self.round += 1

            return params_prime

        return agg_info.g_params


    def to_json(self) -> Dict:
        return {
            "name":   "FADAS",
            "K":      self.max_buffer_size,
            "lr":     self.lr,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "eps":    self.eps,
        }


    def _compute_model_step(self, g_params: Parameters) -> Parameters:
        bias_correction_1 = 1 - self.beta_1 ** self.round
        bias_correction_2 = 1 - self.beta_2 ** self.round

        lr_t = self._apply_delay_correction()
        step_size = lr_t / bias_correction_1

        params_prime: Parameters = {}
        for k in g_params.keys():
            denom = (self.v_hat[k].sqrt() / math.sqrt(bias_correction_2)).add_(self.eps)
            params_prime[k] = torch.addcdiv(g_params[k], self.m[k], denom, value=step_size)

        print(f"MSE={parameter_mse(g_params, params_prime)}")

        return params_prime


    def _apply_delay_correction(self) -> float:
        if self.delay_adaptive is False:
            return self.lr

        if self.max_staleness <= self.max_delay:
            return self.lr
        else:
            return min(self.lr, (self.lr / self.max_staleness))


    def _update_amsgrad(self, agg_info: AggregationInfo) -> None:
        self._init_moments_if_needed(agg_info.g_params)

        for k in self.buffer.keys():
            self.m[k].mul_(self.beta_1).add_(self.buffer[k], alpha=(1 - self.beta_1))
            self.v[k].mul_(self.beta_2).addcmul_(self.buffer[k], self.buffer[k], value=(1 - self.beta_2))
            torch.maximum(self.v_hat[k], self.v[k], out=self.v_hat[k])


    def _init_moments_if_needed(self, params: Parameters) -> None:
        if len(self.m) == 0:
            for name, tensor in params.items():
                self.m[name] = torch.zeros_like(tensor, dtype=torch.float32)

        if len(self.v) == 0:
            for name, tensor in params.items():
                self.v[name] = torch.zeros_like(tensor, dtype=torch.float32)

        if len(self.v_hat) == 0:
            for name, tensor in params.items():
                self.v_hat[name] = torch.zeros_like(tensor, dtype=torch.float32)
