from typing import Dict, List
import random

from ADFL.model import Parameters, add_parameters, add_parameters_inpace, parameter_mse

from .base import AggregationInfo, Strategy, CommType


class FedBuff(Strategy):
    """FedBuff.

    Note: If you are using central evaluation, you might want to reduce to round threshold to 1 or 2.

    TODO: Staleness needs to be computed differently due to buffering.
    They also mention in the paper that a client can only contribute once to a buffer.
    """
    def __init__(self, max_buffer_size: int, lr: float, apply_staleness: bool) -> None:
        self.lr = lr

        self.buffer: Parameters = {}
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0

        self.free_clients: List[int] = []
        self.client_working_status: List[bool] = []
        self.finished_clients: List[int] = []

        self.round = 0

        # See paper
        self.a = 0.5
        self.apply_staleness = apply_staleness


    def get_comm_type(self) -> CommType:
        return CommType.DELTA


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

        # Finished client will be added back into the selection pool after aggregation
        self.finished_clients.append(client_id)


    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        assert len(agg_info.all_c_params) == 1
        c_update = agg_info.all_c_params[0]

        if self.apply_staleness:
            for tensor in c_update.values():
                # This will not affect the integer tensors
                tensor.float().mul_(self._staleness_factor(agg_info.staleness))

        if self.buffer_size == 0:
            self.buffer = c_update
        else:
            add_parameters_inpace(self.buffer, c_update, 1, 1, to_float=False)

        self.buffer_size += 1
        if self.buffer_size >= self.max_buffer_size:
            # Add the finished clients from this buffer back to the selection pool
            self.free_clients += self.finished_clients
            self.finished_clients.clear()

            for tensor in self.buffer.values():
                # This will not affect the integer tensors
                tensor.float().div_(self.max_buffer_size)

            params_prime = add_parameters(agg_info.g_params, self.buffer, 1.0, self.lr)

            # print(f"MSE={parameter_mse(agg_info.g_params, params_prime)}")

            self.buffer.clear()
            self.buffer_size = 0
            self.round += 1

            return params_prime

        return agg_info.g_params


    def to_json(self) -> Dict:
        return {
            "name":      "FedBuff",
            "K":         self.max_buffer_size,
            "lr":        self.lr,
            "staleness": self.apply_staleness,
        }


    def _staleness_factor(self, staleness: int) -> float:
        return (1 + staleness) ** (-self.a)

