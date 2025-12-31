from typing import List, Dict
import random

from ADFL.model import Parameters, simple_aggregate, add_parameters

from .base import Strategy, CommType, AggregationInfo


class Simple(Strategy):
    """Simple.

    Just simple aggregation. Average the update.

    Note: Since the architecture change to a client pool system, the Simple async implementation has not been tested
    with client selection. If you want to use a simple async system, you can just use FedAsync with alpha=0.5 and
    constant staleness.
    """
    def __init__(self, comm_type: CommType, sync: bool) -> None:
        self.comm_type = comm_type
        self.sync = sync

        self.free_clients: List[int] = []
        self.client_working_status: List[bool] = []

        self.sync_finished_clients: List[int] = []

        self.round = 0


    def get_comm_type(self) -> CommType:
        return self.comm_type


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

        if self.sync:
            # Finished clients will be reset upon aggregation
            self.sync_finished_clients.append(client_id)
            return
        else:
            self.free_clients.append(client_id)


    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        self.round += 1

        if self.sync:
            self.free_clients += self.sync_finished_clients
            self.sync_finished_clients.clear()
            return self._sync_update(agg_info.g_params, agg_info.all_c_params)

        assert len(agg_info.all_c_params) == 1
        return self._async_update(agg_info.g_params, agg_info.all_c_params[0])


    def to_json(self) -> Dict:
        return {"name": "Simple", "comm_type": self.comm_type.value, "sync": self.sync}


    def _sync_update(self, g_params: Parameters, all_c_params: List[Parameters]) -> Parameters:
        if self.comm_type == CommType.NORMAL:
            return simple_aggregate(all_c_params)

        elif self.comm_type == CommType.DELTA:
            avg_delta = simple_aggregate(all_c_params)
            return add_parameters(g_params, avg_delta, 1.0, 1.0)

        else:
            assert False, "Should not happen"


    def _async_update(self, g_params: Parameters, c_params: Parameters) -> Parameters:
        if self.comm_type == CommType.NORMAL:
            return simple_aggregate([g_params, c_params])

        elif self.comm_type == CommType.DELTA:
            return add_parameters(g_params, c_params, 1.0, 1.0)

        else:
            assert False, "Should not happen"
