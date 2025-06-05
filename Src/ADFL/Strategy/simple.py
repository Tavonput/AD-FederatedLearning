from typing import List, Dict

from ADFL.model import Parameters, simple_aggregate, add_parameters

from .base import Strategy, CommType, AggregationInfo


class Simple(Strategy):
    """Simple.

    Just simple aggregation. Average the update.
    """
    def __init__(self, comm_type: CommType, sync: bool) -> None:
        self.comm_type = comm_type
        self.sync = sync

        self.round = 1


    def get_comm_type(self) -> CommType:
        return self.comm_type


    def get_round(self) -> int:
        return self.round


    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        self.round += 1

        if self.sync:
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
