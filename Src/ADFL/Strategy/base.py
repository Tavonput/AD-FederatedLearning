from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
from dataclasses import dataclass

from ADFL.model import Parameters


class CommType(Enum):
    NORMAL = "normal"
    DELTA  = "delta"


@dataclass
class AggregationInfo:
    g_params:     Parameters
    all_c_params: List[Parameters]
    staleness:    int


class Strategy(ABC):
    """Base Strategy Interface.

    A Strategy is responsible for computing the parameters for the next global step. It is also responsible for
    maintaining the global round. Usually, it would make more sense for the server to manage the global round, but
    different aggregation algorithms (specifically the buffered ones) maintain the global round differently.
    """
    @abstractmethod
    def get_comm_type(self) -> CommType:
        """Get the method."""
        pass


    @abstractmethod
    def get_round(self) -> int:
        """Get the round."""
        pass


    @abstractmethod
    def select_client(self, num_clients: int) -> int:
        """Select a client."""
        pass


    @abstractmethod
    def on_client_finish(self, client_id: int) -> None:
        """Notify that a client has finished to update the client selection process."""
        pass


    @abstractmethod
    def produce_update(self, agg_info: AggregationInfo) -> Parameters:
        """Produce the parameters for the next update."""
        pass


    @abstractmethod
    def to_json(self) -> Dict:
        """Return serialization for json."""
        pass
