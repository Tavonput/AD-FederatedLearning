from dataclasses import dataclass
from typing import Optional

from .model import Parameters, CompressedParameters
from .types import RoundResults


@dataclass
class AsyncClientTrainMessage:
    parameters: Optional[CompressedParameters]
    epochs:     int
    g_round:    int


@dataclass
class ClientUpdateMessage:
    parameters:    CompressedParameters
    client_id:     int
    client_round:  int
    g_round:       int
    round_results: RoundResults
    num_examples:  int          = 0


@dataclass
class ServerUpdate:
    parameters: CompressedParameters
    server_id:  int


@dataclass
class EvalMessage:
    parameters: Parameters
    client_id:  int
    g_time:     float
