from dataclasses import dataclass

from .model import Parameters
from .types import CompressedParameters, RoundResults


@dataclass
class AsyncClientTrainMessage:
    parameters: CompressedParameters
    epochs:     int
    g_round:    int


@dataclass
class ClientUpdate:
    parameters:    CompressedParameters
    client_id:     int
    client_round:  int
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
