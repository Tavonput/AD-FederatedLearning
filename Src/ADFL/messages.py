from dataclasses import dataclass

from .types import Parameters, RoundResults


@dataclass
class ClientUpdate:
    parameters:    Parameters
    client_id:     int
    client_round:  int
    round_results: RoundResults
    num_examples:  int          = 0


@dataclass
class ServerUpdate:
    parameters: Parameters
    server_id:  int


@dataclass
class EvalMessage:
    parameters: Parameters
    client_id:  int
    g_time:     float
