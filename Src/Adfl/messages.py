from dataclasses import dataclass

from .types import Parameters


@dataclass
class ClientUpdate:
    parameters: Parameters
    client_id: int
    client_round: int
