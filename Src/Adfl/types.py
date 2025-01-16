from typing import Dict
from dataclasses import dataclass

import torch


Parameters = Dict[str, torch.Tensor]

@dataclass
class TrainingConfig:
    num_rounds: int
    num_epochs: int
    num_clients: int
    batch_size: int
    max_rounds: int