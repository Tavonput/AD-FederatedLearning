from typing import Dict, Union, Tuple, List
from dataclasses import dataclass, field

import torch

Scalar = Union[int, float]
ScalarPair = Tuple[Scalar, Scalar]

Parameters = Dict[str, torch.Tensor]


@dataclass
class EvalConfig:
    method:     str             = "round"
    threshold:  Scalar          = 1
    num_actors: int             = 1
    client_map: List[List[int]] = None


@dataclass
class TrainingConfig:
    num_rounds:   int             = 1
    num_epochs:   int             = 1
    num_clients:  int             = 1
    num_servers:  int             = 1
    batch_size:   int             = 32
    max_rounds:   int             = 1
    timeout:      float           = 30
    slowness_map: List[float]     = None
    sc_map:       List[List[int]] = None


@dataclass
class Accuracy:
    value:  float = 0
    g_time: float = 0


@dataclass
class TrainResults:
    running_loss: List[ScalarPair] = field(default_factory=list)
    average_loss: float            = 0
    sum_loss:     float            = 0
    accuracy:     float            = 0
    elapsed_time: float            = 0
    g_start_time: float            = 0
    

@dataclass
class RoundResults:
    train_round:   int                = 0
    train_results: List[TrainResults] = field(default_factory=list)
    round_time:    float              = 0
    epochs:        int                = 0
    g_start_time:  float              = 0


@dataclass
class ClientResults:
    client_id:  int                = -1
    rounds:     List[RoundResults] = field(default_factory=list)
    accuracies: List[Accuracy]     = field(default_factory=list)


@dataclass
class FederatedResults:
    paradigm:       str                 = "this should be filled in"
    train_config:   TrainingConfig      = field(default_factory=TrainingConfig)
    eval_config:    EvalConfig          = field(default_factory=EvalConfig)
    g_start_time:   float               = 0
    client_results: List[ClientResults] = field(default_factory=list)
