from typing import Dict, Union, Tuple, List
from typing import Optional as Op
from enum import Enum
from dataclasses import dataclass, field

import torch
import numpy as np

from .Strategy.base import Strategy


Scalar = Union[int, float]
ScalarPair = Tuple[Scalar, Scalar]

NDArrayT2 = Tuple[np.ndarray, np.ndarray]


@dataclass
class QuantParameter:
    data:    bytes
    bits:    int
    scale:   float
    shape:   torch.Size
    dtype:   torch.dtype
    q_dtype: torch.dtype


@dataclass
class QuantParameters:
    params: Dict[str, QuantParameter]
    size:   int


@dataclass
class ByteParameter:
    data:  bytes
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class ByteParameters:
    params: Dict[str, ByteParameter]
    size:   int


@dataclass
class MixedParameters:
    params: Dict[str, Union[QuantParameter, ByteParameter]]
    size:   int


CompressedParameters = Union[QuantParameters, ByteParameters, MixedParameters]


@dataclass
class EvalConfig:
    method:     str    = "round"
    central:    bool   = False
    threshold:  Scalar = 1
    num_actors: int    = 1
    client_map: Union[List[List[int]], None] = None


# Deprecated
class TCMethod(Enum):
    NONE     = "none"
    NORMAL   = "normal"
    DELTA    = "delta"


class TCDataset(Enum):
    NONE    = "none"
    MNIST   = "mnist"
    CIFAR10 = "cifar10"


@dataclass
class TrainingConfig:
    method:         TCMethod            = TCMethod.NORMAL  # Deprecated
    dataset:        TCDataset           = TCDataset.MNIST
    iid:            bool                = True
    dirichlet_a:    float               = 0.1
    strategy:       Op[Strategy]        = None
    num_rounds:     int                 = 1
    num_epochs:     int                 = 1
    num_clients:    int                 = 1
    num_servers:    int                 = 1
    batch_size:     int                 = 32
    max_rounds:     int                 = 1
    timeout:        float               = 30
    compress:       str                 = "byte"
    quant_lvl_1:    int                 = 8
    quant_lvl_2:    int                 = 8
    slowness_map:   Op[List[float]]     = None
    sc_map:         Op[List[List[int]]] = None
    slowness_sigma: Op[float]           = None


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
    mse:           float              = 0


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
    c_accuracies:   List[Accuracy]      = field(default_factory=list)

