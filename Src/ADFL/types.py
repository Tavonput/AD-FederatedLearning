from typing import Union, Tuple, List
from typing import Optional as Op
from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from .Channel import Channel, IdentityChannel
from .Strategy.base import Strategy
from .model import Parameters


Scalar = Union[int, float]
ScalarPair = Tuple[Scalar, Scalar]

NDArrayT2 = Tuple[np.ndarray, np.ndarray]


@dataclass
class EvalConfig:
    method:     str    = "round"
    central:    bool   = False
    threshold:  Scalar = 1
    num_actors: int    = 1
    client_map: Union[List[List[int]], None] = None


@dataclass
class TrainingConfig:

    @dataclass
    class Delay:
        server_mbps:   Op[float]            = None  # None for no server-to-client delay
        compute_sigma: Op[float]            = None
        network_sigma: Op[float]            = None
        network_shift: float                = 0
        delay_map:     Op[List[ScalarPair]] = None
        sc_map:        Op[List[List[int]]]  = None  # Deprecated sort of

    @dataclass
    class Metrics:
        staleness:  bool = False
        q_error:    bool = False
        model_dist: bool = False
        fetch_raw:  bool = False
        fetch_freq: int  = 1

    class Dataset(Enum):
        NONE    = "none"
        MNIST   = "mnist"
        FMNIST  = "fmnist"
        CIFAR10 = "cifar10"
        SENT140 = "sent140"

    dataset:          Dataset             = Dataset.MNIST
    data_dir:         Op[str]             = None
    train_file:       Op[str]             = None
    test_file:        Op[str]             = None
    iid:              bool                = True
    dirichlet_a:      float               = 0.1
    strategy:         Op[Strategy]        = None
    channel:          Channel             = IdentityChannel(no_compute_time=True)
    model:            str                 = "mobile_net_v3_small"
    num_rounds:       int                 = 1
    num_epochs:       int                 = 1
    num_clients:      int                 = 1
    num_cur_clients:  int                 = 1
    num_servers:      int                 = 1
    batch_size:       int                 = 32
    max_rounds:       int                 = 1
    timeout:          float               = 30
    delay:            Delay               = field(default_factory=Delay)
    model_save:       Op[str]             = None
    model_load:       Op[str]             = None
    num_client_pools: int                 = 1
    metrics:          Metrics             = field(default_factory=Metrics)


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
    compute_time:  float              = 0
    network_time:  float              = 0
    epochs:        int                = 0
    g_start_time:  float              = 0
    mse:           float              = 0
    q_error_mse:   float              = 0
    q_error_cos:   float              = 0
    model_dist:    ScalarPair         = (0, 0)
    accuracy:      Op[Accuracy]       = None
    sent_eval_req: bool               = False


@dataclass
class ClientResults:
    client_id:  int                = -1
    rounds:     List[RoundResults] = field(default_factory=list)
    accuracies: List[Accuracy]     = field(default_factory=list)


@dataclass
class AsyncClientWorkerResults:
    worker_id:    int         = -1
    fetch_times:  List[float] = field(default_factory=list)
    fetch_mean:   float       = 0
    fetch_std:    float       = 0
    fetch_min:    float       = 0
    fetch_max:    float       = 0
    uptime:       float       = 0
    num_eval_req: int         = 0  # Number of requests sent to an evaluator


@dataclass
class FederatedResults:
    paradigm:        str                            = "this should be filled in"
    train_config:    TrainingConfig                 = field(default_factory=TrainingConfig)
    eval_config:     EvalConfig                     = field(default_factory=EvalConfig)
    g_start_time:    float                          = 0
    g_end_time:      float                          = 0
    client_results:  List[ClientResults]            = field(default_factory=list)
    c_accuracies:    List[Accuracy]                 = field(default_factory=list)
    total_g_rounds:  int                            = 0
    q_errors_mse:    List[float]                    = field(default_factory=list)
    q_errors_cos:    List[float]                    = field(default_factory=list)
    model_dists:     List[ScalarPair]               = field(default_factory=list)
    trainer_results: List[AsyncClientWorkerResults] = field(default_factory=list)
    staleness:       List[int]                      = field(default_factory=list)


@dataclass
class AsyncServerResults:
    model:          Parameters          = field(default_factory=dict)
    model_dist:     List[ScalarPair]    = field(default_factory=list)
    client_results: List[ClientResults] = field(default_factory=list)
    accuracies:     List[Accuracy]      = field(default_factory=list)
    g_rounds:       int                 = 0
    q_errors_mse:   List[float]         = field(default_factory=list)
    q_errors_cos:   List[float]         = field(default_factory=list)
    staleness:      List[int]           = field(default_factory=list)
    g_end_time:     float               = 0

