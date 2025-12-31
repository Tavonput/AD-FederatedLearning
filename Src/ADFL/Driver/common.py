import os
import json
from re import A
from typing import Union, Any, Tuple, List, Optional
from enum import Enum
from dataclasses import asdict
from itertools import chain
from abc import ABC, abstractmethod

import ray

import torch.nn as nn
import numpy as np

from ADFL.types import ScalarPair, TrainingConfig, EvalConfig, FederatedResults
from ADFL.model import get_mobile_net_v3_small, get_mobile_net_v3_large, get_resnet50, get_distilbert
from ADFL.sampling import sample_half_normal
import ADFL.dataset as ds

from ADFL.Strategy import Strategy, Simple, FedAsync, FedBuff, FADAS

NDArrayT2 = Tuple[np.ndarray, np.ndarray]


class Driver(ABC):
    """Base Driver class."""

    @abstractmethod
    def init_backend(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        """Initialize the training environment."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run federated learning."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the driver."""
        pass


class FederatedResultsEncoder(json.JSONEncoder):
    """Custom encoder for serializing FederatedResults."""
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_json"):
            return o.to_json()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)


def run_driver(driver: Driver, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
    driver.init_backend()
    driver.init_training(train_config, eval_config)
    driver.run()
    driver.shutdown()


def generate_model(dataset: TrainingConfig.Dataset, model: str) -> nn.Module:
    """Get a model with 10 classes."""
    if dataset == TrainingConfig.Dataset.CIFAR10:
        num_input_channels = 3
    elif dataset == TrainingConfig.Dataset.MNIST or dataset == TrainingConfig.Dataset.FMNIST:
        num_input_channels = 1
    elif dataset == TrainingConfig.Dataset.SENT140:
        num_input_channels = 0  # Not used
    elif dataset == TrainingConfig.Dataset.NONE:
        assert False, "Cannot get model for TrainingConfig.Dataset.NONE"

    if model == "mobile_net_v3_small":
        return get_mobile_net_v3_small(num_classes=10, num_input_channels=num_input_channels)
    elif model == "mobile_net_v3_large":
        return get_mobile_net_v3_large(num_classes=10, num_input_channels=num_input_channels)
    elif model == "resnet_50":
        return get_resnet50(num_classes=10, num_input_channels=num_input_channels)
    elif model == "distilbert":
        return get_distilbert(num_classes=2)
    else:
        assert False, "Invalid model"


def create_datasets(
    dataset: TrainingConfig.Dataset,
    num_splits: int,
    iid: bool,
    data_path: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    alpha: float = 0.1,
    seed: int = 0
) -> ds.DatasetSplit:
    """Get a partitioned dataset."""
    return ds.create_datasets(
        dataset=dataset,
        num_splits=num_splits,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        iid=iid,
        alpha=alpha,
        seed=seed
    )


def init_ray(tmp_path: Union[str, None] = None) -> None:
    """Initialize ray with a given tmp path."""
    if ray.is_initialized():
        ray.shutdown()

    if tmp_path is not None:
        ray.init(
            _temp_dir=tmp_path,
            include_dashboard=False,
            object_store_memory=16 * (1024**3),
        )
    else:
        ray.init()


def get_delay_map(delay: TrainingConfig.Delay, num_clients: int) -> List[ScalarPair]:
    """Get the delay map if needed."""
    if delay.delay_map is not None:
        assert num_clients == len(delay.delay_map)
        return delay.delay_map

    compute_delays: List[float] = []
    if delay.compute_sigma is None:
        compute_delays = [0.0] * num_clients
    else:
        compute_delays = sample_half_normal(num_clients, delay.compute_sigma, seed=1)

    network_delays: List[float] = []
    if delay.network_sigma is None:
        network_delays = [1.0] * num_clients
    else:
        network_delays = sample_half_normal(
            num_clients, delay.network_sigma, shift=delay.network_shift, seed=5, reverse=True
        )

    return list(zip(compute_delays, network_delays))


def check_sc_map(train_config: TrainingConfig) -> None:
    """Asserts that the server-client map layout is correct."""
    if train_config.delay.sc_map is not None:
        assert train_config.num_servers == len(train_config.delay.sc_map)

        all_clients = list(chain.from_iterable(train_config.delay.sc_map))
        assert train_config.num_clients == len(all_clients)


def check_eval_config(eval_config: EvalConfig, train_config: TrainingConfig) -> None:
    """Asserts that the client map layout is correct."""
    if eval_config.central == True:
        assert eval_config.num_actors == 1
        return

    if eval_config.num_actors == 0:
        return

    if eval_config.client_map is None:
        # Create partitions for the client map
        client_ids = list(range(train_config.num_clients))
        clients_per_eval = (train_config.num_clients + eval_config.num_actors - 1) // eval_config.num_actors

        eval_config.client_map = []
        start = 0
        while start < train_config.num_clients:
            end = start + clients_per_eval
            eval_config.client_map.append(client_ids[start:end])
            start = end

    assert len(eval_config.client_map) == eval_config.num_actors

    all_clients = list(chain.from_iterable(eval_config.client_map))
    assert len(all_clients) == train_config.num_clients


def check_strategy(system_sync: bool, strategy: Strategy) -> None:
    """Check if the strategy is compatible with the server."""
    if isinstance(strategy, Simple):
        assert system_sync == strategy.sync

    elif (
        isinstance(strategy, FedAsync) or
        isinstance(strategy, FedBuff)  or
        isinstance(strategy, FADAS)
    ):
        assert system_sync is False
    else:
        assert False, "Encountered invalid Strategy or Strategy is not in the check list"


def federated_results_to_json(path: str, results: FederatedResults) -> None:
    """Create a json file from a RoundResults."""
    _check_directory(path)

    with open(path, "w") as file:
        json.dump(asdict(results), file, indent=4, cls=FederatedResultsEncoder)


def _check_directory(path: str) -> None:
    """Create the directory if it does not exist."""
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
