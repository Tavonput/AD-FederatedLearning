import os
import json
from typing import Union, Any, Tuple, Optional, List
from enum import Enum
from dataclasses import asdict
from itertools import chain
from abc import ABC, abstractmethod

import ray

import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from ADFL.types import TrainingConfig, EvalConfig, FederatedResults, TCDataset
from ADFL.model import get_mobile_net_v3_small
from ADFL.sampling import sample_half_normal
import ADFL.dataset as ds

from ADFL.Strategy import Strategy, Simple, FedAsync, FedBuff, FADAS

NDArrayT2 = Tuple[np.ndarray, np.ndarray]


class NumpyDataset(Dataset):
    """Numpy Dataset.

    Simple PyTorch Dataset from numpy data and labels.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform: Optional[transforms.Compose] = None) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        if self.transform:
            data = self.transform(data)
        return data, label


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


def generate_model(dataset: TCDataset) -> nn.Module:
    """Get a MobileNetV3 Small model with 10 classes."""
    if dataset == TCDataset.CIFAR10:
        return get_mobile_net_v3_small(num_classes=10, num_input_channels=3)
    elif dataset == TCDataset.MNIST:
        return get_mobile_net_v3_small(num_classes=10, num_input_channels=1)
    elif dataset == TCDataset.NONE:
        assert False, "Cannot get model for TCDataset.NONE"


def create_datasets(
    dataset: TCDataset, data_path: str, num_splits: int, iid: bool, alpha: float = 0.1, seed: int = 0
) -> ds.DatasetSplit:
    """Get a partitioned dataset."""
    return ds.create_datasets(dataset, data_path, num_splits, iid, alpha, seed)


def init_ray(tmp_path: Union[str, None] = None) -> None:
    """Initialize ray with a given tmp path."""
    if ray.is_initialized():
        ray.shutdown()

    if tmp_path is not None:
        ray.init(_temp_dir=tmp_path)
    else:
        ray.init()


def get_slowness_map(
    slowness_map: Optional[List[float]], half_normal_sigma: Optional[float], num_clients: int
) -> List[float]:
    """Get the slowness map if needed."""
    if slowness_map is not None:
        assert num_clients == len(slowness_map)
        return slowness_map

    elif half_normal_sigma is None:
        return [0.0] * num_clients

    else:
        return sample_half_normal(num_clients, half_normal_sigma)


def check_sc_map(train_config: TrainingConfig) -> None:
    """Asserts that the server-client map layout is correct."""
    if train_config.sc_map is not None:
        assert train_config.num_servers == len(train_config.sc_map)

        all_clients = list(chain.from_iterable(train_config.sc_map))
        assert train_config.num_clients == len(all_clients)


def check_eval_config(eval_config: EvalConfig, train_config: TrainingConfig) -> None:
    """Asserts that the client map layout is correct."""
    if eval_config.central == True:
        assert eval_config.num_actors == 1
        return

    if eval_config.num_actors == 0:
        return

    if eval_config.client_map is not None:
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
