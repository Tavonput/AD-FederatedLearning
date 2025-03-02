import os
import json
from typing import List
from dataclasses import dataclass, asdict
from itertools import chain
from abc import ABC, abstractmethod

import ray

import torch.nn as nn
from torch.utils.data import Subset, random_split, Dataset
from torchvision import datasets, transforms

from ADFL.types import TrainingConfig, EvalConfig, FederatedResults
from ADFL.model import get_mobile_net_v3_small


@dataclass
class DataSetSplit:
    sets: List[Subset]
    test: Dataset
    full_size: int
    split_size: int


class Driver(ABC):
    """Base Driver class."""

    @abstractmethod
    def init_backend(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    def init_training(self, train_config: TrainingConfig) -> None:
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


def _generate_model() -> nn.Module:
    """Get a MobileNetV3 Small model with 10 classes."""
    return get_mobile_net_v3_small(num_classes=10)


def _create_datasets(data_path: str, num_splits: int) -> DataSetSplit:
    """Get a partitioned CIFAR10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    split_datasets = random_split(
        full_dataset, [len(full_dataset) // num_splits] * num_splits
    )

    test_set = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    return DataSetSplit(split_datasets, test_set, len(full_dataset), len(split_datasets[0]))


def _init_ray(tmp_path: str = None) -> None:
    """Initialize ray with a given tmp path."""
    if tmp_path is not None:
        ray.init(_temp_dir=tmp_path)
    else:
        ray.init()


def _check_slowness_map(train_config: TrainingConfig) -> None:
    """Asserts that the size of the slowness map is correct."""
    if train_config.slowness_map is not None:
        assert train_config.num_clients == len(train_config.slowness_map)


def _check_sc_map(train_config: TrainingConfig) -> None:
    """Asserts that the server-client map layout is correct."""
    if train_config.sc_map is not None:
        assert train_config.num_servers == len(train_config.sc_map)

        all_clients = list(chain.from_iterable(train_config.sc_map))
        assert train_config.num_clients == len(all_clients)


def _check_eval_client_map(eval_config: EvalConfig, train_config: TrainingConfig) -> None:
    """Asserts that the client map layout is correct."""
    if eval_config.num_actors == 0:
        return

    if eval_config.client_map is not None:
        assert len(eval_config.client_map) == eval_config.num_actors

        all_clients = list(chain.from_iterable(eval_config.client_map))
        assert len(all_clients) == train_config.num_clients


def _check_directory(path: str) -> None:
    """Create the directory if it does not exist."""
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def _federated_results_to_json(path: str, results: FederatedResults) -> None:
    """Create a json file from a RoundResults."""
    _check_directory(path)
    with open(path, "w") as file:
        json.dump(asdict(results), file, indent=4)

