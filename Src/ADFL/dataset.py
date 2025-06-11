from typing import List, Optional
from collections.abc import Sized
from dataclasses import dataclass

from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms

import numpy as np

from .types import TCDataset, NDArrayT2
from .sampling import sample_without_replacement


@dataclass
class DatasetSplit:
    sets:       List[Dataset]
    test:       Dataset
    full_size:  int
    split_size: int


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


def create_datasets(
    dataset: TCDataset, data_path: str, num_splits: int, iid: bool, alpha: float = 0.1, seed: int = 0
) -> DatasetSplit:
    """Get a partitioned dataset."""
    if dataset == TCDataset.CIFAR10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    elif dataset == TCDataset.MNIST:
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

    else:
        assert False, "Only CIFAR10 and MNIST are supported"

    if iid:
        split_datasets = _get_iid_splits(full_dataset, num_splits)
    else:
        split_datasets = _get_noniid_splits(full_dataset, num_splits, alpha, seed, transform)

    assert isinstance(split_datasets[0], Sized)
    return DatasetSplit(split_datasets, test_set, len(full_dataset), len(split_datasets[0]))


def _get_iid_splits(dataset: Dataset, num_splits: int) -> List[Dataset]:
    """Generate IID dataset partitions. Note that it is not 100% IID."""
    assert isinstance(dataset, Sized)
    assert len(dataset) % num_splits == 0
    return random_split(dataset, [len(dataset) // num_splits] * num_splits)  # type: ignore


def _get_noniid_splits(
    dataset: Dataset, num_splits: int, alpha: float, seed: int, transform: Optional[transforms.Compose]
) -> List[Dataset]:
    """Generate Non-IID dataset partitions based on a Dirichlet distribution.

    Reference paper: https://arxiv.org/abs/1909.06335
    Reference implementation: https://github.com/adap/flower/tree/main/baselines/fedavgm
    """
    assert hasattr(dataset, "data")
    train_data: np.ndarray = dataset.data  # type: ignore

    assert hasattr(dataset, "targets")
    train_labels = np.asarray(dataset.targets)  # type: ignore

    train_data, train_labels = _shuffle(train_data, train_labels)
    train_data, train_labels = _sort_by_label(train_data, train_labels)

    assert train_data.shape[0] % num_splits == 0, "Number of splits must divide the number of samples"
    samples_per_split = train_data.shape[0] // num_splits
    num_samples = [samples_per_split] * num_splits

    # Get classes and starting indices for each class (assuming train_labels is sorted)
    classes, start_indices = np.unique(train_labels, return_index=True)

    # Split the train data into sub-arrays by class
    list_samples_per_class_np = np.split(train_data, start_indices[1:])
    list_samples_per_class = [_squeeze_numpy_to_list(i) for i in list_samples_per_class_np]

    dirichlet_dist = np.random.default_rng(seed).dirichlet([alpha] * classes.size, num_splits)
    splits: List[NDArrayT2] = [(_, _) for _ in range(num_splits)]  # type: ignore
    empty_classes = [False] * classes.size

    for split_id in range(num_splits):
        splits[split_id], empty_classes = sample_without_replacement(
            dirichlet_dist[split_id].copy(),
            list_samples_per_class,
            num_samples[split_id],
            empty_classes,
        )

    return _splits_to_datasets(splits, transform)


def _shuffle(x: np.ndarray, y: np.ndarray) -> NDArrayT2:
    """Shuffle data and labels."""
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]


def _sort_by_label(x: np.ndarray, y: np.ndarray) -> NDArrayT2:
    """Sort data and labels by labels."""
    idx = np.argsort(y, axis=0).reshape((y.shape[0]))
    return x[idx], y[idx]


def _squeeze_numpy_to_list(array: np.ndarray) -> List[np.ndarray]:
    """Turn the first dimension of a numpy array into a list."""
    return [array[i] for i in range(array.shape[0])]


def _splits_to_datasets(splits: List[NDArrayT2], transform: Optional[transforms.Compose] = None) -> List[Dataset]:
    """Convert a list of data splits into PyTorch Datasets."""
    return [NumpyDataset(data, labels, transform) for data, labels in splits]

