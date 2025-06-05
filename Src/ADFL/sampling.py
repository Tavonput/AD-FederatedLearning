from typing import Tuple, List

import numpy as np

from .types import NDArrayT2


def sample_half_normal(samples: int, sigma: float, seed: int = 0) -> List[float]:
    """Sample from a half-normal distribution centered at 0."""
    values = np.abs(np.random.default_rng(seed).normal(0, sigma, samples))
    return [x.item() for x in values]


def sample_without_replacement(
    distribution: np.ndarray, list_samples: List[List[np.ndarray]], num_samples: int, empty_classes: List[bool]
) -> Tuple[NDArrayT2, List[bool]]:
    """Sample from the list of samples without replacement given a distribution."""
    assert np.sum([len(x) for x in list_samples]) >= num_samples

    distribution = _exclude_classes_and_normalize(distribution, empty_classes)

    data: List[np.ndarray] = []
    target: List[np.intp] = []

    for _ in range(num_samples):
        sample_class = np.argmax(np.random.multinomial(1, distribution))
        sample = list_samples[sample_class].pop()

        data.append(sample)
        target.append(sample_class)

        if len(list_samples[sample_class]) == 0:
            empty_classes[sample_class] = True
            distribution = _exclude_classes_and_normalize(distribution, empty_classes)

    data_array = np.stack(data, axis=0)
    target_array = np.array(target, dtype=np.int64)

    return (data_array, target_array), empty_classes


def _exclude_classes_and_normalize(distribution: np.ndarray, exclude_dims: List[bool], eps: float = 1e-5) -> np.ndarray:
    """Excludes classes from a distribution by setting the probability to 0, followed by a normalize."""
    assert np.isclose(np.sum(distribution), 1.0)
    assert distribution.size == len(exclude_dims)
    assert eps > 0

    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows

    return distribution
