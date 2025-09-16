from typing import List

from ADFL.types import TrainingConfig
from ADFL.dataset import create_datasets

from torch.utils.data import DataLoader

def main() -> None:
    datasets = create_datasets(
        TrainingConfig.Dataset.CIFAR10,
        2,
        False,
        "../Data",
    )

    loader = DataLoader(datasets.sets[0], batch_size=64, shuffle=True)
    batch = next(iter(loader))

    if isinstance(batch, List):
        i, l = batch
        print(i.shape, l.shape)

if __name__ == "__main__":
    main()
