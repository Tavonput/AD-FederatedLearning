from dataclasses import dataclass
from typing import List
from pathlib import Path
import math

from ADFL.types import TCDataset

from ADFL.Utils.federated_results import FederatedResultsExplorer, ScalarPairs


BASE_DIR = "../Output/Results/Deltas/Collapse/Real"


@dataclass
class Config:
    name:           str

    epochs:         int
    num_clients:    int
    hetero:         bool
    dataset:        TCDataset

    accuracies:     ScalarPairs
    final_accuracy: float
    mses:           List[ScalarPairs]


def parse_data_dir(dirpath: str) -> List[Config]:
    configs: List[Config] = []

    dir = Path(dirpath)
    window = 60

    for file in dir.iterdir():
        filepath = str(file.absolute())

        explorer = FederatedResultsExplorer()
        explorer.set_federated_results_from_file(filepath)

        is_hetero = (explorer.fr.train_config.slowness_map is not None)
        config = Config(
            "",  # Filled in later
            explorer.fr.train_config.num_epochs,
            explorer.fr.train_config.num_clients,
            is_hetero,
            explorer.fr.train_config.dataset,
            explorer.get_central_accuracies_raw(),
            explorer.get_central_accuracies_final(window),
            explorer.get_mse_all(),
        )

        # Name is "[Dataset]-[Compute]-e[epochs]-c[clients]"
        compute_name = "hetero" if is_hetero else "homo"
        config.name = f"{config.dataset.value}-{compute_name}-e{config.epochs}-c{config.num_clients}"

        configs.append(config)

    return configs


def is_mse_stable(mses: ScalarPairs, base_window: int, threshold: float) -> bool:
    for (_, mse) in mses[:base_window]:
        if math.isfinite(mse) is False:
            return False

    baseline = min(mses[:base_window], key=lambda x: x[1])[1]
    for (_, mse) in mses[base_window:]:
        if mse >= baseline * threshold:
            return False

    return True


def main():
    configs = parse_data_dir(BASE_DIR)

    for config in configs:
        stable = is_mse_stable(config.mses[0], 10, 100)
        print(config.name, stable, config.final_accuracy)


if __name__ == "__main__":
    main()
