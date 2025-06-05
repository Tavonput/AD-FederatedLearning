import os

from numpy.random import dirichlet
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

from typing import List, Tuple

from ADFL.my_logging import init_logging
from ADFL.types import TCDataset, TrainingConfig, EvalConfig
from ADFL.Driver import *
from ADFL.Strategy import FADAS, FedBuff


def clients_10() -> Tuple[List, List[List]]:
    slowness_map = [
        1.00, 1.00, 1.00, 1.00, 1.00, # Clients 0 - 4
        1.33, 1.33, 1.33,             # Clients 5 - 7
        2.00, 2.00,                   # Clients 8 - 9
    ]
    sc_map = [
        [0, 1, 5, 8], # Server 0
        [2, 3, 6],    # Server 1
        [4, 7, 9],    # Server 2
    ]
    return slowness_map, sc_map


def clients_8() -> Tuple[List, List[List]]:
    slowness_map = [
        1.00, 1.00, 1.00, 1.00, # Clients 0 - 3
        1.33, 1.33, 2.00, 2.00, # Clients 4 - 7
    ]
    sc_map = [
        [0, 1, 4], # Server 0
        [2, 5, 6], # Server 1
        [3, 7],    # Server 2
    ]
    return slowness_map, sc_map


def clients_2() -> Tuple[List, List[List]]:
    slowness_map = [
        1.00, 1.00
    ]
    sc_map = [
        [0], [1]
    ]
    return slowness_map, sc_map


def clients_1() -> Tuple[List, List[List]]:
    slowness_map = [1.00]
    sc_map = [[0]]
    return slowness_map, sc_map


def clients_8_special_case() -> List[List]:
    return [
        [0], [1], [2], [3], [4], [5], [6], [7]
    ]


def run(driver: Driver, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
    driver.init_backend()
    driver.init_training(train_config, eval_config)
    driver.run()
    driver.shutdown()


def main() -> None:
    init_logging()

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 3,
        num_actors = 1,
        client_map = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    slowness_map, _ = clients_8()

    results_base_path = "../Output/Results/Test"
    tmp_path = "/data/tavonputl/tmp/ray"
    timeline_path = "../Output/Timelines/test.json"

    train_config = TrainingConfig(
        strategy     = FedBuff(5, 1, apply_staleness=True),
        dataset      = TCDataset.CIFAR10,
        iid          = False,
        dirichlet_a  = 0.1,
        num_rounds   = 100000,
        num_epochs   = 1,
        num_clients  = 8,
        num_servers  = 1,
        batch_size   = 512,
        max_rounds   = 1000000,  # Not used anymore (needs to be refactored out)
        timeout      = 600,
        compress     = "byte",
        quant_lvl_1  = 8,
        quant_lvl_2  = 4,
        slowness_map = slowness_map,
        sc_map       = None,
    )

    results_path = f"{results_base_path}/test.json"
    driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=False)
    run(driver, train_config, eval_config)
    del driver


if __name__ == "__main__":
    main()
