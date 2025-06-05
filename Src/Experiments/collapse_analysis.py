import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

from typing import List
import time

from ADFL.my_logging import init_logging, get_logger
from ADFL.types import TCMethod, TCDataset, TrainingConfig, EvalConfig
from ADFL.Driver import *
from ADFL.Driver.common import run_driver


def get_compute_for_clients(num_clients: int) -> List[float]:
    if num_clients == 2:
        return [1.00, 2.00]

    elif num_clients == 4:
        return [1.00, 1.00, 1.33, 2.00]

    elif num_clients == 8:
        return [
            1.00, 1.00, 1.00, 1.00,
            1.33, 1.33, 2.00, 2.00,
        ]

    else:
        assert False


def get_eval_threshold(num_clients: int, num_epochs: int) -> int:
    mappings = {
        (2, 1):  4,
        (2, 2):  2,
        (2, 3):  1,
        (2, 5):  1,
        (2, 10): 1,
        (4, 1):  8,
        (4, 2):  4,
        (4, 3):  2,
        (4, 5):  1,
        (4, 10): 1,
        (8, 1):  16,
        (8, 2):  8,
        (8, 3):  4,
        (8, 5):  2,
        (8, 10): 1,
    }
    key = (num_clients, num_epochs)
    assert key in mappings
    return mappings[key]


def main() -> None:
    init_logging()
    log = get_logger("MAIN")

    results_base_path = "../Output/Results/Deltas/Collapse/Real"
    tmp_path = "/data/tavonputl/tmp/ray"
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 8,
        num_actors = 1,
        client_map = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    train_config = TrainingConfig(
        method       = TCMethod.DELTA,
        dataset      = TCDataset.MNIST,
        num_rounds   = 1000000,
        num_epochs   = 1,
        num_clients  = 8,
        num_servers  = 1,
        batch_size   = 512,
        max_rounds   = 1000000,  # Not used anymore (needs to be refactored out)
        timeout      = 600,
        compress     = "byte",
        quant_lvl_1  = 8,
        quant_lvl_2  = 4,
        slowness_map = None,
        sc_map       = None,
    )

    epochs   = [1, 2, 3, 5, 10]
    clients  = [2, 4, 8]
    computes = ["homo", "hetero"]
    datasets = [TCDataset.MNIST, TCDataset.CIFAR10]


    for dataset in datasets:
        for epoch in epochs:
            for client in clients:
                for compute in computes:

                    eval_config.threshold = get_eval_threshold(client, epoch)

                    train_config.dataset = dataset
                    train_config.num_epochs = epoch
                    train_config.num_clients = client

                    if compute == "hetero":
                        train_config.slowness_map = get_compute_for_clients(client)
                    elif compute == "homo":
                        train_config.slowness_map = None

                    results_path = f"{results_base_path}/async_delta_e{epoch}_c{client}_{compute}_{dataset.value}.json"
                    driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=False)
                    run_driver(driver, train_config, eval_config)
                    del driver

                    time.sleep(5)


if __name__ == "__main__":
    main()
