import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

from typing import List, Tuple

from ADFL.types import TrainingConfig, EvalConfig
from ADFL.Driver import (
    Driver, 
    SyncDriver, 
    AsyncDriver, 
    AsyncPeerDriver,
    AsyncPeerDriverV2,
    AsyncHybridDriver,
)


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
    slowness_map, sc_map = clients_8()

    eval_config = EvalConfig(
        method     = "round",
        threshold  = 1,
        num_actors = 2,
        client_map = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    sc_map = clients_8_special_case()

    train_config = TrainingConfig(
        num_rounds   = 10000,
        num_epochs   = 5,
        num_clients  = 8,
        num_servers  = 8,
        batch_size   = 512,
        max_rounds   = 10000,
        timeout      = 60,
        slowness_map = slowness_map,
        sc_map       = sc_map,
    )

    results_base_path = "../Output/Results/Test"
    tmp_path = "/data/tavonputl/tmp/ray"
    timeline_path = None # "../Output/Timelines/async_hybrid_test.json" 

    # results_path = "../Output/Results/Paper/traditional_sc.json"
    # driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=True)
    # run(driver, train_config, eval_config)
    # del driver

    results_path = f"{results_base_path}/async_sc.json"
    driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=False)
    run(driver, train_config, eval_config)
    del driver

    # results_path = f"../Output/Results/Paper/async_peer.json"
    # driver = AsyncPeerDriverV2(timeline_path, tmp_path, results_path)
    # run(driver, train_config, eval_config)
    # del driver

    # results_path = f"{results_base_path}/async_peer_special.json"
    # driver = AsyncHybridDriver(timeline_path, tmp_path, results_path)    
    # run(driver, train_config, eval_config)
    # del driver

if __name__ == "__main__":
    main()
