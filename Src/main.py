import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

from Adfl.types import TrainingConfig
from Adfl.driver import SyncDriver


def run_sync(train_config: TrainingConfig):
    driver = SyncDriver(timeline_path="../Output/timeline.json", tmp_path="/data/tavonputl/tmp/ray")

    driver.init_backend()
    driver.init_training(train_config)

    driver.run()

    driver.shutdown()

def main() -> None:
    train_config = TrainingConfig(
        num_rounds  = 3,
        num_epochs  = 1,
        num_clients = 4,
        batch_size  = 512,
        max_rounds  = 0,
    )

    run_sync(train_config)

if __name__ == "__main__":
    main()
