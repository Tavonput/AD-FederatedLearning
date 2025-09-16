import ray
import memray
import torch

from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import TrainingConfig, EvalConfig
from ADFL.messages import AsyncClientTrainMessage
from ADFL.my_logging import get_logger
from ADFL.memory import MEMRAY_PATH, MEMRAY_RECORD

from .async_sc import AsyncClient


# @ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
@ray.remote(num_cpus=NUM_CPUS)
class AsyncClientWorker:
    """Worker node for an AsyncClient."""
    def __init__(self, worker_id: int, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        if MEMRAY_RECORD:
            memray.Tracker(f"{MEMRAY_PATH}{self.__class__.__name__}_{worker_id}_mem_profile.bin").__enter__()

        self.log = get_logger(f"WORKER {worker_id}")
        self.worker_id = worker_id

        self.device: torch._C.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_config = train_config
        self.eval_config = eval_config

        assert eval_config.central, (
            "Distributed evaluation is currently not supported. The AsyncServer has no way of getting the client " +
            "accuracies. Client accuracies would need to be stored on the server side since clients are no longer " +
            "actors."
        )

        self.ready = False
        self.stop_flag = False


    def initialize(self) -> None:
        self.log.info(f"Initialized: device={self.device}")
        self.ready = True
        return


    def stop(self) -> None:
        """Stop working."""
        if self.stop_flag is True:
            return

        self.log.info("Stopping")
        self.stop_flag = True


    def train_client(self, client: AsyncClient, msg: AsyncClientTrainMessage) -> None:
        if self.stop_flag:
            return

        client.train(msg, self.device)


class AsyncClientWorkerProxy:
    """Proxy class for interacting with an AsyncClientWorker actor."""
    def __init__(self, worker_id: int, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        self.actor = AsyncClientWorker.remote(worker_id, train_config, eval_config)


    def initialize(self) -> None:
        ray.get(self.actor.initialize.remote())  # type: ignore


    def stop(self, block: bool = True) -> None:
        if block:
            ray.get(self.actor.stop.remote())  # type: ignore
        else:
            self.actor.stop.remote()  # type: ignore


    def train_client(self, client: AsyncClient, msg: AsyncClientTrainMessage) -> None:
        self.actor.train_client.remote(client, msg)  # type: ignore
