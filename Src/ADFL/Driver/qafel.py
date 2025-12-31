from ADFL.Server import ServerType
from ADFL.types import TrainingConfig, EvalConfig
from ADFL.Strategy import FedBuff

from .async_sc import AsyncDriver


class QAFeLDriver(AsyncDriver):
    """QAFeL Driver

    Driver to run Quantized Asynchronous Federated Learning. It's the same as AsyncDriver but with a few minor changes:
    - Creates a QAFeLServer instance.
    - Initialized each client's model.
    """
    def __init__(
        self,
        timeline_path: str = None,  # type: ignore
        tmp_path: str = None,  # type: ignore
        results_path: str = "./results.json",
    ) -> None:
        super().__init__(timeline_path, tmp_path, results_path)

        # This will ensure a QAFeLServer is created
        self.server_type = ServerType.QAFEL


    def init_training(self, train_config: TrainingConfig, eval_config: EvalConfig) -> None:
        assert isinstance(train_config.strategy, FedBuff), "QAFeL must use FedBuff"
        super().init_training(train_config, eval_config)

        self.client_pool.init_client_models(self.server.get_model())
