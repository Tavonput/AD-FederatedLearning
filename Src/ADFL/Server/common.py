import time
from typing import Tuple

import torch.nn as nn

from ADFL.messages import AsyncClientTrainMessage, EvalMessage
from ADFL.compression import compress_model
from ADFL.types import Scalar
from ADFL.model import Parameters
from ADFL.eval import EvalActorProxy

from ADFL.Client import AsyncClientProxy


def train_client(
    client: AsyncClientProxy, model: nn.Module, g_round: int, epochs: int, method: str, bits: int
) -> Tuple[float, float]:
    """Train an AsyncClient and get the compression and bandwidth times."""
    c_params, c_time = compress_model(model, method, bits)
    msg = AsyncClientTrainMessage(c_params, epochs, g_round)

    start_time = time.time()
    client.train(msg)
    b_time = time.time() - start_time

    return c_time, b_time


def need_to_eval(method: str, g_round: int, threshold: Scalar) -> bool:
    """Check if we need to eval."""
    if method == "time":
        assert False, "Central time eval is not supported"

    elif method == "round":
        return (g_round % threshold == 0)

    else:
        assert False, "Invalid eval config method"


def send_eval_message(params: Parameters, s_id: int, evaluator: EvalActorProxy) -> None:
    """Send an evaluation message to the evaluator."""
    message = EvalMessage(
        parameters = params,
        client_id  = s_id,
        g_time     = time.time(),
    )
    evaluator.evaluate(message)
