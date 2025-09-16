import time
from typing import Tuple, Optional

from ADFL.messages import AsyncClientTrainMessage, EvalMessage
from ADFL.types import Scalar
from ADFL.model import Parameters
from ADFL.eval import EvalActorProxy

from ADFL.Channel import Channel
from ADFL.Client import AsyncClient, AsyncClientWorkerProxy


def train_client(
    client:    AsyncClient,
    worker:    AsyncClientWorkerProxy,
    params:    Parameters,
    g_round:   int,
    epochs:    int,
    channel:   Channel,
    bandwidth: Optional[float],
) -> Tuple[float, float]:
    """Train an AsyncClient and get the compression and bandwidth times."""
    c_params, c_time = channel.on_server_send(params)
    msg = AsyncClientTrainMessage(c_params, epochs, g_round)

    start_time = time.time()

    if bandwidth is not None:
        _ = channel.simulate_bandwidth(params, bandwidth)

    worker.train_client(client, msg)

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
