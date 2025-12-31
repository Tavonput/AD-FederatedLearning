from typing import Callable, Optional, Tuple, List, Any
import array
import time

import ray
import memray
import torch
import torch.nn as nn
import numpy as np

from ADFL.resources import NUM_CPUS, NUM_GPUS
from ADFL.types import TrainingConfig, EvalConfig, AsyncClientWorkerResults, RoundResults
from ADFL.model import (
    Parameters, CompressedParameters,
    parameter_relative_mse, parameter_cosine_similarity, parameter_mean_var, set_model_parameters
)
from ADFL.messages import AsyncClientTrainMessage, ClientUpdateMessage
from ADFL.my_logging import get_logger
from ADFL.memory import MEMRAY_PATH, MEMRAY_RECORD

from .async_sc import AsyncClientV2
from .pool import DistributedClientPool


# @ray.remote(num_cpus=NUM_CPUS)
@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class AsyncClientWorker:
    """Worker node for an AsyncClient."""
    def __init__(
        self,
        worker_id: int,
        train_config: TrainingConfig,
        eval_config: EvalConfig,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        if MEMRAY_RECORD:
            memray.Tracker(f"{MEMRAY_PATH}{self.__class__.__name__}_{worker_id}_mem_profile.bin").__enter__()

        self.log = get_logger(f"WORKER {worker_id}")
        self.worker_id = worker_id

        self.model = model_fn()
        self.device: torch._C.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.client_pool: Optional[DistributedClientPool] = None

        self.fetch_times: array.array = array.array("d")
        self.fetch_counter = 0
        self.fetch_get_raw = train_config.metrics.fetch_raw
        self.uptime = 0

        self.eval_requests_sent = 0

        self.train_config = train_config
        self.eval_config = eval_config

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


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        self.client_pool = pool
        self.client_pool.set_logger(self.log)


    def get_metrics(self) -> AsyncClientWorkerResults:
        return AsyncClientWorkerResults(
            worker_id    = self.worker_id,
            fetch_times  = (self.fetch_times.tolist() if self.fetch_get_raw else []),
            fetch_mean   = float(np.mean(self.fetch_times)),
            fetch_std    = float(np.std(self.fetch_times)),
            fetch_max    = float(np.max(self.fetch_times)),
            fetch_min    = float(np.min(self.fetch_times)),
            uptime       = self.uptime,
            num_eval_req = self.eval_requests_sent,
        )


    def train_client(self, c_id: int, msg: AsyncClientTrainMessage) -> None:
        if self.stop_flag:
            return

        s_time = time.perf_counter()

        self.log.debug(f"Fetching client {c_id}")
        assert self.client_pool is not None
        client, fetch_time = self.client_pool.get_client(c_id)

        if msg.parameters is None:
            # Client pool should have the model ready
            assert client.model, "Model retrieved from the client pool is empty"
            self.log.debug(f"Setting model from local client state")
            set_model_parameters(self.model, client.model)
            d_time = 0  # No decompression time. For now it's not needed
        else:
            # Simulate parameter transfer across the communication channel
            assert not client.model, "Model parameters should not be stored in its state"

            # TODO: Make this process better
            msg.parameters: CompressedParameters = ray.get(msg.parameters)  # type: ignore
            params, d_time = self.train_config.channel.on_client_receive(msg.parameters)  # type: ignore

            set_model_parameters(self.model, params)
            client.meta.g_model_step = msg.g_round  # Only changes locally

        self.fetch_counter += 1
        if self.fetch_counter % self.train_config.metrics.fetch_freq == 0:
            self.fetch_times.append(fetch_time)

        self._process_client(client.meta, msg, d_time)

        self.uptime += time.perf_counter() - s_time


    def train_client_no_model(self, c_id: int, msg: AsyncClientTrainMessage)-> None:
        assert msg.parameters is None
        self.train_client(c_id, msg)


    def _process_client(self, client: AsyncClientV2, msg: AsyncClientTrainMessage, d_time: float) -> None:
        params_prime, round_results = self._train_pass(client, msg.epochs)
        self._update_client_state(client, round_results)
        self._send_update_to_server(params_prime, round_results, d_time, client)


    def _train_pass(self, client: AsyncClientV2, epochs: int) -> Tuple[Parameters, RoundResults]:
        """Actually train the client."""
        start_time = time.time()

        self.log.debug(f"Training client {client.client_id}")
        params_prime, round_results = client.train(
            self.model, self.device, self.train_config, self.eval_config, epochs
        )
        round_results.g_start_time = start_time

        if round_results.sent_eval_req is True:
            self.eval_requests_sent += 1

        return params_prime, round_results


    def _update_client_state(self, client: AsyncClientV2, round_results: RoundResults) -> None:
        """Update the state of a client after training."""
        cs_update: List[Tuple[str, Any]] = [("round", round_results.train_round)]
        if round_results.accuracy is not None:
            cs_update.append(("accuracy", round_results.accuracy))

        assert self.client_pool is not None
        self.client_pool.update_client_state(client.client_id, cs_update)


    def _send_update_to_server(
        self,
        parameters:    Parameters,
        round_results: RoundResults,
        d_time:        float,
        client:        AsyncClientV2,
    ) -> None:
        """Send a round update to the server."""
        # Simulate parameter transfer across the communication channel (performs compression)
        c_params, c_time = self.train_config.channel.on_client_send(parameters)

        # Simulate delays
        compute_delay, compute_time = self._simulate_compute_delay(d_time, c_time, round_results, client.compute_delay)
        network_delay = self._simulate_network_delay(parameters, client.network_delay)

        round_results.compute_time = compute_delay + compute_time
        round_results.network_time = network_delay
        round_results.round_time = time.time() - round_results.g_start_time

        if self.train_config.metrics.q_error:
            d_params, _ = self.train_config.channel.on_server_receive(c_params)
            round_results.q_error_mse = parameter_relative_mse(parameters, d_params, exclude_bias=True)
            round_results.q_error_cos = parameter_cosine_similarity(parameters, d_params, exclude_bias=True)

        if self.train_config.metrics.model_dist:
            round_results.model_dist = parameter_mean_var(parameters, exclude_bias=True)

        update = ClientUpdateMessage(
            parameters    = c_params,
            client_id     = client.client_id,
            client_round  = client.round,
            g_round       = client.g_model_step,
            round_results = round_results,
            num_examples  = len(client.train_loader.dataset),  # type: ignore
        )

        self.log.debug(f"Sending update to server. model_step={client.g_model_step}")
        client.server.client_update(update)  # type: ignore


    def _simulate_compute_delay(
        self, decompress_time: float, compress_time: float, round_results: RoundResults, compute_delay: float
    ) -> Tuple[float, float]:
        """Simulate compute delay."""
        train_time = 0
        for train_result in round_results.train_results:
            train_time += train_result.elapsed_time

        total_compute_time = train_time + decompress_time + compress_time
        delay = total_compute_time * compute_delay
        time.sleep(delay)

        return delay, total_compute_time


    def _simulate_network_delay(self, params: Parameters, network_delay: float) -> float:
        """Simulate network delay."""
        if self.train_config.delay.network_sigma is not None:
            return self.train_config.channel.simulate_bandwidth(params, network_delay)
        else:
            return 0


class AsyncClientWorkerProxy:
    """Proxy class for interacting with an AsyncClientWorker actor."""
    def __init__(
        self, worker_id: int, train_config: TrainingConfig, eval_config: EvalConfig, model_fn: Callable[[], nn.Module],
    ) -> None:
        self.actor = AsyncClientWorker.remote(worker_id, train_config, eval_config, model_fn)


    def initialize(self) -> None:
        ray.get(self.actor.initialize.remote())  # type: ignore


    def stop(self, block: bool = True) -> None:
        if block:
            ray.get(self.actor.stop.remote())  # type: ignore
        else:
            self.actor.stop.remote()  # type: ignore


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        ray.get(self.actor.attach_client_pool.remote(pool))  # type: ignore


    def get_metrics(self) -> AsyncClientWorkerResults:
        return ray.get(self.actor.get_metrics.remote())  # type: ignore


    def train_client(self, c_id: int, msg: AsyncClientTrainMessage) -> None:
        self.actor.train_client.remote(c_id, msg)  # type: ignore


    def train_client_no_model(self, c_id: int, msg: AsyncClientTrainMessage)-> None:
        self.actor.train_client_no_model.remote(c_id, msg)  # type: ignore
