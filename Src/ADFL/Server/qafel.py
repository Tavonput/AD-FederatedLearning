from typing import Callable, List, Dict, Optional

import torch.nn as nn

import ray

from ADFL import my_logging
from ADFL.types import EvalConfig, TrainingConfig, AsyncServerResults
from ADFL.messages import ClientUpdateMessage, AsyncClientTrainMessage
from ADFL.model import CompressedParameters, Parameters, get_model_parameters, diff_parameters, add_parameters_inpace
from ADFL.flag import RemoteFlag
from ADFL.eval import EvalActorProxy
from ADFL.resources import NUM_CPUS

from ADFL.Client import AsyncClientV2, AsyncClientWorkerProxy, DistributedClientPool

from .common import Server, ServerCore


@ray.remote(num_cpus=NUM_CPUS)
class QAFeLServer(Server):
    """QAFeL Server.

    Server implementation of Quantized Asynchronous Federated Learning.
    """
    def __init__(
        self,
        model_fn:     Callable[[], nn.Module],
        train_config: TrainingConfig,
        eval_config:  EvalConfig,
        evaluator:    Optional[EvalActorProxy],
        stop_flag:    RemoteFlag,
    ):
        self.log = my_logging.get_logger("SERVER")
        self.log.info("Initializing")

        self.core = ServerCore(model_fn, train_config, eval_config, evaluator, stop_flag, self.log)

        # {client_id: worker_id}
        self.client_worker_map: Dict[int, int] = {}

        self.hidden_state: Parameters = get_model_parameters(self.core.global_model)

        self.bandwidth = train_config.delay.server_mbps
        self.broadcast_full = False


    def initialize(self) -> bool:
        return True


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        self.core.add_clients(clients)


    def add_workers(self, workers: List[AsyncClientWorkerProxy]) -> None:
        self.core.add_workers(workers)


    def attach_client_pool(self, pool: DistributedClientPool) -> None:
        self.core.attach_client_pool(pool)


    def get_results(self) -> AsyncServerResults:
        return self.core.get_results()


    def get_model(self) -> Parameters:
        return self.core.get_model()


    def stop(self) -> None:
        self.core.stop()


    def run(self) -> None:
        self.log.info("Starting training")

        for i, _ in enumerate(self.core.workers):
            client_idx = self.core.select_client()
            self._train_client(client_idx, i)


    def client_update(self, client_update: ClientUpdateMessage) -> None:
        """Message: Process a client update."""
        if self.core.stop_flag.state() == True:
            # This will happen if the driver calls for a stop in the event of a timeout.
            return

        # Simulate receiving the parameters over the communication channel (performs decompression).
        client_params, _ = self.core.channel.on_server_receive(client_update.parameters)

        # Notify the strategy that this client has finished.
        self.core.strategy.on_client_finish(client_update.client_id)

        # Get the worker that was working on this client.
        worker_idx = self.client_worker_map[client_update.client_id]
        self.client_worker_map[client_update.client_id] = -1

        # Stop the worker if we hit the maximum number of global communication rounds.
        if self.core.g_round > self.core.train_config.num_rounds:
            # Stop the worker without blocking. At this point the worker should be idle and we won't be sending any
            # more requests to it.
            self.core.workers[worker_idx].stop(block=False)
            self.core.end_of_training()

            self.core.finished_workers += 1
            if self.core.finished_workers >= self.core.train_config.num_cur_clients:
                self.log.info("All workers have finished")
                self.stop()

            return

        self.core.g_round += 1
        self.core.save_update(client_update)

        # This is a bit scuffed, but we will track the strategy's round to see if a buffer flush happened
        current_round = self.core.strategy.get_round()

        self.log.debug(f"Aggregating updates from Client {client_update.client_id}")
        self._aggregate(client_params, client_update.g_round)

        next_round = self.core.strategy.get_round()

        if next_round != current_round:
            self.log.info(f"Broadcasting update to all clients")
            q_update = self._broadcast_update()
            self._update_hidden_state(q_update)
            self.log.debug(f"Finished broadcast")

        self.core.possibly_eval()

        client_idx = self.core.select_client()
        self._train_client(client_idx, worker_idx)


    def _aggregate(self, client_params: Parameters, client_model_step: int) -> None:
        """Aggregate client updates into global model."""
        staleness = self.core.strategy.get_round() - client_model_step
        if self.core.train_config.metrics.staleness:
            self.core.staleness.append(staleness)
        self.core.aggregate([client_params], staleness)


    def _train_client(self, client_id: int, worker_id: int) -> None:
        """Send a client training job to a worker."""
        g_round = self.core.strategy.get_round()
        self.client_worker_map[client_id] = worker_id

        msg = AsyncClientTrainMessage(None, self.core.train_config.num_epochs, g_round)

        self.log.debug(f"Sending training job: client={client_id} worker={worker_id} round={g_round}")
        self.core.workers[worker_id].train_client_no_model(client_id, msg)


    def _broadcast_update(self) -> CompressedParameters:
        """Broadcast the most recent update to all clients."""
        # At this point, the current global model should have progressed to the next step
        g_params = get_model_parameters(self.core.global_model)
        update = diff_parameters(self.hidden_state, g_params)
        q_update, _ = self.core.channel.on_server_send(update)  # Compress

        assert self.core.client_pool is not None
        assert self.bandwidth is not None

        if self.broadcast_full:
            _ = self.core.channel.simulate_bandwidth(g_params, self.bandwidth)
            self.core.client_pool.broadcast_aggregate_client_model(q_update, self.core.channel)
        else:
            for c_id in range(self.core.client_pool.num_clients):
                _ = self.core.channel.simulate_bandwidth(g_params, self.bandwidth)
                self.core.client_pool.aggregate_client_model(c_id, q_update, self.core.channel)

        return q_update


    def _update_hidden_state(self, q_update: CompressedParameters) -> None:
        """Update the hidden state to match the clients."""
        d_update, _ = self.core.channel.on_client_receive(q_update)
        add_parameters_inpace(self.hidden_state, d_update, 1, 1, to_float=False)
