from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import time
from copy import deepcopy

import ray

from ADFL.resources import NUM_CPUS
from ADFL.model import Parameters, add_parameters_inpace, CompressedParameters
from ADFL.my_logging import get_logger
from ADFL.Channel import Channel

from .async_sc import AsyncClientV2


@dataclass
class ClientState:
    meta:  AsyncClientV2
    model: Parameters  # Empty if not used


@ray.remote(num_cpus=NUM_CPUS)
class ClientPool:
    """Client Pool.

    Maintains a set of client states. A client state can be empty if it is not needed.

    """
    def __init__(self, p_id: int) -> None:
        self.log = get_logger(f"POOL {p_id}")

        self.p_id = p_id
        self.clients: List[ClientState] = []


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        for client in clients:
            self.clients.append(ClientState(client, {}))


    def get_client(self, idx: int) -> ClientState:
        return self.clients[idx]


    def get_all_clients(self) -> List[ClientState]:
        return self.clients


    # TODO: Enums?
    def update_client_state(self, idx: int, updates: List[Tuple[str, Any]]) -> None:
        client = self.clients[idx]
        for field, data in updates:
            if field == "round":
                client.meta.round = data
            elif field == "accuracy":
                client.meta.accuracies.append(data)
            else:
                assert False, f"Invalid client update field: {field}"


    def add_to_model(self, idx: int, c_alpha: CompressedParameters, channel: Channel) -> None:
        alpha, _ = channel.on_client_receive(c_alpha)  # May want to store d_time somewhere

        client = self.clients[idx]
        add_parameters_inpace(client.model, alpha, 1, 1, to_float=False)
        client.meta.g_model_step += 1


    def add_to_model_all(self, c_alpha: CompressedParameters, channel: Channel) -> None:
        alpha, _ = channel.on_client_receive(c_alpha)

        for client in self.clients:
            add_parameters_inpace(client.model, alpha, 1, 1, to_float=False)
            client.meta.g_model_step += 1


    def init_models(self, params: Parameters) -> None:
        for client in self.clients:
            client.model = deepcopy(params)


    def log_clients(self) -> None:
        client_ids = [c.meta.client_id for c in self.clients]
        self.log.info(client_ids)


class ClientPoolProxy:
    """Proxy for interfacing with a ClientPool actor."""
    def __init__(self, p_id: int) -> None:
        self._actor = ClientPool.remote(p_id)


    def add_clients(self, clients: List[AsyncClientV2]) -> None:
        ray.get(self._actor.add_clients.remote(clients))  # type: ignore


    def get_client(self, idx: int) -> ClientState:
        return ray.get(self._actor.get_client.remote(idx))  # type: ignore


    def get_all_clients(self) -> List[ClientState]:
        return ray.get(self._actor.get_all_clients.remote())  # type: ignore


    def update_client_state(self, idx: int, updates: List[Tuple[str, Any]]) -> None:
        return ray.get(self._actor.update_client_state.remote(idx, updates))  # type: ignore


    def add_to_model(self, idx: int, c_alpha: CompressedParameters, channel: Channel) -> None:
        self._actor.add_to_model.remote(idx, c_alpha, channel)  # type: ignore


    def add_to_model_all(self, c_alpha: CompressedParameters, channel: Channel) -> None:
        self._actor.add_to_model_all.remote(c_alpha, channel)  # type: ignore


    def init_models(self, params: Parameters) -> None:
        ray.get(self._actor.init_models.remote(params))  # type: ignore


    def log_clients(self) -> None:
        ray.get(self._actor.log_clients.remote())  # type: ignore


class DistributedClientPool:
    """Distributed Client Pool.

    Manages and stores clients in a distributed manner. A distributed client pool of size one is the same a single
    client pool. There is no head/central node. Internally, the distributed client pool holds actor handles to a set of
    ClientPool actors. Helper functions are provided to callers to direct communication traffic to the correct client
    pool.
    """
    def __init__(self, num_clients: int, num_pools: int, logger: logging.Logger) -> None:
        self.log = logger

        self.num_clients = num_clients
        self.num_pools = num_pools

        self.pools: List[ClientPoolProxy] = []


    def set_logger(self, logger: logging.Logger) -> None:
        self.log = logger


    def init_pools(self, clients: List[AsyncClientV2], verbose: bool = False) -> None:
        assert len(clients) == self.num_clients

        pool_partitions = self._get_pool_partition()
        self.log.info(f"Client pools sizes: {pool_partitions}")

        idx = 0
        for i, pool_size in enumerate(pool_partitions):
            pool = ClientPoolProxy(i)
            pool.add_clients(clients[idx:idx + pool_size])
            idx += pool_size
            self.pools.append(pool)

        if verbose:
            [p.log_clients() for p in self.pools]


    def init_client_models(self, params: Parameters) -> None:
        self.log.info("Initializing model on all clients. This should spike memory")
        for pool in self.pools:
            pool.init_models(params)


    def update_client_state(self, idx: int, updates: List[Tuple[str, Any]]) -> None:
        """Update a client's state. Should block? Probably should block."""
        pool, offset = self._get_pool_and_offset(idx)
        self.pools[pool].update_client_state(offset, updates)


    def get_client(self, c_id: int) -> Tuple[ClientState, float]:
        """Get a client by their id and the fetch time."""
        pool, offset = self._get_pool_and_offset(c_id)

        s_time = time.perf_counter()
        client = self.pools[pool].get_client(offset)

        return client, (time.perf_counter() - s_time)


    def get_all_clients_meta(self) -> List[AsyncClientV2]:
        """Get the meta data (AsyncClientV2) for all clients."""
        # TODO: Maybe this shouldn't collect the entire client, just the client state. Collecting the entire client
        # could cause memory issues, and it's slower.

        client_states: List[ClientState] = []
        for pool in self.pools:
            client_states += pool.get_all_clients()

        return [cs.meta for cs in client_states]


    def aggregate_client_model(self, c_id: int, c_alpha: CompressedParameters, channel: Channel) -> None:
        """Add alpha to a client's model."""
        pool, offset = self._get_pool_and_offset(c_id)
        self.pools[pool].add_to_model(offset, c_alpha, channel)


    def broadcast_aggregate_client_model(self, c_alpha: CompressedParameters, channel: Channel) -> None:
        """Send a broadcast to all client pools."""
        for pool in self.pools:
            pool.add_to_model_all(c_alpha, channel)


    def _get_pool_partition(self) -> List[int]:
        """Get the number of clients per pool."""
        base = self.num_clients // self.num_pools
        remainder = self.num_clients % self.num_pools

        if remainder != 0:
            self.log.warning("Number of pools does not evenly divide the number of clients")

        # Distributes the remainder across the front pools
        return ([base + 1] * remainder) + ([base] * (self.num_pools - remainder))


    def _get_pool_and_offset(self, c_id: int) -> Tuple[int, int]:
        """Get the pool and offset for a given client id."""
        base = self.num_clients // self.num_pools
        remainder = self.num_clients % self.num_pools

        # If c_id is within the larger pools
        if c_id < (base + 1) * remainder:
            pool = c_id // (base + 1)
            offset = c_id % (base + 1)
        else:
            # Now we are looking for the client within the smaller pools
            relative_c_id = c_id - (base + 1) * remainder
            pool = remainder + (relative_c_id // base)
            offset = relative_c_id % base

        return pool, offset
