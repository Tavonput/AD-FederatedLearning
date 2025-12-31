from ADFL.Client.pool import *
from ADFL.my_logging import get_logger
from ADFL.Client import AsyncClientV2
from ADFL.types import Accuracy

class TestDistributedClientPool:
    @staticmethod
    def init_pool():
        logger = get_logger("LOG")

        clients = [
            AsyncClientV2(i, None, None, 0, 0, None, None) for i in range(10)  # type: ignore
        ]
        dcp = DistributedClientPool(10, 3, logger)
        dcp.init_pools(clients, verbose=True)

        # Ensure that the client pools own the clients
        del clients

        client, _ = dcp.get_client(8)
        assert client.meta.client_id == 8

        client, _ = dcp.get_client(0)
        assert client.meta.client_id == 0


    @staticmethod
    def update_client_state():
        logger = get_logger("LOG")

        clients = [
            AsyncClientV2(i, None, None, 0, 0, None, None) for i in range(3)  # type: ignore
        ]
        dcp = DistributedClientPool(3, 2, logger)
        dcp.init_pools(clients, verbose=True)

        CLIENT_ID = 1

        update = [
            ("round", 1),
            ("accuracy", Accuracy(60, 2)),
        ]
        dcp.update_client_state(CLIENT_ID, update)
        client, _ = dcp.get_client(CLIENT_ID)
        logger.info(f"Round: {client.meta.round}, Accuracy: {client.meta.accuracies}")

        update = [
            ("round", 2),
            ("accuracy", Accuracy(50, 2)),
        ]
        dcp.update_client_state(CLIENT_ID, update)
        client, _ = dcp.get_client(CLIENT_ID)
        logger.info(f"Round: {client.meta.round}, Accuracy: {client.meta.accuracies}")
