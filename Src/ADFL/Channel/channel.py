import time
from typing import Tuple, Dict
from abc import ABC, abstractmethod

import torch

from ADFL.model import Parameters, CompressedParameters, ByteParameter, ByteParameters, get_parameter_info


class Channel(ABC):
    """Base Channel interface."""
    @abstractmethod
    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        """Logic during a server send request. Returns the parameters to send and the time."""
        pass


    @abstractmethod
    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        """Logic during a server receive. Returns the parameters received and the time."""
        pass


    @abstractmethod
    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        """Logic during a client send request. Returns the parameters to send and the time."""
        pass


    @abstractmethod
    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        """Logic during a client receive. Returns the parameters received and the time."""
        pass


    @abstractmethod
    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """Simulate bandwidth based on the transmission data. Returns the transfer time."""
        pass


    @abstractmethod
    def to_json(self) -> Dict:
        """Return serialization for json."""
        pass


class IdentityChannel(Channel):
    """Identity Channel.

    Just serializes and deserializes parameters over the channel.
    """
    def __init__(self, no_compute_time: bool) -> None:
        self.no_compute_time = no_compute_time


    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        b_params = self._serialize_params(params)
        return b_params, self._finalize_compute_time(s_time)


    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, ByteParameters)
        s_time = time.time()
        params = self._deserialize_params(c_params)
        return params, self._finalize_compute_time(s_time)


    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        b_params = self._serialize_params(params)
        return b_params, self._finalize_compute_time(s_time)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, ByteParameters)
        s_time = time.time()
        params = self._deserialize_params(c_params)
        return params, self._finalize_compute_time(s_time)


    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """All weights are considered 32 bits."""
        p_info = get_parameter_info(params)
        num_elements = p_info.num_bias_w + p_info.num_non_bias_w
        num_bytes = num_elements * 4

        bytes_per_second = mbps * 1_000_000 / 8
        transfer_time = num_bytes / bytes_per_second

        time.sleep(transfer_time)
        return transfer_time


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
        }



    def _serialize_params(self, params: Parameters) -> ByteParameters:
        """Serialize a set of parameters."""
        b_params = ByteParameters({}, 0)
        for name, param in params.items():
            b_params.params[name] = ByteParameter(
                data  = param.numpy().tobytes(),
                shape = param.shape,
                dtype = param.dtype,
            )
            b_params.size += len(b_params.params[name].data)

        return b_params


    def _deserialize_params(self, b_params: ByteParameters) -> Parameters:
        """Deserialize a set of byte parameters."""
        return {name: self._deserialize_tensor(b_param) for name, b_param in b_params.params.items()}


    def _deserialize_tensor(self, b_param: ByteParameter) -> torch.Tensor:
        """Deserialize a byte tensor"""
        b_tensor = bytearray(b_param.data)
        return torch.frombuffer(b_tensor, dtype=b_param.dtype).reshape(b_param.shape)


    def _finalize_compute_time(self, s_time: float) -> float:
        """Return a compute time of 0 if no_compute_time is True."""
        if self.no_compute_time:
            return 0.0
        else:
            return time.time() - s_time

