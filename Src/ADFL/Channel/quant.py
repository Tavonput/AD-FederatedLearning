"""
TODO: QSGD, RQSGD, and CNAT can be merged into the same class.
"""

import time
from typing import Tuple, Dict

import torch

from ADFL.model import Parameters, CompressedParameters, QuantParameter, QuantParameters, get_parameter_info

from .channel import Channel, IdentityChannel


class SLQChannel(Channel):
    """SLQ Channel.

    Bi-directional symmetric linear quantization.
    """
    def __init__(self, bits: int) -> None:
        self.bits = bits


    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """self.bits for weights, 32 bits for biases, 32 bits for scales."""
        p_info = get_parameter_info(params)
        num_bytes  = p_info.num_non_bias_w * self.bits / 8  # Weights
        num_bytes += p_info.num_bias_w * 4  # Biases
        num_bytes += p_info.num_non_bias_t * 4  # Scales

        bytes_per_second = mbps * 1_000_000 / 8
        transfer_time = num_bytes / bytes_per_second

        time.sleep(transfer_time)
        return transfer_time


    def _send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        q_params = self._quantize_params(params, self.bits)
        return q_params, time.time() - s_time


    def _receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, QuantParameters)
        s_time = time.time()
        params = {name: self._dequantize_tensor(c_param) for name, c_param in c_params.params.items()}
        return params, time.time() - s_time


    def _quantize_params(self, params: Parameters, bits: int) -> QuantParameters:
        """Quantize parameters with a given bit width. Biases and running metrics are not quantized."""
        q_params = QuantParameters({}, 0)
        for name, param in params.items():
            if param.ndim > 1:
                print(f"[W] {name}")
                q_param, scale = self._quantize_tensor(param, bits)
            else:
                print(f"[B] {name}")
                q_param, scale = param, 1

            q_params.params[name] = QuantParameter(
                data    = q_param,
                bits    = bits,
                scale   = scale,
                signs   = torch.zeros(1, dtype=torch.uint8),  # Not used
                shape   = param.shape,
                dtype   = param.dtype,
                q_dtype = q_param.dtype,
            )
            q_params.size += q_param.nbytes

        return q_params


    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
        """Quantize a tensor."""
        q_max = 2**(bits - 1) - 1
        scale = torch.max(torch.abs(tensor)) / q_max

        quantizer = torch.quantize_per_tensor
        q_tensor = quantizer(tensor, float(scale), 0, dtype=torch.qint8)
        return q_tensor, float(scale)


    def _dequantize_tensor(self, q_param: QuantParameter) -> torch.Tensor:
        """De-quantize a parameter."""
        if q_param.data.ndim > 1:
            return q_param.data.data.dequantize()
        else:
            return q_param.data.data


class USLQChannel(SLQChannel):
    """USLQ Channel.

    Uni-directional symmetric linear quantization (only client-to-server). IMPORTANT: Ensure that
    TrainingConfig.delay.server_mbps=None so that the server does not compute a communication delay.

    Overrides on_server_send and on_client_receive to use the IdentityChannel.
    """
    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_server_send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_client_receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


class QSGDChannel(Channel):
    """QSGD Channel.

    Bi-directional stochastic quantization based on QSGD.
    """
    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.levels = 2**bits - 1


    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """self.bits + 1 for weights and signs, 32 bits for biases, 32 bits for norms."""
        p_info = get_parameter_info(params)
        num_bytes  = p_info.num_non_bias_w * (self.bits + 1) / 8  # Weights + signs
        num_bytes += p_info.num_bias_w * 4  # Biases
        num_bytes += p_info.num_non_bias_t * 4  # Norms

        bytes_per_second = mbps * 1_000_000 / 8
        transfer_time = num_bytes / bytes_per_second

        time.sleep(transfer_time)
        return transfer_time


    def _send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        q_params = self._quantize_params(params, self.bits)
        return q_params, time.time() - s_time


    def _receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, QuantParameters)
        s_time = time.time()
        params = {name: self._dequantize_tensor(c_param) for name, c_param in c_params.params.items()}
        return params, time.time() - s_time


    def _quantize_params(self, params: Parameters, bits: int) -> QuantParameters:
        """Quantize parameters with a given bit width. Biases and running metrics are not quantized."""
        q_params = QuantParameters({}, 0)
        for name, param in params.items():
            if param.ndim > 1:
                q_levels, signs, norm = self._quantize_tensor(param, self.levels)
            else:
                q_levels, signs, norm = param, torch.zeros(1, dtype=torch.uint8), 0

            q_params.params[name] = QuantParameter(
                data    = q_levels,
                bits    = bits,
                scale   = norm,
                signs   = signs,
                shape   = param.shape,
                dtype   = param.dtype,
                q_dtype = q_levels.dtype,
            )
            q_params.size += q_levels.nbytes

        return q_params


    def _quantize_tensor(self, x: torch.Tensor, s: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Quantize a tensor. Returns the quantization level indices, sign tensor, and norm."""
        # Euclidean norm on the flattened x
        norm = torch.linalg.vector_norm(x, ord=2, dim=None)
        if norm == 0:
            return torch.zeros_like(x, dtype=torch.uint8), torch.ones_like(x, dtype=torch.int8), norm

        scaled = s * torch.abs(x) / norm
        l = torch.floor(scaled)
        prob = scaled - l

        rand_vals = torch.rand_like(prob)
        round_up = (rand_vals < prob).float()  # 1 means round up, 0 means stay
        q_levels = (l + round_up).to(torch.uint8)

        signs = torch.sign(x).to(torch.int8)

        return q_levels, signs, norm.item()


    def _dequantize_tensor(self, q_param: QuantParameter) -> torch.Tensor:
        """De-quantize a parameter. Scale is the norm."""
        if q_param.data.ndim <= 1:
            return q_param.data.data
        else:
            if q_param.scale == 0:
                return torch.zeros_like(q_param.data, dtype=torch.float32)

            magnitude = q_param.scale * q_param.data.float() / self.levels
            return magnitude * q_param.signs.float()


class UQSGDChannel(QSGDChannel):
    """UQSGD Channel.

    Uni-directional QSGD (only client-to-server). IMPORTANT: Ensure that TrainingConfig.delay.server_mbps=None so that
    the server does not compute a communication delay.

    Overrides on_server_send and on_client_receive to use the IdentityChannel.
    """
    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_server_send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_client_receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


class RQSGDChannel(Channel):
    """RQSGD Channel.

    Bi-directional revised stochastic quantization.
    """
    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.levels = 2**bits - 1


    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """self.bits + 1 for weights and signs, 32 bits for biases, 32 bits for norms."""
        p_info = get_parameter_info(params)
        num_bytes  = p_info.num_non_bias_w * (self.bits + 1) / 8  # Weights + signs
        num_bytes += p_info.num_bias_w * 4  # Biases
        num_bytes += p_info.num_non_bias_t * 8  # Norms + minimum factors

        bytes_per_second = mbps * 1_000_000 / 8
        transfer_time = num_bytes / bytes_per_second

        time.sleep(transfer_time)
        return transfer_time


    def _send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        q_params = self._quantize_params(params, self.bits)
        return q_params, time.time() - s_time


    def _receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, QuantParameters)
        s_time = time.time()
        params = {name: self._dequantize_tensor(c_param) for name, c_param in c_params.params.items()}
        return params, time.time() - s_time


    def _quantize_params(self, params: Parameters, bits: int) -> QuantParameters:
        """Quantize parameters with a given bit width. Biases and running metrics are not quantized."""
        q_params = QuantParameters({}, 0)
        for name, param in params.items():
            if param.ndim > 1:
                q_levels, signs, norm, min_factor = self._quantize_tensor(param, self.levels)
            else:
                q_levels, signs, norm, min_factor = param, torch.zeros(1, dtype=torch.uint8), 0, 0

            q_params.params[name] = QuantParameter(
                data    = q_levels,
                bits    = bits,
                scale   = norm,
                signs   = signs,
                shape   = param.shape,
                dtype   = param.dtype,
                q_dtype = q_levels.dtype,
                scale_2 = min_factor,
            )
            q_params.size += q_levels.nbytes

        return q_params


    def _quantize_tensor(self, x: torch.Tensor, s: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Quantize a tensor. Returns the quantization level indices, sign tensor, and norm."""
        # Infinity norm on the flattened x
        norm = torch.linalg.vector_norm(x, ord=float("inf"), dim=None)
        if norm == 0:
            return torch.zeros_like(x, dtype=torch.uint8), torch.ones_like(x, dtype=torch.int8), norm, 0

        scaled = s * torch.abs(x) / norm
        l = torch.floor(scaled)
        prob = scaled - l

        rand_vals = torch.rand_like(prob)
        round_up = (rand_vals < prob).float()  # 1 means round up, 0 means stay
        q_levels = (l + round_up).to(torch.uint8)

        signs = torch.sign(x).to(torch.int8)
        minimum_factor = torch.linalg.vector_norm(x, ord=-float("inf"), dim=None)

        return q_levels, signs, norm.item(), minimum_factor.item()


    def _dequantize_tensor(self, q_param: QuantParameter) -> torch.Tensor:
        """De-quantize a parameter. Scale is the norm scale_2 is the minimum factor."""
        if q_param.data.ndim <= 1:
            return q_param.data.data
        else:
            if q_param.scale == 0:
                return torch.zeros_like(q_param.data, dtype=torch.float32)

            result = q_param.scale * q_param.signs.float() * q_param.data.float() / self.levels

            # Set the zero elements to the minimum factor using a mask
            zero_mask = q_param.data == 0
            result[zero_mask] = q_param.scale_2 * q_param.signs[zero_mask].float()
            return result


class URQSGDChannel(RQSGDChannel):
    """URQSGD Channel.

    Uni-directional RQSGD (only client-to-server). IMPORTANT: Ensure that TrainingConfig.delay.server_mbps=None so that
    the server does not compute a communication delay.

    Overrides on_server_send and on_client_receive to use the IdentityChannel.
    """
    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_server_send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_client_receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


class CNATChannel(Channel):
    """CNAT Channel.

    Bi-directional natural compression.
    """
    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.levels = 2**bits - 1


    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_server_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def on_client_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        return self._send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        return self._receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }


    def simulate_bandwidth(self, params: Parameters, mbps: float) -> float:
        """self.bits + 1 for weights and signs, 32 bits for biases, 32 bits for norms."""
        p_info = get_parameter_info(params)
        num_bytes  = p_info.num_non_bias_w * (self.bits + 1) / 8  # Weights + signs
        num_bytes += p_info.num_bias_w * 4  # Biases
        num_bytes += p_info.num_non_bias_t * 4  # Norms

        bytes_per_second = mbps * 1_000_000 / 8
        transfer_time = num_bytes / bytes_per_second

        time.sleep(transfer_time)
        return transfer_time


    def _send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        s_time = time.time()
        q_params = self._quantize_params(params, self.bits)
        return q_params, time.time() - s_time


    def _receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        assert isinstance(c_params, QuantParameters)
        s_time = time.time()
        params = {name: self._dequantize_tensor(c_param) for name, c_param in c_params.params.items()}
        return params, time.time() - s_time


    def _quantize_params(self, params: Parameters, bits: int) -> QuantParameters:
        """Quantize parameters with a given bit width. Biases and running metrics are not quantized."""
        q_params = QuantParameters({}, 0)
        for name, param in params.items():
            if param.ndim > 1:
                q_levels, signs, norm = self._quantize_tensor(param)
            else:
                q_levels, signs, norm = param, torch.zeros(1, dtype=torch.uint8), 0

            q_params.params[name] = QuantParameter(
                data    = q_levels,
                bits    = bits,
                scale   = norm,
                signs   = signs,
                shape   = param.shape,
                dtype   = param.dtype,
                q_dtype = q_levels.dtype,
            )
            q_params.size += q_levels.nbytes

        return q_params


    def _quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Quantize a tensor. Returns the quantization level indices, sign tensor, and norm."""
        # Euclidean norm on the flattened x
        norm = torch.linalg.vector_norm(x, ord=2, dim=None)
        if norm == 0:
            return torch.zeros_like(x, dtype=torch.uint8), torch.ones_like(x, dtype=torch.int8), norm

        signs = torch.sign(x).to(torch.int8)
        x_abs = torch.abs(x)

        min_exp = -2 ** (self.bits - 1)
        max_exp = 2 ** (self.bits - 1) - 1

        eps = torch.finfo(x.dtype).eps
        log_vals = torch.log2(x_abs + eps)

        floor_vals = torch.floor(log_vals)
        ceil_vals = torch.ceil(log_vals)

        prob = (2**ceil_vals - x_abs) / 2**floor_vals

        is_lower = torch.rand_like(prob) < prob
        final_exponents = torch.where(is_lower, floor_vals, ceil_vals).clamp_(min_exp, max_exp)
        final_exponents[x == 0] = min_exp

        return final_exponents.to(torch.int8), signs, norm.item()  # type: ignore


    def _dequantize_tensor(self, q_param: QuantParameter) -> torch.Tensor:
        """De-quantize a parameter. Scale is the norm."""
        if q_param.data.ndim <= 1:
            return q_param.data.data
        else:
            if q_param.scale == 0:
                return torch.zeros_like(q_param.data, dtype=torch.float32)

            return q_param.scale * q_param.signs.float() * (2 ** q_param.data.float())


class UCNATChannel(CNATChannel):
    """UCNAT Channel.

    Uni-directional CNAT (only client-to-server). IMPORTANT: Ensure that TrainingConfig.delay.server_mbps=None so that
    the server does not compute a communication delay.

    Overrides on_server_send and on_client_receive to use the IdentityChannel.
    """
    def on_server_send(self, params: Parameters) -> Tuple[CompressedParameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_server_send(params)


    def on_client_receive(self, c_params: CompressedParameters) -> Tuple[Parameters, float]:
        identity_channel = IdentityChannel(no_compute_time=True)
        return identity_channel.on_client_receive(c_params)


    def to_json(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "bits": self.bits
        }
