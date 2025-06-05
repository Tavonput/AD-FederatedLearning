import time
from typing import Tuple, Dict

import torch
import torch.nn as nn

from ADFL.model import get_model_parameters

from .model import Parameters
from .types import (
    MixedParameters, QuantParameters, QuantParameter, ByteParameters, ByteParameter, CompressedParameters
)


KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


# ======================================================================================================================
# Quantization
# ======================================================================================================================
def quantize_tensor(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor."""
    q_min = -2**(bits - 1)
    q_max = 2**(bits - 1) - 1

    scale = q_max / torch.max(tensor)
    q_tensor = (scale * tensor).round_().clamp_(q_min, q_max).to(torch.int8)

    return q_tensor, scale.item()


def pack_4bit(q_tensor: torch.Tensor) -> torch.Tensor:
    """Pack a 4 bit tensor into an 8 bit tensor.

    TODO: Optimize this function.
    """
    q_tensor = q_tensor.view(-1)

    if q_tensor.numel() & 1:  # Is odd
        q_tensor = torch.cat([q_tensor, torch.zeros(1, dtype=torch.int8)])

    # Shift to a positive range: [-8, 7] to [0, 15]
    q_tensor = q_tensor + 8

    return (q_tensor[0::2] << 4) | q_tensor[1::2]


def unpack_4bit(b_tensor: bytearray, shape: torch.Size) -> torch.Tensor:
    """Unpack a 4 bit tensor.

    TODO: Optimize this function.
    """
    packed = torch.frombuffer(b_tensor, dtype=torch.int8)

    # For gathering the high values (left 4 bits), we have to mask the left for bits after the right shift to
    # accommodate for signed right shifts.
    high = ((packed >> 4) & 0x0F) - 8
    low = (packed & 0x0F) - 8

    q_tensor = torch.stack([high, low], dim=1).view(-1)
    q_tensor = q_tensor[:shape.numel()]

    return q_tensor.reshape(shape)


def dequantize_tensor(q_param: QuantParameter) -> torch.Tensor:
    """De-quantize a parameter."""
    raw_q_tensor = bytearray(q_param.data)

    if q_param.bits == 4:
        q_tensor = unpack_4bit(raw_q_tensor, q_param.shape).to(torch.float32)
    else:
        q_tensor = torch.frombuffer(raw_q_tensor, dtype=q_param.q_dtype).reshape(q_param.shape).to(torch.float32)

    return (q_tensor / q_param.scale).to(q_param.dtype)


def quantize_params(params: Parameters, bits: int) -> QuantParameters:
    """Quantize parameters with a given bit width.

    TODO: What would happen if we don't quantize the running metrics.
    """
    q_params = QuantParameters({}, 0)
    for name, param in params.items():
        q_param, scale = quantize_tensor(param, bits)

        if bits == 4:
            q_param = pack_4bit(q_param)

        q_params.params[name] = QuantParameter(
            data    = q_param.numpy().tobytes(),
            bits    = bits,
            scale   = scale,
            shape   = param.shape,
            dtype   = param.dtype,
            q_dtype = q_param.dtype,
        )
        q_params.size += len(q_params.params[name].data)

    return q_params


def dequantize_params(q_params: QuantParameters) -> Parameters:
    """De-quantize parameters."""
    return {name: dequantize_tensor(q_param) for name, q_param in q_params.params.items()}


# ======================================================================================================================
# Serialization
# ======================================================================================================================
def deserialize_tensor(b_param: ByteParameter) -> torch.Tensor:
    """Deserialize a byte tensor"""
    b_tensor = bytearray(b_param.data)
    return torch.frombuffer(b_tensor, dtype=b_param.dtype).reshape(b_param.shape)


def serialize_params(params: Parameters) -> ByteParameters:
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


def deserialize_params(b_params: ByteParameters) -> Parameters:
    """Deserialize a set of byte parameters."""
    return {name: deserialize_tensor(b_param) for name, b_param in b_params.params.items()}



# ======================================================================================================================
# Mixed
# ======================================================================================================================
def compress_mixed_params(params: Parameters, quant_map: Dict[str, bool], bits: int) -> MixedParameters:
    """Compress mixes parameters with a given bit width."""
    assert params.keys() == quant_map.keys()

    c_params = MixedParameters({}, 0)
    for name, param in params.items():
        if quant_map[name] == True:
            q_param, scale = quantize_tensor(param, bits)

            if bits == 4:
                q_param = pack_4bit(q_param)

            c_params.params[name] = QuantParameter(
                data    = q_param.numpy().tobytes(),
                bits    = bits,
                scale   = scale,
                shape   = param.shape,
                dtype   = param.dtype,
                q_dtype = q_param.dtype,
            )
        else:
            c_params.params[name] = ByteParameter(
                data  = param.numpy().tobytes(),
                shape = param.shape,
                dtype = param.dtype,
            )

        c_params.size += len(c_params.params[name].data)

    return c_params


def decompress_mixed_params(c_params: MixedParameters) -> Parameters:
    """Decompress a set of mixed parameters."""
    params = {}

    for name, c_param in c_params.params.items():
        if isinstance(c_param, ByteParameter):
            params[name] = deserialize_tensor(c_param)
        elif isinstance(c_param, QuantParameter):
            params[name] = dequantize_tensor(c_param)

    return params


# ======================================================================================================================
# Compression
# ======================================================================================================================
def decompress_params(c_params: CompressedParameters) -> Tuple[Parameters, float]:
    """Dequantize or deserialize parameters."""
    start_time = time.time()

    if isinstance(c_params, ByteParameters):
        params = deserialize_params(c_params)
    elif isinstance(c_params, QuantParameters):
        params = dequantize_params(c_params)
    elif isinstance(c_params, MixedParameters):
        params = decompress_mixed_params(c_params)

    return params, (time.time() - start_time)


def compress_model(model: nn.Module, method: str, bits: int = 8) -> Tuple[CompressedParameters, float]:
    """Quantize or serialize a model."""
    params = get_model_parameters(model)

    if method == "mixed":
        start_time = time.time()

        named_params = {name for name, _ in model.named_parameters()}
        quant_map = {name: (name in named_params) for name in params.keys()}
        c_params = compress_mixed_params(params, quant_map, bits)

        return c_params, (time.time() - start_time)
    else:
        return compress_params(params, method, bits)


def compress_params(params: Parameters, method: str, bits: int = 8) -> Tuple[CompressedParameters, float]:
    """Quantize or serialize parameters."""
    start_time = time.time()

    if method == "quant":
        c_params = quantize_params(params, bits)
    elif method == "byte":
        c_params = serialize_params(params)
    elif method == "mixed":
        assert False, "method 'mixed' is not support. Use compress_model()"
    else:
        assert False, "Compress method not supported"

    return c_params, (time.time() - start_time)
    
