from typing import List, Tuple, Dict, Union, Optional
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet50, ResNet50_Weights,
    vit_l_16, ViT_L_16_Weights,
)

from transformers import DistilBertForSequenceClassification


Parameters = Dict[str, torch.Tensor]


@dataclass
class QuantParameter:
    data:    torch.Tensor
    bits:    int
    scale:   float
    signs:   torch.Tensor  # Used for QSGD
    shape:   torch.Size
    dtype:   torch.dtype
    q_dtype: torch.dtype
    scale_2: float = 0      # Used for RQSGD


@dataclass
class QuantParameters:
    params: Dict[str, QuantParameter]
    size:   int


@dataclass
class ByteParameter:
    data:  bytes
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class ByteParameters:
    params: Dict[str, ByteParameter]
    size:   int


@dataclass
class MixedParameters:
    params: Dict[str, Union[QuantParameter, ByteParameter]]
    size:   int


CompressedParameters = Union[QuantParameters, ByteParameters, MixedParameters]


@dataclass
class ParameterInfo:
    num_non_bias_w: int
    num_non_bias_t: int
    num_bias_w:     int
    num_bias_t:     int


ImageBatch = List[torch.Tensor]
TokenBatch = Dict[str, torch.Tensor]
Batch = Union[ImageBatch, TokenBatch]


@dataclass
class ForwardResults:
    logits:    torch.Tensor
    loss:      torch.Tensor
    correct:   int
    n_samples: int


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 32 * 32 * 3)
        return self.fc(x)


def model_forward(
    model: nn.Module, batch: Batch, device: torch._C.device, criterion: Optional[nn.Module] = None
) -> ForwardResults:
    """Forward pass through the model."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if isinstance(batch, Dict):
        # This is a TokenBatch
        input_ids = batch["input_ids"].to(device)
        attentention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Uses cross entropy loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attentention_mask,
            labels=labels
        )

        preds = outputs.logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        return ForwardResults(outputs.logits, outputs.loss, correct, labels.size(0))

    elif isinstance(batch, List):
        # This is an ImageBatch
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()

        return ForwardResults(outputs, loss, correct, labels.size(0))


def get_mobile_net_v3_small(num_classes: int, num_input_channels: int = 3) -> nn.Module:
    """Get MobileNetV3 Small."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, num_classes)

    if num_input_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_input_channels,
            old_conv.out_channels,  # type: ignore
            old_conv.kernel_size,  # type: ignore
            old_conv.stride,  # type: ignore
            old_conv.padding,  # type: ignore
            bias=False,
        )

    return model


def get_mobile_net_v3_large(num_classes: int, num_input_channels: int = 3) -> nn.Module:
    """Get MobileNetV3 Large."""
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(1280, num_classes)

    if num_input_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_input_channels,
            old_conv.out_channels,  # type: ignore
            old_conv.kernel_size,  # type: ignore
            old_conv.stride,  # type: ignore
            old_conv.padding,  # type: ignore
            bias=False,
        )

    return model


def get_resnet50(num_classes: int, num_input_channels: int = 3) -> nn.Module:
    """Get ResNet50."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_classes)

    if num_input_channels != 3:
        model.conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

    return model


def get_vit_l_16(num_classes: int) -> nn.Module:
    """Get ViT L 16."""
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(1024, num_classes)
    return model


def get_distilbert(num_classes: int) -> nn.Module:
    """Get DistilBert."""
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_classes
    )


def get_model_parameters(model: nn.Module) -> Parameters:
    """Get the model parameters on the CPU."""
    return {name: param.clone().detach().cpu() for name, param in model.state_dict().items()}


def set_model_parameters(model: nn.Module, parameters: Parameters):
    """Set the model parameters on the CPU."""
    model.to("cpu")
    model.load_state_dict(parameters)


def get_parameter_info(params: Parameters) -> ParameterInfo:
    """Get the tensor information of parameters."""
    p_info = ParameterInfo(0, 0, 0, 0)

    for tensor in params.values():
        if tensor.ndim > 1:
            p_info.num_non_bias_t += 1
            p_info.num_non_bias_w += tensor.numel()
        else:
            p_info.num_bias_t += 1
            p_info.num_bias_w += tensor.numel()

    return p_info


def simple_aggregate(parameters: List[Parameters]) -> Parameters:
    """Simple aggregation."""
    assert len(parameters) > 0

    state_dict = OrderedDict()

    with torch.no_grad():
        for name in parameters[0].keys():
            contributions = [param[name] for param in parameters]
            stacked_contributions = torch.stack(contributions, dim=0)
            aggregated = torch.sum(stacked_contributions, dim=0) / len(contributions)
            state_dict[name] = aggregated

    return state_dict


def weighted_aggregate(parameters: List[Tuple[Parameters, int]]) -> Parameters:
    """Weighted aggregation."""
    assert len(parameters) > 0

    state_dict = OrderedDict()

    with torch.no_grad():
        total_examples = sum([param[1] for param in parameters])

        state_dict = OrderedDict()
        for name in parameters[0][0].keys():
            contributions = [param[1] * param[0][name] for param in parameters]
            stacked_contributions = torch.stack(contributions, dim=0)
            aggregated = torch.sum(stacked_contributions, dim=0) / total_examples
            state_dict[name] = aggregated

    return state_dict


def parameter_relative_mse(params_a: Parameters, params_b: Parameters, exclude_bias: bool) -> float:
    """Compute the relative MSE between two sets of parameters."""
    zero_params = {name: torch.zeros_like(tensor) for name, tensor in params_a.items()}
    num = parameter_mse(params_a, params_b, exclude_bias)
    den = parameter_mse(params_a, zero_params, exclude_bias)
    return num / den if den > 0 else 0.0


def parameter_mse(params_a: Parameters, params_b: Parameters, exclude_bias: bool) -> float:
    """Compute the MSE between two sets of parameters."""
    assert params_a.keys() == params_b.keys()

    mse = 0.0
    num_params = 0

    for key in params_a:
        param_a = params_a[key]
        param_b = params_b[key]

        assert param_a.shape == param_b.shape

        if exclude_bias and (param_a.ndim <= 1 or param_b.ndim <= 1):
            continue

        mse += torch.sum((param_a - param_b) ** 2).item()
        num_params += param_a.numel()

    return mse / num_params if num_params > 0 else 0.0


def parameter_mean_var(params: Parameters, exclude_bias: bool) -> Tuple[float, float]:
    """Compute the mean and variance of parameters."""
    all_param_tensors: List[torch.Tensor] = []
    for _, param in params.items():
        if exclude_bias and param.ndim <= 1:
            continue

        all_param_tensors.append(param.flatten())

    all_params = torch.cat(all_param_tensors)
    mean = torch.mean(all_params)
    var = torch.var(all_params)

    return mean.item(), var.item()


def parameter_cosine_similarity(params_a: Parameters, params_b: Parameters, exclude_bias: bool) -> float:
    """Compute the cosine similarity between two sets of parameters."""
    assert params_a.keys() == params_b.keys()

    tensors_a: List[torch.Tensor] = []
    tensors_b: List[torch.Tensor] = []

    for key in params_a:
        param_a = params_a[key]
        param_b = params_b[key]

        assert param_a.shape == param_b.shape
        if exclude_bias and (param_a.ndim <= 1 or param_b.ndim <= 1):
            continue

        tensors_a.append(param_a.flatten())
        tensors_b.append(param_b.flatten())

    vector_a = torch.cat(tensors_a)
    vector_b = torch.cat(tensors_b)

    return nn.functional.cosine_similarity(vector_a, vector_b, dim=0).item()


def add_parameters(params_a: Parameters, params_b: Parameters, alpha: float, beta: float) -> Parameters:
    assert set(params_a.keys()) == set(params_b.keys())

    params_sum: Parameters = {}
    with torch.no_grad():
        for key in params_b:
            params_sum[key] = (alpha * params_a[key]) + (beta * params_b[key])

    return params_sum


def add_parameters_inpace(
    params_a: Parameters, params_b: Parameters, alpha: float, beta: float, to_float: bool
) -> None:
    assert set(params_a.keys()) == set(params_b.keys())

    with torch.no_grad():
        for key in params_b:
            if to_float:
                params_a[key].float().mul_(alpha).add_(params_b[key], alpha=beta)
            else:
                params_a[key].mul_(alpha).add_(params_b[key], alpha=beta)


def diff_parameters(params_a: Parameters, params_b: Parameters) -> Parameters:
    """Return b - a."""
    assert set(params_a.keys()) == set(params_b.keys())

    diff: Parameters = {}
    with torch.no_grad():
        for key in params_a:
            diff[key] = params_b[key] - params_a[key]

    return diff


