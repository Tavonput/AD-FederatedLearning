from typing import List, Tuple, Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    resnet50, ResNet50_Weights,
    vit_l_16, ViT_L_16_Weights,
)

Parameters = Dict[str, torch.Tensor]


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 32 * 32 * 3)
        return self.fc(x)


def get_mobile_net_v3_small(num_classes: int, num_input_channels: int = 3) -> nn.Module:
    """Get MobileNetV3 Small."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, num_classes)

    if num_input_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_input_channels,
            old_conv.out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
            bias=False,
        )

    return model


def get_resnet50(num_classes: int) -> nn.Module:
    """Get ResNet50."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_classes)
    return model


def get_vit_l_16(num_classes: int) -> nn.Module:
    """Get ViT L 16."""
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(1024, num_classes)
    return model


def get_model_parameters(model: nn.Module) -> Parameters:
    """Get the model parameters on the CPU."""
    return {name: param.clone().detach().cpu() for name, param in model.state_dict().items()}


def set_model_parameters(model: nn.Module, parameters: Parameters):
    """Set the model parameters on the CPU."""
    model.to("cpu")
    model.load_state_dict(parameters)


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


def parameter_mse(params_a: Parameters, params_b: Parameters) -> float:
    """Compute the MSE between two sets of parameters."""
    assert params_a.keys() == params_b.keys()

    mse = 0.0
    num_params = 0

    for key in params_a:
        param_a = params_a[key]
        param_b = params_b[key]

        assert param_a.shape == param_b.shape

        mse += torch.sum((param_a - param_b) ** 2).item()
        num_params += param_a.numel()

    return mse / num_params if num_params > 0 else 0.0


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
