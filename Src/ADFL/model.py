from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .types import Parameters


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 32 * 32 * 3)
        return self.fc(x)


def get_mobile_net_v3_small(num_classes: int) -> nn.Module:
    """Get MobileNetV3 Small."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model


def get_model_parameters(model: nn.Module) -> Parameters:
    """Get the model parameters on the CPU."""
    return {name: param.cpu() for name, param in model.state_dict().items()}


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