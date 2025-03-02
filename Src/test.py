import torch

from ADFL.model import get_mobile_net_v3_small
from ADFL.client import _get_model_parameters

model = get_mobile_net_v3_small(10)

for k, v in model.state_dict().items():
    print(k, v.dtype)

print("------------------------")

for k, v in model.named_parameters():
    print(k, v.dtype)