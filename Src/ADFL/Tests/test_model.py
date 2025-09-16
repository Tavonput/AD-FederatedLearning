import torch

from ADFL.model import *


def test_get_parameter_info():
    params = {
        "non_bias_1": torch.randn(2, 2),
        "non_bias_2": torch.randn(3, 2, 2),
        "bias_1": torch.randn(10),
        "bias_2": torch.randn(4),
    }
    p_info = get_parameter_info(params)

    assert p_info.num_non_bias_w == (2 * 2) + (3 * 2 * 2)
    assert p_info.num_non_bias_t == 2
    assert p_info.num_bias_w == 10 + 4
    assert p_info.num_bias_t == 2

    print("[o] get_parameter_info")
