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


def test_diff_parameters():
    model = get_mobile_net_v3_small(10)
    params = get_model_parameters(model)

    # Just ones
    delta = {name: torch.ones_like(t) for name, t in params.items()}

    classifer = "classifier.3.weight"
    num_batches_tracked = "features.0.1.num_batches_tracked"

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)

    # params - 1
    params = diff_parameters(delta, params)

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)


def test_add_parameters():
    model = get_mobile_net_v3_small(10)
    params = get_model_parameters(model)

    # Just ones
    delta = {name: torch.ones_like(t).float() for name, t in params.items()}

    classifer = "classifier.3.weight"
    num_batches_tracked = "features.0.1.num_batches_tracked"

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)

    params = add_parameters(params, delta, 1, 1)

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)


def test_add_parameters_inplace():
    model = get_mobile_net_v3_small(10)
    params = get_model_parameters(model)

    # Just ones
    delta = {name: torch.ones_like(t) for name, t in params.items()}

    classifer = "classifier.3.weight"
    num_batches_tracked = "features.0.1.num_batches_tracked"

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)

    add_parameters_inpace(params, delta, 1, 1, to_float=False)

    print(params[classifer], params[classifer].dtype)
    print(params[num_batches_tracked], params[num_batches_tracked].dtype)


def test_model_parameter_casting():
    """Testing that setting the state dict will automatically cast a tensor to its correct data type."""
    model = get_mobile_net_v3_small(10)

    params = get_model_parameters(model)
    print(params["features.0.1.num_batches_tracked"])

    params = {n: t.float() for n, t in params.items()}
    set_model_parameters(model, params)

    params = get_model_parameters(model)
    print(params["features.0.1.num_batches_tracked"])
