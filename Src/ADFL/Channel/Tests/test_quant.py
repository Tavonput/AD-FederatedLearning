import torch

from ADFL.model import parameter_mse
from ADFL.Channel import *


def test_slq_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    channel = SLQChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 32 bits
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (432 / 1_000_000))  # 432 bps
    print(f"\tt_time={t_time}")


def test_uslq_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    channel = USLQChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 32 bits
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (432 / 1_000_000))  # 432 bps
    print(f"\tt_time={t_time}")


def test_uqsgd_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    channel = UQSGDChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 32 bits + 10 bit
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (442 / 1_000_000))  # 442 bps
    print(f"\tt_time={t_time}")


def test_qsgd_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    channel = QSGDChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 32 bits + 10 bit
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (442 / 1_000_000))  # 442 bps
    print(f"\tt_time={t_time}")


def test_rqsgd_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    y[0, 0] = 0.001
    channel = RQSGDChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 64 bits + 10 bit
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (474 / 1_000_000))  # 474 bps
    print(f"\tt_time={t_time}")


def test_cnat_channel():
    torch.manual_seed(0)

    x = torch.randn(2, 5) * 0.001
    y = torch.randn(2, 5) * 0.001
    x[0, 0] = 0

    channel = CNATChannel(bits=8)

    _send_over_channel(x, y, channel)

    params = {
        "non_bias": torch.randn(2, 5),  # 80 bits + 32 bits + 10 bit
        "bias": torch.randn(10),        # 320 bits
    }

    t_time = channel.simulate_bandwidth(params, (442 / 1_000_000))  # 432 bps
    print(f"\tt_time={t_time}")

    # See if rounding is correct. 0.6 should have a 80% chance to round to -1 and 20% change to 0.
    x = torch.zeros(10000)
    x += 0.6
    q_tensor, _, _ = channel._quantize_tensor(x)

    zero_percentage = (q_tensor == 0).sum().item() / q_tensor.numel() * 100
    print(f"On quantizing 0.6 (should be around 20/80): 0={zero_percentage:.4f} -1={100 - zero_percentage:.4f}")


def _send_over_channel(x: torch.Tensor, y: torch.Tensor, channel: Channel) -> None:
    x_params = {"hi": x}
    y_params = {"hi": y}

    print("Initial server tensor")
    print("=====================")
    print(x)

    c_params, sc_c_time = channel.on_server_send(x_params)
    d_params, sc_d_time = channel.on_client_receive(c_params)
    sc_q_error = parameter_mse(d_params, x_params, exclude_bias=True)

    print()
    print("Tensor after s-c transmission")
    print("=============================")
    print(d_params["hi"])

    print()
    print("Initial client tensor")
    print("=====================")
    print(y)

    c_params, cs_c_time = channel.on_client_send(y_params)
    d_params, cs_d_time = channel.on_server_receive(c_params)
    cs_q_error = parameter_mse(d_params, y_params, exclude_bias=True)

    print()
    print("Tensor after c-s transmission")
    print("=============================")
    print(d_params["hi"])

    print()
    print("Stats")
    print("=====")
    print(f"\tsc_c_time={sc_c_time:.4e} sc_d_time={sc_d_time:.4e} sc_q_error={sc_q_error:.4e}")
    print(f"\tcs_c_time={cs_c_time:.4e} cs_d_time={cs_d_time:.4e} cs_q_error={cs_q_error:.4e}")
