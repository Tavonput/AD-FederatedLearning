import torch

from ADFL.Channel import IdentityChannel


def test_identity_channel():
    x = torch.randn(2, 5)
    params = {"hi": x}
    channel = IdentityChannel(no_com_time=False)

    print("Initial server tensor")
    print("=====================")
    print(x)

    c_params, c_time = channel.on_server_send(params)
    d_params, d_time = channel.on_client_receive(c_params)

    print()
    print("Tensor after s-c transmission")
    print("=============================")
    print(d_params["hi"])

    assert torch.equal(x, d_params["hi"])

    y = torch.randn(2, 5)
    params = {"hi": y}

    print()
    print("Initial client tensor")
    print("=====================")
    print(y)

    c_params, c_time = channel.on_client_send(params)
    d_params, d_time = channel.on_server_receive(c_params)

    print()
    print("Tensor after c-s transmission")
    print("=============================")
    print(d_params["hi"])

    assert torch.equal(y, d_params["hi"])

    print()
    print("Stats")
    print("=====")
    print(f"\tc_time={c_time:.4e}")
    print(f"\td_time={d_time:.4e}")

    transfer_time = channel.simulate_bandwidth(params, 0.00032)  # 40 bytes/s
    print(f"\tt_time={transfer_time:.4e}")
