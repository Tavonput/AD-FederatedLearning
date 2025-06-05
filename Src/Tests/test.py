import ray
import time

from ADFL import compression
from ADFL.model import get_mobile_net_v3_small, get_model_parameters, get_resnet50, get_model_parameters
from ADFL.my_logging import get_logger

from ADFL.Client.common import  diff_parameters
from ADFL.Server.common import add_parameters


@ray.remote
class Actor:
    def __init__(self, name: str) -> None:
        self.log = get_logger(name)
        self.model = get_resnet50(10)
        self.other_actor = None

        self.simulate_network = True
        self.bps = 10 * compression.MiB * 8


    def add_actor(self, actor):
        self.other_actor = actor


    def receive_model(self, b_params, is_quant):
        s_time = time.time()

        if is_quant:
            _ = compression.dequantize_params(b_params)
        else:
            _ = compression.deserialize_params(b_params)

        return (time.time() - s_time) * 1000


    def send_full_model(self):
        params = get_model_parameters(self.model)

        s_time = time.time()
        b_params = compression.serialize_params(params)
        c_time = (time.time() - s_time) * 1000

        s_time = time.time()
        self.simulate_network_delay(b_params.size, self.bps)
        d_time = self.other_actor.receive_model.remote(b_params, False)  # type: ignore
        b_time = (time.time() - s_time) * 1000

        d_time = ray.get(d_time)

        return b_params.size, c_time, b_time, d_time


    def send_quant_model(self, bits):
        params = get_model_parameters(self.model)

        s_time = time.time()
        q_params = compression.quantize_params(params, bits)
        c_time = (time.time() - s_time) * 1000

        s_time = time.time()
        self.simulate_network_delay(q_params.size, self.bps)
        d_time = self.other_actor.receive_model.remote(q_params, True)  # type: ignore
        b_time = (time.time() - s_time) * 1000

        d_time = ray.get(d_time)

        return q_params.size, c_time, b_time, d_time


    def simulate_network_delay(self, data_size: int, bps: float) -> None:
        if self.simulate_network:
            time.sleep(data_size / (bps / 8))


def bandwidth_test():
    log = get_logger("MAIN")

    tmp_path = "/data/tavonputl/tmp/ray"
    ray.init(_temp_dir=tmp_path)

    actor_a = Actor.remote("Actor A")
    actor_b = Actor.remote("Actor B")
    ray.get(actor_a.add_actor.remote(actor_b))  # type: ignore

    def print_metric(size, c_time, b_time, d_time):
        log.info(f"    Model Size:    {(size / compression.MiB):.2f}MB")
        log.info(f"    Compression:   {c_time:.2f}ms")
        log.info(f"    Bandwidth:     {b_time:.2f}ms")
        log.info(f"    Decompression: {d_time:.2f}ms")
        log.info(f"    Total:         {(c_time + b_time + d_time):.2f}ms")

    size, c_time, b_time, d_time = ray.get(actor_a.send_full_model.remote())  # type: ignore
    log.info("Full Model - 32 bit")
    print_metric(size, c_time, b_time, d_time)

    log.info("")
    size, c_time, b_time, d_time = ray.get(actor_a.send_quant_model.remote(8))  # type: ignore
    log.info("Quantized Model - 8 bit")
    print_metric(size, c_time, b_time, d_time)

    log.info("")
    size, c_time, b_time, d_time = ray.get(actor_a.send_quant_model.remote(4))  # type: ignore
    log.info("Quantized Model - 4 bit")
    print_metric(size, c_time, b_time, d_time)

    ray.shutdown()


def main():
    import copy
    model = get_mobile_net_v3_small(10)
    model_params = get_model_parameters(model)

    model_prime = copy.deepcopy(model)
    model_prime_params = get_model_parameters(model_prime)

    for v in model_prime_params.values():
        v.add_(1)

    diff = diff_parameters(model_params, model_prime_params)
    agg = add_parameters(model_params, diff)

    model.load_state_dict(agg)
    model_params_new = get_model_parameters(model)

    for k in diff.keys():
        print(model_params[k])
        print(model_params_new[k])

if __name__ == "__main__":
    main()
