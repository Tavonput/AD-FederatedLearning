import os
import time
import random
from dataclasses import dataclass

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_PROFILING_MODE"] = "1"

import torch
import numpy

from ADFL.my_logging import init_logging
from ADFL.types import TrainingConfig, EvalConfig
from ADFL.model import (
    get_mobile_net_v3_small, get_resnet50, get_vit_l_16, get_model_parameters, get_parameter_info
)
from ADFL.Driver import *
from ADFL.Strategy import *
from ADFL.Strategy.base import CommType
from ADFL.Channel import *

# TMP_PATH = "/data/tavonputl/tmp/ray"
TMP_PATH = "/home/tavonput.luangphasy/tmp"

def test():
    results_base_path = "../Output/Results/Test"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 10,
        num_actors = 1,
        client_map = None
    )
    delay = TrainingConfig.Delay(
         compute_sigma = None,
    )
    metrics = TrainingConfig.Metrics(
        staleness  = False,
        q_error    = False,
        model_dist = False,
        fetch_raw  = False,
        fetch_freq = 1,
    )
    channel = SLQChannel(8)
    # channel = IdentityChannel(True)

    strategy = FedBuff(max_buffer_size=2, lr=1, apply_staleness=True)
    # strategy = Simple(CommType.NORMAL, sync=True)

    train_config = TrainingConfig(
        strategy         = strategy,
        channel          = channel,
        dataset          = TrainingConfig.Dataset.MNIST,
        data_dir         = "../Data",
        train_file       = "../Data/sent140_small/train.pt",
        test_file        = "../Data/sent140_small/test.pt",
        iid              = True,
        dirichlet_a      = 0.3,
        model            = "mobile_net_v3_small",
        num_rounds       = 4,
        num_epochs       = 1,
        num_clients      = 10,
        num_cur_clients  = 2,
        num_servers      = 1,
        batch_size       = 32,
        max_rounds       = 1000000,  # Not used anymore (needs to be refactored out)
        timeout          = 7200,
        delay            = delay,
        num_client_pools = 2,
        metrics          = metrics,
    )

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    results_path = f"{results_base_path}/test.json"
    driver = QAFeLDriver(timeline_path, tmp_path, results_path)
    run_driver(driver, train_config, eval_config)
    del driver


def fetching():
    results_base_path = "../Output/Results/Kamiak/Engine/Sync"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 1000,
        num_actors = 1,
        client_map = None
    )
    delay = TrainingConfig.Delay(
         compute_sigma = None,
    )
    metrics = TrainingConfig.Metrics(
        staleness  = False,
        q_error    = False,
        model_dist = False,
        fetch_raw  = False,
        fetch_freq = 1,
    )

    channel = IdentityChannel(True)
    strategy = Simple(CommType.NORMAL, sync=True)

    train_config = TrainingConfig(
        strategy         = strategy,
        channel          = channel,
        dataset          = TrainingConfig.Dataset.MNIST,
        data_dir         = "../Data",
        train_file       = "../Data/sent140_small/train.pt",
        test_file        = "../Data/sent140_small/test.pt",
        iid              = True,
        dirichlet_a      = 0.3,
        model            = "mobile_net_v3_small",
        num_rounds       = 100,
        num_epochs       = 1,
        num_clients      = 100,
        num_cur_clients  = 20,
        num_servers      = 1,
        batch_size       = 32,
        max_rounds       = 1000000,  # Not used anymore (needs to be refactored out)
        timeout          = 7200,
        delay            = delay,
        num_client_pools = 2,
        metrics          = metrics,
    )

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    results_path = f"{results_base_path}/2.json"
    driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=True)
    run_driver(driver, train_config, eval_config)
    del driver


def eval_throughput():
    results_base_path = "../Output/Results/Kamiak/Engine/Eval/Sync"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = False,
        threshold  = 1,
        num_actors = 2,
        client_map = None
    )
    delay = TrainingConfig.Delay(
         compute_sigma = None,
    )
    metrics = TrainingConfig.Metrics(
        staleness  = False,
        q_error    = False,
        model_dist = False,
        fetch_raw  = False,
        fetch_freq = 1,
    )

    channel = IdentityChannel(True)
    strategy = Simple(CommType.NORMAL, sync=True)

    train_config = TrainingConfig(
        strategy         = strategy,
        channel          = channel,
        dataset          = TrainingConfig.Dataset.MNIST,
        data_dir         = "../Data",
        train_file       = "../Data/sent140_small/train.pt",
        test_file        = "../Data/sent140_small/test.pt",
        iid              = True,
        dirichlet_a      = 0.3,
        model            = "mobile_net_v3_small",
        num_rounds       = 2,
        num_epochs       = 1,
        num_clients      = 100,
        num_cur_clients  = 10,
        num_servers      = 1,
        batch_size       = 32,
        max_rounds       = 1000000,  # Not used anymore (needs to be refactored out)
        timeout          = 7200,
        delay            = delay,
        num_client_pools = 2,
        metrics          = metrics,
    )

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    results_path = f"{results_base_path}/client_self.json"
    driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=True)
    run_driver(driver, train_config, eval_config)
    del driver


def qafel_test():
    results_base_path = "../Output/Results/Kamiak/Bi/"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 2,
        num_actors = 1,
        client_map = None
    )
    delay = TrainingConfig.Delay(
         server_mbps   = 30,
         compute_sigma = 4,
         network_sigma = 8,
         network_shift = 10,
    )
    metrics = TrainingConfig.Metrics(
        staleness  = False,
        q_error    = False,
        model_dist = False,
        fetch_raw  = False,
        fetch_freq = 1,
    )
    channel = QSGDChannel(8)
    strategy = FedBuff(max_buffer_size=10, lr=1, apply_staleness=True)

    train_config = TrainingConfig(
        strategy         = strategy,
        channel          = channel,
        dataset          = TrainingConfig.Dataset.MNIST,
        data_dir         = "../Data",
        train_file       = "../Data/sent140_small/train.pt",
        test_file        = "../Data/sent140_small/test.pt",
        iid              = True,
        dirichlet_a      = 0.3,
        model            = "mobile_net_v3_small",
        num_rounds       = 100,
        num_epochs       = 1,
        num_clients      = 100,
        num_cur_clients  = 10,
        num_servers      = 1,
        batch_size       = 32,
        max_rounds       = 1000000,  # Not used anymore (needs to be refactored out)
        timeout          = 7200,
        delay            = delay,
        num_client_pools = 2,
        metrics          = metrics,
    )

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    results_path = f"{results_base_path}/qafel_nonbroadcast.json"
    driver = QAFeLDriver(timeline_path, tmp_path, results_path)
    run_driver(driver, train_config, eval_config)
    del driver


def main_ex():
    results_base_path = "../Output/Results/Kamiak/Quant/Real"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 10,
        num_actors = 1,
        client_map = None
    )

    delay = TrainingConfig.Delay(
         server_mbps   = None,
         compute_sigma = 4,
         network_sigma = 8,
         network_shift = 10,
    )

    train_config = TrainingConfig(
        strategy        = Simple(CommType.DELTA, sync=True),
        channel         = SLQChannel(bits=8),
        dataset         = TrainingConfig.Dataset.CIFAR10,
        data_dir        = "../Data",
        train_file      = "../Data/sent140_small/train.pt",
        test_file       = "../Data/sent140_small/test.pt",
        iid             = False,
        dirichlet_a     = 0.3,
        model           = "mobile_net_v3_small",
        num_rounds      = 10000000,
        num_epochs      = 1,
        num_clients     = 200,
        num_cur_clients = 10,
        num_servers     = 1,
        batch_size      = 64,
        max_rounds      = 1000000,  # Not used anymore (needs to be refactored out)
        timeout         = 14400,
        delay           = delay,
    )

    @dataclass
    class Config:
        name: str
        eval_threshold: int
        strategy: Strategy
        channel: Channel
        model: str = "distilbert"

    configs = [
        # Config("fedasync_base_m",         5,  FedAsync(FedAsync.Method.POLY), IdentityChannel(True)),
        # Config("fedasync_quant_slq_8_m",  10, FedAsync(FedAsync.Method.POLY), USLQChannel(8),       ),
        # Config("fedasync_quant_qsgd_8_m", 10, FedAsync(FedAsync.Method.POLY), UQSGDChannel(8),      ),

        Config("fedavg_base",         2, Simple(CommType.NORMAL, True), IdentityChannel(True)),

        Config("fedbuff_base",           2,   FedBuff(5, 1, True),            IdentityChannel(True)),
        # Config("fedbuff_quant_slq_8",    10,  FedBuff(5, 1, True),            USLQChannel(8)),
        # Config("fedbuff_quant_qsgd_8_2",   10,  FedBuff(5, 1, True),            UQSGDChannel(8)),
        # Config("fedbuff_quant_rqsgd_8",  10,  FedBuff(5, 1, True),            URQSGDChannel(8)),
        # Config("fedbuff_quant_cnat_8",   10,  FedBuff(5, 1, True),            UCNATChannel(8)),
        # Config("fedbuff_quant_slq_4",    20,  FedBuff(10, 1, True),            USLQChannel(4)),
        # Config("fedbuff_quant_qsgd_4",   20,  FedBuff(10, 1, True),            UQSGDChannel(4)),
        # Config("fedbuff_quant_rqsgd_4",  20,  FedBuff(10, 1, True),            URQSGDChannel(4)),
        # Config("fedbuff_quant_cnat_4",   20,  FedBuff(10, 1, True),            UCNATChannel(4)),

        # Config("fedavg_quant_slq_8_m",  1, Simple(CommType.NORMAL, True), SLQChannel(8)),
        # Config("fedavg_quant_qsgd_8_m", 1, Simple(CommType.NORMAL, True), QSGDChannel(8)),

        # Config("fedavgd_base_m",        1, Simple(CommType.DELTA, True),  IdentityChannel(True)),
        # Config("fedavgd_quant_slq_8_m",   1, Simple(CommType.DELTA, True),  SLQChannel(8)),
        # Config("fedavgd_quant_qsgd_m",  1, Simple(CommType.DELTA, True),  QSGDChannel(8)),
    ]

    datasets = [TrainingConfig.Dataset.SENT140]
    for dataset in datasets:
        train_config.dataset = dataset

        for config in configs:
            torch.manual_seed(0)
            numpy.random.seed(0)
            random.seed(0)

            eval_config.threshold = config.eval_threshold
            train_config.strategy = config.strategy
            train_config.channel = config.channel
            train_config.model = config.model
            train_config.model_save = f"{results_base_path}/{config.model}/{dataset.value}/{config.name}.pth"
            results_path = f"{results_base_path}/{config.model}/{dataset.value}/{config.name}.json"

            if isinstance(config.strategy, Simple):
                sync = True
            else:
                sync = False

            driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=sync)
            run_driver(driver, train_config, eval_config)
            del driver

            time.sleep(10)


def param_vs_delay() -> None:
    results_base_path = "../Output/Results/Kamiak/Quant/Delta"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 10,
        num_actors = 1,
        client_map = None
    )

    delay = TrainingConfig.Delay(
         compute_sigma = 4,
    )

    train_config = TrainingConfig(
        dataset         = TrainingConfig.Dataset.CIFAR10,
        data_dir        = "../Data",
        iid             = True,
        dirichlet_a     = 0.3,
        model           = "resnet_50",
        num_rounds      = 15000,
        num_epochs      = 1,
        num_clients     = 200,
        num_cur_clients = 20,
        num_servers     = 1,
        batch_size      = 64,
        max_rounds      = 1000000,  # Not used anymore (needs to be refactored out)
        timeout         = 7200,
        delay           = delay,
    )

    @dataclass
    class Config:
        name: str
        eval_threshold: int
        strategy: Strategy
        channel: Channel

    configs = [
        Config("fedavg_32",        5, Simple(CommType.NORMAL, True),  IdentityChannel(True)),
        Config("fedavg_8_slq",     5, Simple(CommType.NORMAL, True),  USLQChannel(8)),
        Config("fedavg_8_qsgd",    5, Simple(CommType.NORMAL, True),  UQSGDChannel(8)),
        Config("fedasync_32",     30, FedAsync(FedAsync.Method.POLY), IdentityChannel(True)),
        Config("fedasync_8_slq",  30, FedAsync(FedAsync.Method.POLY), USLQChannel(8)),
        Config("fedasync_8_qsgd", 30, FedAsync(FedAsync.Method.POLY), UQSGDChannel(8)),
        Config("fedbuff_32",       5, FedBuff(10, 1, True),           IdentityChannel(True)),
        Config("fedbuff_8_slq",    5, FedBuff(10, 1, True),           USLQChannel(8)),
        Config("fedbuff_8_qsgd",   5, FedBuff(10, 1, True),           UQSGDChannel(8)),
    ]

    for config in configs:
        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)

        eval_config.threshold = config.eval_threshold
        train_config.strategy = config.strategy
        train_config.channel = config.channel
        train_config.model_save = f"{results_base_path}/{train_config.model}/{train_config.dataset.value}/{config.name}.pth"
        results_path = f"{results_base_path}/{train_config.model}/{train_config.dataset.value}/{config.name}.json"

        if isinstance(config.strategy, Simple):
            sync = True
        else:
            sync = False

        driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=sync)
        run_driver(driver, train_config, eval_config)
        del driver

        time.sleep(10)


def quant_error_ex():
    results_base_path = "../Output/Results/Quant/Real/Error"
    tmp_path = TMP_PATH
    timeline_path = "../Output/Timelines/test.json"

    eval_config = EvalConfig(
        method     = "round",
        central    = True,
        threshold  = 10,
        num_actors = 1,
        client_map = None
    )

    delay = TrainingConfig.Delay(
        server_mbps   = 30,
        compute_sigma = 10,
        network_sigma = 8,
        network_shift = 10,
    )

    train_config = TrainingConfig(
        strategy        = Simple(CommType.DELTA, sync=True),
        channel         = SLQChannel(bits=8),
        dataset         = TrainingConfig.Dataset.FMNIST,
        iid             = True,
        dirichlet_a     = 0.3,
        model           = "mobile_net_v3_small",
        num_rounds      = 10000000,
        num_epochs      = 1,
        num_clients     = 100,
        num_cur_clients = 16,
        num_servers     = 1,
        batch_size      = 64,
        max_rounds      = 1000000,  # Not used anymore (needs to be refactored out)
        timeout         = 100000,
        delay           = delay,
    )

    @dataclass
    class Config:
        name: str
        strategy: Strategy
        channel: Channel

    configs_sync = [
        Config("fedavg_p_8_qsgd", Simple(CommType.NORMAL, True), QSGDChannel(8)),
        Config("fedavg_p_8_slq", Simple(CommType.NORMAL, True), SLQChannel(8)),
        Config("fedavg_d_8_qsgd", Simple(CommType.DELTA, True), QSGDChannel(8)),
        Config("fedavg_d_8_slq", Simple(CommType.DELTA, True), SLQChannel(8)),
    ]

    configs_async = [
        # Config("fedasync_8_qsgd", FedAsync(FedAsync.Method.POLY), QSGDChannel(8)),
        # Config("fedasync_8_slq", FedAsync(FedAsync.Method.POLY), SLQChannel(8)),
        # Config("fedbuff_8_qsgd", FedBuff(5, 1, True), QSGDChannel(8)),
        # Config("fedbuff_8_slq", FedBuff(5, 1, True), SLQChannel(8)),
    ]

    for iid in [True]:
        train_config.iid = iid

        # Async
        train_config.num_rounds = 1000
        for config in configs_async:
            train_config.strategy = config.strategy
            train_config.channel = config.channel

            if iid:
                results_path = f"{results_base_path}/IID/Async/{config.name}.json"
            else:
                results_path = f"{results_base_path}/NonIID/Async/{config.name}.json"

            if isinstance(train_config.strategy, FedAsync):
                eval_config.threshold = 5
            else:
                eval_config.threshold = 1

            driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=False)
            run_driver(driver, train_config, eval_config)
            del driver

            time.sleep(5)

        # Sync (no need for delays since we are not tracking running time)
        train_config.num_rounds = 100

        train_config.delay.server_mbps = None
        train_config.delay.compute_sigma = None
        train_config.delay.network_sigma = None
        train_config.delay.network_shift = 0
        for config in configs_sync:
            eval_config.threshold = 1
            train_config.strategy = config.strategy
            train_config.channel = config.channel

            if iid:
                results_path = f"{results_base_path}/IID/Sync/{config.name}.json"
            else:
                results_path = f"{results_base_path}/NonIID/Sync/{config.name}.json"

            driver = AsyncDriver(timeline_path, tmp_path, results_path, traditional=True)
            run_driver(driver, train_config, eval_config)
            del driver

            time.sleep(5)


def bandwidth_model_size_comp():
    for model in [get_mobile_net_v3_small(1000), get_resnet50(1000), get_vit_l_16(1000)]:
        for bit in [8, 4, 2]:
            params = get_model_parameters(model)
            info = get_parameter_info(params)

            channel = SLQChannel(bit)
            slq_time = channel.simulate_bandwidth(params, 1000)

            channel = QSGDChannel(bit)
            qsgd_time = channel.simulate_bandwidth(params, 1000)

            print(info.num_non_bias_w, info.num_non_bias_t, qsgd_time/slq_time)


def main() -> None:
    init_logging()
    eval_throughput()

if __name__ == "__main__":
    main()
