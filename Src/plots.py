import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Optional
from bisect import bisect_right

import numpy as np
from prettytable import PrettyTable

from ADFL.Utils.federated_results import FederatedResultsExplorer, ScalarPairs
from ADFL.sampling import sample_half_normal

from ExtraUtils.display import *


set_sns_theme(style="white")
set_global_rcparams()
FIGSIZE = (7, 6)


@dataclass
class Data:
    name: str
    accuracies: ScalarPairs
    final_acc: Tuple[float, float]
    g_rounds: int
    q_errors_mse: List[float]
    q_errors_cos: List[float]
    model_dists: ScalarPairs
    a_cmpt: Tuple[float, float]
    a_comm: Tuple[float, float]
    throughput: float


def parse_data(name: str, filepath: str) -> Optional[Data]:
    if not os.path.exists(filepath):
        print(f"parse_data: Could not find path {filepath}")
        return None

    explorer = FederatedResultsExplorer()
    explorer.set_federated_results_from_file(filepath)

    final_acc, final_acc_std = explorer.get_central_accuracies_final(120, "mean")

    (comm_mean, comm_std), (cmpt_mean, cmpt_std) = explorer.get_client_network_compute_times(method="mean")

    return Data(
        name=name,
        accuracies=explorer.get_central_accuracies_raw(),
        final_acc=(round(final_acc, 2), round(final_acc_std, 2)),
        g_rounds=explorer.fr.total_g_rounds,
        q_errors_mse=explorer.fr.q_errors_mse,
        q_errors_cos=explorer.fr.q_errors_cos,
        model_dists=explorer.fr.model_dists,
        a_cmpt=(round(cmpt_mean, 3), round(cmpt_std, 3)),
        a_comm=(round(comm_mean, 3), round(comm_std, 3)),
        throughput=round(explorer.get_round_throughput(), 3),
    )


def plot_q_error(data: List[Data], save_path: str) -> None:
    names = [d.name for d in data]

    xs, ys = [], []
    for d in data:
        y = d.q_errors_mse
        x = list(range(1, len(y) + 1))
        xs.append(x)
        ys.append(y)

    config = PlotConfig(
        title="Quantization Error on FedAvg (100 Clients, 16 Concurrent)",
        ylabel="Quantization Error",
        xlabel="Communication Event",
    )

    plot_line(
        x=xs,
        y=ys,
        config=config,
        labels=names,  # type: ignore
        show=False,
        save_path=save_path,
    )


def table_q_errors() -> None:
    PATH_BASE = "../Output/Results/Quant/Real/Error"
    paths = [
        ("FedAvg P8 QSGD",  "Sync/fedavg_p_8_qsgd.json"),
        ("FedAvg P8 SLQ",   "Sync/fedavg_p_8_slq.json"),
        ("FedAvg D8 QSGD",  "Sync/fedavg_d_8_qsgd.json"),
        ("FedAvg D8 SLQ",   "Sync/fedavg_d_8_slq.json"),
        ("FedAsync 8 QSGD", "Async/fedasync_8_qsgd.json"),
        ("FedAsync 8 SLQ",  "Async/fedasync_8_slq.json"),
        ("FedBuff 8 QSGD",  "Async/fedbuff_8_qsgd.json"),
        ("FedBuff 8 SLQ",   "Async/fedbuff_8_slq.json"),
    ]
    table = PrettyTable([
        "Config",
        "IID ACC", "IID MSE M", "IID MSE S", "IID COS M", "IID COS S",
        "NIID ACC", "NIID MSE M", "NIID MSE S", "NIID COS M", "NIID COS S"
    ])

    num_entrys = 1000
    def mean_std(x: List[float]) -> Tuple[float, float]:
        # assert len(x) >= num_entrys
        return float(np.mean(x)), float(np.std(x))

    for name, path in paths:
        iid_data = parse_data(name, f"{PATH_BASE}/IID/{path}")
        niid_data = parse_data(name, f"{PATH_BASE}/NonIID/{path}")

        assert iid_data is not None
        assert niid_data is not None

        iid_mse_mean, iid_mse_std = mean_std(iid_data.q_errors_mse)
        iid_cos_mean, iid_cos_std = mean_std(iid_data.q_errors_cos)
        niid_mse_mean, niid_mse_std = mean_std(niid_data.q_errors_mse)
        niid_cos_mean, niid_cos_std = mean_std(niid_data.q_errors_cos)

        table.add_row([
            name,
            iid_data.final_acc, iid_mse_mean, iid_mse_std, iid_cos_mean, iid_cos_std,
            niid_data.final_acc, niid_mse_mean, niid_mse_std, niid_cos_mean, niid_cos_std,
        ])

    print(table)


def table_model_dists(data: List[Data]) -> None:
    names = [d.name for d in data]

    for i, d in enumerate(data):
        ms, vs = zip(*d.model_dists)
        ms, vs = list(ms), list(vs)

        ms_mean = np.mean(ms)
        ms_std = np.std(ms)

        vs_mean = np.mean(vs)
        vs_std = np.std(vs)

        print(f"{names[i]}: mean={ms_mean:.2e}+-{ms_std:.2e} | var={vs_mean:.2e}+-{vs_std:.2e}")


def main_async_experiment() -> None:
    PATH_BASE = "../Output/Results/Kamiak/Quant/Real/resnet_50"
    configs_8 = [
        ("FedAvg 32", "fedavg_base.json"),
        ("FedBuff 32", "fedbuff_base.json"),
        ("QSGD 8", "fedbuff_quant_qsgd_8_2.json"),
        ("RQSGD 8", "fedbuff_quant_rqsgd_8.json"),
        ("CNAT 8", "fedbuff_quant_cnat_8.json"),
        ("SLQ 8", "fedbuff_quant_slq_8.json"),
    ]

    configs_4 = [
        ("FedAvg 32", "fedavg_base.json"),
        ("FedBuff 32", "fedbuff_base.json"),
        ("QSGD 4", "fedbuff_quant_qsgd_4_2.json"),
        ("RQSGD 4", "fedbuff_quant_rqsgd_4.json"),
        ("CNAT 4", "fedbuff_quant_cnat_4.json"),
        ("SLQ 4", "fedbuff_quant_slq_4.json"),

    ]

    for configs, bits in [(configs_8, 8), (configs_4, 4)]:
        # Parse all info needed
        all_data: Dict[str, List[Data]] = defaultdict(list)
        for dataset in ["fmnist", "cifar10"]:
            for name, path in configs:
                data = parse_data(name, f"{PATH_BASE}/{dataset}/{path}")
                if data is not None:
                    all_data[dataset].append(data)

        # Plotting info
        xs: Dict[str, List[ScalarList]] = defaultdict(list)
        ys: Dict[str, List[ScalarList]] = defaultdict(list)
        names: Dict[str, List[str]] = defaultdict(list)
        lines = ["--", "--", "-", "-", "-", "-"]

        # Parse plotting info
        for dataset, datas in all_data.items():
            for data in datas:
                x, y = zip(*data.accuracies)
                xs[dataset].append(list(x))
                # xs[dataset].append(list(range(len(y))))
                ys[dataset].append(list(y))
                names[dataset].append(data.name)

        config = PlotConfig(
            title=f"{bits}-bit Wall Clock Accuracy",
            ylabel="Accuracy",
            xlabel="Time (s)",
            figsize=FIGSIZE,
            # ylim=(40, 85),
        )
        # plot_line(
        #     x=xs["fmnist"],
        #     y=ys["fmnist"],
        #     config=config,
        #     linestyles=lines,
        #     labels=names["fmnist"],  # type: ignore
        #     show=False,
        #     save_path=f"./tmp2/fmnist_acc_{bits}.png",
        # )

        # plot_line(
        #     x=xs["cifar10"],
        #     y=ys["cifar10"],
        #     config=config,
        #     linestyles=lines,
        #     labels=names["cifar10"],  # type: ignore
        #     show=False,
        #     save_path=f"./tmp2/cifar_acc_{bits}.png",
        # )

        # Add sent140 for 8 bits
        if bits == 8:
            for name, path in configs:
                data = parse_data(name, f"{PATH_BASE}/../distilbert/sent140/{path}")
                if data is not None:
                    all_data["sent140"].append(data)

        for dataset, datas in all_data.items():
            table = PrettyTable([
                "Method", "CMPT", "COMM", "THROPT"
            ])
            for data in datas:
                table.add_row([data.name, data.a_cmpt, data.a_comm, data.throughput])

            print(bits, dataset)
            print(table)


def param_vs_delta() -> None:
    PATH_BASE = "../Output/Results/Kamiak/Quant/Delta/resnet_50/cifar10/"
    configs_fedavg = [
        ("FedAvg 32", "fedavg_32.json"),
        ("FedAvg SLQ 8", "fedavg_8_slq.json"),
        ("FedAvg QSGD 8", "fedavg_8_qsgd.json"),
    ]

    configs_fedasync = [
        ("FedAsync 32", "fedasync_32.json"),
        ("FedAsync SLQ 8", "fedasync_8_slq.json"),
        ("FedAsync QSGD 8", "fedasync_8_qsgd.json"),
    ]

    configs_fedbuff = [
        ("FedBuff 32", "fedbuff_32.json"),
        ("FedBuff SLQ 8", "fedbuff_8_slq.json"),
        ("FedBuff QSGD 8", "fedbuff_8_qsgd.json"),
    ]

    experiments = [
        ("FedAvg", configs_fedavg, 5 * 20), ("FedAsync", configs_fedasync, 30), ("FedBuff", configs_fedbuff, 5 * 10)
    ]

    ROUND_THRESH = 15000
    for experiment, configs, eval_fix in experiments:
        all_data: List[Data] = []
        for name, path in configs:
            data = parse_data(name, f"{PATH_BASE}/{path}")
            if data is not None:
                all_data.append(data)

        # Plotting info
        xs: List[ScalarList] = []
        ys: List[ScalarList] = []
        names: List[str] = []

        # Parse plotting info
        for data in all_data:
            _, y = zip(*data.accuracies)

            x_per_acc_eval = list(range(len(y)))
            x_per_com_round = list(map(lambda x: x * eval_fix, x_per_acc_eval))
            drop_idx = bisect_right(x_per_com_round, ROUND_THRESH)

            xs.append(x_per_com_round[:drop_idx])
            ys.append(list(y)[:drop_idx])
            names.append(data.name)

        config = PlotConfig(
            title=f"{experiment} Quantization Accuracy",
            ylabel="Accuracy",
            xlabel="Communication Rounds",
            figsize=FIGSIZE,
            # ylim=(40, 85),
        )
        plot_line(
            x=xs,
            y=ys,
            config=config,
            labels=names,  # type: ignore
            show=False,
            save_path=f"./tmp2/{experiment}_quant.png",
        )


def table_target_accuracy():
    PATH_BASE = "../Output/Results/Kamiak/Quant/Real/distilbert/"
    configs = [
        ("QSGD 8", "fedbuff_quant_qsgd_8_2.json"),
        ("RQSGD 8", "fedbuff_quant_rqsgd_8.json"),
        ("CNAT 8", "fedbuff_quant_cnat_8.json"),
        ("SLQ 8", "fedbuff_quant_slq_8.json"),
        # ("QSGD 4", "fedbuff_quant_qsgd_4_2.json"),
        # ("RQSGD 4", "fedbuff_quant_rqsgd_4.json"),
        # ("CNAT 4", "fedbuff_quant_cnat_4.json"),
        # ("SLQ 4", "fedbuff_quant_slq_4.json"),
    ]

    # targets = [("fmnist", 80), ("cifar10", 60)]
    targets = [("sent140", 82)]
    for dataset, target in targets:
        table = PrettyTable(["Method", "Time (m)"])
        print(dataset)

        for name, path in configs:
            explorer = FederatedResultsExplorer()
            explorer.set_federated_results_from_file(f"{PATH_BASE}/{dataset}/{path}")

            t_time = explorer.get_time_to_target_accuracy(target_accuracy=target)
            table.add_row([name, round(t_time / 60, 2)])

        print(table)


def delay_plot():
    # This is the setup for the main experiment
    compute = sample_half_normal(200, sigma=4, shift=0, seed=1, reverse=False)
    network = sample_half_normal(200, sigma=8, shift=10, seed=5, reverse=True)

    config = PlotConfig(
        title="Delay Simulation",
        xlabel="Delay",
        ylabel="Count",
    )
    plot_histogram(
        {"Compute": compute, "Connectivity": network},
        config,
        element="step",
        show=False,
        save_path="./tmp2/delay_hist.png"
    )

    config = PlotConfig(
        title="Client Compute and Network Delay",
        xlabel="Compute Delay Factor",
        ylabel="Bandwidth (Mbps)",
        figsize=FIGSIZE,
    )
    plot_scatter(
        compute,
        network,
        config,
        alpha=0.9,
        marker="o",
        label="Client",
        show=False,
        save_path="./tmp2/delay_scatter.png",
    )


def table_communication_cost():
    table = PrettyTable(["(n, b)", "QSGD", "SLQ"])
    for n in [10, 100, 1000]:
        for b in [8, 4, 2]:
            qsgd = (b + 1) * n
            slq = b * n
            table.add_row([f"({n}, {b})", qsgd / 8, slq / 8])

    print(table)


# table_q_errors()
table_target_accuracy()
# table_communication_cost()
# main_async_experiment()
# main_sync_experiment()
# delay_plot()
# param_vs_delta()
