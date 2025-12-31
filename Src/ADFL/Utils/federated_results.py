import json
import os
from typing import Dict, TypeAlias, List, Any, Literal, Tuple, Optional

import numpy as np

from ADFL.types import (
    ClientResults, RoundResults, TrainResults, FederatedResults,
    Accuracy, TrainingConfig, EvalConfig, ScalarPair, AsyncClientWorkerResults
)


ScalarPairs = List[ScalarPair]
A_ScalarPair: TypeAlias = ScalarPair


"""Type aliases to help with readability since we don't have python 3.12."""
t_accuracy: TypeAlias = Dict[str, float]
t_train_config: TypeAlias = Dict[str, Any]
t_eval_config: TypeAlias = Dict[str, Any]

t_train_results: TypeAlias = Dict[str, Any]
t_round_results: TypeAlias = Dict[str, Any]
t_client_result: TypeAlias = Dict[str, Any]
t_trainer_result: TypeAlias = Dict[str, Any]


class FederatedResultsExplorer:
    """Federated Results Explorer.

    Helper class to explore a FederatedResults output from a federated learning run.
    """
    def __init__(self) -> None:
        self.fr = FederatedResults()


    def set_federated_results_from_file(self, filepath: str) -> None:
        """Set the current FederatedResults from a file."""
        data = self._open_json(filepath)
        self.filepath = filepath

        self.fr.paradigm = data["paradigm"]
        self.fr.g_start_time = data["g_start_time"]
        self.fr.g_end_time = self._get(data, "g_end_time")
        self.fr.total_g_rounds = self._get(data, "total_g_rounds")
        self.fr.q_errors_mse = self._get(data, "q_errors_mse")
        self.fr.q_errors_cos = self._get(data, "q_errors_cos")
        self.fr.model_dists = self._get(data, "model_dists")
        self._set_c_accuracies(data)
        self._set_train_config(data)
        self._set_eval_config(data)
        self._set_client_results(data)
        self._set_trainer_results(data)


    def get_time_to_target_accuracy(self, target_accuracy: float) -> float:
        """Get the time taken to reach a target accuracy. 0 if not reached."""
        accuracies = self.get_central_accuracies_raw()

        for t, accuracy in accuracies:
            if accuracy >= target_accuracy:
                return t

        return 0


    def get_central_accuracies_final(self, window: int, method: Literal["mean", "median"]) -> Tuple[float, float]:
        """Get the final central accuracy with the std."""
        central_accuracies = self.get_central_accuracies_raw()
        if len(central_accuracies) == 0:
            return (0.0, 0.0)

        latest_time = central_accuracies[-1][0]
        recent_accuracies: List[float] = []

        for (r_time, acc) in reversed(central_accuracies):
            if latest_time - r_time >= window:
                break

            recent_accuracies.append(acc)

        m_accuracy = 0
        if method == "mean":
            m_accuracy = np.mean(recent_accuracies)
        elif method == "median":
            m_accuracy = np.median(recent_accuracies)

        return float(m_accuracy), float(np.std(recent_accuracies))


    def get_central_accuracies_raw(self) -> ScalarPairs:
        """Get central accuracies in the form (time, accuracy)."""
        accuracies: ScalarPairs = []

        for accuracy in self.fr.c_accuracies:
            r_time = accuracy.g_time - self.fr.g_start_time
            accuracies.append((r_time, accuracy.value))

        accuracies.sort(key=lambda x: x[0])
        return accuracies


    def get_mse_all(self) -> List[ScalarPairs]:
        """Get MSE for all clients."""
        all_mses: List[ScalarPairs] = []

        for client_results in self.fr.client_results:
            client_mses: ScalarPairs = []

            for round_result in client_results.rounds:
                r_time = round_result.g_start_time - self.fr.g_start_time
                client_mses.append((r_time, round_result.mse))

            client_mses.sort(key=lambda x: x[0])
            all_mses.append(client_mses)

        return all_mses


    def get_round_throughput(self) -> float:
        """Get the round throughput."""
        # Find the time of the last round completed
        last_round_time = self.fr.g_start_time
        for client_result in self.fr.client_results:
            for round_result in client_result.rounds:
                round_end_time = round_result.g_start_time + round_result.round_time
                if round_end_time > last_round_time:
                    last_round_time = round_end_time

        total_elapsed_time = last_round_time - self.fr.g_start_time
        return self.fr.total_g_rounds / total_elapsed_time


    def get_client_network_compute_ratios(self) -> List[float]:
        """Get the ratio between network and compute times."""
        ratios: List[float] = []

        for client_result in self.fr.client_results:
            total_compute_time, total_network_time = 0, 0

            for round_result in client_result.rounds:
                total_compute_time += round_result.compute_time
                total_network_time += round_result.network_time

            if total_compute_time == 0 or total_network_time == 0:
                continue

            ratios.append(total_network_time / total_compute_time)

        return ratios


    def get_per_client_network_compute_times(self, method: Literal["total", "mean"]) -> ScalarPairs:
        """Get the network and compute times per client, either total or mean."""
        times: ScalarPairs = []

        for client_result in self.fr.client_results:
            total_compute_time, total_network_time = 0, 0

            for round_result in client_result.rounds:
                total_compute_time += round_result.compute_time
                total_network_time += round_result.network_time

            if len(client_result.rounds) == 0:
                times.append((0, 0))

            if method == "total":
                times.append((total_network_time, total_compute_time))
            elif method == "mean":
                times.append((
                    total_network_time / len(client_result.rounds),
                    total_compute_time / len(client_result.rounds),
                ))

        return times


    def get_client_network_compute_times(self, method: Literal["total", "mean"]) -> Tuple[ScalarPair, ScalarPair]:
        """Get the network and compute times across all clients, either total or mean along with the std."""
        compute_times, network_times, total_rounds = [], [], 0

        for client_result in self.fr.client_results:
            for round_result in client_result.rounds:
                compute_times.append(round_result.compute_time)
                network_times.append(round_result.network_time)
                total_rounds += 1

        assert total_rounds == self.fr.total_g_rounds

        compute_times_std = np.std(compute_times).item()
        network_times_std = np.std(network_times).item()

        if method == "total":
            return (
                (np.sum(network_times).item(), network_times_std),
                (np.sum(compute_times).item(), compute_times_std),
            )
        elif method == "mean":
            return (
                (np.mean(network_times).item(), network_times_std),
                (np.mean(compute_times).item(), compute_times_std),
            )


    def get_total_train_time(self) -> float:
        """Get the total training time."""
        return self.fr.g_end_time - self.fr.g_start_time


    def get_average_client_uptime(self) -> float:
        """Get the average client uptime."""
        return np.mean([r.uptime for r in self.fr.trainer_results]).item()


    def _set_train_config(self, data: Dict) -> None:
        d_train_config: t_train_config = data["train_config"]

        dataset = self._get(d_train_config, "dataset", "none")

        self.fr.train_config = TrainingConfig(
            dataset         = TrainingConfig.Dataset(dataset),
            channel         = self._get(d_train_config, "channel"),
            strategy        = None,  # Not parsed
            num_rounds      = self._get(d_train_config, "num_rounds"),
            num_epochs      = self._get(d_train_config, "num_epochs"),
            num_clients     = self._get(d_train_config, "num_clients"),
            num_cur_clients = self._get(d_train_config, "num_cur_clients"),
            num_servers     = self._get(d_train_config, "num_servers"),
            batch_size      = self._get(d_train_config, "batch_size"),
            max_rounds      = self._get(d_train_config, "max_rounds"),
            timeout         = self._get(d_train_config, "timeout"),
            delay           = self._get(d_train_config, "delay"),
        )


    def _set_eval_config(self, data: Dict) -> None:
        d_eval_config: t_eval_config = data["eval_config"]

        self.fr.eval_config = EvalConfig(
            self._get(d_eval_config, "method"),
            self._get(d_eval_config, "central"),
            self._get(d_eval_config, "threshold"),
            self._get(d_eval_config, "num_actors"),
            self._get(d_eval_config, "client_map"),
        )


    def _set_c_accuracies(self, data: Dict) -> None:
        d_c_accuracies: List[t_accuracy] = data["c_accuracies"]

        for d_accuracy in d_c_accuracies:
            accuracy = Accuracy(self._get(d_accuracy, "value"), self._get(d_accuracy, "g_time"))
            self.fr.c_accuracies.append(accuracy)


    def _set_client_results(self, data: Dict) -> None:
        d_client_results: List[t_client_result] = data["client_results"]

        for d_client_result in d_client_results:
            client_result = ClientResults()
            client_result.client_id = self._get(d_client_result, "client_id")

            # RoundResults
            d_round_results: List[t_round_results] = self._get(d_client_result, "rounds")
            for d_round_result in d_round_results:
                round_result = self._parse_round_result(d_round_result)
                client_result.rounds.append(round_result)

            # Accuracy
            d_accuracies: List[t_accuracy] = self._get(d_client_result, "accuracies")
            for d_accuracy in d_accuracies:
                accuracy = Accuracy(self._get(d_accuracy, "value"), self._get(d_accuracy, "g_time"))
                client_result.accuracies.append(accuracy)

            self.fr.client_results.append(client_result)


    def _parse_round_result(self, d_round_result: t_round_results) -> RoundResults:
        train_results: List[TrainResults] = []
        d_train_results: List[t_train_results] = self._get(d_round_result, "train_results")
        for d_train_result in d_train_results:
            train_result = self._parse_train_result(d_train_result)
            train_results.append(train_result)

        g = lambda x: self._get(d_round_result, x)

        return RoundResults(
            train_round   = g("train_round"),
            train_results = train_results,
            round_time    = g("round_time"),
            compute_time  = g("compute_time"),
            network_time  = g("network_time"),
            epochs        = g("epochs"),
            g_start_time  = g("g_start_time"),
            mse           = g("mse"),
        )


    def _parse_train_result(self, d_train_result: t_train_results) -> TrainResults:
        return TrainResults(
            self._get(d_train_result, "running_loss"),
            self._get(d_train_result, "average_loss"),
            self._get(d_train_result, "sum_loss"),
            self._get(d_train_result, "accuracy"),
            self._get(d_train_result, "elapsed_time"),
            self._get(d_train_result, "g_start_time"),
        )


    def _open_json(self, filepath: str) -> Dict:
        self._assert_file_exists(filepath)

        with open(filepath, "r") as file:
            data = json.load(file)
        return data


    def _assert_file_exists(self, filepath: str) -> None:
        assert os.path.exists(filepath)


    def _get(self, data: Dict, key: str, default: Any = None) -> Any:
        if key in data:
            return data[key]
        else:
            print(f"{self.filepath}: Failed to get key={key}")
            return default

    def _set_trainer_results(self, data):
        d_trainer_results: Optional[List[t_trainer_result]] = self._get(data, "trainer_results")
        if d_trainer_results is None:
            return

        for d_trainer_result in d_trainer_results:
            g = lambda x: self._get(d_trainer_result, x)

            self.fr.trainer_results.append(AsyncClientWorkerResults(
                worker_id=g("worker_id"),
                fetch_times=g("fetch_times"),
                fetch_mean=g("fetch_mean"),
                fetch_std=g("fetch_std"),
                fetch_min=g("fetch_min"),
                fetch_max=g("fetch_max"),
                uptime=g("uptime"),
            ))

