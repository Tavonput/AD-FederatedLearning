import json
import os
from typing import Dict, TypeAlias, List, Any

import numpy as np

from ADFL.types import (
    ClientResults, RoundResults, TrainResults, FederatedResults,
    Accuracy, TrainingConfig, TCDataset, TCMethod, EvalConfig, ScalarPair
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
        self._set_c_accuracies(data)
        self._set_train_config(data)
        self._set_eval_config(data)
        self._set_client_results(data)


    def get_central_accuracies_final(self, window: int) -> float:
        """Get the median final central accuracy."""
        central_accuracies = self.get_central_accuracies_raw()

        latest_time = central_accuracies[-1][0]
        recent_accuracies: List[float] = []

        for (r_time, acc) in reversed(central_accuracies):
            if latest_time - r_time >= window:
                break

            recent_accuracies.append(acc)

        return np.median(recent_accuracies)  # type: ignore


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


    def _set_train_config(self, data: Dict) -> None:
        d_train_config: t_train_config = data["train_config"]

        dataset = self._get(d_train_config, "dataset", "none")

        self.fr.train_config = TrainingConfig(
            method       = self._get(d_train_config, "method", TCMethod.NONE),
            dataset      = TCDataset(dataset),
            strategy     = None,  # Not parsed
            num_rounds   = self._get(d_train_config, "num_rounds"),
            num_epochs   = self._get(d_train_config, "num_epochs"),
            num_clients  = self._get(d_train_config, "num_clients"),
            num_servers  = self._get(d_train_config, "num_servers"),
            batch_size   = self._get(d_train_config, "batch_size"),
            max_rounds   = self._get(d_train_config, "max_rounds"),
            timeout      = self._get(d_train_config, "timeout"),
            compress     = self._get(d_train_config, "compress"),
            quant_lvl_1  = self._get(d_train_config, "quant_lvl_1"),
            quant_lvl_2  = self._get(d_train_config, "quant_lvl_2"),
            slowness_map = self._get(d_train_config, "slowness_map"),
            sc_map       = self._get(d_train_config, "sc_map"),
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

        return RoundResults(
            self._get(d_round_result, "train_round"),
            train_results,
            self._get(d_round_result, "round_time"),
            self._get(d_round_result, "epochs"),
            self._get(d_round_result, "g_start_time"),
            self._get(d_round_result, "mse"),
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
        value = data.get(key)

        if value is None:
            print(f"{self.filepath}: Failed to get key={key}")
            return default

        return value
