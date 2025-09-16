import json
from typing import List, Dict
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

TWEET_IDX = 4

def explore_data(filepath: Path) -> None:
    print("-----------------------------")
    print(f"Loading dataset: {filepath}")

    with open(filepath, "r") as file:
        data = json.load(file)

    users: List[str] = data.get("users")
    num_samples: List[int] = data.get("num_samples")
    user_data: Dict = data.get("user_data")

    num_users = len(users)
    total_samples = np.sum(num_samples)
    min_samples, max_samples = np.min(num_samples), np.max(num_samples)

    print(f"Number of users: {num_users}")
    print(f"Number of samples: {total_samples}")
    print(f"Min/Max samples per user: ({min_samples}/{max_samples})")

    num_pos = 0
    for data in user_data.values():
        for _, y in zip(data["x"], data["y"]):
            num_pos += y

    print(f"Number of positive: {num_pos}")
    print(f"Number of negative: {total_samples - num_pos}")

    print("Examples:")
    num_tweets = 5
    i = 0
    for data in user_data.values():
        i += 1
        if i > num_tweets:
            break

        print(f"\t[{data["y"][0]}] {data["x"][0][TWEET_IDX]}")


def parse_arguments() -> Path:
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to data directory.")
    return Path(parser.parse_args().data_dir)

def main() -> None:
    data_path = parse_arguments()
    explore_data(data_path / "train.json")
    explore_data(data_path / "test.json")


if __name__ == "__main__":
    main()
