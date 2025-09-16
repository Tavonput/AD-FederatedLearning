import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from transformers import DistilBertTokenizer


TWEET_IDX = 4
MAX_LENGTH = 128


@dataclass
class Samples:
    texts:  List[str]
    labels: List[int]


def load_data(filepath: Path) -> Samples:
    with open(filepath, "r") as file:
        data = json.load(file)
    user_data: Dict = data.get("user_data")

    samples = Samples([], [])
    for data in user_data.values():
        for x, y in zip(data["x"], data["y"]):
            samples.texts.append(x[TWEET_IDX])
            samples.labels.append(y)

    return samples


def trim(array: List, multiple: int) -> List:
    remainder = len(array) % multiple
    if remainder == 0:
        return array
    return array[:-remainder]


def prepare_and_save_data(
    input_file: Path, output_file: Path, tokenizer: DistilBertTokenizer, multiple: Optional[int]
) -> None:
    print(f"Loading data from {input_file}")
    samples = load_data(input_file)

    if multiple is not None:
        n_samples = len(samples.texts)
        samples.texts = trim(samples.texts, multiple)
        samples.labels = trim(samples.labels, multiple)
        print(f"Trimmed data from length {n_samples} to {len(samples.texts)}")

    print(f"Tokenizing data with embedding size {MAX_LENGTH}. This may take a while...")
    encodings = tokenizer(
        samples.texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(samples.labels, dtype=torch.long),
    }

    torch.save(data, output_file)
    print(f"Saved embeddings to {output_file}")


def parse_args() -> Tuple[Path, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="The path to the directory containing Sent140 data.")
    parser.add_argument("--multiple", type=int, help="Trim the end of the train set to this multiple.")
    args = parser.parse_args()

    dir_path = Path(args.data_dir)

    assert os.path.exists(dir_path)
    assert os.path.exists(dir_path / "train.json")
    assert os.path.exists(dir_path / "test.json")

    if args.multiple is not None:
        assert args.multiple > 0

    return dir_path, args.multiple


def main() -> None:
    data_dir, multiple = parse_args()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    prepare_and_save_data(data_dir / "train.json", data_dir / "train.pt", tokenizer, multiple)
    prepare_and_save_data(data_dir / "test.json", data_dir / "test.pt", tokenizer, None)


if __name__ == "__main__":
    main()
