import os
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification
from tqdm import tqdm


Batch = Dict[str, torch.Tensor]

@dataclass
class Args:
    data_dir:    Path
    save_dir:    Path
    num_classes: int
    batch_size:  int
    epochs:      int
    lr:          float
    verbose:     bool


@dataclass
class TrainResults:
    time_per_epoch: List[float] = field(default_factory=list)
    train_loss:     List[float] = field(default_factory=list)
    val_loss:       List[float] = field(default_factory=list)
    accuracy:       List[float] = field(default_factory=list)


class TokenizedDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        print(f"Loading dataset {data_path}")
        self.data = torch.load(data_path)


    def __len__(self) -> int:
        return self.data["input_ids"].size(0)


    def __getitem__(self, index) -> Dict:
        return {
            "input_ids": self.data["input_ids"][index],
            "attention_mask": self.data["attention_mask"][index],
            "labels": self.data["labels"][index],
        }


class ResultsEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_results(model: DistilBertForSequenceClassification, results: TrainResults, config: Args) -> None:
    data = asdict(config)
    data["results"] = asdict(results)

    model.save_pretrained(config.save_dir)
    with open(config.save_dir / "results.json", "w") as file:
        json.dump(data, file, indent=4, cls=ResultsEncoder)
    print(f"Model saved to {config.save_dir}")


def model_forward(
    model: nn.Module, batch: Batch, device: torch._C.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    input_ids = batch["input_ids"].to(device)
    attentention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attentention_mask,
        labels=labels
    )

    preds = outputs.logits.argmax(dim=1)
    correct = (preds == labels).sum()

    return outputs.logits, outputs.loss, correct, labels.size(0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch._C.device,
    epochs: int = 3,
    lr: float = 2e-5,
    verbose: bool = True
) -> TrainResults:
    print(f"Training model on device {device}")
    results = TrainResults()

    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        s_time = time.time()

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=(not verbose)):
            _, loss, _, _ = model_forward(model, batch, device)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation", disable=(not verbose)):
                _, loss, correct, total = model_forward(model, batch, device)

                total_val_loss += loss.item()
                total_correct += correct.item()
                total_samples += total

        avg_val_loss = total_val_loss / len(test_loader)
        accuracy = total_correct / total_samples
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        results.train_loss.append(avg_train_loss)
        results.val_loss.append(avg_val_loss)
        results.accuracy.append(accuracy)
        results.time_per_epoch.append(time.time() - s_time)

    model.to("cpu")
    return results


def parge_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, required=True, help="Path to data directory contains train/test.pt.")
    parser.add_argument("-s", type=str, required=True, help="Path to save directory.")
    parser.add_argument("-nc", type=int, required=True, help="Number of classes.")
    parser.add_argument("-bs", type=int, default=16, help="Batch size.")
    parser.add_argument("-ep", type=int, default=3, help="Epochs.")
    parser.add_argument("-lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--verbose", action="store_true", help="Show tqdm progress bar.")

    args = parser.parse_args()

    assert os.path.exists(Path(args.d) / "train.pt")
    assert os.path.exists(Path(args.d) / "test.pt")

    if not os.path.exists(args.s):
        os.makedirs(args.s, exist_ok=True)

    return Args(
        data_dir=Path(args.d),
        save_dir=Path(args.s),
        num_classes=args.nc,
        batch_size=args.bs,
        epochs=args.ep,
        lr=args.lr,
        verbose=args.verbose,
    )


def main() -> None:
    config = parge_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TokenizedDataset(config.data_dir / "train.pt")
    test_dataset = TokenizedDataset(config.data_dir/ "test.pt")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=config.num_classes
    )
    results = train_model(
        model,
        train_loader,
        test_loader,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        verbose=config.verbose,
    )

    save_results(model, results, config)


if __name__ == "__main__":
    main()
