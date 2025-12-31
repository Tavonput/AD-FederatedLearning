import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from ADFL.types import TrainResults
from ADFL.model import Parameters, model_forward


LR = 0.001  # CNN
# LR = 0.00002  # LLM


def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch._C.device,
    save_running_loss: bool = False
) -> TrainResults:
    """Train a model for one epoch."""
    model.to(device)
    model.train()

    results = TrainResults()
    start_time = time.time()
    results.g_start_time = start_time

    correct, total = 0, 0

    for batch in dataloader:
        forward_results = model_forward(model, batch, device, criterion)

        results.running_loss.append((forward_results.loss.item(), time.time() - start_time))

        optimizer.zero_grad()
        forward_results.loss.backward()
        optimizer.step()

        total += forward_results.n_samples
        correct += forward_results.correct

    train_time = time.time() - start_time

    results.accuracy = correct / total
    results.sum_loss = sum(loss[0] for loss in results.running_loss)
    results.average_loss = results.sum_loss / len(dataloader.dataset)  # type: ignore
    results.elapsed_time = train_time

    if save_running_loss is False:
        results.running_loss.clear()

    return results


def evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch._C.device,
) -> float:
    """Evaluate a model."""
    model.to(device)
    model.eval()

    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            forward_results = model_forward(model, batch, device)

            num_samples += forward_results.n_samples
            num_correct += forward_results.correct

    return num_correct / num_samples * 100


# Just stuff for type hinting I guess
class ServerProxy:
    # Defined in ADFL/Server/proxy.py
    pass
