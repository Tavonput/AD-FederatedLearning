import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from ADFL.types import TrainResults


LR = 0.001


def _train_epoch(
    model: nn.Module, 
    optimizer: Optimizer, 
    criterion: nn.Module, 
    dataloader: DataLoader, 
    device: str,
    slowness: float,
    save_running_loss: bool = False
) -> TrainResults:
    """Train a model for one epoch with a given slowness."""
    model.to(device)
    model.train()
    
    results = TrainResults()
    start_time = time.time()
    results.g_start_time = start_time
    
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        results.running_loss.append((loss.item(), time.time() - start_time))
        total += labels.size(0)
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    train_time = time.time() - start_time

    # Simulate slowness
    sleep_time = train_time * (slowness - 1)
    time.sleep(sleep_time)

    results.accuracy = correct / total
    results.sum_loss = sum(loss[0] for loss in results.running_loss)
    results.average_loss = results.sum_loss / len(dataloader.dataset)
    results.elapsed_time = train_time + sleep_time

    if save_running_loss is False:
        results.running_loss.clear()

    return results


def _evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     str,
) -> float:
    """Evaluate a model."""
    model.to(device)
    model.eval()

    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.argmax(dim=1)

            num_samples += labels.size(0)
            num_correct += (outputs == labels).sum()
    
    return (num_correct / num_samples * 100).item()

# Just stuff for type hinting I guess
class AsyncServer:
    # Defined in ADFL/server.py
    pass

