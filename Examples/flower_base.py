from typing import Tuple, List
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.datasets as datasets

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda:0")

NUM_CLIENTS = 8

NUM_ROUNDS = 10
EPOCHS = 10
BATCH_SIZE = 512

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

full_dataset = datasets.CIFAR10(root="../Data", train=True, transform=transform, download=True)
split_datasets = random_split(
    full_dataset, [len(full_dataset) // NUM_CLIENTS] * NUM_CLIENTS
)

test_set = datasets.CIFAR10(root="../Data", train=False, transform=transform, download=True)


class Loaders:
    def __init__(
        self, 
        train: DataLoader = None,
        test: DataLoader = None,
    ) -> None:
        self.train: DataLoader = train
        self.test: DataLoader = test

def load_datasets(partition_id: int, num_partitions: int) -> Loaders:
    train_loader = DataLoader(split_datasets[partition_id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return Loaders(train_loader, test_loader)

def get_mobile_net_v3_small(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model

def train(net: nn.Module, trainloader: DataLoader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net: nn.Module, testloader: DataLoader) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def _evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     str,
) -> float:
    """Evaluate a model."""
    model.to(device)
    model.eval()

    criterion   = torch.nn.CrossEntropyLoss()
    num_samples = 0
    num_correct = 0
    loss        = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            
            outputs = outputs.argmax(dim=1)

            num_samples += labels.size(0)
            num_correct += (outputs == labels).sum()
    
    return (loss / len(dataloader.dataset)), (num_correct / num_samples * 100).item()

def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict)

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]

class FlowerClient(NumPyClient):
    def __init__(self, model: nn.Module, dataloaders: Loaders, partition_id: int) -> None:
        super().__init__()
        self.model = model
        self.dataloaders = dataloaders
        self.partition_id = partition_id

    def get_parameters(self, config):
        # print(f"\t[Client {self.partition_id}] get parameters")
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        if self.partition_id == 0:
            print(f"\t[Client {self.partition_id} fit, config: {config}]")
            print(f"\t[TRAINING WITH {len(self.dataloaders.train.dataset)}]")

        set_parameters(self.model, parameters)
        train(self.model, self.dataloaders.train, EPOCHS, True)

        return get_parameters(self.model), len(self.dataloaders.train.dataset), {}

    def evaluate(self, parameters, config):
        if self.partition_id == 0:
            print(f"\t[Client {self.partition_id}] evaluate, config: {config}")
            print(f"\t[EVALUATING WITH {len(self.dataloaders.test.dataset)}]")

        set_parameters(self.model, parameters)
        loss, accuracy = _evaluate(self.model, self.dataloaders.test, DEVICE)

        if self.partition_id == 0:
            print(f"\t{loss} {accuracy}")

        return loss, len(self.dataloaders.test.dataset), {"accuracy": accuracy}

def client_fn(context: Context) -> Client:
    model = mobilenet_v3_small(num_classes=10).to(DEVICE)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    data_loaders = load_datasets(partition_id=partition_id, num_partitions=num_partitions)
    return FlowerClient(model, data_loaders, partition_id).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    return {"accuracy": metrics[0][1]["accuracy"]}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    model = get_mobile_net_v3_small(10)
    params = get_parameters(model)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(params),
        inplace=False,
    )

    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)

def main() -> None:
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.2}}

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        verbose_logging=True,
    )
    
if __name__ == "__main__":
    main()