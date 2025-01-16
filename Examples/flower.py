from typing import Tuple, List
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda:0")

NUM_CLIENTS = 10
BATCH_SIZE = 32

class Loaders:
    def __init__(
        self, 
        train: DataLoader = None,
        val: DataLoader = None,
        test: DataLoader = None,
    ) -> None:
        self.train: DataLoader = train
        self.val: DataLoader = val
        self.test: DataLoader = test

def verify_installation() -> None:
    print(f"Training device: {DEVICE}")
    print(f"Flower: {flwr.__version__}")
    print(f"PyTorch: {torch.__version__}")

def load_datasets(partition_id: int, num_partitions: int) -> Loaders:
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True) 

    val_loader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    test_set = fds.load_split("test").with_transform(apply_transforms)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    return Loaders(train_loader, val_loader, test_loader)

def train(net: nn.Module, trainloader: DataLoader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

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

@torch.inference_mode()
def test(net: nn.Module, testloader: DataLoader) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    for batch in testloader:
        images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

        outputs = net(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

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
        # print(f"\t[Client {self.partition_id} fit, config: {config}]")
        set_parameters(self.model, parameters)
        train(self.model, self.dataloaders.train, 1)
        return get_parameters(self.model), len(self.dataloaders.train.dataset), {}

    def evaluate(self, parameters, config):
        # print(f"\tClient {self.partition_id} evaluate, config: {config}")
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.dataloaders.val)
        return loss, len(self.dataloaders.val.dataset), {"accuracy": accuracy}

def client_fn(context: Context) -> Client:
    model = resnet18(num_classes=10).to(DEVICE)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    data_loaders = load_datasets(partition_id=partition_id, num_partitions=num_partitions)
    return FlowerClient(model, data_loaders, partition_id).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    params = get_parameters(model)

    strategy = FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(params),
    )

    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

def main() -> None:
    verify_installation()

    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.5}}

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        verbose_logging=True,
    )
    
if __name__ == "__main__":
    main()