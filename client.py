import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

import torch.nn.functional as F

# Define the CNN model in PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Shared randomness seed
shared_seed = 42
torch.manual_seed(shared_seed)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split the training data to simulate federated learning
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

# Data loaders
train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=32, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

# Define the client class with compression
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        params = [param.cpu().numpy() for param in self.model.parameters()]
        compressed_params = self.compress_parameters(params)
        return compressed_params

    def set_parameters(self, compressed_parameters):
        decompressed_params = self.decompress_parameters(compressed_parameters)
        for param, new_param in zip(self.model.parameters(), decompressed_params):
            param.data = torch.tensor(new_param, dtype=param.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Train for 1 epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

    def compress_parameters(self, params):
        # Compress parameters to single-bit representation using shared randomness
        np.random.seed(shared_seed)
        compressed_params = [(param > np.random.rand(*param.shape)).astype(np.int8) for param in params]
        return compressed_params

    def decompress_parameters(self, compressed_params):
        # Decompress single-bit parameters back to the original scale
        np.random.seed(shared_seed)
        decompressed_params = [param.astype(np.float32) * 2 - 1 for param in compressed_params]
        return decompressed_params

# Initialize the model
model = CNNModel()

# Create a client
client = FLClient(model, train_loader, val_loader)

# Start Flower client
fl.client.start_numpy_client(server_address="localhost:8080", client=client)
