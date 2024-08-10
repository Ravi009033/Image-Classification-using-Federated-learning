import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

# Define the Logistic Regression model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        outputs = self.linear(x)
        return outputs

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
                data = data.view(data.size(0), -1)
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
                data = data.view(data.size(0), -1)
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

# Model parameters
input_dim = 28 * 28  # MNIST images are 28x28 pixels
output_dim = 10      # 10 classes for the digits 0-9

# Initialize the model
model = LogisticRegressionModel(input_dim, output_dim)

# Create a client
client = FLClient(model, train_loader, val_loader)

# Start Flower client
fl.client.start_numpy_client(server_address="localhost:8080", client=client)
