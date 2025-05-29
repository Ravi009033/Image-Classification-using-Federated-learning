## Federated Learning with Convolutional Neural Networks (CNN)

# 📘Overview

This repository contains an implementation of a Convolutional Neural Network (CNN) for federated learning using PyTorch and Flower (FLWR). Federated learning enables collaborative model training across multiple decentralized devices while keeping the data localized on each device, enhancing data privacy.

# 🚀Key Features

- Federated Learning: Train a shared global CNN model across multiple clients without transferring raw data.      
- Convolutional Neural Network: Deep learning architecture ideal for image classification tasks     
- Data Compression: Supports single-bit gradient compression to minimize communication overhead.     
- PyTorch & Flower: Utilizes PyTorch for model development and training, and Flower for FL orchestration      

# 🔁 Federated Learning Workflow

Federated learning enables training a global model across multiple clients, each holding local datasets. The server coordinates the process without requiring clients to share their data, thus maintaining data privacy.

- Initialization: The central server initializes the global CNN model.
- Local Training: Each client receives the model, trains it locally on their private dataset, and returns the updated parameters.        
- Aggregation: The server aggregates client updates using Federated Averaging (FedAvg).       
- Iteration: This process is repeated for several rounds until the global model converges.             

# 🧠 Model Architecture

- Type: Convolutional Neural Network
- Use Case: Image classification (e.g., handwritten digit recognition using MNIST)
- Layers: Conv2D, ReLU, MaxPooling, Fully Connected layers
- Activation: ReLU (Hidden Layers), Softmax (Output Layer)

# 🏋️ Model Training Configuration
- Optimizer: Stochastic Gradient Descent (SGD)
- Loss Function: Cross-Entropy Loss
- Dataset: MNIST (or any image classification dataset)
- Evaluation: Local accuracy is evaluated at each client; global accuracy is aggregated at the server.


# 📊 Results
- Use Case Tested: MNIST digit classification
- Achieved Accuracy: ~96% training accuracy after federated training
Note: Actual performance may vary based on the number of clients, local epochs, communication rounds, and system configurations.
