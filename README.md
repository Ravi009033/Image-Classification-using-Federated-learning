Federated Learning with Convolutional Neural Networks (CNN)

Overview

This repository contains an implementation of a Convolutional Neural Network (CNN) for federated learning using PyTorch and Flower (FLWR). Federated learning enables collaborative model training across multiple decentralized devices while keeping the data localized on each device, enhancing data privacy.

Key Features

Federated Learning: Train a CNN model collaboratively across multiple clients without sharing raw data.      
Convolutional Neural Network: A deep learning model that is particularly effective for image classification tasks.       
Data Compression: Optional single-bit compression to reduce communication overhead between clients and the server.     
PyTorch & Flower: Uses PyTorch for model definition and training, and Flower for federated learning orchestration.         

Federated Learning

Federated learning enables training a global model across multiple clients, each holding local datasets. The server coordinates the process without requiring clients to share their data, thus maintaining data privacy.

How It Works

Initialization: The server initializes a global CNN model.        
Local Training: Clients receive the current global model, train it on their local data, and send the updated model parameters back to the server.        
Aggregation: The server aggregates the parameters from all clients to update the global model.        
Iteration: This process is repeated for several rounds until the global model converges.             

Model

The Convolutional Neural Network (CNN) used in this implementation is designed for image classification tasks, such as digit recognition with the MNIST dataset.

Model Training

Optimizer: The model is trained using Stochastic Gradient Descent (SGD).      
Loss Function: Cross-Entropy Loss is used, which is suitable for classification tasks.       
Data: The model is trained on image data, such as MNIST.          

Results

The CNN model is well-suited for tasks like digit classification with the MNIST dataset.
Training Accuracy: ~96%    
(Note: Results may vary depending on the dataset, number of clients, and federated learning configuration.)
