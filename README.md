Logistic Regression for Federated Learning

Overview

This repository contains an implementation of Logistic Regression for federated learning using PyTorch and Flower (FLWR). 
Federated learning allows training a global model across multiple decentralized devices while keeping data local on each device, thus enhancing privacy.

Key Features

Federated Learning: Train a logistic regression model across multiple clients without sharing raw data.   
PyTorch: Utilizes PyTorch for defining and training the logistic regression model.    
Flower (FLWR): Leveraging the Flower framework for federated learning orchestration.      
Data Compression: Includes optional single-bit compression to reduce communication overhead between clients and the server.    

Federated Learning

In federated learning, the server coordinates multiple clients to train a model collaboratively without sharing data between them. Each client trains a model locally on its data and sends only the model updates (gradients) to the server. The server aggregates these updates to refine a global model.

How It Works

Initialization: The server initializes the global model.         
Local Training: Clients receive the current global model, train it on their local data, and send the updates back to the server.           
Aggregation: The server aggregates the updates and improves the global model.          
Iteration: The process repeats for several rounds until the model converges.         

Model

The logistic regression model is a simple yet effective linear model used for binary classification. It computes the probability of a sample belonging to a particular class and makes decisions based on a threshold (typically 0.5).

Model Training

Optimizer: Stochastic Gradient Descent (SGD) is used for training.             
Loss Function: The model is trained using Cross-Entropy Loss, suitable for classification tasks.

Customization

customize the implementation by modifying the following:    
Model Architecture: Replace the logistic regression model with any other model (e.g., CNN).   
Federated Learning Strategy: Modify the aggregation strategy on the server side.   
Data Compression: Enable or modify the single-bit compression logic for model updates.    

Results

After training for several federated learning rounds, the model achieve accuracy like:  
Training Accuracy: ~93%           
(Note: Actual results may vary based on the dataset and federated learning setup.)
