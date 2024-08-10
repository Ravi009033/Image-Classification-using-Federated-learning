import flwr as fl

# Define a strategy
strategy = fl.server.strategy.FedAvg()

# Start the server
fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 3},  # Number of federated learning rounds
    strategy=strategy,
)
