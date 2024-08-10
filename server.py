import flwr as fl

# Define a strategy with a simple FedAvg aggregation
class CompressedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # Optionally: Convert back to the float32 format for the global model after aggregation
        return aggregated_weights

# Start the server
strategy = CompressedFedAvg()
fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 3},
    strategy=strategy,
)
