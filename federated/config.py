class FLConfig:
    """
    Configuration parameters for EdgeExplain Federated Learning simulations.
    """
    DEFAULT_CLIENTS = 3
    DEFAULT_ROUNDS = 5
    DEFAULT_EPOCHS = 10
    BATCH_SIZE = 32
    RANDOM_SEED = 42
    SIMULATION_MODE = True
    DEFAULT_AGGREGATOR = "FedAvg"
    SUPPORTED_AGGREGATORS = ["FedAvg", "FedProx", "FedNova", "SCAFFOLD", "FedDyn", "MOON"]
