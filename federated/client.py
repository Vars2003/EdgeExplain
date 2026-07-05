import pandas as pd
import time
from typing import Dict, Any

class FederatedClient:
    """
    State actor representing a single federated learning client.
    Conducts local training steps and exports metric payloads to the server.
    """
    def __init__(self, client_id: int, df_partition: pd.DataFrame):
        self.client_id = client_id
        self.df_partition = df_partition
        self.local_model = None
        self.local_accuracy = 0.0
        self.local_loss = 0.0
        self.training_time_ms = 0.0
        self.local_model_weights = None

    def receive_model(self, global_model_weights: Any) -> None:
        """
        Loads global model parameters.
        """
        self.local_model_weights = global_model_weights

    def train_locally(self, model_name: str, target: str = None, local_epochs: int = 1) -> Dict[str, Any]:
        """
        Invokes local trainer to fit a classifier on the local partition.
        """
        from federated.trainer import LocalTrainer
        from federated.models import get_weights
        start_time = time.time()
        
        model, acc, loss, count = LocalTrainer.train(
            self.df_partition, model_name, target,
            initial_weights=self.local_model_weights,
            local_epochs=local_epochs
        )
        
        self.local_model = model
        self.local_accuracy = acc
        self.local_loss = loss
        self.training_time_ms = float((time.time() - start_time) * 1000)
        
        if model is not None:
            self.local_model_weights = get_weights(model)
        else:
            self.local_model_weights = None
            
        return self.export_update()

    def export_update(self) -> Dict[str, Any]:
        """
        Packs update metrics to send to the coordinator.
        """
        return {
            "client_id": self.client_id,
            "samples": len(self.df_partition) if self.df_partition is not None else 0,
            "accuracy": self.local_accuracy,
            "loss": self.local_loss,
            "training_time_ms": self.training_time_ms,
            "weights": self.local_model_weights
        }
