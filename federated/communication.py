from typing import Dict, Any, List

class LocalCommunicationBus:
    """
    In-memory message transmission bus simulating server-to-client broadcasts.
    Eliminates socket/HTTP overhead by swapping in-memory Python objects.
    """
    def __init__(self):
        self.payload_history = []
        self.total_bytes_transferred = 0
        self.total_latency_ms = 0.0

    @staticmethod
    def estimate_bytes(weights: Any) -> int:
        """
        Estimates the serialized size of the model parameters.
        Float64 weights are 8 bytes each.
        """
        if weights is None:
            return 120 # Nominal bytes for client handshake / empty payload
        if isinstance(weights, dict):
            coef_size = weights.get("coef").size if hasattr(weights.get("coef"), "size") else 0
            intercept_size = weights.get("intercept").size if hasattr(weights.get("intercept"), "size") else 0
            classes_size = weights.get("classes").size if hasattr(weights.get("classes"), "size") else 0
            return int((coef_size + intercept_size + classes_size) * 8) + 120
        return 120

    def broadcast_to_clients(self, global_model_weights: Any, client_list: List[Any]) -> int:
        """
        Simulates broadcasting the global model parameters to all clients.
        """
        client_count = len(client_list)
        weight_bytes = self.estimate_bytes(global_model_weights)
        broadcast_bytes = weight_bytes * client_count
        
        # Simulating 8ms latency overhead per client broadcast
        latency = 8.0 * client_count
        
        self.total_bytes_transferred += broadcast_bytes
        self.total_latency_ms += latency
        
        for client in client_list:
            client.receive_model(global_model_weights)
            
        return broadcast_bytes

    def collect_from_clients(self, client_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Simulates harvesting local updates (trained weights, accuracies, sample counts) from clients.
        """
        updates = []
        for client in client_list:
            update = client.export_update()
            weight_bytes = self.estimate_bytes(update.get("weights"))
            
            # Simulating 8ms latency upload overhead per client upload
            latency = 8.0
            
            self.total_bytes_transferred += weight_bytes
            self.total_latency_ms += latency
            
            updates.append(update)
            self.payload_history.append(update)
            
        return updates
