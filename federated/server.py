from typing import List, Dict, Any
from federated.aggregator import BaseAggregator

class FederatedServer:
    """
    Coordinator server orchestrating FL rounds and client registrations.
    Decoupled from specific aggregation mechanics by referencing only the BaseAggregator interface.
    """
    def __init__(self, aggregator_name: str = "FedAvg"):
        self.clients = []
        self.current_round = 0
        self.global_model = None
        self.aggregator = self._resolve_aggregator(aggregator_name)

    def _resolve_aggregator(self, name: str) -> BaseAggregator:
        """
        Dynamically instantiates aggregator strategies based on name.
        """
        from federated.aggregator import (
            FedAvgAggregator,
            FedProxAggregator,
            FedNovaAggregator,
            SCAFFOLDAggregator,
            FedDynAggregator,
            MOONAggregator
        )
        clean_name = name.lower().strip()
        if "prox" in clean_name:
            return FedProxAggregator()
        elif "nova" in clean_name:
            return FedNovaAggregator()
        elif "scaffold" in clean_name:
            return SCAFFOLDAggregator()
        elif "dyn" in clean_name:
            return FedDynAggregator()
        elif "moon" in clean_name:
            return MOONAggregator()
        else:
            return FedAvgAggregator()

    def register_client(self, client: Any) -> None:
        """
        Registers client instances in the coordination list.
        """
        if client not in self.clients:
            self.clients.append(client)

    def run_round(self, model_name: str, target: str = None, local_epochs: int = 1) -> Dict[str, Any]:
        """
        Schedules broadcast updates, commands client local fits, and invokes strategy aggregates.
        """
        import time
        from federated.communication import LocalCommunicationBus
        from federated.models import load_model, set_weights, get_weights
        
        self.current_round += 1
        start_round_time = time.time()
        
        # Broadcast global parameters to clients (in-memory bus wrapper)
        bus = LocalCommunicationBus()
        global_weights = get_weights(self.global_model) if self.global_model is not None else None
        
        broadcast_bytes = bus.broadcast_to_clients(global_weights, self.clients)
        
        # Command client training fits locally
        for client in self.clients:
            client.train_locally(model_name, target, local_epochs)
            
        collect_updates = bus.collect_from_clients(self.clients)
        
        # Validate client updates before running aggregations
        is_valid = self.aggregator.validate(collect_updates)
        
        # Aggregate parameters through base contract interface
        agg_result = {}
        agg_time_ms = 0.0
        
        if is_valid:
            start_agg = time.time()
            agg_result = self.aggregator.aggregate(collect_updates)
            agg_time_ms = float((time.time() - start_agg) * 1000)
            
            # If aggregation succeeded, update global model
            if agg_result.get("status") == "SUCCESS" and agg_result.get("global_weights") is not None:
                if self.global_model is None:
                    self.global_model = load_model(model_name)
                set_weights(self.global_model, agg_result["global_weights"])
            
        summary = self.aggregator.summarize()
        round_time_ms = float((time.time() - start_round_time) * 1000)
        
        # Total bytes and latency for this round
        round_bytes = bus.total_bytes_transferred
        round_latency_ms = bus.total_latency_ms
        
        return {
            "round": self.current_round,
            "validation_passed": is_valid,
            "aggregation_result": agg_result,
            "summary": summary,
            "client_updates": collect_updates,
            "aggregation_time_ms": agg_time_ms,
            "bytes_transferred": round_bytes,
            "latency_ms": round_latency_ms,
            "round_time_ms": round_time_ms
        }

    def evaluate_global_model(self, df: Any, target: str = None) -> Dict[str, float]:
        """
        Evaluates the current global model on the complete validation dataset.
        Returns accuracy and loss.
        """
        if self.global_model is None:
            return {"accuracy": 0.0, "loss": 1.0}
            
        if not target:
            target = df.columns[-1]
            
        import pandas as pd
        X = df.drop(columns=[target])
        y = df[target]
        
        # Preprocessing: Impute numeric nulls and encode object categoricals
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_encoded[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X_encoded[col] = X[col].astype(str).astype("category").cat.codes
                
        y_encoded = y.fillna(y.mode()[0] if not y.isna().all() else 0)
        if not pd.api.types.is_numeric_dtype(y_encoded):
            y_encoded = y_encoded.astype(str).astype("category").cat.codes
            
        try:
            acc = float(self.global_model.score(X_encoded, y_encoded))
            
            # Loss metric (Approximated log loss)
            loss = 0.0
            if hasattr(self.global_model, "predict_proba"):
                try:
                    from sklearn.metrics import log_loss
                    probs = self.global_model.predict_proba(X_encoded)
                    loss = float(log_loss(y_encoded, probs))
                except Exception:
                    loss = float(1.0 - acc)
            else:
                loss = float(1.0 - acc)
                
            return {"accuracy": acc, "loss": loss}
        except Exception:
            return {"accuracy": 0.0, "loss": 1.0}
