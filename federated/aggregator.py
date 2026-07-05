from typing import List, Dict, Any

class BaseAggregator:
    """
    Abstract base class for all Federated Learning aggregation strategies.
    Ensures Strategy Pattern compliance across model aggregations.
    """
    def __init__(self):
        self.history = []

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate client weights/updates.
        To be implemented by concrete strategy classes.
        """
        raise NotImplementedError("Aggregators must implement aggregate()")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        """
        Validate updates before running aggregation logic.
        """
        raise NotImplementedError("Aggregators must implement validate()")

    def summarize(self) -> Dict[str, Any]:
        """
        Return summary statistics of the aggregation round.
        """
        raise NotImplementedError("Aggregators must implement summarize()")


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg (Federated Averaging) Aggregator Strategy.
    Calculates weighted parameters based on sample counts.
    (Phase 6.1 infrastructure stub).
    """
    def __init__(self):
        super().__init__()
        self.last_summary = {}

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.validate(client_updates):
            return {"status": "Validation Failed", "global_weights": None}
            
        import numpy as np
        
        # Check if first client update is None or has no weights (i.e. tree model)
        first_weight = client_updates[0].get("weights")
        if first_weight is None or "coef" not in first_weight:
            self.last_summary = {
                "status": "UNSUPPORTED_FOR_FEDAVG",
                "message": "True Federated Averaging is currently implemented only for Logistic Regression. Tree-based models require alternative aggregation strategies and will be supported in future phases."
            }
            return self.last_summary

        total_samples = sum(u.get("samples", 0) for u in client_updates)
        if total_samples == 0:
            return {"status": "Zero Samples Error", "global_weights": None}

        # Weighted aggregation
        global_coef = np.zeros_like(first_weight["coef"], dtype=float)
        global_intercept = np.zeros_like(first_weight["intercept"], dtype=float)

        for u in client_updates:
            w = u.get("weights")
            samples = u.get("samples", 0)
            weight_factor = samples / total_samples
            
            global_coef += weight_factor * w["coef"]
            global_intercept += weight_factor * w["intercept"]

        global_weights = {
            "coef": global_coef,
            "intercept": global_intercept,
            "classes": first_weight["classes"]
        }

        self.last_summary = {
            "strategy": "FedAvg",
            "clients_aggregated": len(client_updates),
            "total_samples": total_samples,
            "status": "SUCCESS"
        }
        self.history.append(self.last_summary)
        
        return {"status": "SUCCESS", "global_weights": global_weights}

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        # Check updates are list and not empty
        return isinstance(client_updates, list) and len(client_updates) > 0

    def summarize(self) -> Dict[str, Any]:
        return self.last_summary or {"status": "No aggregation executed yet"}


class FedProxAggregator(BaseAggregator):
    """
    FedProx Aggregator Strategy. Handles client heterogeneity via proximal terms.
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("FedProx is not implemented in Phase 6.1. Coming in future release.")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()

    def summarize(self) -> Dict[str, Any]:
        return {"status": "FedProx Unavailable"}


class FedNovaAggregator(BaseAggregator):
    """
    FedNova Aggregator Strategy. Corrects for objective shifts in non-uniform local steps.
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("FedNova is not implemented in Phase 6.1. Coming in future release.")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()

    def summarize(self) -> Dict[str, Any]:
        return {"status": "FedNova Unavailable"}


class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD Aggregator Strategy. Uses control variates to correct client drift.
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("SCAFFOLD is not implemented in Phase 6.1. Coming in future release.")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()

    def summarize(self) -> Dict[str, Any]:
        return {"status": "SCAFFOLD Unavailable"}


class FedDynAggregator(BaseAggregator):
    """
    FedDyn Aggregator Strategy. Dynamic regularization aggregation.
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("FedDyn is not implemented in Phase 6.1. Coming in future release.")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()

    def summarize(self) -> Dict[str, Any]:
        return {"status": "FedDyn Unavailable"}


class MOONAggregator(BaseAggregator):
    """
    MOON (Model-Contrastive Federated Learning) Aggregator Strategy.
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("MOON is not implemented in Phase 6.1. Coming in future release.")

    def validate(self, client_updates: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()

    def summarize(self) -> Dict[str, Any]:
        return {"status": "MOON Unavailable"}
