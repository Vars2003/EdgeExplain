import datetime
from typing import List, Dict, Any

class FederatedExperimentManager:
    """
    Manages historical and runtime records for Federated Learning experiments.
    Stores metadata summaries, snapshots, and client leaderboard results.
    """
    _experiments = []

    @classmethod
    def create_experiment(cls, dataset_name: str, algorithm: str, aggregator: str, rounds: int, clients: int, model: str) -> str:
        """
        Generates a unique Experiment ID and registers a new experiment session.
        Example ID: EXP_20260705_001
        """
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        
        # Count experiments started today
        today_count = sum(1 for exp in cls._experiments if exp["id"].startswith(f"EXP_{date_str}"))
        exp_id = f"EXP_{date_str}_{today_count + 1:03d}"
        
        exp = {
            "id": exp_id,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": dataset_name,
            "algorithm": algorithm,
            "aggregator": aggregator,
            "rounds": rounds,
            "clients": clients,
            "model": model,
            "duration": 0.0,
            "final_accuracy": 0.0,
            "final_loss": 1.0,
            "snapshots": [],
            "leaderboard": []
        }
        cls._experiments.append(exp)
        return exp_id

    @classmethod
    def update_experiment(cls, exp_id: str, duration: float, final_accuracy: float, final_loss: float, 
                          snapshots: List[Dict[str, Any]], leaderboard: List[Dict[str, Any]]) -> None:
        """
        Saves final metrics, timelines, and round snapshots for a completed experiment.
        """
        for exp in cls._experiments:
            if exp["id"] == exp_id:
                exp["duration"] = duration
                exp["final_accuracy"] = final_accuracy
                exp["final_loss"] = final_loss
                exp["snapshots"] = snapshots
                exp["leaderboard"] = leaderboard
                break

    @classmethod
    def get_experiment(cls, exp_id: str) -> Dict[str, Any]:
        """
        Retrieves a registered experiment by ID.
        """
        for exp in cls._experiments:
            if exp["id"] == exp_id:
                return exp
        return None

    @classmethod
    def get_all_experiments(cls) -> List[Dict[str, Any]]:
        """
        Returns all registered experiments.
        """
        return cls._experiments
