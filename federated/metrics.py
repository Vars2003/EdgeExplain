from typing import List, Dict, Any

class FederatedMetrics:
    """
    Logs local client performance indicators (accuracies, loss parameters, sample counts, duration ms)
    and global aggregation summaries.
    """
    def __init__(self):
        self.num_clients = 0
        self.active_clients = 0
        self.dataset_sizes = {}
        self.round_number = 0
        self.local_accuracies = {}
        self.local_losses = {}
        self.training_times_ms = {}
        
        # Phase 6.2 metrics
        self.global_accuracy = 0.0
        self.global_loss = 0.0
        self.aggregation_time_ms = 0.0
        self.communication_cost_bytes = 0
        self.communication_latency_ms = 0.0
        self.total_training_time_ms = 0.0
        
        self.communication_history = []
        self.convergence_history = []
        self.global_model_history = []
        self.training_timeline = []
        self.history = []

    def log_round(self, round_num: int, client_updates: List[Dict[str, Any]], 
                  global_acc: float = 0.0, global_loss: float = 0.0,
                  agg_time_ms: float = 0.0, comm_bytes: int = 0, 
                  comm_latency_ms: float = 0.0, round_time_ms: float = 0.0) -> None:
        """
        Records details for a single completed round.
        """
        self.round_number = round_num
        self.active_clients = len(client_updates)
        
        self.global_accuracy = global_acc
        self.global_loss = global_loss
        self.aggregation_time_ms = agg_time_ms
        self.communication_cost_bytes += comm_bytes
        self.communication_latency_ms += comm_latency_ms
        self.total_training_time_ms += round_time_ms
        
        accs = {}
        losses = {}
        times = {}
        
        for u in client_updates:
            c_id = u.get("client_id")
            accs[c_id] = u.get("accuracy", 0.0)
            losses[c_id] = u.get("loss", 0.0)
            times[c_id] = u.get("training_time_ms", 0.0)
            self.dataset_sizes[c_id] = u.get("samples", 0)
            
        self.local_accuracies = accs
        self.local_losses = losses
        self.training_times_ms = times
        
        # Track communication history
        self.communication_history.append({
            "round": round_num,
            "bytes": comm_bytes,
            "latency_ms": comm_latency_ms
        })
        
        # Track round-wise convergence history
        self.convergence_history.append({
            "round": round_num,
            "accuracy": global_acc,
            "loss": global_loss,
            "bytes": comm_bytes,
            "latency_ms": comm_latency_ms,
            "aggregation_time_ms": agg_time_ms,
            "training_time_ms": sum(times.values()) / len(times) if times else 0.0
        })

        import datetime
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        
        self.global_model_history.append({
            "round": round_num,
            "version": f"global_v{round_num}",
            "accuracy": global_acc,
            "loss": global_loss,
            "communication_cost": comm_bytes,
            "aggregation_time_ms": agg_time_ms,
            "training_time_ms": round_time_ms,
            "participating_clients": self.active_clients,
            "aggregator": "FedAvg",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        })
        
        self.training_timeline.append({
            "time": time_str,
            "event": "Round Started",
            "round": round_num
        })
        for u in client_updates:
            self.training_timeline.append({
                "time": time_str,
                "event": "Client Training Completed",
                "client": u.get("client_id")
            })
        self.training_timeline.append({
            "time": time_str,
            "event": "FedAvg Aggregation Finished",
            "round": round_num
        })
        self.training_timeline.append({
            "time": time_str,
            "event": "Global Model Updated",
            "version": f"global_v{round_num}"
        })
        
        self.history.append({
            "round": round_num,
            "active_clients": self.active_clients,
            "accuracies": accs,
            "losses": losses,
            "training_times": times,
            "global_accuracy": global_acc,
            "global_loss": global_loss
        })

    def export_summary(self) -> Dict[str, Any]:
        """
        Returns a dictionary summary of current metrics.
        """
        return {
            "round_number": self.round_number,
            "num_clients": self.num_clients,
            "active_clients": self.active_clients,
            "dataset_sizes": self.dataset_sizes,
            "accuracies": self.local_accuracies,
            "losses": self.local_losses,
            "training_times_ms": self.training_times_ms,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
            "aggregation_time_ms": self.aggregation_time_ms,
            "communication_cost_bytes": self.communication_cost_bytes,
            "communication_latency_ms": self.communication_latency_ms,
            "total_training_time_ms": self.total_training_time_ms,
            "communication_history": self.communication_history,
            "convergence_history": self.convergence_history,
            "global_model_history": self.global_model_history,
            "training_timeline": self.training_timeline
        }
