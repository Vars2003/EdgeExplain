from typing import Dict, Any, List
import numpy as np

class FederatedAnalyticsEngine:
    """
    Computes global performance statistics, aggregation rates, 
    and returns standardized scientific evaluation summaries.
    """
    @staticmethod
    def analyze_metrics(cached_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs analytics over convergence history records.
        """
        hist = cached_metrics.get("convergence_history", [])
        if not hist:
            return {
                "score": 0.0,
                "rating": "N/A",
                "confidence": 100,
                "evidence": ["No history records available."],
                "explanation": "No training runs have been completed yet."
            }
            
        accs = [h.get("accuracy", 0.0) for h in hist]
        losses = [h.get("loss", 0.0) for h in hist]
        times = [h.get("training_time_ms", 0.0) for h in hist]
        agg_times = [h.get("aggregation_time_ms", 0.0) for h in hist]
        bytes_list = [h.get("bytes", 0) for h in hist]
        
        best_round_idx = int(np.argmax(accs) + 1)
        worst_round_idx = int(np.argmin(accs) + 1)
        
        avg_acc = float(np.mean(accs))
        avg_loss = float(np.mean(losses))
        avg_time = float(np.mean(times))
        avg_agg = float(np.mean(agg_times))
        
        # Calculate communication efficiency (accuracy improvement per MB transferred)
        total_bytes = sum(bytes_list)
        total_bytes_mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 1.0
        acc_gain = accs[-1] - accs[0]
        comm_efficiency = float(acc_gain / total_bytes_mb)
        
        # Performance rating logic
        score = int(accs[-1] * 100)
        rating = "Excellent" if score >= 90 else "Good" if score >= 75 else "Moderate" if score >= 50 else "Poor"
        
        evidence = [
            f"Best validation accuracy achieved in Round {best_round_idx}: {accs[best_round_idx-1]*100:.2f}%",
            f"Worst validation accuracy occurred in Round {worst_round_idx}: {accs[worst_round_idx-1]*100:.2f}%",
            f"Average Round time was {avg_time/1000.0:.2f} seconds",
            f"Average Aggregation duration was {avg_agg:.2f} ms",
            f"Total bytes transferred: {total_bytes:,} bytes",
            f"Communication efficiency score: {comm_efficiency:.4f} accuracy gain per MB"
        ]
        
        explanation = (
            f"The global federated model converged to a final validation accuracy of {accs[-1]*100:.2f}% "
            f"and validation loss of {losses[-1]:.4f} over {len(hist)} communication rounds. "
            f"Model convergence peak occurred at Round {best_round_idx}."
        )
        
        return {
            "score": score,
            "rating": rating,
            "confidence": 95, # high confidence in deterministic counts
            "evidence": evidence,
            "explanation": explanation,
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "best_round": best_round_idx,
            "worst_round": worst_round_idx,
            "comm_efficiency": comm_efficiency,
            "avg_round_time_ms": avg_time,
            "avg_agg_time_ms": avg_agg
        }
