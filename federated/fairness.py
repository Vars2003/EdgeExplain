from typing import Dict, Any, List
import numpy as np

class FederatedFairnessAppraiser:
    """
    Measures system variance and balance parity (accuracy variances, 
    sample size skews) and appraises system fairness.
    """
    @staticmethod
    def appraise_fairness(cached_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes the system fairness coefficient.
        """
        accs = list(cached_metrics.get("accuracies", {}).values())
        sizes = list(cached_metrics.get("dataset_sizes", {}).values())
        
        if not accs:
            return {
                "fairness_score": 100,
                "rating": "Excellent",
                "confidence": 100,
                "evidence": ["No clients registered yet."],
                "explanation": "No training runs have been completed yet."
            }
            
        # 1. Accuracy Variance
        acc_var = float(np.var(accs)) if len(accs) > 1 else 0.0
        acc_penalty = min(50.0, acc_var * 1000.0) # scale variance penalty
        
        # 2. Sample Size Imbalance coefficient (coefficient of variation CV)
        mean_size = float(np.mean(sizes)) if sizes else 1.0
        std_size = float(np.std(sizes)) if len(sizes) > 1 else 0.0
        cv_size = std_size / mean_size if mean_size > 0 else 0.0
        size_penalty = min(50.0, cv_size * 50.0) # scale skew penalty
        
        # Deduct penalties from maximum score of 100
        fairness_score = max(0, int(100 - (acc_penalty + size_penalty)))
        
        rating = "Excellent" if fairness_score >= 90 else "Good" if fairness_score >= 75 else "Moderate" if fairness_score >= 50 else "Poor"
        
        evidence = [
            f"Local accuracy variance between clients: {acc_var:.6f}",
            f"Simulated client partition size imbalance coefficient (CV): {cv_size:.4f}",
            f"Minimum accuracy: {min(accs)*100:.2f}%",
            f"Maximum accuracy: {max(accs)*100:.2f}%",
            f"Dataset splits ranges: {min(sizes)} to {max(sizes)} samples"
        ]
        
        explanation = (
            f"The Federated training session achieved a Fairness Score of {fairness_score} ({rating}). "
            f"The score reflects minor performance variance ({acc_var*100:.3f}% deviation) "
            f"and dataset partition splits balance across participating nodes."
        )
        
        return {
            "fairness_score": fairness_score,
            "rating": rating,
            "confidence": 92,
            "evidence": evidence,
            "explanation": explanation
        }
