from typing import Dict, Any, List

class FederatedRecommendationEngine:
    """
    Formulates structured optimization recommendations based on 
    communication efficiency, local accuracies, and imbalances.
    """
    @staticmethod
    def generate_recommendations(cached_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs diagnostic rules and generates recommendations.
        """
        hist = cached_metrics.get("convergence_history", [])
        accs = [h.get("accuracy", 0.0) for h in hist]
        
        recs = []
        
        # Rule 1: Increase Rounds if accuracy has not converged
        if len(hist) < 10 and accs and accs[-1] < 0.90:
            recs.append({
                "title": "Increase Communication Rounds",
                "reason": "Current validation accuracy is below 90% and training run was limited to few rounds.",
                "expected_impact": "Allows local parameters more cycles to stabilize and reach optimal bounds.",
                "confidence": 85
            })
            
        # Rule 2: Increase Local Epochs if training converges slowly
        if len(hist) > 1 and accs[-1] - accs[0] < 0.05:
            recs.append({
                "title": "Increase Local Optimization Epochs (E)",
                "reason": "Slow convergence rate observed across training cycles.",
                "expected_impact": "Accelerates weight convergence at client nodes before server aggregation.",
                "confidence": 80
            })
            
        # Rule 3: Balance Client dataset partition weights
        sizes = list(cached_metrics.get("dataset_sizes", {}).values())
        if sizes and (max(sizes) - min(sizes)) > 50:
            recs.append({
                "title": "Balance Dataset Partitions across Clients",
                "reason": "High variance in local client training sample counts detected.",
                "expected_impact": "Prevents client parameter weights from biasing the global averaged model.",
                "confidence": 93
            })
            
        # Default fallback recommendation
        if not recs:
            recs.append({
                "title": "Maintain Current Training Bounds",
                "reason": "Global accuracy and client distribution balances are within optimal targets.",
                "expected_impact": "Preserves stable, balanced collaborative convergence rates.",
                "confidence": 95
            })
            
        return recs
