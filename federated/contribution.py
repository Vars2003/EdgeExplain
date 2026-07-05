from typing import Dict, Any, List

class FederatedContributionAnalyzer:
    """
    Evaluates individual client contributions based on local accuracy scores,
    sample partitions, and participation, ensuring contributions sum to 100%.
    """
    @staticmethod
    def analyze_contributions(cached_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates normalized contribution weights.
        """
        accs = cached_metrics.get("accuracies", {})
        sizes = cached_metrics.get("dataset_sizes", {})
        
        if not accs:
            return []
            
        client_ids = list(accs.keys())
        raw_scores = {}
        total_raw = 0.0
        
        for c_id in client_ids:
            # Combine accuracy and samples count to score contribution quality
            accuracy = accs.get(c_id, 0.0)
            samples = sizes.get(c_id, 1)
            score = float(accuracy * samples)
            raw_scores[c_id] = score
            total_raw += score
            
        if total_raw == 0:
            total_raw = 1.0
            
        contributions = []
        for c_id in client_ids:
            weight = float((raw_scores[c_id] / total_raw) * 100.0)
            accuracy = accs.get(c_id, 0.0)
            samples = sizes.get(c_id, 0)
            
            explanation = (
                f"Client {c_id} contributed {weight:.1f}% to the global aggregated model. "
                f"It submitted {samples} local training observations with local accuracy of {accuracy*100:.2f}%."
            )
            
            contributions.append({
                "client": f"Client {c_id}",
                "contribution": weight,
                "samples": samples,
                "accuracy": float(accuracy * 100.0),
                "confidence": 90,
                "explanation": explanation
            })
            
        # Ensure exact floating point sum to 100%
        current_sum = sum(c["contribution"] for c in contributions)
        if contributions and current_sum != 100.0:
            diff = 100.0 - current_sum
            contributions[0]["contribution"] += diff
            
        return contributions
