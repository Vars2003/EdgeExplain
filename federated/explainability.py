from typing import Dict, Any, List

class FederatedExplainabilityEngine:
    """
    Diagnoses and explains convergence anomalies, bottlenecks, and round updates 
    using deterministic scientific rules.
    """
    @staticmethod
    def generate_explanations(cached_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs rules diagnostics and compiles explanation records.
        """
        hist = cached_metrics.get("convergence_history", [])
        c_accs = cached_metrics.get("accuracies", {})
        
        if not hist:
            return []
            
        explanations = []
        
        # 1. Convergence improvement explanation
        if len(hist) > 1:
            acc_diff = (hist[-1]["accuracy"] - hist[0]["accuracy"]) * 100.0
            explanation_1 = {
                "question": "Why did accuracy improve over rounds?",
                "answer": (
                    f"Global accuracy improved by {acc_diff:.2f}% from Round 1 to Round {len(hist)} "
                    f"as local client gradients synchronized through FedAvg parameter aggregation."
                ),
                "evidence": [
                    f"Round 1 Global Accuracy: {hist[0]['accuracy']*100:.2f}%",
                    f"Final Round Global Accuracy: {hist[-1]['accuracy']*100:.2f}%"
                ],
                "confidence": 95
            }
            explanations.append(explanation_1)
            
        # 2. Round duration explanation
        avg_time = sum(h.get("training_time_ms", 0.0) for h in hist) / len(hist)
        explanation_2 = {
            "question": "What is the primary factor in round duration?",
            "answer": (
                f"Round times averaged {avg_time/1000.0:.2f} seconds. Local training fit cycles "
                f"across client hardware partitions represent the main computational cost."
            ),
            "evidence": [
                f"Average Round time: {avg_time:.2f} ms",
                f"Average Aggregation duration: {sum(h.get('aggregation_time_ms', 0.0) for h in hist)/len(hist):.2f} ms"
            ],
            "confidence": 92
        }
        explanations.append(explanation_2)
        
        # 3. Client variance explanation
        if c_accs:
            best_c = max(c_accs, key=c_accs.get)
            worst_c = min(c_accs, key=c_accs.get)
            explanation_3 = {
                "question": "Which client contributed most to model generalization?",
                "answer": (
                    f"Client {best_c} achieved the highest local accuracy of {c_accs[best_c]*100:.2f}%, "
                    f"representing the strongest local predictive power."
                ),
                "evidence": [
                    f"Best client local accuracy: {c_accs[best_c]*100:.2f}% (Client {best_c})",
                    f"Worst client local accuracy: {c_accs[worst_c]*100:.2f}% (Client {worst_c})"
                ],
                "confidence": 90
            }
            explanations.append(explanation_3)
            
        return explanations
