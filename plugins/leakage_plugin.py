import pandas as pd
import numpy as np
from typing import Dict, Any, List
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.leakage")

class LeakageDetectionPlugin(BasePlugin):
    """
    Plugin scanning features for information leakage risks (e.g. suspicious target correlation, future dates).
    """

    def __init__(self):
        super().__init__("Leakage Detection", ["pandas", "numpy"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        target_candidates = memory_obj.get("context", {}).get("candidate_targets", [])
        target = target_candidates[0] if target_candidates else df.columns[-1]
        
        leakage_alerts = []
        overall_risk = "Low"
        
        # 1. Suspicious target correlations (>0.95)
        dep_data = memory_obj.get("metrics", {}).get("dependency", {})
        if dep_data:
            cols = dep_data.get("columns", [])
            matrix = dep_data.get("matrix", [])
            
            if target in cols:
                t_idx = cols.index(target)
                for j, col in enumerate(cols):
                    if t_idx != j:
                        weight = matrix[t_idx][j]
                        if weight > 0.95:
                            leakage_alerts.append({
                                "feature": col,
                                "type": "Target Leakage",
                                "severity": "High",
                                "detail": f"Feature '{col}' exhibits extremely high association score ({weight:.4f}) with target label '{target}'. This feature likely contains direct downstream target information."
                            })
                            overall_risk = "High"
                            
        # 2. High Cardinality ID risk
        for col in df.columns:
            if col != target:
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio > 0.98 and not pd.api.types.is_numeric_dtype(df[col]):
                    # If it's a string/object and highly unique, it's an ID
                    leakage_alerts.append({
                        "feature": col,
                        "type": "ID Leakage / Overfitting Risk",
                        "severity": "Medium",
                        "detail": f"Feature '{col}' has a unique ratio of {unique_ratio*100:.1f}%. It acts as an identifier and must be removed to prevent model memorization."
                    })
                    if overall_risk != "High":
                        overall_risk = "Medium"
                        
        # 3. Future Leakage (dates/times containing future indicators)
        date_keywords = ["after", "post", "future", "result", "outcome", "date_completed"]
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in date_keywords):
                leakage_alerts.append({
                    "feature": col,
                    "type": "Temporal Future Leakage",
                    "severity": "Medium",
                    "detail": f"Feature '{col}' contains terminology associated with future events. Models using this feature may suffer from lookahead bias."
                })
                if overall_risk != "High":
                    overall_risk = "Medium"
                    
        return {
            "result": {
                "overall_risk_level": overall_risk,
                "alerts": leakage_alerts
            },
            "confidence": 90,
            "evidence": [f"Scanned {df.shape[1]} columns. Flagged {len(leakage_alerts)} suspicious features against target column '{target}'."]
        }
