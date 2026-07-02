import pandas as pd
import numpy as np
from typing import Dict, Any, List
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.bias")

class BiasAnalysisPlugin(BasePlugin):
    """
    Plugin analyzing dataset balance, sampling bias, and class imbalances.
    """

    def __init__(self):
        super().__init__("Bias & Balance Analysis", ["pandas", "numpy"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        target_candidates = memory_obj.get("context", {}).get("candidate_targets", [])
        target = target_candidates[0] if target_candidates else df.columns[-1]
        
        bias_alerts = []
        recommendations = []
        
        # 1. Class Imbalance (Target Column)
        if target in df.columns:
            counts = df[target].value_counts(normalize=True).to_dict()
            if len(counts) > 1 and len(counts) <= 10: # Classification targets
                min_ratio = min(counts.values())
                max_ratio = max(counts.values())
                imbalance_ratio = min_ratio / max_ratio
                
                if imbalance_ratio < 0.20:
                    bias_alerts.append({
                        "category": "Target Class Imbalance",
                        "severity": "High",
                        "detail": f"Target column '{target}' is highly imbalanced. The minority class accounts for only {min_ratio*100:.2f}% of samples (ratio: {imbalance_ratio:.2f})."
                    })
                    recommendations.append("Apply SMOTE minority oversampling or class-weight cost-sensitive training.")
                elif imbalance_ratio < 0.50:
                    bias_alerts.append({
                        "category": "Target Class Imbalance",
                        "severity": "Medium",
                        "detail": f"Target column '{target}' is moderately imbalanced. Minority class ratio is {imbalance_ratio:.2f}."
                    })
                    recommendations.append("Use stratified cross-validation folds and monitor F1-Score instead of accuracy.")
            elif len(counts) > 10 and pd.api.types.is_numeric_dtype(df[target]): # Regression Target Skew
                skew = df[target].skew()
                if abs(skew) > 1.5:
                    bias_alerts.append({
                        "category": "Target Distribution Skew",
                        "severity": "Medium",
                        "detail": f"Continuous target '{target}' exhibits high skewness ({skew:.2f}). Large values may bias model fits."
                    })
                    recommendations.append("Apply logarithmic or Box-Cox power transformation to the target variable.")

        # 2. Categorical Imbalance (Nominal columns)
        semantic_types = memory_obj.get("schema", {})
        for col, info in semantic_types.items():
            if col != target and info["type"] in ["nominal", "binary"]:
                counts = df[col].value_counts(normalize=True).to_dict()
                if counts:
                    max_key = max(counts, key=counts.get)
                    max_ratio = counts[max_key]
                    if max_ratio > 0.90:
                        bias_alerts.append({
                            "category": "Feature Category Imbalance",
                            "severity": "Medium",
                            "detail": f"Column '{col}' is heavily dominated by category '{max_key}' ({max_ratio*100:.1f}% of observations)."
                        })
                        recommendations.append(f"Consider dropping feature '{col}' or merging minority categories to reduce zero-variance segments.")
                        
        # 3. Default recommendation if all balanced
        if not recommendations:
            recommendations.append("No active sampling adjustments needed. Maintain standard randomized cross-validation.")
            
        return {
            "result": {
                "alerts": bias_alerts,
                "recommendations": recommendations
            },
            "confidence": 92,
            "evidence": [f"Evaluated target column '{target}' and nominal features. Found {len(bias_alerts)} imbalance factors."]
        }
