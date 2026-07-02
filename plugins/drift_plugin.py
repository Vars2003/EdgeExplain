import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.drift")

class DatasetDriftPlugin(BasePlugin):
    """
    Plugin implementing local statistical drift detection between two datasets (Dataset A vs Dataset B).
    """

    def __init__(self):
        super().__init__("Dataset Drift Detection", ["pandas", "numpy", "scipy"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        # During single dataset ingestion, we compare the current df with a split or historical baseline.
        # However, to support general comparisons, if st.session_state contains two datasets, we use them.
        # For unit testing/default plugin running, we split the active dataset in half to simulate A and B.
        import streamlit as st
        df_a = df
        df_b = None
        
        # Check if secondary dataset is cached in session state
        try:
            if hasattr(st, "session_state") and st.session_state is not None:
                if st.session_state.get("df_comparison", None) is not None:
                    df_b = st.session_state.df_comparison
        except Exception:
            pass
            
        if df_b is None:
            # Simulate comparison by splitting active dataset in half
            logger.info("No comparison dataset found. Simulating drift check on split halves.")
            split_idx = len(df) // 2
            df_a = df.iloc[:split_idx]
            df_b = df.iloc[split_idx:]
            
        drifted_features = {}
        total_features = 0
        drifted_count = 0
        
        common_cols = list(set(df_a.columns).intersection(set(df_b.columns)))
        
        for col in common_cols:
            col_drifted = False
            p_val = 1.0
            test_method = "None"
            
            # Continuous columns (Kolmogorov-Smirnov Test)
            if pd.api.types.is_numeric_dtype(df_a[col]) and df_a[col].nunique() > 10:
                test_method = "Kolmogorov-Smirnov (KS) Test"
                a_clean = df_a[col].dropna()
                b_clean = df_b[col].dropna()
                
                if len(a_clean) > 0 and len(b_clean) > 0:
                    res = stats.ks_2samp(a_clean, b_clean)
                    p_val = float(res.pvalue)
                    col_drifted = p_val < 0.05
                    
            # Categorical/Discrete columns (proportion comparison or Chi-Square approximation)
            else:
                test_method = "Categorical Ratio Delta Check"
                # Calculate absolute proportion difference of top labels
                a_counts = df_a[col].value_counts(normalize=True).to_dict()
                b_counts = df_b[col].value_counts(normalize=True).to_dict()
                
                max_delta = 0.0
                all_keys = set(a_counts.keys()).union(set(b_counts.keys()))
                for k in all_keys:
                    delta = abs(a_counts.get(k, 0.0) - b_counts.get(k, 0.0))
                    if delta > max_delta:
                        max_delta = delta
                        
                col_drifted = max_delta > 0.15 # Shift > 15%
                p_val = 1.0 - max_delta # Simulated p-value for uniform schema reporting
                
            total_features += 1
            if col_drifted:
                drifted_count += 1
                
            drifted_features[col] = {
                "drift_detected": col_drifted,
                "test_method": test_method,
                "metric_p_value": p_val,
                "null_pct_a": float((df_a[col].isna().sum() / len(df_a)) * 100),
                "null_pct_b": float((df_b[col].isna().sum() / len(df_b)) * 100)
            }
            
        target_candidates = memory_obj.get("context", {}).get("candidate_targets", [])
        target = target_candidates[0] if target_candidates else None
        target_drift_status = "Not Analyzed"
        if target and target in drifted_features:
            target_drift_status = "Drifted" if drifted_features[target]["drift_detected"] else "Stable"
            
        overall_drift_score = (drifted_count / total_features) * 100 if total_features > 0 else 0.0
        drift_level = "Low"
        if overall_drift_score > 50:
            drift_level = "High"
        elif overall_drift_score > 25:
            drift_level = "Medium"
            
        return {
            "result": {
                "overall_drift_score": overall_drift_score,
                "drift_level": drift_level,
                "target_drift": target_drift_status,
                "features": drifted_features
            },
            "confidence": 95,
            "evidence": [f"Compared Dataset A (rows: {len(df_a)}) with Dataset B (rows: {len(df_b)}). Detected {drifted_count} drifted columns out of {total_features}."]
        }
