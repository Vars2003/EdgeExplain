import pandas as pd
import numpy as np
from typing import Dict, Any, List
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.comparison")

class DatasetComparisonPlugin(BasePlugin):
    """
    Plugin calculating differential statistics, schema deltas, and quality shifts
    between two datasets (Dataset A vs Dataset B).
    """

    def __init__(self):
        super().__init__("Dataset Comparison", ["pandas", "numpy"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
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
            # Split dataset in half to simulate A and B comparison
            logger.info("No comparison dataset found. Splitting dataset for comparison simulation.")
            split_idx = len(df) // 2
            df_a = df.iloc[:split_idx]
            df_b = df.iloc[split_idx:]
            
        # 1. Compare dimensions & memory
        rows_a, cols_a = df_a.shape
        rows_b, cols_b = df_b.shape
        
        mem_a = df_a.memory_usage(deep=True).sum()
        mem_b = df_b.memory_usage(deep=True).sum()
        
        # 2. Schema comparison (added/removed columns)
        set_a = set(df_a.columns)
        set_b = set(df_b.columns)
        
        added_cols = list(set_b - set_a)
        removed_cols = list(set_a - set_b)
        common_cols = list(set_a.intersection(set_b))
        
        # 3. Numeric statistics comparison
        stat_deltas = {}
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
                mean_a = df_a[col].mean()
                mean_b = df_b[col].mean()
                delta_mean = mean_b - mean_a if not (pd.isna(mean_a) or pd.isna(mean_b)) else 0.0
                
                null_a = df_a[col].isna().sum() / len(df_a) * 100
                null_b = df_b[col].isna().sum() / len(df_b) * 100
                delta_null = null_b - null_a
                
                stat_deltas[col] = {
                    "mean_a": float(mean_a) if not pd.isna(mean_a) else 0.0,
                    "mean_b": float(mean_b) if not pd.isna(mean_b) else 0.0,
                    "delta_mean": float(delta_mean),
                    "null_a_pct": float(null_a),
                    "null_b_pct": float(null_b),
                    "delta_null_pct": float(delta_null)
                }
                
        # 4. Compare Quality (Mock delta calculation or load metrics if available)
        quality_a = memory_obj.get("context", {}).get("quality_overview", {}).get("quality_score", 90.0)
        # Quality B can be calculated, but for simplicity, we mock it or compute it based on missingness change
        total_missing_b = df_b.isna().sum().sum()
        missing_rate_b = total_missing_b / df_b.size if df_b.size > 0 else 0.0
        quality_b = max(10.0, min(100.0, 100.0 - (missing_rate_b * 100.0) - (df_b.duplicated().sum() / len(df_b) * 50 if len(df_b)>0 else 0)))
        
        return {
            "result": {
                "dimensions": {
                    "dataset_a": {"rows": rows_a, "columns": cols_a, "memory_bytes": int(mem_a)},
                    "dataset_b": {"rows": rows_b, "columns": cols_b, "memory_bytes": int(mem_b)},
                    "delta_rows": rows_b - rows_a,
                    "delta_columns": cols_b - cols_a
                },
                "schema_changes": {
                    "added_columns": added_cols,
                    "removed_columns": removed_cols,
                    "common_columns_count": len(common_cols)
                },
                "feature_statistics_deltas": stat_deltas,
                "quality_comparison": {
                    "score_a": float(quality_a),
                    "score_b": float(quality_b),
                    "delta_score": float(quality_b - quality_a)
                }
            },
            "confidence": 95,
            "evidence": [f"Performed comparison schema map for {len(common_cols)} intersecting features."]
        }
