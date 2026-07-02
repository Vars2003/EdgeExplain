from typing import Dict, Any, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger("intelligence.context_builder")

class ContextBuilder:
    """
    Consolidates structural profiling metrics and business inferences 
    into a queryable DatasetContext schema.
    """

    @staticmethod
    def build_context(df: pd.DataFrame, filename: str, metrics: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]], domain_info: Dict[str, Any], task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assembles all parameters into a unified context structure.
        """
        logger.info(f"Assembling context summary for dataset: {filename}")
        rows, cols = df.shape
        
        # 1. Infer candidate target columns
        target_candidates = []
        for col, info in semantic_types.items():
            col_lower = col.lower()
            if any(t in col_lower for t in ["target", "label", "class", "clicked", "price", "churn", "fraud", "output", "target_col"]):
                target_candidates.append(col)
                
        # 2. Determine recommended ML task based on targets and dataset characteristics
        ml_task = "Unsupervised Analytics (Clustering / Association)"
        if target_candidates:
            primary_target = target_candidates[0]
            target_type = semantic_types.get(primary_target, {}).get("type", "nominal")
            if target_type in ["nominal", "binary", "boolean", "discrete_numerical"]:
                ml_task = "Supervised Classification"
            elif target_type == "continuous_numerical":
                ml_task = "Supervised Regression"
        elif task_info.get("type") == "Time Series":
            ml_task = "Time Series Forecasting"
            
        # 3. Pull quality risks
        key_risks = []
        readiness_checks = metrics.get("ml_readiness", {}).get("checks", {})
        for risk_name, check in readiness_checks.items():
            if check.get("status") in ["WARNING", "ERROR"]:
                key_risks.append({
                    "risk": risk_name.replace("_", " ").title(),
                    "severity": "High" if check.get("score", 100) < 60 else "Medium",
                    "detail": check.get("detail", "")
                })
                
        # Handle outliers risk explicitly
        total_outliers = metrics.get("outliers", {}).get("total_outliers", 0)
        outliers_pct = (total_outliers / df.size) * 100 if df.size > 0 else 0
        if outliers_pct > 2.0:
            key_risks.append({
                "risk": "Outliers Detected",
                "severity": "Medium",
                "detail": f"{total_outliers} cell outliers flagged ({outliers_pct:.2f}% of cells)."
            })
            
        context = {
            "dataset_name": filename,
            "row_count": rows,
            "column_count": cols,
            "memory_readable": metrics.get("basic_metrics", {}).get("memory_readable", "N/A"),
            "domain": domain_info.get("domain", "Generic Tabular").replace(" Dataset", ""),
            "dataset_type": task_info.get("type", "Mixed Dataset"),
            "candidate_targets": target_candidates,
            "recommended_ml_task": ml_task,
            "quality_overview": {
                "quality_score": metrics.get("quality_score", 0.0),
                "ml_readiness_score": metrics.get("ml_readiness", {}).get("score", 0.0),
                "missing_cells_pct": 100.0 - metrics.get("missing_data", {}).get("completeness_pct", 100.0),
                "duplicate_rows_count": metrics.get("basic_metrics", {}).get("duplicate_rows", 0)
            },
            "key_risks": key_risks
        }
        
        logger.info("Successfully constructed dataset context summary.")
        return context
