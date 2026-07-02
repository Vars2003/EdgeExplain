import pandas as pd
import numpy as np
from typing import Dict, Any, List
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.automl")

class AutoMLRecommendationPlugin(BasePlugin):
    """
    Plugin building complete ML pipeline recommendations for 9 core algorithms.
    """

    def __init__(self):
        super().__init__("AutoML Recommendations", ["pandas", "numpy"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        context = memory_obj.get("context", {})
        metrics = memory_obj.get("metrics", {})
        semantic_types = memory_obj.get("schema", {})
        
        rows, cols = df.shape
        missing_pct = context.get("quality_overview", {}).get("missing_cells_pct", 0.0)
        is_regression = "regression" in context.get("recommended_ml_task", "").lower()
        
        # 1. Determine universal pipeline recommendations based on dataset
        impute_strategy = "No imputation required (dataset complete)."
        impute_reason = "No null cells found."
        if missing_pct > 0:
            impute_strategy = "Median imputation for numerical variables, Mode imputation for categoricals."
            impute_reason = "Protects data from mean-biases in skewed distributions."
            
        enc_cols = [col for col, info in semantic_types.items() if info["type"] in ["nominal", "binary"]]
        encoding = "No encoding needed (all numeric features)."
        encoding_reason = "Dataset contains zero nominal/binary columns."
        if enc_cols:
            encoding = "One-Hot Encoding for cardinality < 10, Target Encoding otherwise."
            encoding_reason = f"Resolves categories in columns {enc_cols[:3]} without creating sparse dimensions."
            
        outliers_flag = metrics.get("outliers", {}).get("total_outliers", 0) > 0
        scaling = "StandardScaler (Z-Score scaling)."
        scaling_reason = "Standardizes continuous variables for uniform boundaries."
        if outliers_flag:
            scaling = "RobustScaler (IQR-based scaling)."
            scaling_reason = "Prevents outlier extremes from compressing the main feature variance."
            
        selection = "Correlation threshold selector (remove features with correlation coefficient > 0.90)."
        selection_reason = "Reduces multicollinearity and shrinks redundant variances."
        
        validation = "Stratified 5-Fold Cross-Validation" if not is_regression else "K-Fold Cross-Validation (K=5)"
        validation_reason = "Provides stable offline evaluation across folds."
        if rows < 1000:
            validation = "Stratified 10-Fold Cross-Validation"
            validation_reason = "Maximizes training samples in small data pools."
        elif rows > 20000:
            validation = "Holdout Split (80% Train, 20% Test)"
            validation_reason = "Saves computation time on large record counts."
            
        metric = "ROC AUC (for classification)" if not is_regression else "R-Squared (R2)"
        metric_reason = "Standard performance metric for this task type."
        
        # 2. Score suitability for the 9 algorithms
        algorithms_list = [
            "Random Forest", "XGBoost", "LightGBM", "CatBoost", 
            "SVM", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes"
        ]
        
        recommendations = []
        for algo in algorithms_list:
            score = 75 # Default base
            pros = []
            cons = []
            algo_scaling = scaling
            algo_scale_reason = scaling_reason
            
            if algo == "Random Forest":
                score = 95 if rows < 15000 else 85
                pros = ["Handles non-linear relationships natively", "Robust to outliers and collinearity"]
                cons = ["Slower inference on large depth trees"]
                algo_scaling = "None (Tree-based model)"
                algo_scale_reason = "Tree splits are scaling invariant."
            elif algo == "XGBoost":
                score = 98 if rows > 2000 else 80
                pros = ["High prediction accuracy via gradient boosting", "Internal handling of missing values"]
                cons = ["Higher hyperparameter tuning complexity"]
                algo_scaling = "None (Tree-based model)"
                algo_scale_reason = "Tree splits are scaling invariant."
            elif algo == "LightGBM":
                score = 96 if rows > 10000 else 70
                pros = ["Fast training speed and low memory usage", "Leaf-wise tree growth"]
                cons = ["Overfits easily on small datasets (<1000 rows)"]
                algo_scaling = "None (Tree-based model)"
                algo_scale_reason = "Tree splits are scaling invariant."
            elif algo == "CatBoost":
                score = 97 if enc_cols else 80
                pros = ["Handles categorical features out-of-the-box", "Prevents overfitting via symmetric trees"]
                cons = ["Slow training compared to LightGBM"]
                algo_scaling = "None (Tree-based model)"
                algo_scale_reason = "Tree splits are scaling invariant."
            elif algo == "SVM":
                score = 80 if rows < 5000 else 60
                pros = ["Effective in high-dimensional spaces", "Robust kernel tricks"]
                cons = ["Inference scales cubically with row count"]
            elif algo == "Logistic Regression":
                score = 85 if not is_regression else 0 # (Becomes Linear Regression for continuous target)
                if is_regression:
                    algo = "Linear Regression"
                    score = 85
                pros = ["Extremely interpretable coefficients", "Very fast training"]
                cons = ["Underfits complex non-linear patterns"]
            elif algo == "Decision Tree":
                score = 78
                pros = ["High transparency and explainability", "Requires zero scaling"]
                cons = ["Prone to high variance and overfitting"]
                algo_scaling = "None (Tree-based model)"
                algo_scale_reason = "Tree splits are scaling invariant."
            elif algo == "KNN":
                score = 70 if rows < 10000 else 50
                pros = ["Simple distance-based lazy learner", "No training phase"]
                cons = ["High memory cost for inference on large datasets"]
            elif algo == "Naive Bayes":
                score = 65 if not is_regression else 30
                pros = ["Extremely fast classification", "Good baseline for text categorical frequencies"]
                cons = ["Assumes independent feature relationships (rarely holds)"]

            recommendations.append({
                "algorithm": algo,
                "suitability_score": score,
                "pipeline": {
                    "imputation": impute_strategy,
                    "imputation_reason": impute_reason,
                    "encoding": encoding,
                    "encoding_reason": encoding_reason,
                    "scaling": algo_scaling,
                    "scaling_reason": algo_scale_reason,
                    "feature_selection": selection,
                    "feature_selection_reason": selection_reason,
                    "validation_strategy": validation,
                    "validation_strategy_reason": validation_reason,
                    "evaluation_metric": metric,
                    "evaluation_metric_reason": metric_reason
                },
                "pros": pros,
                "cons": cons
            })
            
        # Sort by score descending
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return {
            "result": {
                "automl_pipelines": recommendations
            },
            "confidence": 95,
            "evidence": [f"Ranked 9 algorithms based on {rows} observations and {len(enc_cols)} categorical features."]
        }
