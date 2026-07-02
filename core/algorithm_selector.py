from typing import List, Dict, Any
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("core.algorithm_selector")

class AlgorithmSelector:
    """
    Evaluates dataset profiles and recommends offline machine learning algorithms
    using heuristics based on dimensionality, missingness, scales, and target types.
    """

    @staticmethod
    def recommend_algorithms(df: pd.DataFrame, dataset_type_info: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Computes compatibility ratings for ML algorithms based on metadata metrics.
        """
        logger.info("Running machine learning algorithm selector.")
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return []
            
        task_type = dataset_type_info.get("type", "Mixed Dataset")
        
        # If task_type is not Classification or Regression, look for target columns to classify
        is_classification = "classification" in task_type.lower() or task_type == "Classification"
        is_regression = "regression" in task_type.lower() or task_type == "Regression"
        
        # Fallback target estimation
        if not is_classification and not is_regression:
            # Let's see if we have binary/nominal targets or numeric targets
            target_candidates = [col for col in df.columns if any(t in col.lower() for t in ["target", "label", "class", "price", "churn", "fraud"])]
            if target_candidates:
                t_col = target_candidates[0]
                t_type = semantic_types.get(t_col, {}).get("type", "nominal")
                if t_type in ["nominal", "binary", "boolean"]:
                    is_classification = True
                else:
                    is_regression = True
            else:
                # Default to Classification for mixed datasets
                is_classification = True
                
        # Gather characteristics
        missing_pct = metrics.get("missing_data", {}).get("completeness_pct", 100.0)
        missing_count = metrics.get("missing_data", {}).get("total_missing", 0)
        has_missing = missing_count > 0
        
        cat_count = sum(1 for col, info in semantic_types.items() if info["type"] in ["nominal", "ordinal", "binary"])
        has_categorical = cat_count > 0
        
        outliers_count = metrics.get("outliers", {}).get("total_outliers", 0)
        has_outliers = outliers_count > 0
        
        recs = []
        
        if is_classification:
            # 1. Random Forest Classifier
            rf_score = 85
            rf_reasons = ["Handles mixed numerical and categorical features extremely well."]
            if has_outliers:
                rf_score += 5
                rf_reasons.append("Robust to numeric outliers since it relies on tree splits.")
            if has_missing:
                rf_score += 5
                rf_reasons.append("Can handle missing values implicitly via tree structures.")
            if rows < 100:
                rf_score -= 10
                rf_reasons.append("Small sample size may lead to tree variance issues.")
            rf_score = min(98, max(40, rf_score))
            recs.append({
                "algorithm": "Random Forest Classifier",
                "compatibility_score": rf_score,
                "pros": ["Handles mixed data types", "Robust to outliers", "Implicitly handles missing values", "Low risk of overfitting"],
                "cons": ["Slow to predict on high tree counts", "High memory footprint on large datasets"],
                "reasoning": " ".join(rf_reasons)
            })
            
            # 2. XGBoost Classifier
            xgb_score = 80
            xgb_reasons = ["State-of-the-art performance on tabular datasets."]
            if has_missing:
                xgb_score += 10
                xgb_reasons.append("Handles missing values natively during split calculations.")
            if rows < 500:
                xgb_score -= 15
                xgb_reasons.append("High risk of overfitting on small training samples (<500 rows).")
            if cols > 100:
                xgb_score += 5
                xgb_reasons.append("Highly efficient with wide datasets (column subsampling).")
            xgb_score = min(98, max(30, xgb_score))
            recs.append({
                "algorithm": "XGBoost Classifier",
                "compatibility_score": xgb_score,
                "pros": ["State-of-the-art accuracy", "Natively handles missing values", "Regularization controls overfitting"],
                "cons": ["Complex hyperparameter tuning", "Black-box complexity"],
                "reasoning": " ".join(xgb_reasons)
            })
            
            # 3. Logistic Regression
            lr_score = 75
            lr_reasons = ["Extremely fast training and low resource requirement."]
            if has_missing:
                lr_score -= 25
                lr_reasons.append("Cannot handle missing values natively. Requires full imputation first.")
            if has_categorical:
                lr_score -= 10
                lr_reasons.append("Requires one-hot encoding for categorical variables which expands dimensions.")
            if has_outliers:
                lr_score -= 10
                lr_reasons.append("Linear coefficients are highly sensitive to numeric outliers.")
            lr_score = min(95, max(20, lr_score))
            recs.append({
                "algorithm": "Logistic Regression",
                "compatibility_score": lr_score,
                "pros": ["Highly interpretable", "Low computation footprint", "Outputs class probabilities"],
                "cons": ["Assumes linear boundaries", "Sensitive to outliers and multicollinearity"],
                "reasoning": " ".join(lr_reasons)
            })
            
            # 4. Decision Tree Classifier
            dt_score = 70
            dt_reasons = ["Offers direct visual rules for business users."]
            if rows < 200:
                dt_score += 10
                dt_reasons.append("Good baseline for small, quick classifications.")
            if rows > 20000:
                dt_score -= 10
                dt_reasons.append("A single tree will overfit significantly on large data volumes.")
            dt_score = min(90, max(40, dt_score))
            recs.append({
                "algorithm": "Decision Tree Classifier",
                "compatibility_score": dt_score,
                "pros": ["Extremely interpretable", "No scaling required", "Fast training times"],
                "cons": ["Highly unstable to small data shifts", "Prone to high overfitting"],
                "reasoning": " ".join(dt_reasons)
            })
            
        elif is_regression:
            # 1. Random Forest Regressor
            rf_score = 85
            rf_reasons = ["Superb default model for non-linear target distributions."]
            if has_outliers:
                rf_score += 5
                rf_reasons.append("Robust to outliers in feature values.")
            if rows < 100:
                rf_score -= 10
            rf_score = min(98, max(40, rf_score))
            recs.append({
                "algorithm": "Random Forest Regressor",
                "compatibility_score": rf_score,
                "pros": ["Handles non-linear patterns", "Robust to features outliers", "Low parameter sensitivity"],
                "cons": ["Cannot extrapolate beyond training target ranges", "Slow inference on big trees"],
                "reasoning": " ".join(rf_reasons)
            })
            
            # 2. XGBoost Regressor
            xgb_score = 80
            xgb_reasons = ["Excellent prediction accuracy on continuous tabular predictions."]
            if has_missing:
                xgb_score += 10
            if rows < 500:
                xgb_score -= 15
            xgb_score = min(98, max(30, xgb_score))
            recs.append({
                "algorithm": "XGBoost Regressor",
                "compatibility_score": xgb_score,
                "pros": ["Excellent accuracy", "Natively supports missing features", "Fast training speed"],
                "cons": ["Black-box model", "Requires parameter tuning"],
                "reasoning": " ".join(xgb_reasons)
            })
            
            # 3. Linear Regression (OLS)
            lr_score = 70
            lr_reasons = ["Simple baseline model with direct coefficient weights."]
            if has_missing:
                lr_score -= 25
                lr_reasons.append("Requires full missing value imputation before training.")
            if has_outliers:
                lr_score -= 15
                lr_reasons.append("OLS optimization is heavily biased by outlier residuals.")
            lr_score = min(95, max(15, lr_score))
            recs.append({
                "algorithm": "Linear Regression (OLS)",
                "compatibility_score": lr_score,
                "pros": ["Instant calculations", "Perfect transparency and coefficient values"],
                "cons": ["Cannot model non-linear boundaries", "Highly sensitive to multicollinearity"],
                "reasoning": " ".join(lr_reasons)
            })
            
        # Sort recommendations by compatibility score (descending)
        recs.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return recs
