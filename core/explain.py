import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

logger = get_logger("core.explain")

class ExplainabilityEngine:
    """
    Core interface for Explainable AI (XAI). Provides live decision tree-based
    feature importance and structure outlines for future SHAP/LIME frameworks.
    """

    @staticmethod
    def calculate_feature_importance(df: pd.DataFrame, target_col: str, semantic_types: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Fits a local Decision Tree to extract instant feature importance rankings offline.
        """
        logger.info(f"Calculating feature importances targeting: {target_col}")
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping feature importance.")
            return {}
            
        # 1. Clean data for decision tree fitting
        df_clean = df.copy()
        
        # Drop rows with null target
        df_clean = df_clean.dropna(subset=[target_col])
        if len(df_clean) < 10:
            logger.warning("Insufficient rows for feature importance calculation.")
            return {}
            
        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])
        
        # Remove datetime, identifier, or text features
        cols_to_drop = []
        for col in X.columns:
            t = semantic_types.get(col, {}).get("type", "nominal")
            if t in ["datetime", "identifier", "text"]:
                cols_to_drop.append(col)
        X = X.drop(columns=cols_to_drop)
        
        if X.empty:
            logger.warning("No valid predictive features remaining for tree fitting.")
            return {}
            
        # Impute missing cells and encode categorical values
        for col in X.columns:
            col_type = semantic_types.get(col, {}).get("type", "nominal")
            if X[col].isna().any():
                if col_type in ["continuous_numerical", "discrete_numerical"]:
                    X[col] = X[col].fillna(X[col].median() if not X[col].dropna().empty else 0)
                else:
                    mode_val = X[col].mode()
                    X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Missing")
                    
            if col_type in ["nominal", "ordinal", "binary"]:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
        # Handle target label encoding if categorical
        target_type = semantic_types.get(target_col, {}).get("type", "nominal")
        if target_type in ["nominal", "binary", "boolean"]:
            y = LabelEncoder().fit_transform(y.astype(str))
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            
        try:
            model.fit(X, y)
            importances = model.feature_importances_
            
            # Map importances
            rankings = {col: float(imp) for col, imp in zip(X.columns, importances)}
            # Sort descending
            sorted_rankings = dict(sorted(rankings.items(), key=lambda item: item[1], reverse=True))
            logger.info("Successfully calculated feature importances.")
            return sorted_rankings
        except Exception as e:
            logger.exception(f"Failed to fit feature importance tree: {e}")
            return {}

    # --- Architectural Placeholders for Future SHAP / LIME Integrations ---

    @staticmethod
    def get_shap_explanation_placeholder(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        API interface for future SHAP tree/linear explainer integration.
        """
        return {
            "status": "Architectural Placeholder",
            "future_library": "SHAP (will be imported locally in Phase 2)",
            "required_api": "shap.TreeExplainer(model).shap_values(X)",
            "output_format": "DataFrame mapping features to mean SHAP force values."
        }

    @staticmethod
    def explain_individual_prediction_placeholder(model: Any, sample_row: pd.Series) -> Dict[str, Any]:
        """
        API interface for the future interactive 'Why Button' on prediction pages.
        """
        return {
            "status": "Architectural Placeholder",
            "future_library": "SHAP / LIME",
            "ui_integration": "Renders waterfall chart illustrating positive/negative pushes.",
            "natural_language": "Generates textual card: 'Prediction is Fraud because TransactionAmount was 4x average.'"
        }
