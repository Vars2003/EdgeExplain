import pandas as pd
import numpy as np
from typing import Dict, Any
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.shap")

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
    logger.info("Official SHAP library successfully import-checked.")
except ImportError:
    logger.warning("SHAP library is not installed. SHAP explanations will display as placeholders.")

class ShapExplainabilityPlugin(BasePlugin):
    """
    Plugin wrapper for official SHAP explainability.
    """

    def __init__(self):
        super().__init__("SHAP Explainability", ["pandas", "numpy", "scikit-learn"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not HAS_SHAP:
            return {
                "result": {
                    "status": "UNAVAILABLE",
                    "message": "The official SHAP library is not installed. To activate real global and local feature importance models, run 'pip install shap' in your Python environment."
                },
                "confidence": 100,
                "evidence": ["Checked sys.modules for shap"]
            }

        # Select target variable
        target_candidates = memory_obj.get("context", {}).get("candidate_targets", [])
        target = target_candidates[0] if target_candidates else df.columns[-1]
        
        # Prepare numeric features
        X = df.drop(columns=[target])
        y = df[target]
        
        # Fill missing values and convert categories to codes for model training
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_encoded[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X_encoded[col] = X[col].astype(str).astype("category").cat.codes
                
        y_encoded = y.fillna(y.mode()[0] if not y.isna().all() else 0)
        if not pd.api.types.is_numeric_dtype(y_encoded):
            y_encoded = y_encoded.astype(str).astype("category").cat.codes

        # Train a fast decision tree
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        is_classifier = len(np.unique(y_encoded)) <= 10
        
        try:
            if is_classifier:
                model = DecisionTreeClassifier(max_depth=4, random_state=42)
                model.fit(X_encoded, y_encoded)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_encoded)
                # Handle binary vs multiclass shape differentials
                if isinstance(shap_values, list):
                    shap_vals_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                elif len(shap_values.shape) == 3:
                    shap_vals_matrix = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
                else:
                    shap_vals_matrix = shap_values
            else:
                model = DecisionTreeRegressor(max_depth=4, random_state=42)
                model.fit(X_encoded, y_encoded)
                explainer = shap.TreeExplainer(model)
                shap_vals_matrix = explainer.shap_values(X_encoded)
                
            # Global importance (mean absolute SHAP)
            mean_shap = np.abs(shap_vals_matrix).mean(axis=0)
            global_importance = {col: float(mean_shap[idx]) for idx, col in enumerate(X_encoded.columns)}
            
            # Local importance (first row values as example)
            row_idx = 0
            local_shap = {col: float(shap_vals_matrix[row_idx, idx]) for idx, col in enumerate(X_encoded.columns)}
            local_feature_values = {col: float(X_encoded.iloc[row_idx, idx]) for idx, col in enumerate(X_encoded.columns)}
            
            result_payload = {
                "status": "AVAILABLE",
                "target_used": target,
                "model_type": "Classifier" if is_classifier else "Regressor",
                "global_importance": global_importance,
                "local_explanation": {
                    "row_index": row_idx,
                    "shap_values": local_shap,
                    "feature_values": local_feature_values,
                    "base_value": float(explainer.expected_value[1]) if (is_classifier and isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1) else float(explainer.expected_value)
                }
            }
            
            return {
                "result": result_payload,
                "confidence": 95,
                "evidence": [f"Trained Scikit-Learn tree using target '{target}'", f"Computed shap.TreeExplainer on {X_encoded.shape[0]} rows"]
            }
        except Exception as ex:
            logger.exception(f"SHAP explanation calculation failed: {ex}")
            return {
                "result": {
                    "status": "ERROR",
                    "message": f"Official SHAP calculation failed. Details: {ex}"
                },
                "confidence": 50,
                "evidence": [f"Error occurred: {ex}"]
            }
