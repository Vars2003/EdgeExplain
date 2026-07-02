import pandas as pd
import numpy as np
from typing import Dict, Any
from plugins import BasePlugin
from utils.logger import get_logger

logger = get_logger("plugins.lime")

HAS_LIME = False
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
    logger.info("Official LIME library successfully import-checked.")
except ImportError:
    logger.warning("LIME library is not installed. LIME explanations will display placeholders.")

class LimeExplainabilityPlugin(BasePlugin):
    """
    Plugin wrapper for official LIME tabular predictions explainer.
    """

    def __init__(self):
        super().__init__("LIME Explainability", ["pandas", "numpy", "scikit-learn"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not HAS_LIME:
            return {
                "result": {
                    "status": "UNAVAILABLE",
                    "message": "The official LIME library is not installed. To activate local prediction attributions, run 'pip install lime' in your Python environment."
                },
                "confidence": 100,
                "evidence": ["Checked sys.modules for lime"]
            }

        # Select target variable
        target_candidates = memory_obj.get("context", {}).get("candidate_targets", [])
        target = target_candidates[0] if target_candidates else df.columns[-1]
        
        # Prepare numeric features
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encoding categorical features and filling nulls
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_encoded[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X_encoded[col] = X[col].astype(str).astype("category").cat.codes
                
        y_encoded = y.fillna(y.mode()[0] if not y.isna().all() else 0)
        is_classifier = len(np.unique(y_encoded)) <= 10
        if not pd.api.types.is_numeric_dtype(y_encoded):
            y_encoded = y_encoded.astype(str).astype("category").cat.codes

        # Train a random forest model
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        try:
            row_idx = 0
            row_to_explain = X_encoded.iloc[row_idx]
            
            if is_classifier:
                model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                model.fit(X_encoded, y_encoded)
                predict_fn = model.predict_proba
                mode = "classification"
                class_names = [str(c) for c in np.unique(y_encoded)]
            else:
                model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                model.fit(X_encoded, y_encoded)
                predict_fn = model.predict
                mode = "regression"
                class_names = [target]
                
            # Create LIME Explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_encoded),
                feature_names=list(X_encoded.columns),
                class_names=class_names,
                mode=mode,
                random_state=42
            )
            
            exp = explainer.explain_instance(
                data_row=np.array(row_to_explain),
                predict_fn=predict_fn,
                num_features=min(5, len(X_encoded.columns))
            )
            
            # Extract feature weights
            weights = exp.as_list()
            influential_features = [{"feature_rule": rule, "weight": float(w)} for rule, w in weights]
            
            result_payload = {
                "status": "AVAILABLE",
                "target_used": target,
                "mode": mode,
                "row_index": row_idx,
                "influential_features": influential_features,
                "prediction_value": list(predict_fn(np.array([row_to_explain]))[0]) if is_classifier else float(predict_fn(np.array([row_to_explain]))[0])
            }
            
            return {
                "result": result_payload,
                "confidence": 92,
                "evidence": [f"Trained RandomForest {mode} model", "Ran LimeTabularExplainer.explain_instance"]
            }
        except Exception as ex:
            logger.exception(f"LIME calculation failed: {ex}")
            return {
                "result": {
                    "status": "ERROR",
                    "message": f"Official LIME calculation failed. Details: {ex}"
                },
                "confidence": 50,
                "evidence": [f"Error occurred: {ex}"]
            }
