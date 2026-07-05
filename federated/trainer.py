import pandas as pd
import numpy as np
from typing import Any, Tuple
from utils.logger import get_logger

logger = get_logger("federated.trainer")

class LocalTrainer:
    """
    Executes local Scikit-Learn training fits on client partition DataFrames.
    Contains no networking or server aggregation code.
    """
    @staticmethod
    def train(df: pd.DataFrame, model_name: str, target: str = None, initial_weights: dict = None, local_epochs: int = 1) -> Tuple[Any, float, float, int]:
        if df is None or len(df) == 0:
            return None, 0.0, 0.0, 0
            
        if not target:
            target = df.columns[-1]
            
        X = df.drop(columns=[target])
        y = df[target]
        
        # Preprocessing: Impute numeric nulls and encode object categoricals
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_encoded[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X_encoded[col] = X[col].astype(str).astype("category").cat.codes
                
        y_encoded = y.fillna(y.mode()[0] if not y.isna().all() else 0)
        if not pd.api.types.is_numeric_dtype(y_encoded):
            y_encoded = y_encoded.astype(str).astype("category").cat.codes

        from federated.models import load_model, set_weights
        model = load_model(model_name)
        
        # If Logistic Regression and initial_weights are provided, we warm-start from global parameters
        if "logistic" in model_name.lower() and initial_weights is not None:
            try:
                # 1. Quick fit to initialize internal shapes
                model.fit(X_encoded.iloc[:max(5, len(X_encoded))], y_encoded.iloc[:max(5, len(y_encoded))])
                # 2. Overwrite weights with global ones
                set_weights(model, initial_weights)
                # 3. Enable warm start optimization
                model.warm_start = True
            except Exception as e:
                logger.warning(f"Failed to initialize warm start weights: {e}")

        # Note: Scikit-Learn classifiers (LR, DT, RF) are non-incremental.
        # We perform a single full fitting optimization cycle per communication round.
        # If local_epochs > 1, this single fit represents the full convergence cycle starting from initial weights.
        try:
            model.fit(X_encoded, y_encoded)
            
            # Accuracy metric
            accuracy = float(model.score(X_encoded, y_encoded))
            
            # Loss metric (Approximated log loss)
            loss = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    from sklearn.metrics import log_loss
                    probs = model.predict_proba(X_encoded)
                    loss = float(log_loss(y_encoded, probs))
                except Exception:
                    loss = float(1.0 - accuracy)
            else:
                loss = float(1.0 - accuracy)
                
            return model, accuracy, loss, len(df)
        except Exception as e:
            logger.exception(f"Local training fit failed for model '{model_name}': {e}")
            return None, 0.0, 1.0, len(df)
