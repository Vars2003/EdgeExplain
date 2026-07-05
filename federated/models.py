from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_model(name: str):
    """
    Loads and returns a new instance of a supported classifier model.
    """
    clean_name = name.lower().replace("_", " ").strip()
    
    if "logistic" in clean_name:
        return LogisticRegression(max_iter=100, random_state=42)
    elif "decision tree" in clean_name:
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    elif "random forest" in clean_name:
        return RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    else:
        # Default fallback
        return RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

def get_weights(model) -> dict:
    """
    Extracts coefficient and intercept weights from a fitted model.
    Only supports Logistic Regression parameters.
    """
    if isinstance(model, LogisticRegression):
        if hasattr(model, "coef_"):
            return {
                "coef": model.coef_.copy(),
                "intercept": model.intercept_.copy(),
                "classes": model.classes_.copy()
            }
    return None

def set_weights(model, weights: dict) -> None:
    """
    Sets coefficient and intercept weights on a model instance.
    """
    if isinstance(model, LogisticRegression) and weights is not None:
        model.coef_ = weights["coef"].copy()
        model.intercept_ = weights["intercept"].copy()
        model.classes_ = weights["classes"].copy()
