from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger("core.reasoning")

class ReasoningEngine:
    """
    Evaluates metadata observations and provides structured explanations
    detailing the mathematical and logic boundaries of platform decisions.
    """

    @staticmethod
    def explain_dataset_classification(dataset_type_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explains why the dataset was categorized into its selected operational focus.
        """
        dtype = dataset_type_info["type"]
        confidence = dataset_type_info["confidence"]
        evidence = dataset_type_info["evidence"]
        
        explanation = ""
        if dtype == "Time Series":
            explanation = (
                "The presence of datetime sequence indexes suggests chronological tracking. "
                "Models must maintain historical dependency matrices rather than standard independent data splits."
            )
        elif dtype == "Geospatial Dataset":
            explanation = (
                "At least two columns contain geographic coordinate markers (Latitude and Longitude). "
                "Spatial clustering (K-Means, DBSCAN) or map-based visual layers are required to interpret these values."
            )
        elif dtype == "Classification":
            explanation = (
                f"We detected a target-like column with categorical or discrete outputs ({evidence[0]}). "
                "The target has distinct groups, making the machine learning task classification (predicting discrete class memberships)."
            )
        elif dtype == "Regression":
            explanation = (
                f"We detected a continuous numeric target-like column ({evidence[0]}). "
                "The values span a real-valued continuous scale, indicating a regression task (forecasting values)."
            )
        else:
            explanation = (
                "The columns consist of a mix of numerical indices and strings, without a prominent "
                "datetime or spatial theme. This is a generic tabular analysis pipeline."
            )
            
        return {
            "decision": f"Dataset Type = {dtype}",
            "confidence": confidence,
            "evidence": evidence,
            "explanation": explanation
        }

    @staticmethod
    def explain_feature_classification(col_name: str, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explains why a specific column was labeled with its semantic type.
        """
        ftype = type_info["type"]
        confidence = type_info["confidence"]
        evidence = type_info["evidence"]
        
        reasons = {
            "datetime": "Values conform to standard date/time formatting templates and were parsed as chronologically indexable records.",
            "identifier": "Each record in the column is completely unique and name keywords match primary database key schemas, meaning it contains no generalizable intelligence.",
            "boolean": "The column is binary-state, consisting only of standard true/false labels or 0/1 indicator bits.",
            "binary": "Exactly two categorical classes are present, indicating a binary feature (e.g. Yes/No).",
            "geographic_coordinate": "The numeric values align with coordinate bounds and column headers match coordinate tags.",
            "continuous_numerical": "The column contains real numbers with a high number of unique values, indicating continuous measurements.",
            "discrete_numerical": "The values are integers within a small, bounded range, suggesting count statistics.",
            "text": "High unique ratio paired with long string lengths indicates free-form text values suitable for tokenization.",
            "ordinal": "Values have hierarchical sequence flags (like grades or ranks), indicating order is mathematically relevant.",
            "nominal": "Standard textual categories with no inherent ordinal hierarchy."
        }
        
        return {
            "decision": f"Feature '{col_name}' = {ftype}",
            "confidence": confidence,
            "evidence": evidence,
            "explanation": reasons.get(ftype, "Generic feature type rule classification.")
        }

    @staticmethod
    def explain_preprocessing_need(rec_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Justifies why specific transformations (scaling, encoding, imputation) are suggested.
        """
        features = details.get("features", [])
        evidence = details.get("evidence", [])
        confidence = details.get("confidence", 90)
        
        explanation = ""
        if rec_type == "normalization":
            explanation = (
                f"Numerical scales of {features} vary by more than one order of magnitude. "
                "Distance-based algorithms (KNN, SVM, K-Means) will be heavily biased toward "
                "larger-range features if features are not normalized or standardized first."
            )
        elif rec_type == "encoding":
            explanation = (
                f"Features {features} are categorical. "
                "Standard machine learning libraries require inputs to be numerical matrices. "
                "One-hot encoding is suggested for low cardinality features, and target encoding for high cardinality."
            )
        elif rec_type == "imputation":
            explanation = (
                f"Columns {features} contain missing entries. "
                "Empty cells will crash training routines in scikit-learn. Imputing values using mean/median/mode "
                "restores matrix integrity without dropping entire observations."
            )
        elif rec_type == "redundancy_removal":
            explanation = (
                f"High multicollinearity detected between {features} (correlation > 0.90). "
                "This leads to overfitting, destabilizes regression coefficients, and inflates variance."
            )
        elif rec_type == "outlier_handling":
            explanation = (
                f"Significant outliers detected in {features}. "
                "Outliers distort mean calculations and bias linear regressions. Robust scaling or "
                "Winsorization is recommended to limit impact."
            )
        else:
            explanation = "General data health preprocessing recommended to ensure matrix alignment."
            
        return {
            "decision": f"Recommendation = {rec_type.upper()}",
            "confidence": confidence,
            "evidence": evidence,
            "explanation": explanation
        }
