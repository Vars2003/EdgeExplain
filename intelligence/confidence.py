import numpy as np
import pandas as pd
from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger("intelligence.confidence")

class ConfidenceEngine:
    """
    Formulates and calculates confidence scores (0-100%) for semantic categorizations,
    target variables, preprocessing recommendations, and model suggestions.
    """

    @staticmethod
    def calculate_domain_confidence(hit_count: int, total_matches: int) -> int:
        """
        Derives confidence for domain estimation based on vocabulary hit ratios.
        """
        if total_matches == 0:
            return 50
        ratio = hit_count / total_matches
        # Scale: weight proportion is 80%, absolute hit quantity gives up to 20% boost
        score = int((ratio * 80) + min(20, hit_count * 5))
        return min(99, max(50, score))

    @staticmethod
    def calculate_task_confidence(df: pd.DataFrame, task_type: str, has_time: bool, target_found: bool) -> int:
        """
        Derives confidence for machine learning task classifications.
        """
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return 100
            
        score = 70
        if task_type == "Time Series":
            score = 95 if df.index.name == "Timestamp" or any("time" in str(col).lower() for col in df.columns) else 80
        elif task_type == "Geospatial Dataset":
            score = 90
        elif target_found:
            score = 90
            if rows > 100:
                score += 5
            if cols > 2:
                score += 3
        return min(99, score)

    @staticmethod
    def calculate_target_confidence(col_name: str, unique_ratio: float, cardinality: int) -> int:
        """
        Calculates confidence that a column is an target label.
        """
        col_lower = col_name.lower()
        score = 50
        
        # Keyword matching boosts confidence
        if any(t in col_lower for t in ["target", "label", "class", "clicked", "price", "churn", "fraud"]):
            score += 35
            
        # Target characteristics
        if cardinality == 2: # Binary classification
            score += 10
        elif 2 < cardinality <= 15: # Multiclass or discrete numeric
            score += 5
        elif unique_ratio > 0.95: # Too unique (like an ID) - lower target confidence
            score -= 40
            
        return min(99, max(10, score))

    @staticmethod
    def calculate_recommendation_confidence(rec_type: str, severity: float) -> int:
        """
        Calculates confidence of a preprocessing recommendation based on defect severity.
        """
        if rec_type in ["Imputation", "Feature Removal"]:
            # Missingness checks
            score = int(severity * 2 + 50) if severity <= 40 else int(severity)
            return min(99, max(50, score))
        elif rec_type == "Row De-duplication":
            return 99 # Exact check
        elif rec_type == "Encoding":
            return 95 # Always high if nominal columns exist
        elif rec_type == "Standardization":
            return 92
        elif rec_type == "Outlier Handling":
            return 85
        elif rec_type == "Feature Selection":
            return int(min(99, severity * 100))
        return 90
