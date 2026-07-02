from typing import List, Dict, Any
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("core.recommendation")

class RecommendationEngine:
    """
    Evaluates dataset metrics and rules to compile a list of actionable,
    grounded preprocessing recommendations.
    """

    @staticmethod
    def generate_recommendations(df: pd.DataFrame, metrics: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans data attributes to output an ordered set of preprocessing recommendations.
        """
        logger.info("Generating preprocessing recommendations.")
        recs = []
        rows = len(df)
        if rows == 0:
            return []
            
        missing_data = metrics.get("missing_data", {})
        basic_metrics = metrics.get("basic_metrics", {})
        outliers = metrics.get("outliers", {})
        statistics = metrics.get("statistics", {})
        dependency = metrics.get("dependency", {})
        
        # 1. Missing Value Treatment
        if missing_data:
            percentages = missing_data.get("percentages", {})
            for col, pct in percentages.items():
                if 0.1 <= pct <= 40.0:
                    col_type = semantic_types.get(col, {}).get("type", "unknown")
                    strategy = "mode"
                    if col_type in ["continuous_numerical", "discrete_numerical"]:
                        # Skewness based imputation selection
                        col_stats = statistics.get(col, {})
                        skew = col_stats.get("skewness", 0)
                        strategy = "median" if abs(skew) > 1.0 else "mean"
                        
                    recs.append({
                        "type": "Imputation",
                        "features": [col],
                        "suggested_fix": f"Impute missing cells using '{strategy}' strategy.",
                        "reason": f"Feature '{col}' contains {pct:.1f}% missing values. Imputing restores data alignment.",
                        "confidence": int(min(95, pct * 2 + 50))
                    })
                elif pct > 40.0:
                    recs.append({
                        "type": "Feature Removal",
                        "features": [col],
                        "suggested_fix": "Drop column from the dataset.",
                        "reason": f"Column '{col}' is missing {pct:.1f}% of its values, which is too high to impute reliably.",
                        "confidence": int(min(99, pct))
                    })

        # 2. Duplicate Removal
        if basic_metrics:
            dup_count = basic_metrics.get("duplicate_rows", 0)
            dup_pct = basic_metrics.get("duplicate_pct", 0)
            if dup_count > 0:
                recs.append({
                    "type": "Row De-duplication",
                    "features": [],
                    "suggested_fix": "Execute drop_duplicates() on the dataset.",
                    "reason": f"Found {dup_count} duplicate rows ({dup_pct:.1f}%). De-duplication prevents target leakage during validations.",
                    "confidence": 99
                })

        # 3. Categorical Encodings
        categorical_columns = []
        high_cardinality_cats = []
        for col, info in semantic_types.items():
            col_type = info["type"]
            if col_type in ["nominal", "ordinal", "binary"]:
                categorical_columns.append(col)
                card = df[col].nunique()
                if card > 20:
                    high_cardinality_cats.append(col)
                    
        for col in categorical_columns:
            col_type = semantic_types[col]["type"]
            if col in high_cardinality_cats:
                recs.append({
                    "type": "Encoding",
                    "features": [col],
                    "suggested_fix": "Apply Target Encoding or Frequency Encoding.",
                    "reason": f"Feature '{col}' is nominal categorical with high cardinality ({df[col].nunique()} unique labels). One-hot encoding would blow up dimensions.",
                    "confidence": 90
                })
            elif col_type == "ordinal":
                recs.append({
                    "type": "Encoding",
                    "features": [col],
                    "suggested_fix": "Apply Ordinal Encoder (preserve ranks).",
                    "reason": f"Feature '{col}' has sequential ordinal characteristics. Preserving hierarchy is mathematically optimal.",
                    "confidence": 95
                })
            else:
                recs.append({
                    "type": "Encoding",
                    "features": [col],
                    "suggested_fix": "Apply One-Hot Encoding.",
                    "reason": f"Feature '{col}' is categorical with low cardinality. One-hot encoding maps this to binary columns.",
                    "confidence": 95
                })

        # 4. Feature Scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            stds = {}
            for col in numeric_cols:
                std = df[col].std()
                if std > 0 and np.isfinite(std):
                    stds[col] = std
                    
            if stds:
                max_std_col = max(stds, key=stds.get)
                min_std_col = min(stds, key=stds.get)
                ratio = stds[max_std_col] / stds[min_std_col]
                
                if ratio > 10.0:
                    # Scaling is recommended
                    recs.append({
                        "type": "Standardization",
                        "features": list(stds.keys()),
                        "suggested_fix": "Apply StandardScaler or MinMaxScaler.",
                        "reason": f"Feature variance scales differ by {ratio:.1f}x. Standardizing values is required for linear and distance-based algorithms.",
                        "confidence": 92
                    })

        # 5. Outliers Treatment
        if outliers:
            for col, count in outliers.get("columns", {}).items():
                pct = (count / rows) * 100
                if pct > 5.0:
                    recs.append({
                        "type": "Outlier Handling",
                        "features": [col],
                        "suggested_fix": "Apply Robust Scaling or Winsorization (clipping values).",
                        "reason": f"Feature '{col}' contains {count} outliers ({pct:.1f}%). Adjusting tails protects model integrity.",
                        "confidence": 85
                    })

        # 6. Feature Redundancy (Multicollinearity)
        if dependency:
            cols = dependency.get("columns", [])
            matrix = dependency.get("matrix", [])
            
            redundant_pairs = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = matrix[i][j]
                    if val > 0.90:
                        redundant_pairs.append((cols[i], cols[j], val))
                        
            for c1, c2, score in redundant_pairs:
                recs.append({
                    "type": "Feature Selection",
                    "features": [c1, c2],
                    "suggested_fix": f"Drop '{c2}' and keep '{c1}' to eliminate redundant variance.",
                    "reason": f"Extreme multicollinearity detected between '{c1}' and '{c2}' (Dependency score: {score:.2f}).",
                    "confidence": int(score * 100)
                })
                
        # 7. Constant Column Removal
        constant_cols = basic_metrics.get("constant_columns", [])
        if constant_cols:
            for col in constant_cols:
                recs.append({
                    "type": "Feature Removal",
                    "features": [col],
                    "suggested_fix": "Drop column from the dataset.",
                    "reason": f"Column '{col}' is a constant column with zero variance.",
                    "confidence": 99
                })
                
        # Fallback empty recommendation
        if len(recs) == 0:
            recs.append({
                "type": "None",
                "features": [],
                "suggested_fix": "No preprocessing actions required.",
                "reason": "Dataset statistics indicate ideal distribution shapes, zero duplicates, and complete rows.",
                "confidence": 100
            })
            
        return recs
