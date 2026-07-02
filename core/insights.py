from typing import List, Dict, Any
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("core.insights")

class InsightsEngine:
    """
    Scans dataset summary metrics, correlation maps, and distributions 
    to automatically extract natural language observations and quality findings.
    """

    @staticmethod
    def generate_insights(df: pd.DataFrame, metrics: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes dataset profiles to compile a list of structured quality insights.
        """
        logger.info("Generating automated dataset insights.")
        insights = []
        rows = len(df)
        if rows == 0:
            return []
            
        # 1. Missing Values Insights
        missing_data = metrics.get("missing_data", {})
        if missing_data:
            percentages = missing_data.get("percentages", {})
            for col, pct in percentages.items():
                if pct > 40.0:
                    insights.append({
                        "title": f"Extreme Missingness in '{col}'",
                        "description": f"Feature '{col}' is missing {pct:.1f}% of its values. Consider removing this column as it lacks sufficient training signal.",
                        "severity": "High",
                        "category": "Missing Values",
                        "columns": [col]
                    })
                elif pct > 10.0:
                    insights.append({
                        "title": f"Moderate Missingness in '{col}'",
                        "description": f"Feature '{col}' has {pct:.1f}% missing values. Imputation is required before modeling.",
                        "severity": "Medium",
                        "category": "Missing Values",
                        "columns": [col]
                    })
                    
        # 2. Outliers Insights
        outlier_data = metrics.get("outliers", {})
        if outlier_data:
            col_outliers = outlier_data.get("columns", {})
            for col, count in col_outliers.items():
                pct = (count / rows) * 100
                if pct > 8.0:
                    insights.append({
                        "title": f"Significant Outliers in '{col}'",
                        "description": f"Column '{col}' has {count} outliers ({pct:.1f}% of observations). May bias linear model training.",
                        "severity": "Medium",
                        "category": "Outliers",
                        "columns": [col]
                    })

        # 3. Duplicate Records
        basic = metrics.get("basic_metrics", {})
        if basic:
            dup_pct = basic.get("duplicate_pct", 0)
            if dup_pct > 5.0:
                insights.append({
                    "title": "High Row Duplication Detected",
                    "description": f"The dataset contains {basic['duplicate_rows']} duplicate rows ({dup_pct:.1f}% of rows). Deduping is recommended to prevent validation leakage.",
                    "severity": "High",
                    "category": "Duplicates",
                    "columns": []
                })
            elif basic['duplicate_rows'] == 0:
                insights.append({
                    "title": "No Duplicate Records",
                    "description": "Rows are clean and unique. No duplicates found.",
                    "severity": "Low",
                    "category": "Duplicates",
                    "columns": []
                })
                
        # 4. Skewness and distribution
        stats = metrics.get("statistics", {})
        if stats:
            for col, col_stats in stats.items():
                skew = col_stats.get("skewness", 0)
                if not np.isnan(skew):
                    if abs(skew) > 2.0:
                        insights.append({
                            "title": f"Heavy Distribution Skewness in '{col}'",
                            "description": f"Column '{col}' is heavily skewed ({skew:.2f}). A log transform or robust scaler is recommended.",
                            "severity": "Medium",
                            "category": "Distribution",
                            "columns": [col]
                        })

        # 5. Identifier Columns
        for col, info in semantic_types.items():
            if info["type"] == "identifier":
                insights.append({
                    "title": f"Identifier Feature Found: '{col}'",
                    "description": f"'{col}' functions as a unique key identifier and holds no generalizable prediction patterns. It should be removed from feature sets.",
                    "severity": "Medium",
                    "category": "Metadata",
                    "columns": [col]
                })
                
        # 6. High Cardinality Categorical
        for col, info in semantic_types.items():
            if info["type"] == "nominal":
                card = df[col].nunique()
                if card > 30:
                    insights.append({
                        "title": f"High Cardinality in '{col}'",
                        "description": f"Categorical column '{col}' has {card} unique values. One-hot encoding will inflate feature dimensionality. Consider target encoding.",
                        "severity": "Medium",
                        "category": "Cardinality",
                        "columns": [col]
                    })
                    
        # 7. Strong Relationships
        dependency = metrics.get("dependency", {})
        if dependency:
            cols = dependency.get("columns", [])
            matrix = dependency.get("matrix", [])
            methods = dependency.get("methods", {})
            
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    weight = matrix[i][j]
                    if weight > 0.85:
                        insights.append({
                            "title": f"Strong Relationship: '{cols[i]}' and '{cols[j]}'",
                            "description": f"Strong association detected ({weight:.2f}) using {methods.get(f'{cols[i]} <-> {cols[j]}', 'association checks')}. Flagged for multicollinearity.",
                            "severity": "High",
                            "category": "Relationships",
                            "columns": [cols[i], cols[j]]
                        })
                        
        # Default fallback if clean
        if len(insights) == 0:
            insights.append({
                "title": "Healthy Structural Base",
                "description": "No major data quality defects, extreme missingness, or duplicate concerns were flagged.",
                "severity": "Low",
                "category": "Quality",
                "columns": []
            })
            
        # Sort insights by severity (High -> Medium -> Low)
        severity_map = {"High": 0, "Medium": 1, "Low": 2}
        insights.sort(key=lambda x: severity_map.get(x["severity"], 3))
        
        return insights
