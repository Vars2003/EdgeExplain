from typing import List, Dict, Any
import pandas as pd
from utils.logger import get_logger
from core.recommendation import RecommendationEngine
from intelligence.confidence import ConfidenceEngine

logger = get_logger("intelligence.rule_engine")

class IntelligenceRuleEngine:
    """
    Centralized decision logic manager. Processes raw recommendations 
    and appends standardized alternatives and detailed evidence.
    """

    @staticmethod
    def get_enriched_recommendations(df: pd.DataFrame, metrics: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieves raw recommendations and enriches them with alternative paths and evidence lists.
        """
        logger.info("Enriching preprocessing recommendations via AI Rule Engine.")
        
        # Pull raw recommendations using core engine
        raw_recs = RecommendationEngine.generate_recommendations(df, metrics, semantic_types)
        enriched_recs = []
        
        for r in raw_recs:
            rec_type = r["type"]
            features = r["features"]
            suggested_fix = r["suggested_fix"]
            reason = r["reason"]
            confidence = r["confidence"]
            
            # Map evidence and alternatives based on recommendation type
            evidence = []
            alternatives = []
            
            if rec_type == "Imputation":
                feat_name = features[0] if features else ""
                miss_pct = metrics.get("missing_data", {}).get("percentages", {}).get(feat_name, 0.0)
                evidence = [f"Feature '{feat_name}' missingness rate: {miss_pct:.2f}%"]
                alternatives = ["Row Deletion (drop rows containing missing cells)", "K-Nearest Neighbors (KNN) Imputation"]
                
            elif rec_type == "Feature Removal":
                feat_name = features[0] if features else ""
                if feat_name in metrics.get("missing_data", {}).get("percentages", {}):
                    miss_pct = metrics["missing_data"]["percentages"][feat_name]
                    evidence = [f"Feature '{feat_name}' missingness rate: {miss_pct:.2f}% (exceeds 40% threshold)"]
                else:
                    evidence = [f"Feature '{feat_name}' cardinality is 1 (constant column)"]
                alternatives = ["Iterative Imputer (MICE)", "Re-ingest database to check data capture pipeline"]
                
            elif rec_type == "Row De-duplication":
                dup_count = metrics.get("basic_metrics", {}).get("duplicate_rows", 0)
                evidence = [f"Duplicate row count: {dup_count} instances"]
                alternatives = ["Keep duplicates but apply sample weights during modeling", "No action (if duplicates are valid identical events)"]
                
            elif rec_type == "Encoding":
                feat_name = features[0] if features else ""
                cardinality = df[feat_name].nunique() if feat_name in df.columns else 0
                evidence = [f"Categorical variable '{feat_name}' has cardinality: {cardinality}"]
                if cardinality > 20:
                    alternatives = ["Frequency Encoding", "One-Hot Encoding (caution: high dimensions)"]
                else:
                    alternatives = ["Target Encoding", "Binary Encoding"]
                    
            elif rec_type == "Standardization":
                evidence = ["Scale standard deviation ratios differ by > 10x across numeric features"]
                alternatives = ["RobustScaler (recommended if outliers exist)", "MinMaxScaler (bounds inputs between 0 and 1)", "No scaling (if using tree models like Random Forest)"]
                
            elif rec_type == "Outlier Handling":
                feat_name = features[0] if features else ""
                outlier_count = metrics.get("outliers", {}).get("columns", {}).get(feat_name, 0)
                evidence = [f"Outlier count in '{feat_name}': {outlier_count} values (IQR Tukey method)"]
                alternatives = ["Log / Box-Cox transformations", "Quantile clipping / Winsorization"]
                
            elif rec_type == "Feature Selection":
                evidence = [f"Collinearity correlation coefficient between {features} exceeds 0.90"]
                alternatives = ["Apply Principal Component Analysis (PCA)", "Keep both and use Ridge L2 regularization to shrink coefficients"]
                
            else:
                evidence = ["General dataset hygiene check"]
                alternatives = ["No action required"]
                
            enriched_recs.append({
                "type": rec_type,
                "features": features,
                "suggested_fix": suggested_fix,
                "reason": reason,
                "evidence": evidence,
                "confidence": confidence,
                "alternatives": alternatives
            })
            
        return enriched_recs
