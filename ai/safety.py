from typing import List
from utils.logger import get_logger

logger = get_logger("ai.safety")

class SafetyFilter:
    """
    Validates user queries against dataset bounds.
    Blocks out-of-scope prompts (e.g. general knowledge, unrelated coding requests).
    """

    @staticmethod
    def is_query_safe(query: str, df_columns: List[str]) -> bool:
        """
        Runs token-matching validation. Returns True if query is about the dataset, False otherwise.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return False
            
        # 1. Always safe if a column name is mentioned
        for col in df_columns:
            if col.lower() in query_lower:
                logger.debug(f"Safety passed: Query references column '{col}'")
                return True
                
        # 2. Check general analytical vocabulary keywords
        domain_vocabulary = [
            "row", "col", "dataset", "data", "quality", "missing", "outlier", 
            "statistic", "correlation", "normalize", "algorithm", "model", 
            "imputation", "readiness", "score", "metric", "type", "domain", 
            "risk", "shape", "distribution", "skew", "kurtosis", "regression", 
            "classification", "clustering", "encoding", "clean", "analysis", 
            "variable", "feature", "summary", "report", "file", "target", "predict",
            "mean", "median", "std", "maximum", "minimum", "null", "nan", "vif",
            "anova", "cramer", "pearson", "spearman", "mutual information",
            "split", "test", "train", "tree", "forest"
        ]
        
        for kw in domain_vocabulary:
            if kw in query_lower:
                logger.debug(f"Safety passed: Query references vocabulary keyword '{kw}'")
                return True
                
        logger.warning(f"Safety block fired: Query '{query}' has no relation to dataset parameters.")
        return False
