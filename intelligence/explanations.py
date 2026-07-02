from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger("intelligence.explanations")

class ExplanationEngine:
    """
    Constructs standardized explanations (Why/How/Evidence) matching 
    a strict schema for UI representation and LLM prompts.
    """

    @staticmethod
    def construct_explanation(title: str, reason: str, evidence: List[str], confidence: int, alternatives: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Builds a standard Explanation Schema object.
        """
        return {
            "title": title,
            "reason": reason,
            "evidence": evidence,
            "confidence": confidence,
            "alternatives": alternatives,
            "references": references
        }

    @classmethod
    def explain_recommendation(cls, rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a preprocessing recommendation into the standard Explanation Schema.
        """
        rec_type = rec["type"]
        features = rec["features"]
        suggested_fix = rec["suggested_fix"]
        reason = rec["reason"]
        confidence = rec["confidence"]
        evidence = rec.get("evidence", [])
        alternatives = rec.get("alternatives", [])
        
        # Determine references in local knowledge JSONs
        ref_map = {
            "Imputation": ["knowledge/preprocessing.json#imputation"],
            "Feature Removal": ["knowledge/preprocessing.json#imputation"],
            "Row De-duplication": ["knowledge/preprocessing.json#imputation"],
            "Encoding": ["knowledge/preprocessing.json#encoding"],
            "Standardization": ["knowledge/preprocessing.json#scaling"],
            "Outlier Handling": ["knowledge/preprocessing.json#scaling"],
            "Feature Selection": ["knowledge/preprocessing.json#preprocessing"]
        }
        references = ref_map.get(rec_type, ["knowledge/preprocessing.json"])
        
        title = f"Apply {rec_type} for {', '.join(features)}" if features else f"Apply {rec_type}"
        full_reason = f"{reason} How to resolve: {suggested_fix}"
        
        return cls.construct_explanation(
            title=title,
            reason=full_reason,
            evidence=evidence,
            confidence=confidence,
            alternatives=alternatives,
            references=references
        )
