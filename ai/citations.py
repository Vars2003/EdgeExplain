from typing import List, Dict, Any, Tuple
from utils.logger import get_logger

logger = get_logger("ai.citations")

class CitationEngine:
    """
    Identifies and formats knowledge sources and data nodes 
    supporting the AI's generated response.
    """

    @staticmethod
    def compile_citations(retrieved_contexts: List[Tuple[Dict[str, Any], int, str]]) -> List[str]:
        """
        Extracts source tags from retrieved contexts.
        Returns a deduplicated, clean list of source citations.
        """
        citations = []
        for item, score, source in retrieved_contexts:
            # Clean and format citations for display
            clean_source = source
            if source.startswith("KnowledgeGraph.feat_"):
                col = source.replace("KnowledgeGraph.feat_", "")
                clean_source = f"Knowledge Graph Node (Feature: '{col}')"
            elif source.startswith("RuleEngine."):
                rec = source.replace("RuleEngine.", "").title()
                clean_source = f"Rule Decision Engine (Preprocessing: '{rec}')"
            elif source.startswith("KnowledgeGraph.algo_"):
                algo = source.replace("KnowledgeGraph.algo_", "").replace("_", " ")
                clean_source = f"Compatibility Index (Algorithm: '{algo}')"
            elif source == "MemoryObject.context":
                clean_source = "Dataset Context Metadata & Metrics"
                
            citations.append(clean_source)
            
        # Deduplicate
        seen = set()
        deduped = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
                
        logger.debug(f"Compiled {len(deduped)} citation sources: {deduped}")
        return deduped
