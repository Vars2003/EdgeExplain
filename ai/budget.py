import json
from typing import List, Dict, Any, Tuple
from utils.logger import get_logger

logger = get_logger("ai.budget")

class TokenBudgetManager:
    """
    Manages prompt sizes and compresses retrieved contexts to fit model constraints.
    Prioritizes contexts: Recommendations ➔ Evidence ➔ Knowledge Graph ➔ Summaries.
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Approximates token count using character-to-token ratio (4 chars = 1 token).
        """
        return max(1, len(text) // 4)

    @classmethod
    def compress_context(cls, retrieved_contexts: List[Tuple[Dict[str, Any], int, str]], max_tokens: int = 2048) -> Tuple[str, List[str]]:
        """
        Incrementally builds a context string from retrieved context nodes.
        Prioritizes items by importance categories.
        Returns a tuple: (compressed_context_string, citation_list)
        """
        logger.info(f"Compressing context within token budget: {max_tokens}")
        
        # 1. Group retrieved contexts by category type
        recommendations = []
        evidence_facts = []
        graph_nodes = []
        summaries = []
        citations = []
        
        for item, score, source in retrieved_contexts:
            node_type = item.get("type", "")
            node_id = item.get("id", "")
            
            # Record citation source
            citations.append(source)
            
            if node_type == "recommendation":
                recommendations.append(item)
            elif node_type == "dependency" or "weight" in item.get("metadata", {}):
                evidence_facts.append(item)
            elif node_type == "feature" or node_type == "algorithm":
                graph_nodes.append(item)
            else:
                summaries.append(item)
                
        # 2. Build context incrementally based on priority lists
        context_str = ""
        current_tokens = 0
        
        # Priority 1: Recommendations
        if recommendations:
            block = "\n### Recommendations & Suggested Fixes:\n"
            for r in recommendations:
                item_str = f"- {r.get('label')}: {r.get('metadata', {}).get('fix', '')} (Why: {r.get('metadata', {}).get('reason', '')})\n"
                if current_tokens + cls.estimate_tokens(block + item_str) <= max_tokens:
                    block += item_str
            context_str += block
            current_tokens = cls.estimate_tokens(context_str)
            
        # Priority 2: Evidence Facts
        if evidence_facts:
            block = "\n### Feature Dependencies & Evidence Statistics:\n"
            for ef in evidence_facts:
                item_str = f"- {ef.get('label')}: Score {ef.get('metadata', {}).get('weight', 0.0)}\n"
                if current_tokens + cls.estimate_tokens(block + item_str) <= max_tokens:
                    block += item_str
            context_str += block
            current_tokens = cls.estimate_tokens(context_str)
            
        # Priority 3: Graph Nodes
        if graph_nodes:
            block = "\n### Feature Details & Model Compatibilities:\n"
            for gn in graph_nodes:
                meta = gn.get("metadata", {})
                item_str = f"- {gn.get('label')} (Type: {gn.get('type')}, Meta: {json.dumps(meta)})\n"
                if current_tokens + cls.estimate_tokens(block + item_str) <= max_tokens:
                    block += item_str
            context_str += block
            current_tokens = cls.estimate_tokens(context_str)
            
        # Priority 4: Summaries
        if summaries:
            block = "\n### General Dataset Summaries:\n"
            for s in summaries:
                meta = s.get("metadata", {})
                item_str = f"- Dataset Focus: {meta.get('dataset_type', 'Generic')}, Domain: {meta.get('domain', 'Unknown')}, Quality Score: {meta.get('quality_score', 0.0)}%\n"
                if current_tokens + cls.estimate_tokens(block + item_str) <= max_tokens:
                    block += item_str
            context_str += block
            current_tokens = cls.estimate_tokens(context_str)
            
        logger.info(f"Compression complete. Result size: {current_tokens} tokens / {len(context_str)} characters.")
        return context_str.strip(), citations
