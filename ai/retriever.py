from typing import List, Dict, Any, Tuple
import pandas as pd
from utils.logger import get_logger
from intelligence.knowledge_graph import KnowledgeGraph

logger = get_logger("ai.retriever")

class GraphRetriever:
    """
    Retrieves grounded context facts from the cached MemoryObject using 
    Intelligence Layer query APIs, ranking results by token relevance.
    """

    @staticmethod
    def retrieve_relevant_context(prompt: str, memory_obj: Dict[str, Any], max_items: int = 5) -> List[Tuple[Dict[str, Any], int, str]]:
        """
        Scans prompt tokens against Knowledge Graph APIs.
        Computes relevance scores, ranks matches, and returns top results.
        Returns a list of tuples: (context_data, relevance_score, source_tag)
        """
        logger.info(f"Running ranked retriever search for query: '{prompt}'")
        prompt_lower = prompt.lower()
        
        # Instantiate temporary KG populated with memory object contents
        kg = KnowledgeGraph()
        kg.nodes = memory_obj.get("graph", {}).get("nodes", [])
        kg.edges = memory_obj.get("graph", {}).get("edges", [])
        
        retrieved_contexts: List[Tuple[Dict[str, Any], int, str]] = []
        
        # 1. Search features
        schema_cols = memory_obj.get("schema", {}).keys()
        for col in schema_cols:
            col_lower = col.lower()
            score = 0
            # Exact match
            if f" {col_lower} " in f" {prompt_lower} " or prompt_lower.startswith(col_lower) or prompt_lower.endswith(col_lower):
                score += 60
            # Substring match
            elif col_lower in prompt_lower:
                score += 30
                
            if score > 0:
                feat_context = kg.find_feature(col)
                if feat_context and feat_context.get("node"):
                    retrieved_contexts.append((
                        feat_context["node"],
                        score,
                        f"KnowledgeGraph.feat_{col}"
                    ))
                    
        # 2. Search preprocessing concepts
        concept_keywords = {
            "imputation": ["missing", "null", "impute", "empty", "nan", "fill"],
            "removal": ["drop", "remove", "constant", "useless"],
            "encoding": ["encode", "category", "nominal", "string", "text", "one-hot", "dummy"],
            "standardization": ["scale", "normalize", "standardize", "range", "std", "mean"],
            "outlier": ["outlier", "anomaly", "tukey", "iqr", "extreme"],
            "selection": ["multicollinearity", "redundant", "correlation", "vif", "features"]
        }
        
        for rec_type, keywords in concept_keywords.items():
            score = 0
            for kw in keywords:
                if kw in prompt_lower:
                    score += 40
            if score > 0:
                recs = kg.find_recommendation(rec_type)
                for r in recs:
                    retrieved_contexts.append((
                        r,
                        score,
                        f"RuleEngine.{rec_type}"
                    ))

        # 3. Search model/algorithms
        algo_keywords = ["algorithm", "model", "prediction", "train", "classification", "regression", "rf", "xgboost", "tree", "forest"]
        has_algo_intent = any(ak in prompt_lower for ak in algo_keywords)
        if has_algo_intent:
            algos = memory_obj.get("algorithms", [])
            for idx, a in enumerate(algos):
                score = 35
                algo_name = a.get("algorithm", "")
                if algo_name.lower() in prompt_lower:
                    score += 25
                algo_node = kg.find_algorithm(algo_name)
                if algo_node:
                    retrieved_contexts.append((
                        algo_node,
                        score,
                        f"KnowledgeGraph.algo_{algo_name.replace(' ', '_')}"
                    ))

        # 4. Search general summaries
        summary_keywords = ["about", "dataset", "overview", "summary", "explain", "describe", "metadata", "executive", "general"]
        has_summary_intent = any(sk in prompt_lower for sk in summary_keywords)
        if has_summary_intent or not retrieved_contexts:
            ds_node = kg.find_summary()
            if ds_node:
                score = 30
                if any(sk in prompt_lower for sk in summary_keywords):
                    score += 20
                retrieved_contexts.append((
                    ds_node,
                    score,
                    "MemoryObject.context"
                ))
                
        # Sort retrieved contexts by relevance score descending
        retrieved_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicate nodes in retrieval
        unique_contexts = []
        seen_ids = set()
        for item in retrieved_contexts:
            node_id = item[0].get("id", "")
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_contexts.append(item)
                
        logger.info(f"Retrieved {len(unique_contexts)} unique contexts. Truncating to top {max_items}.")
        return unique_contexts[:max_items]
