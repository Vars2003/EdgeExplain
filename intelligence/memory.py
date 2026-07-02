import datetime
import hashlib
import json
import pandas as pd
from typing import Dict, Any, List
from utils.logger import get_logger
from intelligence.context_builder import ContextBuilder
from intelligence.knowledge_graph import KnowledgeGraph
from intelligence.rule_engine import IntelligenceRuleEngine
from intelligence.summarizer import DatasetSummarizer
from intelligence.plugins import plugin_manager

logger = get_logger("intelligence.memory")

class MemoryObject:
    """
    Serializes all data intelligence profiles into a unified, versioned JSON payload.
    Serves as the grounding context for offline LLM prompts.
    """

    @staticmethod
    def calculate_dataset_hash(df: pd.DataFrame, filename: str) -> str:
        """
        Generates a deterministic MD5 hash of the dataset to serve as a dataset_id.
        """
        rows, cols = df.shape
        # Hash filename, dimensions, and first few values to identify modifications
        head_str = str(df.head(5).to_dict())
        payload = f"{filename}_{rows}_{cols}_{head_str}"
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    @classmethod
    def compile_memory_object(cls, df: pd.DataFrame, filename: str, metrics: Dict[str, Any], semantic_types: Dict[str, Dict[str, Any]], domain_info: Dict[str, Any], task_info: Dict[str, Any], algo_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates all Context, Rules, Graphs, and Summaries, caching the final structure.
        """
        logger.info("Compiling unified JSON Memory Object...")
        
        # 1. Dataset ID
        dataset_id = cls.calculate_dataset_hash(df, filename)
        
        # 2. Context Summary
        context = ContextBuilder.build_context(df, filename, metrics, semantic_types, domain_info, task_info)
        
        # 3. Rule recommendations (with alternatives)
        recs = IntelligenceRuleEngine.get_enriched_recommendations(df, metrics, semantic_types)
        
        # 4. Knowledge Graph
        kg = KnowledgeGraph()
        kg.build_graph(context, semantic_types, metrics, recs, algo_recs)
        graph_dict = {
            "nodes": kg.nodes,
            "edges": kg.edges
        }
        
        # 5. Summaries
        summaries = DatasetSummarizer.compile_all_summaries(context, algo_recs)
        
        # 6. Build final payload
        memory_payload = {
            "metadata": {
                "version": "1.0.0",
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "dataset_id": dataset_id,
                "application_version": "1.0.0"
            },
            "context": context,
            "schema": semantic_types,
            "graph": graph_dict,
            "summaries": summaries,
            "recommendations": recs,
            "algorithms": algo_recs,
            "plugins": {} # Future plugin outputs will append here
        }
        
        # 7. Register Plugins dynamically
        try:
            from plugins.shap_plugin import ShapExplainabilityPlugin
            from plugins.lime_plugin import LimeExplainabilityPlugin
            from plugins.automl_plugin import AutoMLRecommendationPlugin
            from plugins.drift_plugin import DatasetDriftPlugin
            from plugins.leakage_plugin import LeakageDetectionPlugin
            from plugins.bias_plugin import BiasAnalysisPlugin
            from plugins.comparison_plugin import DatasetComparisonPlugin
            from plugins.evaluation_plugin import AIEvaluationPlugin
            from plugins.report_plugin import ProfessionalReportPlugin
            
            plugin_manager.register("shap", ShapExplainabilityPlugin())
            plugin_manager.register("lime", LimeExplainabilityPlugin())
            plugin_manager.register("automl", AutoMLRecommendationPlugin())
            plugin_manager.register("drift", DatasetDriftPlugin())
            plugin_manager.register("leakage", LeakageDetectionPlugin())
            plugin_manager.register("bias", BiasAnalysisPlugin())
            plugin_manager.register("comparison", DatasetComparisonPlugin())
            plugin_manager.register("evaluation", AIEvaluationPlugin())
            plugin_manager.register("report", ProfessionalReportPlugin())
        except Exception as e:
            logger.error(f"Failed to dynamically register plugins: {e}")
            
        # 8. Execute Plugins
        plugin_outputs = plugin_manager.run_all(df, memory_payload)
        memory_payload["plugins"] = plugin_outputs
        
        logger.info(f"Memory Object compiled. Version: {memory_payload['metadata']['version']}, ID: {dataset_id}")
        return memory_payload
