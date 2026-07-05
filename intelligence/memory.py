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
        
        # 5.5 Construct federated payload block
        default_model = None
        if algo_recs:
            supported_algos = ["logistic regression", "decision tree", "random forest"]
            for rec_item in algo_recs:
                rec_name = rec_item.get("algorithm", "").lower()
                if any(sa in rec_name for sa in supported_algos):
                    if "logistic" in rec_name:
                        default_model = "Logistic Regression"
                    elif "decision tree" in rec_name:
                        default_model = "Decision Tree"
                    else:
                        default_model = "Random Forest"
                    break
        if not default_model:
            default_model = "Random Forest"

        num_c = 3
        dataset_partitions = []
        try:
            rows_split = len(df) // num_c
            for c_idx in range(1, num_c + 1):
                dataset_partitions.append({
                    "client_id": c_idx,
                    "rows": rows_split,
                    "columns": df.shape[1],
                    "partition_type": "equal"
                })
        except Exception:
            pass

        # Check if we have completed metrics cached in session state
        import streamlit as st
        cached_metrics = None
        try:
            if hasattr(st, "session_state") and st.session_state is not None:
                cached_metrics = st.session_state.get("federated_metrics", None)
        except Exception:
            pass

        # Defaults or cached metrics
        clients = num_c
        rounds = 5
        strategy = "equal"
        status = "Initialized"
        client_metrics = []
        convergence_history = []
        global_acc = None
        global_loss = None
        comm_cost = None
        training_time = None

        if cached_metrics is not None:
            clients = cached_metrics.get("num_clients", clients)
            rounds = cached_metrics.get("round_number", rounds)
            status = "Training Complete"
            
            # Map client metrics
            accs = cached_metrics.get("accuracies", {})
            losses = cached_metrics.get("losses", {})
            times = cached_metrics.get("training_times_ms", {})
            sizes = cached_metrics.get("dataset_sizes", {})
            for c_id in accs.keys():
                client_metrics.append({
                    "client_id": c_id,
                    "accuracy": accs[c_id],
                    "loss": losses[c_id],
                    "training_time_ms": times.get(c_id, 0.0),
                    "samples": sizes.get(c_id, 0)
                })
                
            convergence_history = cached_metrics.get("convergence_history", [])
            global_model_history = cached_metrics.get("global_model_history", [])
            training_timeline = cached_metrics.get("training_timeline", [])
            global_acc = cached_metrics.get("global_accuracy", None)
            global_loss = cached_metrics.get("global_loss", None)
            comm_cost = cached_metrics.get("communication_cost_bytes", None)
            training_time = cached_metrics.get("total_training_time_ms", 0.0) / 1000.0 # Convert to seconds
            
        else:
            global_model_history = []
            training_timeline = []

        # Get active experiment and leaderboard summaries
        active_exp_id = "EXP_INITIAL"
        experiments_list = []
        try:
            if hasattr(st, "session_state") and st.session_state is not None:
                active_exp_id = st.session_state.get("active_experiment_id", "EXP_INITIAL")
                from federated.experiment import FederatedExperimentManager
                experiments_list = FederatedExperimentManager.get_all_experiments()
        except Exception:
            pass

        # Compile Phase 6.4 standardized telemetry schemas
        from federated.analytics import FederatedAnalyticsEngine
        from federated.contribution import FederatedContributionAnalyzer
        from federated.fairness import FederatedFairnessAppraiser
        from federated.heterogeneity import FederatedHeterogeneityAnalyzer
        from federated.explainability import FederatedExplainabilityEngine
        from federated.recommendations import FederatedRecommendationEngine

        analytics_data = {}
        contributions_data = []
        fairness_data = {}
        heterogeneity_data = {}
        explanations_data = []
        recommendations_data = []

        if cached_metrics is not None:
            analytics_data = FederatedAnalyticsEngine.analyze_metrics(cached_metrics)
            contributions_data = FederatedContributionAnalyzer.analyze_contributions(cached_metrics)
            fairness_data = FederatedFairnessAppraiser.appraise_fairness(cached_metrics)
            heterogeneity_data = FederatedHeterogeneityAnalyzer.analyze_heterogeneity(cached_metrics)
            explanations_data = FederatedExplainabilityEngine.generate_explanations(cached_metrics)
            recommendations_data = FederatedRecommendationEngine.generate_recommendations(cached_metrics)
        else:
            # Fallbacks for uninitialized runs
            analytics_data = {
                "score": 0, "rating": "N/A", "confidence": 100, 
                "evidence": ["No history records available."], 
                "explanation": "No training runs have been completed yet."
            }
            fairness_data = {
                "fairness_score": 100, "rating": "Excellent", "confidence": 100,
                "evidence": ["No clients registered yet."], "explanation": "No training runs completed yet."
            }
            heterogeneity_data = {
                "heterogeneity_score": 0, "classification": "IID", "confidence": 100,
                "evidence": ["No splits registered."], "explanation": "No training runs completed yet."
            }

        experiment_snapshots = []
        for h in convergence_history:
            experiment_snapshots.append({
                "round": h.get("round"),
                "accuracy": h.get("accuracy"),
                "loss": h.get("loss"),
                "clients": clients,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            })

        leaderboard_data = []
        if client_metrics:
            sorted_clients = sorted(client_metrics, key=lambda x: x.get("accuracy", 0.0), reverse=True)
            for rank, c in enumerate(sorted_clients, start=1):
                leaderboard_data.append({
                    "rank": rank,
                    "client_id": c["client_id"],
                    "accuracy": c["accuracy"],
                    "loss": c["loss"],
                    "samples": c["samples"],
                    "training_time_ms": c["training_time_ms"]
                })

        experiment_dict = {
            "id": active_exp_id,
            "dataset": st.session_state.filename if hasattr(st, "session_state") and st.session_state.get("filename") else "dataset.csv",
            "aggregator": "FedAvg",
            "rounds": rounds,
            "clients": clients,
            "training_time": training_time,
            "status": status
        }

        federated_dict = {
            "enabled": True,
            "simulation_mode": True,
            "algorithm": {
                "selected": "FedAvg",
                "status": "Training Complete" if cached_metrics is not None else "Infrastructure Ready",
                "implemented": True
            },
            "clients": clients,
            "rounds": rounds,
            "partition_strategy": strategy,
            "selected_model": default_model,
            "dataset_partitions": dataset_partitions,
            "global_model": None,
            "client_metrics": client_metrics,
            "convergence_history": convergence_history,
            "global_model_history": global_model_history,
            "training_timeline": training_timeline,
            "experiment_snapshots": experiment_snapshots,
            "leaderboard": leaderboard_data,
            "experiment": experiment_dict,
            "experiments_history": experiments_list,
            "analytics": analytics_data,
            "contributions": contributions_data,
            "fairness": fairness_data,
            "heterogeneity": heterogeneity_data,
            "explanations": explanations_data,
            "recommendations": recommendations_data,
            "global_metrics": {
                "accuracy": global_acc,
                "loss": global_loss,
                "communication_cost": comm_cost,
                "training_time": training_time
            },
            "aggregation": {
                "algorithm": "FedAvg",
                "supported_models": [
                    "Logistic Regression"
                ],
                "future_models": [
                    "Decision Tree",
                    "Random Forest",
                    "FedProx",
                    "FedNova",
                    "SCAFFOLD",
                    "FedDyn",
                    "MOON"
                ]
            },
            "status": status
        }

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
            "plugins": {}, # Future plugin outputs will append here
            "federated": federated_dict
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
