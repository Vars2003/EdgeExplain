import time
from typing import Generator, List, Dict, Any, Tuple
from utils.logger import get_logger
from ai.model_manager import model_manager
from ai.retriever import GraphRetriever
from ai.budget import TokenBudgetManager
from ai.prompt_builder import PromptBuilder
from ai.safety import SafetyFilter
from ai.citations import CitationEngine
from ai.streaming import TokenStreamer

logger = get_logger("ai.llm")

class OfflineLLMEngine:
    """
    Coordinates conversational requests.
    Validates safety, ranks context nodes, compresses token budgets,
    and runs local LLM generation or fires the grounded rule-based responder.
    """

    @staticmethod
    def answer_question(question: str, memory_obj: Dict[str, Any], persona: str = "Data Analyst", stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Processes query and yields response dictionaries with progress status:
        Yields: { "type": "chunk", "text": "..." }
        Final yield: { "type": "final", "content": "Full response", "metrics": {...}, "citations": [...] }
        """
        logger.info(f"Answering query: '{question}' using persona: {persona}")
        start_time = time.time()
        
        schema_cols = list(memory_obj.get("schema", {}).keys())
        
        # 1. Safety Guardrail check
        if not SafetyFilter.is_query_safe(question, schema_cols):
            msg = "This question is outside the scope of the uploaded dataset. I can only answer questions related to your dataset's schema, quality, statistics, recommendations, and ML modeling suitabilities."
            yield {"type": "chunk", "text": msg}
            yield {
                "type": "final",
                "content": msg,
                "metrics": {
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "retrieved_context_count": 0,
                    "fallback_used": True,
                    "model_name": "SafetyGuardrail"
                },
                "citations": ["Safety Layer Check"]
            }
            return

        # 2. Ranked Context Retrieval
        # Max 5 context items
        retrieved = GraphRetriever.retrieve_relevant_context(question, memory_obj, max_items=5)
        
        # 3. Token Budget Compression
        max_context_tokens = 2048 # Adjust as needed
        context_str, citations = TokenBudgetManager.compress_context(retrieved, max_tokens=max_context_tokens)
        citation_labels = CitationEngine.compile_citations(retrieved)
        
        # 4. Check LLM availability
        provider = model_manager.get_provider()
        available_models = provider.get_installed_models()
        
        # Determine if we should use local model or run fallback
        active_model = None
        selected_model = None
        is_fallback_mode = True
        temp = 0.7
        
        try:
            import streamlit as st
            # Check if session_state is accessible
            if hasattr(st, "session_state") and st.session_state is not None:
                selected_model = st.session_state.get("ai_selected_model", None)
                is_fallback_mode = st.session_state.get("ai_fallback_mode", True)
                temp = st.session_state.get("ai_temperature", 0.7)
        except Exception:
            pass
            
        use_llm = False
        if not is_fallback_mode and selected_model and selected_model in available_models:
            active_model = selected_model
            use_llm = True
            
        full_content = ""
        fallback_triggered = not use_llm
        
        # 5. Core Generation Pipeline
        if use_llm:
            try:
                system_prompt = PromptBuilder.get_system_prompt(persona)
                user_prompt = PromptBuilder.build_user_prompt(question, context_str)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Retrieve temperature from state (already populated safely)
                
                logger.info(f"Invoking active local model: {active_model}")
                token_gen = provider.generate_chat(active_model, messages, temperature=temp, stream=stream)
                
                for token in token_gen:
                    full_content += token
                    yield {"type": "chunk", "text": token}
                    
            except Exception as e:
                logger.exception(f"Local LLM runtime crashed: {e}. Defaulting to rule fallback.")
                fallback_triggered = True
                
        if fallback_triggered:
            logger.info("Executing offline grounded rule-based responder fallback.")
            response_text = OfflineLLMEngine.generate_rule_based_response(question, memory_obj)
            
            # Simulate streaming type writing
            word_stream = TokenStreamer.simulate_streaming(response_text, delay=0.005)
            for chunk in word_stream:
                full_content += chunk
                yield {"type": "chunk", "text": chunk}
                
        # 6. Final return package
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "type": "final",
            "content": full_content,
            "metrics": {
                "response_time_ms": elapsed_ms,
                "retrieved_context_count": len(retrieved),
                "fallback_used": fallback_triggered,
                "model_name": active_model if not fallback_triggered else "GroundedRuleEngine"
            },
            "citations": citation_labels
        }

    @staticmethod
    def generate_rule_based_response(question: str, memory_obj: Dict[str, Any]) -> str:
        """
        Determines structured answers based on query keywords and cached context.
        """
        prompt_lower = question.lower()
        context = memory_obj.get("context", {})
        summaries = memory_obj.get("summaries", {})
        recs = memory_obj.get("recommendations", {})
        algos = memory_obj.get("algorithms", [])
        
        # Keyword 0: Federated Learning
        if any(kw in prompt_lower for kw in ["federated", "round", "global accuracy", "client", "aggregation", "communication", "bytes", "latency", "version", "improved", "timeline", "history", "participate", "leaderboard", "compare", "experiment", "export", "fastest"]):
            fed = memory_obj.get("federated", {})
            if fed and fed.get("enabled"):
                algo_selected = fed.get("algorithm", {}).get("selected", "FedAvg")
                rounds = fed.get("rounds", 0)
                clients = fed.get("clients", 0)
                global_metrics = fed.get("global_metrics", {})
                
                client_metrics = fed.get("client_metrics", [])
                best_client_str = "N/A"
                if client_metrics:
                    best_c = max(client_metrics, key=lambda x: x.get("accuracy", 0.0))
                    best_client_str = f"Client {best_c.get('client_id')} (Accuracy: {best_c.get('accuracy')*100:.2f}%)"
                
                history = fed.get("global_model_history", [])
                timeline = fed.get("training_timeline", [])
                
                # Check specific leaderboard query
                if any(kw in prompt_lower for kw in ["leaderboard", "best client", "best-performing client"]):
                    leaderboard = fed.get("leaderboard", [])
                    if leaderboard:
                        bullets = []
                        for c in leaderboard:
                            bullets.append(f"- **Rank {c['rank']}**: Client {c['client_id']} (Accuracy: {c['accuracy']*100:.2f}%, Loss: {c['loss']:.4f}, Samples: {c['samples']}, Time: {c['training_time_ms']:.1f}ms)")
                        lead_text = "\n".join(bullets)
                        return f"### Client Leaderboard Rankings\nHere are the local client standings:\n\n{lead_text}"
                    return f"No client leaderboard stats recorded yet. Best client: {best_client_str}."

                # Check comparison or export query
                if any(kw in prompt_lower for kw in ["compare", "experiment", "export"]):
                    exps = fed.get("experiments_history", [])
                    active_exp = fed.get("experiment", {})
                    
                    if "export" in prompt_lower:
                        return (
                            f"### Session Export Availability\n"
                            f"You can export active federated training metrics to a JSON format offline via the Download Session JSON action button located in the Overview panel.\n\n"
                            f"**Active Experiment Details**:\n"
                            f"- ID: `{active_exp.get('id', 'N/A')}`\n"
                            f"- Dataset: `{active_exp.get('dataset', 'N/A')}`\n"
                            f"- Aggregator: `{active_exp.get('aggregator', 'N/A')}`"
                        )
                        
                    if exps:
                        bullets = []
                        for e in exps:
                            bullets.append(f"- **{e['id']}** (Model: {e['model']}, Acc: {e['final_accuracy']*100:.2f}%, Loss: {e['final_loss']:.4f}, Rounds: {e['rounds']}, Duration: {e['duration']/1000.0:.2f}s)")
                        exps_text = "\n".join(bullets)
                        
                        comparison_hint = ""
                        if len(exps) >= 2:
                            comparison_hint = f"\n\nTo compare experiments side-by-side, you can select them in the **Experiments** tab of the Control Center dashboard."
                        return f"### Registered Experiments History\nHere are the registered experiment runs:\n\n{exps_text}{comparison_hint}"
                    return f"Currently executing active experiment: `{active_exp.get('id', 'N/A')}`. No other historical experiments registered."

                # Check specific round achievements
                if "highest accuracy" in prompt_lower or "achieved highest accuracy" in prompt_lower:
                    if history:
                        best_round = max(history, key=lambda x: x.get("accuracy", 0.0))
                        return f"Round {best_round['round']} achieved the highest global accuracy of **{best_round['accuracy']*100:.2f}%** (Loss: {best_round['loss']:.4f})."
                    return "No training round history available to analyze."
                    
                if "fastest" in prompt_lower or "finished fastest" in prompt_lower:
                    exps = fed.get("experiments_history", [])
                    if exps:
                        fastest = min(exps, key=lambda x: x.get("duration", float('inf')))
                        return f"Experiment **{fastest['id']}** finished fastest, completing training in **{fastest['duration']/1000.0:.2f} seconds** (Final Accuracy: {fastest['final_accuracy']*100:.2f}%)."
                    return "No historical experiments recorded to compare durations."

                # Check specific timeline query
                if "timeline" in prompt_lower:
                    if timeline:
                        bullets = []
                        for item in timeline[:15]:
                            bullets.append(f"- **{item['time']}**: {item['event']}")
                        timeline_text = "\n".join(bullets)
                        return f"### Federated Training Timeline Activity Log\nHere are the logged events:\n\n{timeline_text}"
                    return "No timeline events have been logged yet."
                    
                # Check round detail query
                for r_idx in range(1, 21):
                    if f"round {r_idx}" in prompt_lower or f"round_{r_idx}" in prompt_lower:
                        match_history = [h for h in history if h.get("round") == r_idx]
                        if match_history:
                            h = match_history[0]
                            return (
                                f"### Federated Round {r_idx} Status Details\n"
                                f"- **Global Version**: `{h['version']}`\n"
                                f"- **Global Accuracy**: `{h['accuracy']*100:.2f}%`\n"
                                f"- **Global Loss**: `{h['loss']:.4f}`\n"
                                f"- **Aggregation Time**: `{h['aggregation_time_ms']:.2f} ms`\n"
                                f"- **Training Time**: `{h['training_time_ms']/1000.0:.2f} s`\n"
                                f"- **Participating Clients**: `{h['participating_clients']}`\n"
                                f"- **Aggregator**: `{h['aggregator']}`"
                            )
                        return f"No global version records found matching Round {r_idx}."
                        
                # Check version details query
                if "version" in prompt_lower:
                    if history:
                        latest = history[-1]
                        return (
                            f"### Latest Global Model Version\n"
                            f"**Active Version**: `{latest['version']}` (Round {latest['round']})\n"
                            f"- **Accuracy**: `{latest['accuracy']*100:.2f}%`\n"
                            f"- **Loss**: `{latest['loss']:.4f}`\n"
                            f"- **Participating Clients**: `{latest['participating_clients']}`\n"
                            f"- **Aggregator**: `{latest['aggregator']}`"
                        )
                    return "No version records have been initialized yet."
                    
                # Check improvement details query
                if "improved" in prompt_lower:
                    if len(history) > 1:
                        first = history[0]
                        latest = history[-1]
                        acc_diff = (latest["accuracy"] - first["accuracy"]) * 100
                        loss_diff = latest["loss"] - first["loss"]
                        return (
                            f"### Global Model Performance Improvement\n"
                            f"Over `{len(history)}` rounds of collaborative training:\n\n"
                            f"- **Accuracy**: `{first['accuracy']*100:.2f}%` ➔ `{latest['accuracy']*100:.2f}%` ({'+' if acc_diff >= 0 else ''}{acc_diff:.2f}% change)\n"
                            f"- **Loss**: `{first['loss']:.4f}` ➔ `{latest['loss']:.4f}` ({'' if loss_diff <= 0 else '+'}{loss_diff:.4f} change)"
                        )
                    return "Need at least 2 training rounds of history data to calculate performance improvements."
                
                # Check history details query
                if "history" in prompt_lower:
                    if history:
                        rows = []
                        for h in history:
                            rows.append(f"| {h['version']} | {h['round']} | {h['accuracy']*100:.2f}% | {h['loss']:.4f} |")
                        table_str = "\n".join(rows)
                        return (
                            f"### Global Model Evolution History\n"
                            f"| Version | Round | Accuracy | Loss |\n"
                            f"| :--- | :--- | :--- | :--- |\n"
                            f"{table_str}"
                        )
                    return "No history records compiled yet."

                return (
                    f"### Federated Learning Summary\n"
                    f"Here is the current state of collaborative training from the Memory Object:\n\n"
                    f"- **Aggregation Algorithm**: `{algo_selected}`\n"
                    f"- **Rounds Completed**: `{rounds}`\n"
                    f"- **Client Count (K)**: `{clients}`\n"
                    f"- **Global Accuracy**: `{f'{global_metrics.get('accuracy', 0.0)*100:.2f}%' if global_metrics.get('accuracy') is not None else 'N/A'}`\n"
                    f"- **Global Loss**: `{f'{global_metrics.get('loss', 0.0):.4f}' if global_metrics.get('loss') is not None else 'N/A'}`\n"
                    f"- **Best Performing Client**: `{best_client_str}`\n"
                    f"- **Total Communication Cost**: `{f'{global_metrics.get('communication_cost', 0):,} bytes' if global_metrics.get('communication_cost') is not None else '0 bytes'}`\n"
                    f"- **Total Training Time**: `{f'{global_metrics.get('training_time', 0.0):.2f} seconds' if global_metrics.get('training_time') is not None else '0.0 seconds'}`"
                )
            else:
                return "Federated Learning has not been initialized or executed yet. Please navigate to the Federated Learning Workspace in the sidebar and trigger collaborative training."

        # Keyword 1: what is this dataset / summary
        if any(kw in prompt_lower for kw in ["what is this dataset", "overview", "summary", "about"]):
            return (
                f"### Dataset Focus: {context.get('dataset_name')}\n"
                f"This is a **{context.get('dataset_type', 'Generic')}** dataset identified with the **{context.get('domain', 'Unknown')}** domain.\n\n"
                f"**Dimensions & Size**:\n"
                f"- Total Observations: {context.get('row_count', 0):,}\n"
                f"- Columns: {context.get('column_count', 0)}\n"
                f"- Memory Footprint: {context.get('memory_readable', 'N/A')}\n\n"
                f"**Health Diagnostics**:\n"
                f"- Data Quality Score: {context.get('quality_overview', {}).get('quality_score', 0.0):.1f}%\n"
                f"- ML Modeling Readiness: {context.get('quality_overview', {}).get('ml_readiness_score', 0.0):.1f}%\n\n"
                f"Suggested ML Task: **{context.get('recommended_ml_task')}**."
            )
            
        # Keyword 2: normalize / scale / standard
        if any(kw in prompt_lower for kw in ["normalize", "scale", "standardize", "range"]):
            scaler_recs = [r for r in recs if r["type"] == "Standardization"]
            if scaler_recs:
                r = scaler_recs[0]
                return (
                    f"### Scaling & Standardization recommendation\n"
                    f"**Suggested Fix**: {r['suggested_fix']}\n\n"
                    f"**Why**: {r['reason']}\n\n"
                    f"**Evidence**: {', '.join(r['evidence'])}\n\n"
                    f"**Alternatives**: {', '.join(r['alternatives'])}"
                )
            return "No standardization or feature scaling recommendations are active for this dataset. This indicates that the numeric features are already in uniform range spreads."

        # Keyword 3: algorithm / model
        if any(kw in prompt_lower for kw in ["algorithm", "model", "suitability", "recommend"]):
            algo_bullets = "\n".join([f"- **{a['algorithm']}** (Compatibility: {a['compatibility_score']}%) — Pros: {a['pros'][0]}. Cons: {a['cons'][0]}." for a in algos[:3]])
            return (
                f"### Model Compatibility Recommendations\n"
                f"Based on the dataset type ({context.get('dataset_type')}) and ML task category ({context.get('recommended_ml_task')}), the following algorithms are recommended:\n\n"
                f"{algo_bullets}\n\n"
                f"Select these algorithms to train classifiers or regressor benchmarks."
            )
            
        # Keyword 4: missing / null
        if any(kw in prompt_lower for kw in ["missing", "null", "empty", "nan"]):
            impute_recs = [r for r in recs if r["type"] in ["Imputation", "Feature Removal"]]
            if impute_recs:
                bullets = "\n".join([f"- Feature `{', '.join(r['features'])}`: {r['suggested_fix']} (Why: {r['reason']}). Alternative: {', '.join(r['alternatives'])}" for r in impute_recs])
                return (
                    f"### Missing Data Analysis\n"
                    f"Total null percentage is **{context.get('quality_overview', {}).get('missing_cells_pct', 0.0):.2f}%**.\n\n"
                    f"We advise addressing missing variables:\n"
                    f"{bullets}"
                )
            return "Excellent. Zero missing values or null cells were found in the dataset structure."

        # Default fallback response: Generic context report
        risk_bullets = ""
        risks = context.get("key_risks", [])
        if risks:
            risk_bullets = "\n".join([f"- **{r['risk']}**: {r['detail']}" for r in risks])
            
        return (
            f"### Grounded Dataset Health Summary\n"
            f"I am EdgeExplain, your offline AI Data Scientist. I can answer questions about column schema, missing values, outliers, correlations, and ML algorithms.\n\n"
            f"- **Data Quality Score**: {context.get('quality_overview', {}).get('quality_score', 0.0):.1f}%\n"
            f"- **ML Readiness**: {context.get('quality_overview', {}).get('ml_readiness_score', 0.0):.1f}%\n"
            f"- **Row count**: {context.get('row_count', 0):,}\n\n"
            f"**Flagged Structural Vulnerabilities**:\n"
            f"{risk_bullets if risk_bullets else '- No high-severity risks identified.'}\n\n"
            f"Please specify a column name or preprocessing topic (e.g. 'normalization' or 'missing values') to inspect deeper."
        )
