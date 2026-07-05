import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from utils.helpers import inject_custom_css
from federated import FederatedSimulator, FederatedClient, FederatedServer, FederatedMetrics

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Federated Learning Workspace
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset uploaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
memory = st.session_state.memory
recs = memory.get("recommendations", [])
fed_meta = memory.get("federated", {})

st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Federated Learning Workspace
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Extensible Strategy Aggregation Framework, Training Convergence, and Metrics Visualization.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 1. EDUCATIONAL WORKFLOW DIAGRAM
# ----------------------------------------------------
st.markdown("### 🕸️ Collaborative Training Pipeline")
st.markdown(
    """
    <div style='display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center; padding: 15px; 
                background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;'>
        <div style='padding: 8px 12px; background: rgba(56, 189, 248, 0.2); border: 1px solid #38BDF8; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #38BDF8;'>Dataset</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(129, 140, 248, 0.2); border: 1px solid #818CF8; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #818CF8;'>Analytics</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(129, 140, 248, 0.2); border: 1px solid #818CF8; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #818CF8;'>Recommendation Engine</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid #A78BFA; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #A78BFA;'>Federated Simulator</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid #A78BFA; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #A78BFA;'>Simulated Clients</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(244, 63, 94, 0.2); border: 1px solid #F43F5E; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #F43F5E;'>Aggregation (FedAvg)</div>
        <div style='color: #94A3B8;'>➔</div>
        <div style='padding: 8px 12px; background: rgba(52, 211, 153, 0.2); border: 1px solid #34D399; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: #34D399;'>Global Model</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

c_left, c_right = st.columns([1.1, 0.9])

with c_left:
    st.markdown("### ⚙️ Federated Configuration")
    
    # Preprocessing Pipeline Summary
    st.markdown("**🔧 Preprocessing Pipeline Summary:**")
    if recs:
        for r in recs[:2]:
            st.markdown(f"- `{r['type']}` for **{r['features']}**: {r['suggested_fix']}")
    else:
        st.markdown("- Clean numerical nulls using median strategy.")
        
    # Default recommended model from AutoML
    default_model = fed_meta.get("selected_model", "Random Forest")
    st.markdown(f"**🤖 Recommended Model (AutoML Selection)**: `{default_model}`")
    
    model_override = st.selectbox(
        "Select Model for Federated Learning:",
        ["Logistic Regression", "Random Forest", "Decision Tree"],
        index=["Logistic Regression", "Random Forest", "Decision Tree"].index(default_model)
    )

    # Strategy Selector
    selected_strategy = st.selectbox(
        "Select Aggregator Strategy:",
        ["FedAvg", "FedProx (Placeholder)", "FedNova (Placeholder)", "SCAFFOLD (Placeholder)", "FedDyn (Placeholder)", "MOON (Placeholder)"],
        index=0
    )

    rounds = st.number_input("Communication Rounds (R):", min_value=1, max_value=20, value=5, step=1)
    local_epochs = st.number_input("Local Optimization Epochs (E):", min_value=1, max_value=20, value=5, step=1)

with c_right:
    st.markdown("### 👥 Simulated Client Partitioning")
    
    num_clients = st.number_input("Number of Simulated Clients (K):", min_value=2, max_value=10, value=3, step=1)
    partition_strategy = st.selectbox(
        "Partition Strategy:",
        ["Equal Split", "Random Split", "Non-IID (Class grouped)"]
    )
    
    strategy_map = {
        "Equal Split": "equal",
        "Random Split": "random",
        "Non-IID (Class grouped)": "non-iid"
    }
    
    # Run partitioning split
    client_dfs = FederatedSimulator.partition_dataset(df, num_clients, strategy_map[partition_strategy])
    
    st.markdown("**📋 Dataset Split Distribution Summary:**")
    split_records = []
    for idx, c_df in enumerate(client_dfs):
        split_records.append({
            "Client ID": f"Simulated Client {idx + 1}",
            "Rows": len(c_df),
            "Cols": c_df.shape[1],
            "Type": partition_strategy
        })
    st.dataframe(pd.DataFrame(split_records), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 2. RUN TRAINING TRIGGER
# ----------------------------------------------------
st.markdown("### 🚀 Federated Execution Control")

btn_train = st.button("Start Federated Training", type="primary")

if btn_train:
    if model_override in ["Random Forest", "Decision Tree"]:
        st.markdown(
            """
            <div style='padding: 15px; border-left: 4px solid #EF4444; background: rgba(239, 68, 68, 0.1); border-radius: 4px; margin-bottom: 20px;'>
                <strong style='color: #F87171;'>⚠️ Model Incompatibility Notice</strong>
                <p style='margin: 6px 0 0 0; font-size: 0.9rem; color: #E2E8F0;'>
                    True FedAvg aggregation is available only for parameterized models such as Logistic Regression. 
                    Tree-based Federated Learning algorithms will be added in future phases.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif selected_strategy != "FedAvg":
        st.markdown(
            """
            <div style='padding: 15px; border-left: 4px solid #EF4444; background: rgba(239, 68, 68, 0.1); border-radius: 4px; margin-bottom: 20px;'>
                <strong style='color: #F87171;'>⚠️ Aggregator Strategy Placeholder</strong>
                <p style='margin: 6px 0 0 0; font-size: 0.9rem; color: #E2E8F0;'>
                    The aggregator strategy selected is defined as an architectural placeholder in Phase 6.1/6.2 and throws a NotImplementedError. 
                    Please use FedAvg to execute active training.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Run FedAvg Training loop
        with st.spinner("Initializing Federated training nodes..."):
            server = FederatedServer(selected_strategy)
            clients = []
            for idx, c_df in enumerate(client_dfs):
                clients.append(FederatedClient(idx + 1, c_df))
                
            for c in clients:
                server.register_client(c)
                
            metrics = FederatedMetrics()
            metrics.num_clients = num_clients
            
            # Setup targets
            target_candidates = memory.get("context", {}).get("candidate_targets", [])
            target = target_candidates[0] if target_candidates else df.columns[-1]
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            for r in range(1, rounds + 1):
                status_text.text(f"Running communication round {r} of {rounds}...")
                
                # Execute Server aggregator round
                round_res = server.run_round(model_override, target, local_epochs)
                
                # Evaluate global model
                eval_dict = server.evaluate_global_model(df, target)
                
                # Log metrics round
                metrics.log_round(
                    round_num=r,
                    client_updates=round_res["client_updates"],
                    global_acc=eval_dict["accuracy"],
                    global_loss=eval_dict["loss"],
                    agg_time_ms=round_res["aggregation_time_ms"],
                    comm_bytes=round_res["bytes_transferred"],
                    comm_latency_ms=round_res["latency_ms"],
                    round_time_ms=round_res["round_time_ms"]
                )
                
                # Yield incremental delays
                time.sleep(0.3)
                progress_bar.progress(float(r / rounds))
                
            status_text.text("Federated training rounds completed successfully!")
            
            # Cache results in session state
            st.session_state.federated_metrics = metrics.export_summary()
            
            # Recompile MemoryObject with completed metrics
            from intelligence import MemoryObject
            metrics_obj = st.session_state.metrics
            semantic_types = st.session_state.semantic_types
            domain_info = st.session_state.domain
            task_info = st.session_state.dataset_type
            
            from core.algorithm_selector import AlgorithmSelector
            algo_recs = AlgorithmSelector.recommend_algorithms(df, task_info, semantic_types, metrics_obj)
            
            # Compile new Memory Object
            st.session_state.memory = MemoryObject.compile_memory_object(
                df, st.session_state.filename, metrics_obj, semantic_types, domain_info, task_info, algo_recs
            )
            
            st.toast("Federated training completed! Metrics saved to MemoryObject.")
            st.rerun()

# ----------------------------------------------------
# 3. METRICS VISUALIZATION PANEL
# ----------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Federated Evaluation & Convergence")

cached_metrics = st.session_state.get("federated_metrics", None)

if cached_metrics is not None:
    st.subheader("🎯 Global Aggregated Metrics")
    
    # Render KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Global Accuracy</div>
                <div class="stat-val">{cached_metrics.get('global_accuracy', 0.0)*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Global Loss</div>
                <div class="stat-val" style='color: #818CF8;'>{cached_metrics.get('global_loss', 0.0):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Communication Cost</div>
                <div class="stat-val" style='color: #A78BFA;'>{cached_metrics.get('communication_cost_bytes', 0):,} B</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Time (s)</div>
                <div class="stat-val" style='color: #34D399;'>{cached_metrics.get('total_training_time_ms', 0.0)/1000.0:.2f}s</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # Plotly Charts
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 Accuracy & Loss Convergence", 
        "💾 Communication & Aggregation", 
        "👥 Client Comparison",
        "📜 Model Evolution & Timeline"
    ])
    
    hist = cached_metrics.get("convergence_history", [])
    hist_df = pd.DataFrame(hist)
    
    with chart_tab1:
        if not hist_df.empty:
            c_left, c_right = st.columns(2)
            with c_left:
                fig_acc = px.line(
                    hist_df, x="round", y="accuracy", markers=True,
                    title="Global Accuracy vs round iteration"
                )
                fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)
            with c_right:
                fig_loss = px.line(
                    hist_df, x="round", y="loss", markers=True,
                    title="Global Loss vs round iteration"
                )
                fig_loss.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No history convergence loaded.")
            
    with chart_tab2:
        if not hist_df.empty:
            c_left, c_right = st.columns(2)
            with c_left:
                # Cumulative bytes transferred
                hist_df["cumulative_bytes"] = hist_df["bytes"].cumsum()
                fig_bytes = px.line(
                    hist_df, x="round", y="cumulative_bytes", markers=True,
                    title="Cumulative bytes transferred vs round"
                )
                fig_bytes.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_bytes, use_container_width=True)
            with c_right:
                fig_agg = px.bar(
                    hist_df, x="round", y="aggregation_time_ms",
                    title="Aggregation time (ms) per round"
                )
                fig_agg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_agg, use_container_width=True)
                
    with chart_tab3:
        # Bar chart comparing final accuracy of clients
        c_accs = cached_metrics.get("accuracies", {})
        c_losses = cached_metrics.get("losses", {})
        
        client_records = []
        for c_id, acc in c_accs.items():
            client_records.append({
                "Client": f"Client {c_id}",
                "Accuracy": acc,
                "Loss": c_losses.get(c_id, 0.0)
            })
        client_df = pd.DataFrame(client_records)
        
        if not client_df.empty:
            fig_client = px.bar(
                client_df, x="Client", y="Accuracy", color="Accuracy",
                color_continuous_scale="Viridis", text_auto=".4f",
                title="Local client accuracy comparisons"
            )
            fig_client.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
            st.plotly_chart(fig_client, use_container_width=True)
            
            st.markdown("**📋 Local Client Run Records:**")
            st.dataframe(client_df, use_container_width=True)

    with chart_tab4:
        st.markdown("### 🧬 Global Model Evolution")
        
        history = cached_metrics.get("global_model_history", [])
        if history:
            latest_v = history[-1]
            st.markdown(
                f"""
                <div class="glass-card" style='padding: 15px !important; margin-bottom: 20px; border-left: 4px solid #38BDF8;'>
                    <strong style='color:#38BDF8;'>🚀 Newest Version Active: {latest_v['version']}</strong>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85rem;'>
                        <div><b>Round:</b> {latest_v['round']}</div>
                        <div><b>Accuracy:</b> {latest_v['accuracy']*100:.2f}%</div>
                        <div><b>Loss:</b> {latest_v['loss']:.4f}</div>
                        <div><b>Comm Cost:</b> {latest_v['communication_cost']:,} B</div>
                        <div><b>Agg Time:</b> {latest_v['aggregation_time_ms']:.2f}ms</div>
                        <div><b>Train Time:</b> {latest_v['training_time_ms']/1000.0:.2f}s</div>
                        <div><b>Participating Clients:</b> {latest_v['participating_clients']}</div>
                        <div><b>Aggregator:</b> {latest_v['aggregator']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("**📋 Version History Table:**")
            hist_records = []
            for item in history:
                hist_records.append({
                    "Version": item["version"],
                    "Round": item["round"],
                    "Accuracy": f"{item['accuracy']*100:.2f}%",
                    "Loss": f"{item['loss']:.4f}",
                    "Participating Clients": item["participating_clients"],
                    "Aggregator": item["aggregator"],
                    "Timestamp": item["timestamp"]
                })
            st.dataframe(pd.DataFrame(hist_records), use_container_width=True)
        else:
            st.info("No global model history recorded yet.")
            
        st.markdown("---")
        
        st.markdown("### ⏱️ Federated Timeline Activity Log")
        timeline = cached_metrics.get("training_timeline", [])
        if timeline:
            timeline_str = ""
            for item in timeline:
                event_name = item["event"]
                details = ""
                icon = "⏱️"
                if "Started" in event_name:
                    details = f"Round {item['round']}"
                    icon = "🏁"
                elif "Completed" in event_name:
                    details = f"Client {item['client']}"
                    icon = "💻"
                elif "Finished" in event_name:
                    details = f"Round {item['round']}"
                    icon = "⚖️"
                elif "Updated" in event_name:
                    details = f"Version {item['version']}"
                    icon = "🔄"
                    
                timeline_str += f"- **{item['time']}** | {icon} **{event_name}** ({details})\n"
            st.markdown(timeline_str)
        else:
            st.info("No timeline events recorded yet.")

    # Reset button to clear simulation run
    if st.button("Reset Simulation Run"):
        st.session_state.federated_metrics = None
        st.rerun()
else:
    st.info("ℹ️ No federated training has been executed yet. Click 'Start Federated Training' above to launch the simulation.")
