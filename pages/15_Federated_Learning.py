import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import datetime
import plotly.express as px
from utils.helpers import inject_custom_css
from federated import (
    FederatedSimulator,
    FederatedClient,
    FederatedServer,
    FederatedMetrics,
    FederatedExperimentManager,
    generate_topology_chart,
    FederatedCommunicationMonitor,
    FederatedReplayController
)

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Federated Learning Control Center
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

# Setup active experiment session ID if not set
if "active_experiment_id" not in st.session_state:
    st.session_state.active_experiment_id = "N/A"

st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Federated Learning Control Center
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Experiment Manager, Client Leaderboards, Network Topologies, and Session Exports.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 1. SETUP CONFIGURATIONS
# ----------------------------------------------------
c_left, c_right = st.columns([1.1, 0.9])

with c_left:
    st.markdown("### ⚙️ Control Center Setup")
    
    default_model = fed_meta.get("selected_model", "Random Forest")
    model_override = st.selectbox(
        "Select model for Federated training run:",
        ["Logistic Regression", "Random Forest", "Decision Tree"],
        index=["Logistic Regression", "Random Forest", "Decision Tree"].index(default_model)
    )

    selected_strategy = st.selectbox(
        "Select Aggregation Strategy:",
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

btn_train = st.button("Start Federated Training Session", type="primary")

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
        # Create experiment session before training round loop
        exp_id = FederatedExperimentManager.create_experiment(
            dataset_name=st.session_state.filename or "dataset.csv",
            algorithm="FedAvg",
            aggregator=selected_strategy,
            rounds=rounds,
            clients=num_clients,
            model=model_override
        )
        st.session_state.active_experiment_id = exp_id
        
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
            
            # Compile duration, final accuracy, and loss
            duration = float(metrics.total_training_time_ms)
            final_acc = float(metrics.global_accuracy)
            final_loss = float(metrics.global_loss)
            
            # Compile snapshots and leaderboard
            snapshots = []
            for h in metrics.convergence_history:
                snapshots.append({
                    "round": h.get("round"),
                    "accuracy": h.get("accuracy"),
                    "loss": h.get("loss"),
                    "clients": num_clients,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                })
                
            leaderboard_data = []
            c_accs = metrics.local_accuracies
            c_losses = metrics.local_losses
            c_times = metrics.training_times_ms
            c_sizes = metrics.dataset_sizes
            for c_id, acc in c_accs.items():
                leaderboard_data.append({
                    "client_id": c_id,
                    "accuracy": acc,
                    "loss": c_losses.get(c_id, 0.0),
                    "samples": c_sizes.get(c_id, 0),
                    "training_time_ms": c_times.get(c_id, 0.0)
                })
                
            FederatedExperimentManager.update_experiment(
                exp_id=exp_id,
                duration=duration,
                final_accuracy=final_acc,
                final_loss=final_loss,
                snapshots=snapshots,
                leaderboard=leaderboard_data
            )
            
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
# 3. METRICS VISUALIZATION PANEL (9 TABS)
# ----------------------------------------------------
st.markdown("### 📊 Federated Evaluation & Convergence")

cached_metrics = st.session_state.get("federated_metrics", None)

if cached_metrics is not None:
    t_overview, t_clients, t_training, t_comm, t_evol, t_timeline, t_replay, t_exp, t_leader, t_analytics, t_contrib, t_fairness, t_hetero, t_explain, t_recommend = st.tabs([
        "👁️ Overview",
        "👥 Clients",
        "📈 Training",
        "💾 Communication",
        "🔄 Model Evolution",
        "⏱️ Timeline",
        "📼 Replay",
        "🧪 Experiments",
        "🏆 Leaderboard",
        "📊 Analytics",
        "🍕 Contribution",
        "⚖️ Fairness",
        "🧬 Heterogeneity",
        "💬 Explainability",
        "💡 Recommendations"
    ])
    
    hist = cached_metrics.get("convergence_history", [])
    hist_df = pd.DataFrame(hist)
    
    # 1. OVERVIEW TAB
    with t_overview:
        st.subheader("🎯 Global Aggregated Metrics")
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
            
        st.markdown(f"**Experiment ID**: `{st.session_state.active_experiment_id}` | **Aggregator**: `FedAvg` | **Clients**: `{cached_metrics.get('num_clients', 0)}`")
        st.markdown("---")
        
        # Plotly Topology Chart
        topology_fig = generate_topology_chart(cached_metrics.get('num_clients', 3))
        st.plotly_chart(topology_fig, use_container_width=True)
        
        # Session Export Download Button
        export_payload = {
            "experiment_metadata": {
                "id": st.session_state.active_experiment_id,
                "dataset": st.session_state.filename,
                "aggregator": "FedAvg",
                "rounds_completed": cached_metrics.get("round_number"),
                "clients_registered": cached_metrics.get("num_clients"),
                "total_time_ms": cached_metrics.get("total_training_time_ms"),
                "global_accuracy": cached_metrics.get("global_accuracy"),
                "global_loss": cached_metrics.get("global_loss")
            },
            "client_metrics": [
                {
                    "client_id": c_id,
                    "accuracy": cached_metrics.get("accuracies", {}).get(c_id),
                    "loss": cached_metrics.get("losses", {}).get(c_id),
                    "samples": cached_metrics.get("dataset_sizes", {}).get(c_id),
                    "training_time_ms": cached_metrics.get("training_times_ms", {}).get(c_id)
                } for c_id in cached_metrics.get("accuracies", {}).keys()
            ],
            "communication_statistics": {
                "bytes_transferred": cached_metrics.get("communication_cost_bytes"),
                "latency_ms": cached_metrics.get("communication_latency_ms")
            },
            "timeline": cached_metrics.get("training_timeline", []),
            "global_model_history": cached_metrics.get("global_model_history", [])
        }
        
        st.download_button(
            label="💾 Download Session JSON",
            data=json.dumps(export_payload, indent=4),
            file_name=f"fl_session_{st.session_state.active_experiment_id}.json",
            mime="application/json"
        )
        
    # 2. CLIENTS MONITOR TAB
    with t_clients:
        st.subheader("👥 Live Client Performance Grid")
        
        c_accs = cached_metrics.get("accuracies", {})
        c_losses = cached_metrics.get("losses", {})
        c_times = cached_metrics.get("training_times_ms", {})
        c_sizes = cached_metrics.get("dataset_sizes", {})
        
        for c_id in c_accs.keys():
            st.markdown(
                f"""
                <div class="glass-card" style='padding: 15px !important; margin-bottom: 12px; border-left: 4px solid #818CF8;'>
                    <strong style='color:#E2E8F0;'>Simulated Client {c_id}</strong> - <span style='color:#34D399; font-weight:600;'>Completed</span>
                    <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 8px; font-size: 0.85rem;'>
                        <div><b>Accuracy:</b> {c_accs[c_id]*100:.2f}%</div>
                        <div><b>Loss:</b> {c_losses.get(c_id, 0.0):.4f}</div>
                        <div><b>Samples:</b> {c_sizes.get(c_id, 0)}</div>
                        <div><b>Train Time:</b> {c_times.get(c_id, 0.0):.2f}ms</div>
                        <div><b>Comm Cost:</b> {cached_metrics.get('communication_cost_bytes', 0) // cached_metrics.get('num_clients', 1):,} B</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # 3. TRAINING TAB
    with t_training:
        if not hist_df.empty:
            c_left, c_right = st.columns(2)
            with c_left:
                fig_acc = px.line(hist_df, x="round", y="accuracy", markers=True, title="Global Accuracy Convergence")
                fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)
            with c_right:
                fig_loss = px.line(hist_df, x="round", y="loss", markers=True, title="Global Loss Convergence")
                fig_loss.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No training metrics generated.")
            
    # 4. COMMUNICATION TAB
    with t_comm:
        st.subheader("💾 Communication Costs & Bandwidth Analytics")
        if not hist_df.empty:
            c_left, c_right = st.columns(2)
            with c_left:
                hist_df["cumulative_bytes"] = hist_df["bytes"].cumsum()
                fig_bytes = px.line(hist_df, x="round", y="cumulative_bytes", markers=True, title="Cumulative Bytes Transferred")
                fig_bytes.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_bytes, use_container_width=True)
            with c_right:
                fig_agg = px.bar(hist_df, x="round", y="aggregation_time_ms", title="Aggregation Time (ms)")
                fig_agg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_agg, use_container_width=True)
                
            # Live stats monitor
            latest_h = hist[-1]
            monitor_stats = FederatedCommunicationMonitor.compile_monitor_stats(
                latest_h.get("round", 1),
                cached_metrics.get("num_clients", 3),
                latest_h.get("bytes", 0),
                latest_h.get("latency_ms", 0.0),
                latest_h.get("training_time_ms", 1.0)
            )
            st.markdown(
                f"""
                <div class="glass-card" style='padding: 15px !important; margin-top: 15px;'>
                    <strong style='color:#A78BFA;'>🛰️ Live Telemetry Metrics</strong>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85rem;'>
                        <div><b>Est. Bandwidth:</b> {monitor_stats['bandwidth_mbps']:.4f} Mbps</div>
                        <div><b>Upload Count:</b> {monitor_stats['upload_count']}</div>
                        <div><b>Download Count:</b> {monitor_stats['download_count']}</div>
                        <div><b>Latency Ratio:</b> {monitor_stats['latency_ratio']*100:.2f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # 5. MODEL EVOLUTION TAB
    with t_evol:
        st.subheader("🧬 Global Model Version History")
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
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            hist_records = []
            for item in history:
                hist_records.append({
                    "Version": item["version"],
                    "Round": item["round"],
                    "Accuracy": f"{item['accuracy']*100:.2f}%",
                    "Loss": f"{item['loss']:.4f}",
                    "Participating Clients": item["participating_clients"],
                    "Timestamp": item["timestamp"]
                })
            st.dataframe(pd.DataFrame(hist_records), use_container_width=True)
            
    # 6. TIMELINE TAB
    with t_timeline:
        st.subheader("⏱️ Chronological activity timeline log")
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
            
    # 7. REPLAY TAB
    with t_replay:
        st.subheader("📼 Historical Round Replay Panel")
        history = cached_metrics.get("global_model_history", [])
        if history:
            round_list = [h.get("round") for h in history]
            selected_round = st.selectbox("Select Round to Replay:", round_list)
            
            snapshot = FederatedReplayController.load_round_snapshot(selected_round, history)
            if snapshot:
                st.markdown(
                    f"""
                    <div class="glass-card" style='padding: 15px !important; border-left: 4px solid #F59E0B;'>
                        <strong style='color:#F59E0B;'>Round {selected_round} Snapshot Loaded</strong>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85rem;'>
                            <div><b>Version:</b> {snapshot['version']}</div>
                            <div><b>Accuracy:</b> {snapshot['accuracy']*100:.2f}%</div>
                            <div><b>Loss:</b> {snapshot['loss']:.4f}</div>
                            <div><b>Participating Clients:</b> {snapshot['participating_clients']}</div>
                            <div><b>Comm Cost:</b> {snapshot['communication_cost']:,} B</div>
                            <div><b>Agg Time:</b> {snapshot['aggregation_time_ms']:.2f}ms</div>
                            <div><b>Train Time:</b> {snapshot['training_time_ms']/1000.0:.2f}s</div>
                            <div><b>Aggregator:</b> {snapshot['aggregator']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No training history available to replay.")
            
    # 8. EXPERIMENTS TAB
    with t_exp:
        st.subheader("🧪 Federated Experiment Manager")
        
        experiments = FederatedExperimentManager.get_all_experiments()
        if len(experiments) >= 2:
            st.markdown("#### Compare Experiments")
            c_left, c_right = st.columns(2)
            with c_left:
                exp_a = st.selectbox("Select Experiment A:", [e["id"] for e in experiments], index=0)
            with c_right:
                exp_b = st.selectbox("Select Experiment B:", [e["id"] for e in experiments], index=1)
                
            e_a = FederatedExperimentManager.get_experiment(exp_a)
            e_b = FederatedExperimentManager.get_experiment(exp_b)
            
            if e_a and e_b:
                comp_records = [
                    {"Metric": "Accuracy", e_a["id"]: f"{e_a['final_accuracy']*100:.2f}%", e_b["id"]: f"{e_b['final_accuracy']*100:.2f}%"},
                    {"Metric": "Loss", e_a["id"]: f"{e_a['final_loss']:.4f}", e_b["id"]: f"{e_b['final_loss']:.4f}"},
                    {"Metric": "Duration (s)", e_a["id"]: f"{e_a['duration']/1000.0:.2f}s", e_b["id"]: f"{e_b['duration']/1000.0:.2f}s"},
                    {"Metric": "Clients", e_a["id"]: e_a["clients"], e_b["id"]: e_b["clients"]},
                    {"Metric": "Rounds", e_a["id"]: e_a["rounds"], e_b["id"]: e_b["rounds"]}
                ]
                st.dataframe(pd.DataFrame(comp_records), use_container_width=True)
                
                # Plotly comparison bar chart
                fig_comp = px.bar(
                    pd.DataFrame([
                        {"Experiment": e_a["id"], "Accuracy": e_a["final_accuracy"]},
                        {"Experiment": e_b["id"], "Accuracy": e_b["final_accuracy"]}
                    ]),
                    x="Experiment", y="Accuracy", color="Experiment",
                    title="Accuracy Comparison Chart"
                )
                fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Run at least 2 training experiments to unlock side-by-side session comparison reports.")
            
        st.markdown("**📋 Experiment History Logs:**")
        exp_records = []
        for e in experiments:
            exp_records.append({
                "ID": e["id"],
                "Model": e["model"],
                "Clients": e["clients"],
                "Rounds": e["rounds"],
                "Accuracy": f"{e['final_accuracy']*100:.2f}%",
                "Loss": f"{e['final_loss']:.4f}",
                "Time": f"{e['duration']/1000.0:.2f}s"
            })
        st.dataframe(pd.DataFrame(exp_records), use_container_width=True)
        
    # 9. LEADERBOARD TAB
    with t_leader:
        st.subheader("🏆 Client Leaderboard")
        
        # Pull leaderboard data sorted descending
        sorted_leaderboard = []
        c_accs = cached_metrics.get("accuracies", {})
        c_losses = cached_metrics.get("losses", {})
        c_times = cached_metrics.get("training_times_ms", {})
        c_sizes = cached_metrics.get("dataset_sizes", {})
        
        for c_id, acc in c_accs.items():
            sorted_leaderboard.append({
                "Client ID": f"Client {c_id}",
                "Accuracy": acc,
                "Loss": c_losses.get(c_id, 0.0),
                "Samples": c_sizes.get(c_id, 0),
                "Train Time": c_times.get(c_id, 0.0)
            })
            
        sorted_leaderboard = sorted(sorted_leaderboard, key=lambda x: x.get("Accuracy", 0.0), reverse=True)
        
        if sorted_leaderboard:
            # Highlight best performer
            best = sorted_leaderboard[0]
            st.markdown(
                f"""
                <div style='padding:15px; border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.1); border-radius: 4px; margin-bottom: 20px;'>
                    <strong style='color:#34D399;'>🥇 Best Performing Client: {best['Client ID']}</strong>
                    <p style='margin: 4px 0 0 0; font-size: 0.9rem; color: #E2E8F0;'>
                        Achieved highest validation accuracy of <b>{best['Accuracy']*100:.2f}%</b> in local fit runs.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            lead_df = []
            for rank, item in enumerate(sorted_leaderboard, start=1):
                lead_df.append({
                    "Rank": rank,
                    "Client": item["Client ID"],
                    "Accuracy": f"{item['Accuracy']*100:.2f}%",
                    "Loss": f"{item['Loss']:.4f}",
                    "Samples Count": item["Samples"],
                    "Training Duration (ms)": f"{item['Train Time']:.2f}ms"
                })
            st.dataframe(pd.DataFrame(lead_df), use_container_width=True)

    # 10. ANALYTICS TAB
    with t_analytics:
        from federated.analytics import FederatedAnalyticsEngine
        analytics_res = FederatedAnalyticsEngine.analyze_metrics(cached_metrics)
        render_standard_schema_section(
            title="Scientific Training Analytics Summary",
            score=analytics_res["score"],
            rating=analytics_res["rating"],
            confidence=analytics_res["confidence"],
            evidence=analytics_res["evidence"],
            explanation=analytics_res["explanation"],
            score_label="Validation Score"
        )
        
    # 11. CONTRIBUTION TAB
    with t_contrib:
        st.subheader("🍕 Client Contribution Analysis")
        from federated.contribution import FederatedContributionAnalyzer
        contributions = FederatedContributionAnalyzer.analyze_contributions(cached_metrics)
        
        if contributions:
            # Render overall score card (sum check)
            total_sum = sum(c["contribution"] for c in contributions)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="stat-card" style='border-left: 4px solid #38BDF8;'>
                        <div class="stat-label">Total Normalized Weight</div>
                        <div class="stat-val">{total_sum:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c2:
                # Plotly Pie Chart
                fig_pie = px.pie(
                    pd.DataFrame(contributions),
                    names="client", values="contribution",
                    title="Normalized client contributions distribution"
                )
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # Render contributions details
            for c in contributions:
                st.markdown(
                    f"""
                    <div class="glass-card" style='padding: 15px !important; margin-bottom: 12px; border-left: 4px solid #38BDF8;'>
                        <strong style='color:#38BDF8;'>{c['client']}</strong>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px; font-size: 0.85rem;'>
                            <div><b>Contribution:</b> {c['contribution']:.1f}%</div>
                            <div><b>Samples:</b> {c['samples']}</div>
                            <div><b>Local Accuracy:</b> {c['accuracy']:.2f}%</div>
                            <div><b>Confidence:</b> {c['confidence']}%</div>
                        </div>
                        <p style='margin: 8px 0 0 0; font-size: 0.85rem; color:#94A3B8;'>{c['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No contribution data generated.")
            
    # 12. FAIRNESS TAB
    with t_fairness:
        from federated.fairness import FederatedFairnessAppraiser
        fairness_res = FederatedFairnessAppraiser.appraise_fairness(cached_metrics)
        render_standard_schema_section(
            title="Federated System Fairness Parity",
            score=fairness_res["fairness_score"],
            rating=fairness_res["rating"],
            confidence=fairness_res["confidence"],
            evidence=fairness_res["evidence"],
            explanation=fairness_res["explanation"],
            score_label="Fairness Index"
        )
        
    # 13. HETEROGENEITY TAB
    with t_hetero:
        from federated.heterogeneity import FederatedHeterogeneityAnalyzer
        hetero_res = FederatedHeterogeneityAnalyzer.analyze_heterogeneity(cached_metrics)
        render_standard_schema_section(
            title="Statistical Heterogeneity Analyzer",
            score=hetero_res["heterogeneity_score"],
            rating=hetero_res["classification"],
            confidence=hetero_res["confidence"],
            evidence=hetero_res["evidence"],
            explanation=hetero_res["explanation"],
            score_label="Heterogeneity Score"
        )
        
    # 14. EXPLAINABILITY TAB
    with t_explain:
        st.subheader("💬 Rule-Based XAI Diagnostics")
        from federated.explainability import FederatedExplainabilityEngine
        explanations = FederatedExplainabilityEngine.generate_explanations(cached_metrics)
        
        for exp in explanations:
            st.markdown(
                f"""
                <div class="glass-card" style='padding: 15px !important; margin-bottom: 12px; border-left: 4px solid #A78BFA;'>
                    <strong style='color:#A78BFA;'>❓ {exp['question']}</strong>
                    <p style='margin: 6px 0; font-size: 0.9rem; color:#E2E8F0;'>{exp['answer']}</p>
                    <div style='font-size:0.8rem; color:#94A3B8;'>Confidence: <b>{exp['confidence']}%</b></div>
                    <ul style='margin: 4px 0 0 0; padding-left: 20px; font-size:0.8rem; color:#94A3B8;'>
                """ + "".join(f"<li>{ev}</li>" for ev in exp["evidence"]) + """
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # 15. RECOMMENDATIONS TAB
    with t_recommend:
        st.subheader("💡 Intelligent Recommendations Engine")
        from federated.recommendations import FederatedRecommendationEngine
        recommendations = FederatedRecommendationEngine.generate_recommendations(cached_metrics)
        
        for r in recommendations:
            st.markdown(
                f"""
                <div class="glass-card" style='padding: 15px !important; margin-bottom: 12px; border-left: 4px solid #10B981;'>
                    <strong style='color:#34D399;'>💡 {r['title']}</strong>
                    <p style='margin: 6px 0; font-size: 0.9rem; color:#E2E8F0;'><b>Reason:</b> {r['reason']}</p>
                    <p style='margin: 0; font-size: 0.9rem; color:#E2E8F0;'><b>Expected Impact:</b> {r['expected_impact']}</p>
                    <div style='margin-top: 6px; font-size:0.8rem; color:#94A3B8;'>Confidence: <b>{r['confidence']}%</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Reset button to clear simulation run
    if st.button("Reset Simulation Run"):
        st.session_state.federated_metrics = None
        st.rerun()
else:
    st.info("ℹ️ No federated training has been executed yet. Click 'Start Federated Training Session' above to launch the simulation.")
