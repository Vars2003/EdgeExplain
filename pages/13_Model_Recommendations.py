import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Model & Pipeline Recommendations
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset uploaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
memory = st.session_state.memory
plugins_data = memory.get("plugins", {})
automl_data = plugins_data.get("automl", {})

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Model & Pipeline Recommendations
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Intelligent suitability rankings, validation partitions, and complete modeling pipeline blueprints.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if automl_data.get("status") == "SUCCESS":
    pipelines = automl_data.get("result", {}).get("automl_pipelines", [])
    
    # 1. Renders ranked suitability scores
    st.subheader("🏆 Model Suitability Leaderboard")
    
    leaderboard_records = []
    for idx, p in enumerate(pipelines):
        leaderboard_records.append({
            "Rank": idx + 1,
            "Algorithm Model": p["algorithm"],
            "Suitability Index": f"{p['suitability_score']}%",
            "Key Strength": p["pros"][0] if p["pros"] else "Stable baseline"
        })
    st.dataframe(pd.DataFrame(leaderboard_records), use_container_width=True)

    st.markdown("---")
    
    # 2. Select Model to display COMPLETE pipeline
    st.subheader("🛠️ Complete Pipeline Blueprint Explorer")
    selected_algo = st.selectbox("Select Model to inspect pipeline:", [p["algorithm"] for p in pipelines])
    
    if selected_algo:
        p_obj = next((p for p in pipelines if p["algorithm"] == selected_algo), None)
        if p_obj:
            pipe = p_obj["pipeline"]
            
            c_flow, c_why = st.columns([1.1, 0.9])
            
            with c_flow:
                st.markdown("**🔧 Pipeline Sequence Stages:**")
                st.markdown(
                    f"""
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #38BDF8; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>1. Missing Value Strategy</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['imputation']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['imputation_reason']}</span>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #38BDF8; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>2. Categorical Encoding</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['encoding']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['encoding_reason']}</span>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #38BDF8; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>3. Feature Scaling</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['scaling']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['scaling_reason']}</span>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #38BDF8; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>4. Feature Selection</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['feature_selection']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['feature_selection_reason']}</span>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #818CF8; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>5. Model Estimator</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{p_obj['algorithm']}</p>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #34D399; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>6. Validation Strategy</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['validation_strategy']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['validation_strategy_reason']}</span>
                    </div>
                    <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #34D399; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                        <strong style='color:#F8FAFC;'>7. Evaluation Metric</strong>
                        <p style='margin:3px 0 0 0; font-size:0.85rem; color:#E2E8F0;'>{pipe['evaluation_metric']}</p>
                        <span style='font-size:0.75rem; color:#94A3B8;'>Reason: {pipe['evaluation_metric_reason']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with c_why:
                st.markdown("**🛡️ Model Justifications & Projections:**")
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style='color:#34D399; margin:0 0 10px 0;'>Key Strengths (Pros)</h4>
                        {''.join([f'<p>- {p}</p>' for p in p_obj['pros']])}
                    </div>
                    <div class="glass-card">
                        <h4 style='color:#F87171; margin:0 0 10px 0;'>Expected Risks (Cons)</h4>
                        {''.join([f'<p>- {c}</p>' for c in p_obj['cons']])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.warning("⚠️ AutoML recommendations failed to execute. Check logs.")
