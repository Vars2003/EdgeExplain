import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import inject_custom_css

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Model Explainability (XAI)
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

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Model Explainability (XAI)
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Interpretable global attributions, local LIME prediction splits, and SHAP trees.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_shap, tab_lime, tab_fallback = st.tabs([
    "🌳 SHAP Global & Local", 
    "🎯 LIME Prediction Splits", 
    "📈 Feature Importance baseline"
])

# 1. SHAP Explanations
with tab_shap:
    shap_data = plugins_data.get("shap", {})
    status = shap_data.get("status", "UNAVAILABLE")
    result = shap_data.get("result", {})
    
    if status == "SUCCESS" and result.get("status") == "AVAILABLE":
        st.subheader("Official SHAP tree explainability")
        st.markdown(f"**Model trained**: `{result['model_type']}` | **Target Target**: `{result['target_used']}`")
        
        # Global Plot
        st.markdown("#### Global Feature Attribution (mean absolute SHAP)")
        glob = result.get("global_importance", {})
        if glob:
            glob_df = pd.DataFrame(list(glob.items()), columns=["Feature", "SHAP Value"]).sort_values(by="SHAP Value", ascending=True)
            fig_glob = px.bar(
                glob_df, x="SHAP Value", y="Feature", orientation="h",
                color="SHAP Value", color_continuous_scale="Viridis", text_auto=".4f"
            )
            fig_glob.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
            st.plotly_chart(fig_glob, use_container_width=True)
            
        # Local Plot
        st.markdown("#### Local Feature Attribution (Instance Row 0)")
        local = result.get("local_explanation", {})
        if local:
            loc_val = local.get("shap_values", {})
            loc_feat = local.get("feature_values", {})
            
            loc_records = []
            for col in loc_val.keys():
                loc_records.append({
                    "Feature": col,
                    "SHAP Attribution": loc_val[col],
                    "Feature Value": loc_feat.get(col, 0.0)
                })
            loc_df = pd.DataFrame(loc_records).sort_values(by="SHAP Attribution", ascending=True)
            
            fig_loc = px.bar(
                loc_df, x="SHAP Attribution", y="Feature", orientation="h",
                color="SHAP Attribution", color_continuous_scale="RdBu", text_auto=".4f",
                hover_data=["Feature Value"]
            )
            fig_loc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
            st.plotly_chart(fig_loc, use_container_width=True)
            st.info(f"💡 Base expected value estimate: `{local.get('base_value', 0.0):.4f}`")
    else:
        # Installation Guide
        st.subheader("Official SHAP Integration Optional Support")
        st.markdown(
            """
            <div class="glass-card" style='border-left: 4px solid #EF4444;'>
                <strong style='color:#EF4444; font-size:1.1rem;'>SHAP Package Missing</strong>
                <p style='margin:10px 0; color:#E2E8F0;'>
                    To view official SHAP summaries and tree waterfall attributions, please install the library 
                    locally. The application remains fully functional and offline.
                </p>
                <code style='display:block; padding:10px; background:#0F172A; border-radius:4px; color:#A78BFA;'>
                    pip install shap
                </code>
            </div>
            """,
            unsafe_allow_html=True
        )

# 2. LIME Explanations
with tab_lime:
    lime_data = plugins_data.get("lime", {})
    status = lime_data.get("status", "UNAVAILABLE")
    result = lime_data.get("result", {})
    
    if status == "SUCCESS" and result.get("status") == "AVAILABLE":
        st.subheader("Official LIME local explainer")
        st.markdown(f"**Target target**: `{result['target_used']}` | **Task**: `{result['mode']}`")
        
        influentials = result.get("influential_features", [])
        if influentials:
            inf_df = pd.DataFrame(influentials).sort_values(by="weight", ascending=True)
            inf_df.rename(columns={"feature_rule": "Decision Boundary Rule", "weight": "Feature Weight"}, inplace=True)
            
            fig_inf = px.bar(
                inf_df, x="Feature Weight", y="Decision Boundary Rule", orientation="h",
                color="Feature Weight", color_continuous_scale="Geyser", text_auto=".4f"
            )
            fig_inf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
            st.plotly_chart(fig_inf, use_container_width=True)
            st.info(f"💡 Local perturbed model prediction outcome value: `{result.get('prediction_value')}`")
    else:
        # Installation Guide
        st.subheader("Official LIME Integration Optional Support")
        st.markdown(
            """
            <div class="glass-card" style='border-left: 4px solid #EF4444;'>
                <strong style='color:#EF4444; font-size:1.1rem;'>LIME Package Missing</strong>
                <p style='margin:10px 0; color:#E2E8F0;'>
                    To view local perturbed instance boundaries and prediction splits, please install the library 
                    locally. The application remains fully functional and offline.
                </p>
                <code style='display:block; padding:10px; background:#0F172A; border-radius:4px; color:#A78BFA;'>
                    pip install lime
                </code>
            </div>
            """,
            unsafe_allow_html=True
        )

# 3. Decision Tree baseline importance
with tab_fallback:
    st.subheader("Vector Baseline Feature Importance (Scikit-Learn split weights)")
    st.markdown("Offline Scikit-Learn tree feature splits, available natively with zero dependency requirements.")
    
    target_candidates = memory.get("context", {}).get("candidate_targets", [])
    target = target_candidates[0] if target_candidates else df.columns[-1]
    
    from core.explain import ExplainabilityEngine
    from core.intelligence import DataIntelligenceEngine
    
    with st.spinner("Compiling decision tree attributions..."):
        types = DataIntelligenceEngine.detect_feature_types(df)
        importances = ExplainabilityEngine.calculate_feature_importance(df, target, types)
        
    if importances:
        imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance Weight"]).sort_values(by="Importance Weight", ascending=True)
        fig_base = px.bar(
            imp_df, x="Importance Weight", y="Feature", orientation="h",
            color="Importance Weight", color_continuous_scale="Plasma", text_auto=".4f"
        )
        fig_base.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
        st.plotly_chart(fig_base, use_container_width=True)
    else:
        st.info("Insufficient samples or features to fit a tree model.")
