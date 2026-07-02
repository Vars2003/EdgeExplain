import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css
from core.reasoning import ReasoningEngine

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                AI Insights & Reasoning
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset uploaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
filename = st.session_state.filename
semantic_types = st.session_state.semantic_types
metrics = st.session_state.metrics
domain_info = st.session_state.domain
task_info = st.session_state.dataset_type
insights = st.session_state.insights

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AI Insights & Reasoning
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Cognitive explanations, domain inference vocabularies, and logical reasoning traces.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col_meta, col_ins = st.columns([1, 1])

# Left column: Inferences & Explanations
with col_meta:
    st.subheader("🧠 Cognitive Inferences")
    
    # Domain Justification
    domain_explanation = ReasoningEngine.explain_dataset_classification({
        "type": domain_info["domain"],
        "confidence": domain_info["confidence"],
        "evidence": domain_info["evidence"]
    })
    
    st.markdown(
        f"""
        <div class="glass-card">
            <h4 style='color: #FBBF24; margin: 0 0 10px 0;'>Domain Inference Justification</h4>
            <p style='font-size: 0.9rem; line-height: 1.5; color: #E2E8F0;'>
                {domain_explanation['explanation']}
            </p>
            <div style='margin-top: 10px;'>
                <span class='tag tag-blue'>Confidence: {domain_info['confidence']}%</span>
                <span class='tag tag-purple'>Evidence: {', '.join(domain_info['evidence'][:4])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Task Justification
    task_explanation = ReasoningEngine.explain_dataset_classification(task_info)
    st.markdown(
        f"""
        <div class="glass-card">
            <h4 style='color: #818CF8; margin: 0 0 10px 0;'>ML Pipeline Task Justification</h4>
            <p style='font-size: 0.9rem; line-height: 1.5; color: #E2E8F0;'>
                {task_explanation['explanation']}
            </p>
            <div style='margin-top: 10px;'>
                <span class='tag tag-blue'>Confidence: {task_info['confidence']}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Interactive Reasoning Explorer
    st.markdown("### ❓ Interactive Justification Explorer")
    col_to_justify = st.selectbox("Select a column to inspect reasoning:", df.columns.tolist())
    
    if col_to_justify:
        col_info = semantic_types[col_to_justify]
        col_reason = ReasoningEngine.explain_feature_classification(col_to_justify, col_info)
        
        st.markdown(
            f"""
            <div class="glass-card" style='border-color: rgba(129, 140, 248, 0.3);'>
                <strong style='color: #A78BFA; font-size:1rem;'>Decision: {col_reason['decision']}</strong>
                <p style='font-size: 0.85rem; color: #E2E8F0; margin: 6px 0 0 0;'>{col_reason['explanation']}</p>
                <div style='margin-top: 8px;'>
                    <span class='tag tag-blue'>Confidence: {col_reason['confidence']}%</span>
                    <span class='tag tag-purple'>Evidence: {', '.join(col_reason['evidence'])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Right column: Automated Insights
with col_ins:
    st.subheader("💡 Automated Insights")
    
    for idx, insight in enumerate(insights):
        sev_class = "tag-red" if insight["severity"] == "High" else "tag-orange" if insight["severity"] == "Medium" else "tag-green"
        st.markdown(
            f"""
            <div class="glass-card" style='padding: 16px !important; margin-bottom: 12px !important;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <strong style='font-size: 1.05rem; color: #F8FAFC;'>{insight['title']}</strong>
                    <span class="tag {sev_class}">{insight['severity']} Risk</span>
                </div>
                <p style='color: #94A3B8; font-size: 0.9rem; margin: 0;'>{insight['description']}</p>
                {f"<div style='margin-top: 8px;'><span class='tag tag-blue'>Affected Columns: {', '.join(insight['columns'])}</span></div>" if insight['columns'] else ""}
            </div>
            """,
            unsafe_allow_html=True
        )
