import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css
from core.profiler import DatasetProfiler
from core.recommendation import RecommendationEngine
from core.insights import InsightsEngine

# Re-inject CSS for visual consistency
inject_custom_css()

st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Analysis Settings
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Configure thresholds, target modeling variables, and offline computing parameters.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c_config, c_ai = st.columns(2)

with c_config:
    st.subheader("⚙️ Global Parameters")
    
    # 1. Target Column Configurator
    if st.session_state.df is not None:
        target_list = [None] + st.session_state.df.columns.tolist()
        default_target = st.session_state.target_column
        default_idx = 0
        if default_target in target_list:
            default_idx = target_list.index(default_target)
            
        new_target = st.selectbox(
            "Global Target Modeling Feature:",
            target_list,
            index=default_idx,
            help="Pre-configures default target labels for Feature Importance and Selector engines."
        )
        st.session_state.target_column = new_target
    else:
        st.info("Ingest a dataset to configure target column selections.")
        
    # 2. Correlation slider
    new_corr = st.slider(
        "Correlation Network Connection Threshold:",
        0.1, 1.0, float(st.session_state.correlation_threshold), 0.05,
        help="Hides edges/links below this value in the circular correlation network."
    )
    st.session_state.correlation_threshold = new_corr
    
    # 3. Outlier boundary slider
    new_outlier = st.slider(
        "Outlier Interquartile Range (IQR) Boundary Factor:",
        1.0, 3.0, float(st.session_state.outlier_threshold), 0.1,
        help="Standard multiplier (default 1.5) for Tukey outlier checks. Higher values detect only extreme anomalies."
    )
    
    # Check if threshold was modified to trigger recalculation
    if new_outlier != st.session_state.outlier_threshold:
        st.session_state.outlier_threshold = new_outlier
        
        # If dataset is loaded, trigger re-evaluation
        if st.session_state.df is not None:
            with st.spinner("Re-evaluating outlier thresholds and quality indexes..."):
                df = st.session_state.df
                metrics = st.session_state.metrics
                semantic_types = st.session_state.semantic_types
                
                # Re-run outlier check with new threshold
                # Wait, standard detect_outliers in profiler.py uses standard 1.5.
                # Let's write the recalculation logic in profiler.py or directly here.
                # Since we want to preserve profiler.py, we can just run the Tukey outlier calculation with the new factor here:
                outlier_counts = {}
                total_outliers = 0
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        series = df[col].dropna()
                        if len(series) >= 4:
                            q1 = series.quantile(0.25)
                            q3 = series.quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - new_outlier * iqr
                            upper = q3 + new_outlier * iqr
                            outliers = series[(series < lower) | (series > upper)]
                            outlier_counts[col] = len(outliers)
                            total_outliers += len(outliers)
                
                # Update metrics dict
                metrics["outliers"] = {
                    "columns": outlier_counts,
                    "total_outliers": total_outliers
                }
                
                # Re-run Quality Score
                metrics["quality_score"] = DatasetProfiler.calculate_data_quality_score(
                    df, metrics["basic_metrics"], metrics["missing_data"], metrics["outliers"]
                )
                
                # Re-run Recommendations & Insights
                st.session_state.recs = RecommendationEngine.generate_recommendations(df, metrics, semantic_types)
                st.session_state.insights = InsightsEngine.generate_insights(df, metrics, semantic_types)
                st.session_state.metrics = metrics
            st.toast("✅ Outlier metrics updated!")

with c_ai:
    st.subheader("🤖 Future Local AI Configuration")
    
    st.selectbox(
        "Theme Style Selector:",
        ["Dark Slate (Active)", "Light Silver (Placeholder)"],
        disabled=True,
        help="Theme changes are pre-configured to dark-slate mode for maximum legibility."
    )
    
    st.selectbox(
        "Select Local Inference Engine:",
        ["llama.cpp (GGML/GGUF) - Placeholder", "Ollama API - Placeholder"],
        index=0,
        disabled=True
    )
    
    st.text_input(
        "Model Folder Path (.gguf file):",
        value="C:/Users/varsh/.cache/lm-studio/models/Llama-3-8B-Instruct.gguf",
        disabled=True
    )
    
    st.slider(
        "Local LLM Context Window:",
        512, 8192, 2048, 512,
        disabled=True
    )
