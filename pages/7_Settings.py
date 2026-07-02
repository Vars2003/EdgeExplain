import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css
from core.profiler import DatasetProfiler
from core.recommendation import RecommendationEngine
from core.insights import InsightsEngine
from ai import model_manager

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
    st.subheader("🤖 Local AI Configuration")
    
    # 1. Fallback / LLM Mode selector
    providers = ["Fallback (Rule-Based Engine)", "Ollama"]
    provider_idx = 1 if not st.session_state.get("ai_fallback_mode", False) else 0
    
    sel_prov = st.selectbox(
        "Active Inference Provider:",
        providers,
        index=provider_idx,
        help="Fallback Mode executes deterministic offline rules, bypassing local Ollama runtimes."
    )
    st.session_state.ai_fallback_mode = (sel_prov == "Fallback (Rule-Based Engine)")
    
    # 2. Model Selection
    available_models = []
    if not st.session_state.ai_fallback_mode:
        try:
            available_models = model_manager.detect_available_models()
        except Exception as e:
            available_models = []
            
    if not st.session_state.ai_fallback_mode and available_models:
        default_model = st.session_state.get("ai_selected_model", None)
        default_idx = 0
        if default_model in available_models:
            default_idx = available_models.index(default_model)
            
        new_model = st.selectbox(
            "Select Local Active Model:",
            available_models,
            index=default_idx
        )
        st.session_state.ai_selected_model = new_model
        
        # Display RAM info
        mem_info = model_manager.estimate_model_memory(new_model)
        st.markdown(
            f"""
            <div style='padding: 8px; margin-top:5px; background:rgba(56, 189, 248, 0.08); border-radius:4px; font-size:0.8rem;'>
                💾 Size: <strong>{mem_info['estimated_file_size_gb']} GB</strong> | 
                ⚙️ Min RAM: <strong>{mem_info['minimum_system_ram_gb']} GB</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif not st.session_state.ai_fallback_mode:
        st.warning("⚠️ No active Ollama models detected. System will force-use Grounded Rule Fallback mode.")
        st.session_state.ai_fallback_mode = True
        
    # 3. Context budget
    new_context = st.slider(
        "Max Context size (Tokens):",
        512, 8192, int(st.session_state.get("ai_max_context", 2048)), 512,
        help="Token boundaries for prompt builder compression."
    )
    st.session_state.ai_max_context = new_context
    
    # 4. Temperature
    new_temp = st.slider(
        "Generation Temperature:",
        0.1, 1.0, float(st.session_state.get("ai_temperature", 0.7)), 0.05,
        help="Controls creativity. Lower values are more deterministic."
    )
    st.session_state.ai_temperature = new_temp
    
    # 5. Persona
    personas = ["Data Analyst", "ML Engineer", "Research Assistant"]
    p_idx = 0
    cur_p = st.session_state.get("ai_persona", "Data Analyst")
    if cur_p in personas:
        p_idx = personas.index(cur_p)
        
    new_persona = st.selectbox(
        "Active System Persona:",
        personas,
        index=p_idx,
        help="Alters system prompt persona behaviors."
    )
    st.session_state.ai_persona = new_persona
    
    # 6. Streaming toggle
    new_stream = st.toggle(
        "Enable Token Streaming",
        value=bool(st.session_state.get("ai_streaming", True)),
        help="Progressively types words instead of complete generation."
    )
    st.session_state.ai_streaming = new_stream
