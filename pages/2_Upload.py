import os
import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css, format_bytes
from core.loader import DataLoader
from core.profiler import DatasetProfiler
from core.statistics import summarize_dataframe
from core.dependency import compute_dependency_matrix
from core.intelligence import DataIntelligenceEngine
from core.recommendation import RecommendationEngine
from core.insights import InsightsEngine
from core.algorithm_selector import AlgorithmSelector
from intelligence import MemoryObject, event_system

# Re-inject css for sub-page rendering consistency
inject_custom_css()

# Header layout with gradient accent
st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Dataset Ingestion
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Select a local database file. Supported formats: CSV, Excel, Parquet, JSON.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload Card Container
uploaded_file = st.file_uploader(
    "Choose a file from your computer", 
    type=["csv", "xlsx", "xls", "parquet", "json"],
    help="Files will be parsed entirely in-memory. Maximum size allowed: 100MB."
)

if uploaded_file is not None:
    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_folder, exist_ok=True)
    
    local_file_path = os.path.join(data_folder, uploaded_file.name)
    
    # Check if we need to load or if it's already loaded
    if st.session_state.filename != uploaded_file.name:
        # Visual Load indicators
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            # Step 1: Save
            progress_text.markdown("📁 *Saving file to local buffer...*")
            progress_bar.progress(10)
            with open(local_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Step 2: Load
            progress_text.markdown("⚙️ *Parsing file structure...*")
            progress_bar.progress(25)
            df = DataLoader.load_dataset(local_file_path)
            
            if df.empty:
                raise ValueError("The uploaded dataset contains zero rows or columns.")
            
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            
            # Step 3: Semantic Types
            progress_text.markdown("🧠 *Running feature intelligence parser...*")
            progress_bar.progress(45)
            semantic_types = DataIntelligenceEngine.detect_feature_types(df)
            st.session_state.semantic_types = semantic_types
            
            # Step 4: Domain & Task
            progress_text.markdown("🔍 *Estimating source domain and machine learning task category...*")
            progress_bar.progress(60)
            domain_info = DataIntelligenceEngine.infer_source_domain(df)
            st.session_state.domain = domain_info
            
            dataset_type_info = DataIntelligenceEngine.detect_dataset_type(df, semantic_types)
            st.session_state.dataset_type = dataset_type_info
            
            # Step 5: Statistics & Dependencies
            progress_text.markdown("🧮 *Calculating custom statistics descriptors (Mean, Median, Skewness, Kurtosis)...*")
            progress_bar.progress(75)
            stats_summary = summarize_dataframe(df)
            dep_summary = compute_dependency_matrix(df, {col: info["type"] for col, info in semantic_types.items()})
            
            # Step 6: Profiling
            progress_text.markdown("🔬 *Diagnosing data quality and outlier bounds...*")
            progress_bar.progress(90)
            basic_metrics = DatasetProfiler.get_basic_metrics(df)
            missing_data = DatasetProfiler.analyze_missing_values(df)
            outliers = DatasetProfiler.detect_outliers(df)
            
            quality_score = DatasetProfiler.calculate_data_quality_score(df, basic_metrics, missing_data, outliers)
            ml_readiness = DatasetProfiler.calculate_ml_readiness_score(df, missing_data, basic_metrics)
            
            metrics = {
                "basic_metrics": basic_metrics,
                "missing_data": missing_data,
                "outliers": outliers,
                "quality_score": quality_score,
                "ml_readiness": ml_readiness,
                "statistics": stats_summary,
                "dependency": dep_summary
            }
            st.session_state.metrics = metrics
            
            # Step 7: Recommendations & Insights
            progress_text.markdown("🛠️ *Compiling recommendations and quality insights...*")
            progress_bar.progress(95)
            recs = RecommendationEngine.generate_recommendations(df, metrics, semantic_types)
            st.session_state.recs = recs
            
            insights = InsightsEngine.generate_insights(df, metrics, semantic_types)
            st.session_state.insights = insights
            
            # Step 8: Memory Compilation & Events
            progress_text.markdown("🧠 *Compiling Unified JSON Memory Object...*")
            progress_bar.progress(98)
            
            algo_recs = AlgorithmSelector.recommend_algorithms(
                df, dataset_type_info, semantic_types, metrics
            )
            memory_obj = MemoryObject.compile_memory_object(
                df, uploaded_file.name, metrics, semantic_types, domain_info, dataset_type_info, algo_recs
            )
            st.session_state.memory = memory_obj
            
            # Dispatch Lifecyle Events
            event_system.dispatch("DatasetUploaded", uploaded_file.name)
            event_system.dispatch("AnalysisCompleted", memory_obj)
            event_system.dispatch("MemoryUpdated", memory_obj)
            event_system.dispatch("SummaryGenerated", memory_obj.get("summaries", {}))
            
            progress_bar.progress(100)
            progress_text.empty()
            st.success("🎉 Ingestion, profiling, and memory compilation complete!")
            
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"Ingestion failed: {e}")
            st.session_state.df = None
            st.session_state.filename = None
            if os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                except:
                    pass
            st.stop()
            
    # Display success dashboard card
    if st.session_state.df is not None:
        df = st.session_state.df
        rows, cols = df.shape
        domain_info = st.session_state.domain
        task_info = st.session_state.dataset_type
        
        st.markdown(
            f"""
            <div class="glass-card" style='border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.05);'>
                <h3 style='color: #10B981; margin: 0;'>✔️ Dataset Ready for Evaluation</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-top: 20px;'>
                    <div class="stat-card">
                        <div class="stat-label">File Name</div>
                        <div class="stat-val" style='font-size:1.1rem; word-break:break-all;'>{st.session_state.filename}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Rows & Columns</div>
                        <div class="stat-val">{rows} x {cols}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Estimated Domain</div>
                        <div class="stat-val" style='font-size:1.1rem;'>{domain_info['domain']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">ML Task</div>
                        <div class="stat-val" style='font-size:1.1rem;'>{task_info['type']}</div>
                    </div>
                </div>
                <div style='margin-top: 20px; font-size: 0.95rem; color: #E2E8F0;'>
                    👉 Go to the <strong>Dashboard Summary</strong> page in the sidebar to review overall statistics and health gauges.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    # Clear session state if file is removed
    st.session_state.df = None
    st.session_state.filename = None
    st.session_state.semantic_types = None
    st.session_state.metrics = None
    st.session_state.domain = None
    st.session_state.dataset_type = None
    st.session_state.recs = None
    st.session_state.insights = None
    
    st.info("Please upload a file to begin the analysis.")
