import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css, format_bytes, normalize_dataframe_for_rendering

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Dataset Comparison & Drift
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No primary dataset uploaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
filename = st.session_state.filename
memory = st.session_state.memory
plugins_data = memory.get("plugins", {})

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Dataset Comparison & Drift: {filename}
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Compare schemas, null allocations, and distribution drift shifts (KS & Chi-Square).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# FILE UPLOADER FOR DATASET B
# ----------------------------------------------------
st.markdown("### 📥 Ingest Comparison Dataset (Dataset B)")
uploaded_b = st.file_uploader(
    "Choose a comparison file", 
    type=["csv", "xlsx", "xls", "parquet", "json"],
    key="uploader_b",
    help="Upload a dataset with similar columns to inspect distribution drifts."
)

# Manage state for comparison dataset
if uploaded_b is not None:
    # Check if we need to load B
    if st.session_state.get("filename_comparison", "") != uploaded_b.name:
        with st.spinner("Parsing comparison file..."):
            from core.loader import DataLoader
            try:
                from core.profiler import DatasetProfiler
                from core.intelligence import DataIntelligenceEngine
                from intelligence import MemoryObject
                
                df_b = DataLoader.load_dataset(uploaded_b)
                st.session_state.df_comparison = df_b
                st.session_state.filename_comparison = uploaded_b.name
                
                # Re-run compile memory on primary dataset to execute plugins with the comparison df
                st.toast("Dataset B ingested! Re-compiling comparative plugins...")
                
                # Retrieve primary parameters
                metrics = st.session_state.metrics
                semantic_types = st.session_state.semantic_types
                domain_info = st.session_state.domain
                task_info = st.session_state.dataset_type
                
                from core.algorithm_selector import AlgorithmSelector
                algo_recs = AlgorithmSelector.recommend_algorithms(df, task_info, semantic_types, metrics)
                
                # This will execute comparison and drift plugins using st.session_state.df_comparison
                memory_obj = MemoryObject.compile_memory_object(
                    df, filename, metrics, semantic_types, domain_info, task_info, algo_recs
                )
                st.session_state.memory = memory_obj
                memory = memory_obj
                plugins_data = memory_obj.get("plugins", {})
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load dataset B: {e}")
                st.session_state.df_comparison = None
                st.session_state.filename_comparison = None
else:
    # Clear comparison states if removed
    if st.session_state.get("df_comparison", None) is not None:
        st.session_state.df_comparison = None
        st.session_state.filename_comparison = None
        st.toast("Comparison dataset cleared.")
        st.rerun()

# ----------------------------------------------------
# COMPARISON DASHBOARD DISPLAY
# ----------------------------------------------------
comp_data = plugins_data.get("comparison", {})
drift_data = plugins_data.get("drift", {})

if comp_data.get("status") == "SUCCESS" and drift_data.get("status") == "SUCCESS":
    c_res = comp_data.get("result", {})
    d_res = drift_data.get("result", {})
    
    # Header card indicating if simulation is active
    if st.session_state.get("df_comparison", None) is None:
        st.info("ℹ️ **Simulation Active**: No comparison file was uploaded. We are currently simulating comparison and drift metrics by splitting your active dataset in half (Dataset A: first 50% rows, Dataset B: second 50% rows).")
    else:
        st.success(f"✔️ **Comparative Ingestion Active**: Comparing Dataset A (`{filename}`) vs Dataset B (`{st.session_state.filename_comparison}`).")

    # 1. Dimensions delta cards
    dims = c_res.get("dimensions", {})
    st.markdown("#### 🗄️ Structural Dimensions")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.markdown(
            f"""
            <div class="stat-card" style='border-color: rgba(56, 189, 248, 0.4);'>
                <div class="stat-label">Dataset A Dimensions</div>
                <div class="stat-val" style='font-size:1.2rem; color: #38BDF8;'>{dims.get('dataset_a', {}).get('rows', 0):,} x {dims.get('dataset_a', {}).get('columns', 0)}</div>
                <div style='font-size:0.75rem; color:#94A3B8; margin-top:4px;'>Size: {format_bytes(dims.get('dataset_a', {}).get('memory_bytes', 0))}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cd2:
        st.markdown(
            f"""
            <div class="stat-card" style='border-color: rgba(129, 140, 248, 0.4);'>
                <div class="stat-label">Dataset B Dimensions</div>
                <div class="stat-val" style='font-size:1.2rem; color: #818CF8;'>{dims.get('dataset_b', {}).get('rows', 0):,} x {dims.get('dataset_b', {}).get('columns', 0)}</div>
                <div style='font-size:0.75rem; color:#94A3B8; margin-top:4px;'>Size: {format_bytes(dims.get('dataset_b', {}).get('memory_bytes', 0))}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cd3:
        delta_rows = dims.get('delta_rows', 0)
        r_color = "#34D399" if delta_rows >= 0 else "#F87171"
        st.markdown(
            f"""
            <div class="stat-card" style='border-color: {r_color}66;'>
                <div class="stat-label">Deltas (B - A)</div>
                <div class="stat-val" style='font-size:1.2rem; color: {r_color};'>{"+" if delta_rows >=0 else ""}{delta_rows:,} rows</div>
                <div style='font-size:0.75rem; color:#94A3B8; margin-top:4px;'>Columns Delta: {dims.get('delta_columns', 0)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # 2. Quality Delta & Schema changes
    col_qual, col_schema = st.columns(2)
    with col_qual:
        st.markdown("#### 🛡️ Quality Score Shift")
        q_comp = c_res.get("quality_comparison", {})
        delta_q = q_comp.get("delta_score", 0.0)
        q_color = "#34D399" if delta_q >= 0 else "#F87171"
        
        st.markdown(
            f"""
            <div class="glass-card" style='text-align: center; border-color: {q_color}44;'>
                <span style='font-size: 0.9rem; color: #94A3B8;'>Quality Score Delta</span>
                <h2 style='font-size: 2.2rem; color: {q_color}; margin: 8px 0;'>{"+" if delta_q >= 0 else ""}{delta_q:.2f}%</h2>
                <div style='font-size:0.85rem; color:#94A3B8;'>
                    Dataset A: {q_comp.get('score_a', 0.0):.1f}% | Dataset B: {q_comp.get('score_b', 0.0):.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_schema:
        st.markdown("#### 📂 Schema Alignment")
        sch = c_res.get("schema_changes", {})
        st.markdown(f"- **Shared Columns**: `{sch.get('common_columns_count', 0)}` features align.")
        st.markdown(f"- **Added Columns in B**: `{sch.get('added_columns', [])}`")
        st.markdown(f"- **Removed Columns in B**: `{sch.get('removed_columns', [])}`")

    # 3. Drift & Shift Details
    st.markdown("---")
    st.markdown("#### 🕸️ Feature Distribution Drift & Shift Matrix")
    st.markdown(
        f"""
        - **Overall Drift Score**: `{d_res.get('overall_drift_score', 0.0):.1f}%` of columns drifted.
        - **Inferred Drift Level**: **{d_res.get('drift_level')}**
        - **Target column Shift**: **{d_res.get('target_drift')}**
        """
    )
    
    drift_records = []
    for col, info in d_res.get("features", {}).items():
        drift_records.append({
            "Column Feature": col,
            "Drift Flagged": "🚨 DRIFTED" if info["drift_detected"] else "✅ STABLE",
            "Statistical Test": info["test_method"],
            "Shift Score / p-value": f"{info['metric_p_value']:.4f}",
            "Null Rate A": f"{info['null_pct_a']:.2f}%",
            "Null Rate B": f"{info['null_pct_b']:.2f}%"
        })
    st.dataframe(normalize_dataframe_for_rendering(pd.DataFrame(drift_records)), use_container_width=True)
else:
    st.warning("⚠️ Comparative plugins have failed to run. Check logs for details.")
