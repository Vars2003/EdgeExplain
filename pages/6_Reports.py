import streamlit as st
import pandas as pd
import json
import os
from utils.helpers import inject_custom_css
from core.profiler import DatasetProfiler

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Export Center
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset uploaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
filename = st.session_state.filename
metrics = st.session_state.metrics
recs = st.session_state.recs

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Export Center: {filename}
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Download clean datasets, statistics matrices, or exploratory profiles.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c_dataset, c_analysis = st.columns(2)

# Column 1: Dataset & Analysis Exports
with c_dataset:
    st.markdown("### 🗄️ Dataset Export")
    st.markdown("Download the active dataset in standard database layouts:")
    
    # CSV download
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=csv_bytes,
        file_name=f"exported_{filename.split('.')[0]}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # JSON download
    json_str = df.to_json(orient="records", indent=2)
    st.download_button(
        label="📥 Download Dataset (JSON)",
        data=json_str,
        file_name=f"exported_{filename.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True
    )

with c_analysis:
    st.markdown("### 🧮 Analysis Export")
    st.markdown("Download computed summaries and preprocessing directives:")
    
    # Statistics JSON download
    stats_json = json.dumps(metrics["statistics"], indent=2)
    st.download_button(
        label="📥 Download Custom Statistics (JSON)",
        data=stats_json,
        file_name=f"statistics_{filename.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Recommendations JSON download
    recs_json = json.dumps(recs, indent=2)
    st.download_button(
        label="📥 Download Recommendations (JSON)",
        data=recs_json,
        file_name=f"recommendations_{filename.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True
    )

st.markdown("---")

c_vis, c_exec = st.columns(2)

# Column 2: Visualizations & Executive Exports
with c_vis:
    st.markdown("### 📈 Visualization Export")
    st.markdown(
        """
        <div class="glass-card" style='padding: 16px !important;'>
            <h5 style='color: #94A3B8; margin: 0;'>Vector Graphic Exporters (Phase 3)</h5>
            <p style='font-size: 0.85rem; color: #94A3B8; margin: 8px 0 0 0;'>
                Supports vector exports for Plotly and Matplotlib assets.
            </p>
            <div style='margin-top: 15px;'>
                <button disabled style='width: 100%; padding: 8px; border-radius: 4px; background: rgba(255,255,255,0.05); color:#94A3B8; border: 1px solid rgba(255,255,255,0.08); cursor: not-allowed;'>
                    Export Plots (PNG / SVG)
                </button>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_exec:
    st.markdown("### 🛡️ Executive Reports")
    
    # HTML Profile report
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    report_file_name = f"profile_report_{filename.replace('.', '_')}.html"
    output_report_path = os.path.join(reports_dir, report_file_name)
    
    # Check if report already exists to allow immediate download without recalculating
    report_exists = os.path.exists(output_report_path)
    
    if report_exists:
        st.success("✔️ HTML Profile report has already been generated.")
        with open(output_report_path, "r", encoding="utf-8") as f:
            html_bytes = f.read()
        st.download_button(
            label="📥 Download Generated HTML Report",
            data=html_bytes,
            file_name=report_file_name,
            mime="text/html",
            use_container_width=True
        )
    else:
        if st.button("🔧 Generate and Download HTML Profile Report", use_container_width=True):
            os.makedirs(reports_dir, exist_ok=True)
            with st.spinner("Generating HTML Report via ydata-profiling..."):
                success = DatasetProfiler.generate_ydata_report(df, output_report_path)
            if success:
                st.success("Report created!")
                with open(output_report_path, "r", encoding="utf-8") as f:
                    html_bytes = f.read()
                st.download_button(
                    label="📥 Download HTML Report Now",
                    data=html_bytes,
                    file_name=report_file_name,
                    mime="text/html",
                    use_container_width=True
                )
            else:
                st.error("HTML profiling failed. Ensure ydata-profiling package is installed.")
                
    st.markdown(
        """
        <div style='margin-top: 15px; padding: 12px; border: 1px dashed rgba(255,255,255,0.08); border-radius: 6px;'>
            <span style='font-size:0.75rem; color: #94A3B8; text-transform: uppercase;'>Future Executive Exports</span>
            <p style='margin: 4px 0 0 0; font-size: 0.8rem; color:#94A3B8;'>PDF Document Compiler placeholder.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
