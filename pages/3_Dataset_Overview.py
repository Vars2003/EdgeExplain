import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.helpers import inject_custom_css, format_bytes
from core.visualizer import (
    plot_histogram, plot_scatter, plot_bar, plot_box, 
    plot_correlation_heatmap, plot_correlation_network
)
from core.statistics import summarize_column
from core.reasoning import ReasoningEngine
from core.algorithm_selector import AlgorithmSelector
from core.explain import ExplainabilityEngine
from core.profiler import DatasetProfiler

def normalize_dataframe_for_rendering(df_to_render: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that any column in the DataFrame does not contain Python lists, 
    tuples, sets, or dicts, which would cause PyArrow serialization errors 
    when rendered in Streamlit.
    """
    df_clean = df_to_render.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple, set)) 
                else str(x) if isinstance(x, dict) 
                else x
            )
    return df_clean

# Re-inject css for sub-page rendering consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Dataset Intelligence
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset uploaded yet. Please go to the Ingestion page to load a database file.")
    st.stop()

# Retrieve session parameters
df = st.session_state.df
filename = st.session_state.filename
semantic_types = st.session_state.semantic_types
metrics = st.session_state.metrics
domain_info = st.session_state.domain
task_info = st.session_state.dataset_type
recs = st.session_state.recs
insights = st.session_state.insights

rows, cols = df.shape

# Header layout with gradient accent
st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Dataset Intelligence: {filename}
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Automated analysis, reasoning, and pre-processing directives running 100% locally.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Global Performance Metrics Bar
quality_score = metrics["quality_score"]
readiness_score = metrics["ml_readiness"]["score"]

q_color = "#10B981" if quality_score > 80 else "#F59E0B" if quality_score > 50 else "#EF4444"
r_color = "#10B981" if readiness_score > 80 else "#F59E0B" if readiness_score > 50 else "#EF4444"

st.markdown(
    f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-bottom: 25px;'>
        <div class="stat-card">
            <div class="stat-label">Dataset Quality Score</div>
            <div class="stat-val" style='color: {q_color};'>{quality_score:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">ML Readiness Index</div>
            <div class="stat-val" style='color: {r_color};'>{readiness_score:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Estimated Domain</div>
            <div class="stat-val" style='color: #FBBF24;'>{domain_info['domain']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Target Task</div>
            <div class="stat-val" style='color: #818CF8;'>{task_info['type']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs navigation
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
    "📈 Overview & Insights", 
    "📋 Data Viewer", 
    "🧮 Custom Statistics", 
    "🕸️ Variable Dependencies",
    "⚠️ Risk Matrix",
    "🛠️ Preprocessing & Models",
    "🔍 Interactive Explainability",
    "📤 Export Report"
])

# ----------------------------------------------------
# TAB 1: OVERVIEW & INSIGHTS
# ----------------------------------------------------
with t1:
    col_l, col_r = st.columns([1.1, 0.9])
    
    with col_l:
        st.subheader("💡 Automated Findings & Insights")
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
                    {f"<div style='margin-top: 8px;'><span class='tag tag-blue'>Affected: {', '.join(insight['columns'])}</span></div>" if insight['columns'] else ""}
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col_r:
        st.subheader("🧠 Cognitive Explanations")
        
        # Explain Domain Inference
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
                    <span class='tag tag-purple'>Evidence Headers: {', '.join(domain_info['evidence'][:4])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Explain Task Inferences
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

# ----------------------------------------------------
# TAB 2: DATA VIEWER
# ----------------------------------------------------
with t2:
    st.subheader("📂 Schema Mapping (Pandas vs Semantic Parse)")
    
    schema_records = []
    for col, info in semantic_types.items():
        schema_records.append({
            "Column Name": col,
            "Pandas Type": str(df[col].dtype),
            "Semantic Category": info["type"].replace("_", " ").title(),
            "Confidence": f"{info['confidence']}%",
            "Reasoning": info["explanation"]
        })
    st.dataframe(normalize_dataframe_for_rendering(pd.DataFrame(schema_records)), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📄 Dataset Samples")
    tab_first, tab_last = st.tabs(["First 20 Records", "Last 20 Records"])
    
    with tab_first:
        st.dataframe(normalize_dataframe_for_rendering(df.head(20)), use_container_width=True)
    with tab_last:
        st.dataframe(normalize_dataframe_for_rendering(df.tail(20)), use_container_width=True)

# ----------------------------------------------------
# TAB 3: CUSTOM STATISTICS
# ----------------------------------------------------
with t3:
    st.subheader("🧮 Custom Descriptors Summary")
    st.markdown(
        "> All calculations (including Skewness and Excess Kurtosis) are computed "
        "directly using our local statistics formulas."
    )
    
    stats_df = pd.DataFrame(metrics["statistics"]).T
    st.dataframe(normalize_dataframe_for_rendering(stats_df), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Interactive Single Feature Plotter")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    plot_col = st.selectbox("Select a column to plot", df.columns.tolist())
    
    if plot_col:
        is_num = pd.api.types.is_numeric_dtype(df[plot_col])
        if is_num:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.plotly_chart(plot_histogram(df, plot_col), use_container_width=True)
            with sub_col2:
                # Custom statistics info
                col_stats = summarize_column(df[plot_col])
                st.markdown("#### Calculated Statistics:")
                for k, v in col_stats.items():
                    if isinstance(v, float):
                        st.markdown(f"- **{k.replace('_', ' ').title()}**: {v:.4f}")
                    else:
                        st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")
        else:
            st.plotly_chart(plot_bar(df, plot_col), use_container_width=True)

# ----------------------------------------------------
# TAB 4: VARIABLE DEPENDENCIES
# ----------------------------------------------------
with t4:
    st.subheader("🕸️ Feature Relationships Mapping")
    
    col_net1, col_net2 = st.columns([1, 1])
    
    with col_net1:
        # Dependency Matrix Heatmap
        dep_data = metrics["dependency"]
        st.plotly_chart(
            plot_correlation_heatmap(dep_data["matrix"], dep_data["columns"]), 
            use_container_width=True
        )
        
    with col_net2:
        # Circular Network Graph
        threshold = st.slider("Association Threshold filter (hide edges below)", 0.0, 1.0, 0.3, 0.05)
        
        # Build network structures
        nodes = []
        for col in dep_data["columns"]:
            nodes.append({"id": col, "label": col, "type": semantic_types.get(col, {}).get("type", "nominal")})
            
        edges = []
        matrix = np.array(dep_data["matrix"])
        cols_list = dep_data["columns"]
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                w = matrix[i, j]
                if w >= threshold:
                    edges.append({
                        "source": cols_list[i],
                        "target": cols_list[j],
                        "weight": float(w)
                    })
                    
        st.plotly_chart(plot_correlation_network(nodes, edges), use_container_width=True)

# ----------------------------------------------------
# TAB 5: RISK MATRIX
# ----------------------------------------------------
with t5:
    st.subheader("⚠️ Data Risks & Health Vulnerabilities")
    
    # Generate risk rows dynamically from metrics
    risk_rows = []
    
    # 1. Multicollinearity risk
    redundancy = metrics["ml_readiness"]["checks"].get("redundancy", {})
    if redundancy.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "Multicollinearity",
            "Severity": "High",
            "Detail": redundancy["detail"],
            "Suggested Fix": "Drop one of the highly correlated columns using clean modules."
        })
        
    # 2. Duplicate rows risk
    dups = metrics["ml_readiness"]["checks"].get("duplicates", {})
    if dups.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "High Row Duplication",
            "Severity": "High",
            "Detail": dups["detail"],
            "Suggested Fix": "Apply deduplication using core cleaner."
        })
        
    # 3. Missing values risk
    miss = metrics["ml_readiness"]["checks"].get("missing_values", {})
    if miss.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "High Missingness",
            "Severity": "Medium",
            "Detail": miss["detail"],
            "Suggested Fix": "Impute values with median/mode or drop features with >40% missingness."
        })
        
    # 4. Outliers risk
    outliers_data = metrics.get("outliers", {})
    if outliers_data:
        total_outliers = outliers_data.get("total_outliers", 0)
        outliers_pct = (total_outliers / df.size) * 100 if df.size > 0 else 0
        if outliers_pct > 2.0:
            risk_rows.append({
                "Risk Category": "Extreme Outliers",
                "Severity": "Medium",
                "Detail": f"{total_outliers} cell outliers flagged ({outliers_pct:.2f}% of cells).",
                "Suggested Fix": "Standardize numerical features using RobustScaler or Winsorization."
            })
            
    # 5. Encoding risk
    enc = metrics["ml_readiness"]["checks"].get("encoding", {})
    if enc.get("status") == "INFO":
        risk_rows.append({
            "Risk Category": "Categorical Columns",
            "Severity": "Low",
            "Detail": enc["detail"],
            "Suggested Fix": "One-hot encode low-cardinality nominals; Target encode high-cardinality nominals."
        })
        
    # 6. Scaling risk
    scale = metrics["ml_readiness"]["checks"].get("scaling", {})
    if scale.get("status") == "INFO":
        risk_rows.append({
            "Risk Category": "Scale Variance Mismatch",
            "Severity": "Low",
            "Detail": scale["detail"],
            "Suggested Fix": "Normalize inputs using StandardScaler or MinMaxScaler."
        })
        
    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)
        # Apply CSS styling for rows
        st.dataframe(normalize_dataframe_for_rendering(risk_df), use_container_width=True)
    else:
        st.success("🎉 No major structural risks flagged. The dataset is clean.")

# ----------------------------------------------------
# TAB 6: PREPROCESSING & MODELS
# ----------------------------------------------------
with t6:
    col_pre, col_models = st.columns(2)
    
    with col_pre:
        st.subheader("🛠️ Preprocessing Directives")
        for r in recs:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong style='color:#38BDF8;'>Type: {r['type']}</strong>
                        <span class="tag tag-blue">Confidence: {r['confidence']}%</span>
                    </div>
                    <p style='margin: 8px 0 0 0; font-size: 0.9rem; color: #E2E8F0;'><strong>Fix</strong>: {r['suggested_fix']}</p>
                    <p style='margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;'><strong>Reason</strong>: {r['reason']}</p>
                    {f"<div style='margin-top:8px;'><span class='tag tag-purple'>Target: {', '.join(r['features'])}</span></div>" if r['features'] else ""}
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col_models:
        st.subheader("🤖 Recommended Machine Learning Models")
        st.markdown("> Compatibility ratings are scored based on scale ratios, sample dimensions, and feature sets.")
        
        algo_recs = AlgorithmSelector.recommend_algorithms(
            df, task_info, semantic_types, metrics
        )
        
        for a in algo_recs:
            score = a["compatibility_score"]
            score_color = "#10B981" if score > 80 else "#F59E0B" if score > 50 else "#EF4444"
            
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong style='color:#818CF8; font-size: 1.05rem;'>{a['algorithm']}</strong>
                        <span class="tag" style='background: rgba(129, 140, 248, 0.2); color:{score_color}; font-weight:700;'>Score: {score}%</span>
                    </div>
                    <p style='margin: 8px 0 0 0; font-size: 0.9rem; color: #E2E8F0;'>{a['reasoning']}</p>
                    <div style='margin-top: 10px; font-size:0.8rem;'>
                        <span style='color:#34D399;'><strong>Pros</strong>: {", ".join(a['pros'][:3])}</span><br>
                        <span style='color:#F87171;'><strong>Cons</strong>: {", ".join(a['cons'][:2])}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------------------------------
# TAB 7: INTERACTIVE EXPLAINABILITY
# ----------------------------------------------------
with t7:
    st.subheader("🔍 Interactive Feature Importance Engine")
    st.markdown(
        "Select a classification or regression target column from the dropdown to fit "
        "an offline decision tree model and compute feature importances."
    )
    
    target_select = st.selectbox(
        "Select Target Column:", 
        [None] + df.columns.tolist(),
        index=0
    )
    
    if target_select:
        with st.spinner("Fitting local model offline..."):
            importances = ExplainabilityEngine.calculate_feature_importance(
                df, target_select, semantic_types
            )
            
        if importances:
            imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            
            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation='h',
                color="Importance",
                color_continuous_scale="Plasma",
                text_auto='.3f'
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(
                "💡 **Interpretability**: A higher score indicates the feature was chosen more frequently "
                "to split the decision tree, representing a stronger predictive relationship."
            )
        else:
            st.warning("⚠️ Could not compute importance. Ensure the chosen target contains sufficient non-null values.")
    else:
        st.write("Please select a target variable to execute the tree explainer.")

    st.markdown("---")
    st.subheader("🔮 Roadmap for Future Explainability Hooks")
    st.markdown(
        """
        - **SHAP Integrations**: Next phases will bundle local SHAP kernels to calculate exact Shapley additive explanations for neural networks and linear models.
        - **Interactive Why Button**: In the prediction screen, a 'Why' widget will translate SHAP outputs into plain English justifications.
        - **LIME Local Fit**: Support for Local Interpretable Model-agnostic Explanations to explain individual unstructured records.
        """
    )

# ----------------------------------------------------
# TAB 8: EXPORT REPORT
# ----------------------------------------------------
with t8:
    st.subheader("📤 Generate Heavy HTML Diagnostics Report")
    st.markdown(
        "If you need a comprehensive offline HTML summary, you can trigger `ydata-profiling` "
        "report generation. This builds a static HTML page in your project's `/reports/` folder."
    )
    
    if st.button("Generate HTML Profile Report"):
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file_name = f"profile_report_{filename.replace('.', '_')}.html"
        output_report_path = os.path.join(reports_dir, report_file_name)
        
        with st.spinner("Generating HTML Report (this may take a few moments for large files)..."):
            success = DatasetProfiler.generate_ydata_report(df, output_report_path)
            
        if success:
            st.success(f"🎉 HTML Report successfully exported to: `{output_report_path}`")
            # Read and prepare for download
            with open(output_report_path, "r", encoding="utf-8") as f:
                html_bytes = f.read()
            st.download_button(
                label="Download HTML Report",
                data=html_bytes,
                file_name=report_file_name,
                mime="text/html"
            )
        else:
            st.error(
                "Generation failed. Check if `ydata-profiling` is installed. "
                "Ensure your system has adequate memory available."
            )
