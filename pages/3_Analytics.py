import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import inject_custom_css, format_bytes, normalize_dataframe_for_rendering
from core.statistics import summarize_column

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Analytics Explorer
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
recs = st.session_state.recs
insights = st.session_state.insights

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Analytics Explorer
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Detailed structural analysis, data health matrices, and searchable column metrics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# FEATURE SEARCH EXPLORER
# ----------------------------------------------------
st.markdown("## 🔍 Searchable Feature Explorer")
col_search = st.selectbox("Search for a column to inspect:", df.columns.tolist())

if col_search:
    st.markdown(f"### Feature Summary: `{col_search}`")
    sec1, sec2 = st.columns([1, 1.2])
    
    with sec1:
        st.markdown("**📊 Custom Statistics**")
        col_type = semantic_types.get(col_search, {}).get("type", "nominal")
        is_num = col_type in ["continuous_numerical", "discrete_numerical", "boolean", "binary"]
        
        if is_num and pd.api.types.is_numeric_dtype(df[col_search]):
            col_stats = summarize_column(df[col_search])
            for k, v in col_stats.items():
                if isinstance(v, float):
                    st.markdown(f"- **{k.replace('_', ' ').title()}**: {v:.4f}")
                else:
                    st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")
        else:
            st.markdown("*Non-numeric categorical feature. Custom numeric statistics are not applicable.*")
            st.markdown(f"- **Cardinality**: {df[col_search].nunique()} unique labels")
            st.markdown(f"- **Missing Value Count**: {df[col_search].isna().sum()}")
            
    with sec2:
        st.markdown("**🕸️ High Dependency Connections**")
        dep_data = metrics["dependency"]
        cols_list = dep_data["columns"]
        matrix = np.array(dep_data["matrix"])
        
        if col_search in cols_list:
            idx = cols_list.index(col_search)
            correlations = []
            for j, c_other in enumerate(cols_list):
                if idx != j:
                    weight = matrix[idx, j]
                    if weight >= st.session_state.correlation_threshold:
                        correlations.append((c_other, weight))
            
            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                for c_other, w in correlations[:5]:
                    st.markdown(f"- Connected with **{c_other}** (Score: `{w:.2f}`)")
            else:
                st.markdown("*No connections above active correlation threshold.*")
        
        st.markdown("**🛠️ Actionable Diagnostics**")
        col_recs = [r for r in recs if col_search in r["features"]]
        col_insights = [i for i in insights if col_search in i["columns"]]
        
        if col_recs:
            for r in col_recs:
                st.markdown(f"- **{r['type']}**: {r['suggested_fix']}")
        else:
            st.markdown("- *No immediate preprocessing recommended.*")
            
        if col_insights:
            for i in col_insights:
                st.markdown(f"- **Insight**: *{i['description']}*")

st.markdown("---")

# Main Section Tabs
tab_schema, tab_stats, tab_risks = st.tabs([
    "📂 Column Schema & Dtypes", 
    "🧮 Custom Statistics Table", 
    "⚠️ Structural Risk Matrix"
])

# 1. Schema mapping
with tab_schema:
    st.subheader("Schema Mapping (Pandas vs Semantic Parse)")
    schema_records = []
    for col, info in semantic_types.items():
        schema_records.append({
            "Column Name": col,
            "Pandas Type": str(df[col].dtype),
            "Semantic Category": info["type"].replace("_", " ").title(),
            "Confidence": f"{info['confidence']}%",
            "Explanation": info["explanation"]
        })
    st.dataframe(normalize_dataframe_for_rendering(pd.DataFrame(schema_records)), use_container_width=True)

# 2. Statistics matrix
with tab_stats:
    st.subheader("All Numerical Column Descriptors")
    stats_df = pd.DataFrame(metrics["statistics"]).T
    st.dataframe(normalize_dataframe_for_rendering(stats_df), use_container_width=True)

# 3. Risk matrix
with tab_risks:
    st.subheader("Identified Health Risks")
    risk_rows = []
    
    # Check redundancy
    redundancy = metrics["ml_readiness"]["checks"].get("redundancy", {})
    if redundancy.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "Multicollinearity",
            "Severity": "High",
            "Detail": redundancy["detail"],
            "Suggested Fix": "Drop one of the highly correlated columns using the clean modules."
        })
        
    # Check duplicates
    dups = metrics["ml_readiness"]["checks"].get("duplicates", {})
    if dups.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "Row Duplication",
            "Severity": "High",
            "Detail": dups["detail"],
            "Suggested Fix": "Apply deduplication using core cleaner."
        })
        
    # Check missing
    miss = metrics["ml_readiness"]["checks"].get("missing_values", {})
    if miss.get("status") == "WARNING":
        risk_rows.append({
            "Risk Category": "Missing Values",
            "Severity": "Medium",
            "Detail": miss["detail"],
            "Suggested Fix": "Impute values with median/mode or drop features with >40% missingness."
        })
        
    # Check outliers
    total_outliers = metrics.get("outliers", {}).get("total_outliers", 0)
    outliers_pct = (total_outliers / df.size) * 100 if df.size > 0 else 0
    if outliers_pct > 2.0:
        risk_rows.append({
            "Risk Category": "Extreme Outliers",
            "Severity": "Medium",
            "Detail": f"{total_outliers} cell outliers flagged ({outliers_pct:.2f}% of cells).",
            "Suggested Fix": "Standardize numerical features using RobustScaler or Winsorization."
        })
        
    if risk_rows:
        st.dataframe(normalize_dataframe_for_rendering(pd.DataFrame(risk_rows)), use_container_width=True)
    else:
        st.success("🎉 No structural risks flagged. The dataset is clean.")
