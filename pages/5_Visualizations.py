import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import inject_custom_css
from core.visualizer import (
    plot_histogram, plot_scatter, plot_bar, plot_box, 
    plot_correlation_heatmap, plot_correlation_network
)
from core.explain import ExplainabilityEngine

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded
if st.session_state.df is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Visualization Studio
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

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Visualization Studio: {filename}
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Interactive Plotly visualizations and neural/tree dependencies.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_dist, tab_rel, tab_corr, tab_dep, tab_imp = st.tabs([
    "📊 Distribution Analysis", 
    "📈 Relationship Analysis", 
    "🗺️ Correlation Analysis", 
    "🕸️ Dependency Analysis",
    "🔍 Feature Importance"
])

# 1. Distribution Analysis
with tab_dist:
    st.subheader("Numerical Variable Spreads")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        dist_col = st.selectbox("Select variable for distribution:", numeric_cols, key="dist_sel")
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.plotly_chart(plot_histogram(df, dist_col), use_container_width=True)
        with sub_c2:
            st.plotly_chart(plot_box(df, dist_col), use_container_width=True)
    else:
        st.info("No numerical variables found for distribution plotting.")

# 2. Relationship Analysis
with tab_rel:
    st.subheader("Categorical Frequencies & Comparisons")
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    col_bar_view, col_scatter_view = st.columns(2)
    
    with col_bar_view:
        if categorical_cols:
            bar_col = st.selectbox("Select categorical variable:", categorical_cols, key="bar_sel")
            st.plotly_chart(plot_bar(df, bar_col), use_container_width=True)
        else:
            st.info("No categorical columns available.")
            
    with col_scatter_view:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis variable:", numeric_cols, index=0)
            y_col = st.selectbox("Select Y-axis variable:", numeric_cols, index=min(1, len(numeric_cols)-1))
            st.plotly_chart(plot_scatter(df, x_col, y_col), use_container_width=True)
        else:
            st.info("At least two numerical columns are required for scatter plotting.")

# 3. Correlation Analysis
with tab_corr:
    st.subheader("Generalized Dependency Matrix Heatmap")
    dep_data = metrics["dependency"]
    st.plotly_chart(
        plot_correlation_heatmap(dep_data["matrix"], dep_data["columns"]), 
        use_container_width=True
    )

# 4. Dependency Analysis (Circular Network Graph)
with tab_dep:
    st.subheader("Feature Dependency Circular Network Graph")
    st.markdown(
        f"This graph maps variable dependencies based on the active Settings threshold (`{st.session_state.correlation_threshold}`)."
    )
    
    nodes = []
    for col in dep_data["columns"]:
        nodes.append({"id": col, "label": col, "type": semantic_types.get(col, {}).get("type", "nominal")})
        
    edges = []
    matrix = np.array(dep_data["matrix"])
    cols_list = dep_data["columns"]
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            w = matrix[i, j]
            if w >= st.session_state.correlation_threshold:
                edges.append({
                    "source": cols_list[i],
                    "target": cols_list[j],
                    "weight": float(w)
                })
                
    st.plotly_chart(plot_correlation_network(nodes, edges), use_container_width=True)

# 5. Feature Importance
with tab_imp:
    st.subheader("Offline Feature Importance")
    st.markdown(
        "Fitted with a decision tree model to estimate feature relevance weights."
    )
    
    # Use target column configured in Settings, or allow overriding here
    default_target = st.session_state.target_column
    target_list = [None] + df.columns.tolist()
    
    # Find index of default target if set
    default_idx = 0
    if default_target in target_list:
        default_idx = target_list.index(default_target)
        
    target_select = st.selectbox(
        "Select Target Column:", 
        target_list,
        index=default_idx,
        key="target_explain_sel"
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
            st.info("💡 Tree split frequency yields the relative feature importance rating.")
        else:
            st.warning("⚠️ Insufficient non-null sample rows or features to compute importances.")
    else:
        st.write("Please select a target variable to plot.")
        if default_target is None:
            st.markdown(
                "*Tip: You can pre-configure a global target column on the **Analysis Settings** page.*"
            )
