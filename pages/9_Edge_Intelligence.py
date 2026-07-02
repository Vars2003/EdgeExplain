import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from utils.helpers import inject_custom_css
from intelligence.exporter import IntelligenceExporter
from intelligence.explanations import ExplanationEngine

# Re-inject CSS for visual consistency
inject_custom_css()

# Validate that a dataset is loaded and memory is cached
if st.session_state.df is None or st.session_state.memory is None:
    st.markdown(
        """
        <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
            <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Edge Intelligence Layer
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("⚠️ No dataset or intelligence context loaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
filename = st.session_state.filename
memory = st.session_state.memory
context = memory["context"]
graph = memory["graph"]
summaries = memory["summaries"]
recs = memory["recommendations"]

st.markdown(
    f"""
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Edge Intelligence: {filename}
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Versioned semantic knowledge bases, queryable graphs, and LLM briefing context.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_context, tab_graph, tab_explanations, tab_summaries, tab_memory = st.tabs([
    "📋 Context Summary", 
    "🕸️ Knowledge Graph Map", 
    "❓ Explanation Directives",
    "📝 Briefing Summaries", 
    "💾 Memory Exporter"
])

# 1. Context Summary
with tab_context:
    st.subheader("Dataset Inferred Context")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style='color:#38BDF8; margin:0 0 10px 0;'>Semantic Information</h4>
                <p>- **Dataset Name**: {context['dataset_name']}</p>
                <p>- **Estimated Business Domain**: {context['domain']}</p>
                <p>- **Structural Focus**: {context['dataset_type']}</p>
                <p>- **Recommended ML Pipeline**: {context['recommended_ml_task']}</p>
                <p>- **Candidate Targets**: {', '.join(context['candidate_targets']) if context['candidate_targets'] else 'None inferred'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_c2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style='color:#818CF8; margin:0 0 10px 0;'>Data Quality Summary</h4>
                <p>- **Data Quality Score**: {context['quality_overview']['quality_score']:.1f}%</p>
                <p>- **ML Readiness Index**: {context['quality_overview']['ml_readiness_score']:.1f}%</p>
                <p>- **Duplicate Row Count**: {context['quality_overview']['duplicate_rows_count']}</p>
                <p>- **Null Cells Percentage**: {context['quality_overview']['missing_cells_pct']:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# 2. Knowledge Graph Map
with tab_graph:
    st.subheader("Semantic Knowledge Network Map")
    st.markdown("Visualizes relationships linking the Dataset, Features, Dependencies, Preprocessing rules, and Models.")
    
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    n = len(nodes)
    if n > 0:
        # Generate layout coordinates
        pos = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            pos[node["id"]] = (np.cos(angle), np.sin(angle))
            
        fig = go.Figure()
        
        # Plot edges
        for edge in edges:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1.5, color="rgba(148, 163, 184, 0.25)"), # Grey
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
        # Plot nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        # Color coding by node category
        type_color_map = {
            "dataset": "#EF4444",        # Red
            "feature": "#1A56DB",        # Blue
            "dependency": "#06B6D4",     # Cyan
            "recommendation": "#7E3AF2", # Purple
            "algorithm": "#10B981"       # Green
        }
        
        for node in nodes:
            x, y = pos[node["id"]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node: {node['label']}<br>Category: {node['type']}")
            node_colors.append(type_color_map.get(node["type"], "#94A3B8"))
            
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node["label"][:15] for node in nodes],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=14,
                line=dict(width=1.5, color='#F8FAFC')
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

# 3. Explanation Directives
with tab_explanations:
    st.subheader("Standardized Explanations Schema")
    st.markdown("Each preprocessing recommendation is explained through structured Q&A logic:")
    
    for r in recs:
        exp = ExplanationEngine.explain_recommendation(r)
        
        st.markdown(
            f"""
            <div class="glass-card" style='border-left: 4px solid #7E3AF2;'>
                <strong style='font-size: 1.1rem; color: #F8FAFC;'>❓ Why and how to apply: {exp['title']}?</strong>
                <p style='margin: 8px 0 0 0; font-size: 0.95rem; color:#E2E8F0;'><strong>Reason</strong>: {exp['reason']}</p>
                <div style='margin-top: 10px; font-size: 0.85rem; color:#94A3B8;'>
                    <span style='color: #38BDF8;'><strong>Evidence</strong>: {', '.join(exp['evidence'])}</span><br>
                    <span style='color: #34D399;'><strong>Alternatives</strong>: {', '.join(exp['alternatives'])}</span><br>
                    <span style='color: #A78BFA;'><strong>References</strong>: {', '.join(exp['references'])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# 4. Briefing Summaries
with tab_summaries:
    st.subheader("Generated Briefing Reports")
    
    with st.expander("Executive Overview", expanded=True):
        st.markdown(summaries["executive"])
    with st.expander("Technical Summary"):
        st.markdown(summaries["technical"])
    with st.expander("ML & Model Recommendations"):
        st.markdown(summaries["ml"])
    with st.expander("Data Quality Diagnostics"):
        st.markdown(summaries["quality"])

# 5. Memory Exporter
with tab_memory:
    st.subheader("JSON Memory Exporter")
    st.markdown("The JSON block below is the exact structured interface used to ground the local LLM prompts in Phase 4.")
    
    # Download JSON Memory Object
    memory_str = json.dumps(memory, indent=2)
    st.download_button(
        label="📥 Download JSON Memory Object",
        data=memory_str,
        file_name=f"memory_{filename.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Download Markdown Briefing
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    brief_file_name = f"briefing_{filename.split('.')[0]}.md"
    brief_file_path = os.path.join(reports_dir, brief_file_name)
    
    # Compute briefing using exporter helper
    IntelligenceExporter.export_markdown_briefing(memory, brief_file_path)
    
    with open(brief_file_path, "r", encoding="utf-8") as f:
        md_bytes = f.read()
        
    st.download_button(
        label="📥 Download Markdown Executive Briefing",
        data=md_bytes,
        file_name=brief_file_name,
        mime="text/markdown",
        use_container_width=True
    )
    
    st.markdown("---")
    st.json(memory)
