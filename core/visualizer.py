import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger("core.visualizer")

# Custom Dark Color Palette (Slate Theme matching settings.py)
COLORS = ["#1A56DB", "#7E3AF2", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899", "#8B5CF6"]
TEMPLATE = "plotly_dark"

def apply_layout_styles(fig: go.Figure, title: str, x_label: str = "", y_label: str = "") -> go.Figure:
    """
    Applies custom styling to a Plotly figure to make it look premium and modern.
    """
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'family': 'Outfit, sans-serif'}
        },
        template=TEMPLATE,
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=x_label,
            gridcolor="#334155",
            linecolor="#334155",
            zerolinecolor="#334155",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title=y_label,
            gridcolor="#334155",
            linecolor="#334155",
            zerolinecolor="#334155",
            tickfont=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="#334155",
            borderwidth=1
        )
    )
    return fig

def plot_histogram(df: pd.DataFrame, column: str, color_col: Optional[str] = None) -> go.Figure:
    """
    Plots an interactive histogram showing data distribution.
    """
    logger.info(f"Generating histogram for column: {column}")
    fig = px.histogram(
        df,
        x=column,
        color=color_col,
        color_discrete_sequence=COLORS,
        marginal="box",
        opacity=0.85
    )
    return apply_layout_styles(fig, f"Distribution of {column}", column, "Count")

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
    """
    Plots an interactive scatter plot showing the relationship between two variables.
    """
    logger.info(f"Generating scatter plot: {x_col} vs {y_col}")
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=COLORS,
        opacity=0.8
    )
    return apply_layout_styles(fig, f"{x_col} vs {y_col} Relationship", x_col, y_col)

def plot_bar(df: pd.DataFrame, column: str, limit: int = 15) -> go.Figure:
    """
    Plots a frequency count bar chart for a categorical column.
    """
    logger.info(f"Generating bar chart for column: {column}")
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, "Count"]
    
    # Cap categories
    if len(value_counts) > limit:
        other_sum = value_counts.iloc[limit:]["Count"].sum()
        value_counts = value_counts.iloc[:limit]
        value_counts.loc[len(value_counts)] = ["Other...", other_sum]
        
    fig = px.bar(
        value_counts,
        x=column,
        y="Count",
        color="Count",
        color_continuous_scale="Viridis",
        text_auto=True
    )
    fig.update_coloraxes(showscale=False)
    return apply_layout_styles(fig, f"Frequency of Categories in {column}", column, "Frequency Count")

def plot_box(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> go.Figure:
    """
    Plots a box plot showing statistical quartiles and outliers.
    """
    logger.info(f"Generating box plot for column: {y_col}")
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=x_col,
        color_discrete_sequence=COLORS
    )
    title = f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else "")
    return apply_layout_styles(fig, title, x_col or "", y_col)

def plot_correlation_heatmap(matrix: List[List[float]], labels: List[str]) -> go.Figure:
    """
    Plots an interactive correlation matrix heatmap.
    """
    logger.info("Generating correlation matrix heatmap.")
    z = np.array(matrix)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    return apply_layout_styles(fig, "Generalized Feature Dependencies", "Features", "Features")

def plot_correlation_network(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders an interactive 2D node-link graph of high correlation edges in Plotly.
    Uses a simple circular layout.
    """
    logger.info("Generating correlation network diagram.")
    fig = go.Figure()
    
    n = len(nodes)
    if n == 0:
        return fig
        
    # Assign coordinates in a circle
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node["id"]] = (np.cos(angle), np.sin(angle))
        
    # Draw edges first (so they sit below nodes)
    for edge in edges:
        x0, y0 = pos[edge["source"]]
        x1, y1 = pos[edge["target"]]
        weight = edge["weight"]
        
        # Edge lines
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight * 3, color=f"rgba(126, 58, 242, {weight})"), # Purple
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    # Map types to node colors
    type_color_map = {
        "continuous_numerical": "#1A56DB", # Blue
        "discrete_numerical": "#06B6D4",   # Cyan
        "nominal": "#7E3AF2",              # Purple
        "binary": "#10B981",               # Green
        "boolean": "#10B981",              # Green
        "identifier": "#EF4444",           # Red
        "datetime": "#F59E0B"              # Orange
    }
    
    for node in nodes:
        x, y = pos[node["id"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Feature: {node['id']}<br>Type: {node['type']}")
        node_colors.append(type_color_map.get(node["type"], "#94A3B8"))
        
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n["label"] for n in nodes],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line=dict(width=2, color='#F8FAFC')
        ),
        showlegend=False
    ))
    
    fig = apply_layout_styles(fig, "Feature Association Network Graph", "", "")
    # Hide axis lines for network
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    )
    return fig
