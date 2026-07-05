import plotly.graph_objects as go
import numpy as np

def generate_topology_chart(num_clients: int) -> go.Figure:
    """
    Generates an interactive Plotly Node-Link network diagram showing
    a central Server hub connected to simulated clients.
    """
    # Node coordinates: Server at center (0,0), Clients distributed in a circle
    x = [0.0]
    y = [0.0]
    labels = ["Server Hub"]
    colors = ["#38BDF8"] # Sky Blue for Server Hub
    
    theta = np.linspace(0, 2*np.pi, num_clients, endpoint=False)
    for i, t in enumerate(theta):
        x.append(np.cos(t))
        y.append(np.sin(t))
        labels.append(f"Client {i+1}")
        colors.append("#818CF8") # Indigo for Clients
        
    # Create links
    edge_x = []
    edge_y = []
    for i in range(1, num_clients + 1):
        edge_x.extend([0.0, x[i], None])
        edge_y.extend([0.0, y[i], None])
        
    fig = go.Figure()
    
    # Add connection lines (edges)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color="rgba(148, 163, 184, 0.4)"),
        hoverinfo="none",
        mode="lines"
    ))
    
    # Add node markers
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        hoverinfo="text",
        text=labels,
        textposition="bottom center",
        textfont=dict(color="#E2E8F0", size=10),
        marker=dict(
            showscale=False,
            color=colors,
            size=[35] + [22]*num_clients,
            line=dict(width=2, color="#0F172A")
        )
    ))
    
    fig.update_layout(
        title=dict(text="Collaborative network topology", x=0.5, font=dict(color="#94A3B8", size=14)),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
