"""
SmartBelt v2 — Real-time Telemetry Charts
Renders Plotly line charts for temperature, vibration RMS, load cell tension,
and fused anomaly risk trajectory.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go


def create_telemetry_history_figure(history_records: List[Dict[str, Any]]) -> go.Figure:
    """Generate multi-channel telemetry plot from recent history buffer."""
    if not history_records:
        fig = go.Figure()
        fig.update_layout(
            title="Awaiting Sensor Telemetry...",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    df = pd.DataFrame(history_records)
    x = df.index

    fig = go.Figure()

    # Temperature
    if "temp_c" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["temp_c"],
            mode="lines", name="Temp (°C)",
            line=dict(color="#FF851B", width=2)
        ))

    # Vibration RMS
    if "vib_rms" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["vib_rms"],
            mode="lines", name="Vib RMS (g)",
            line=dict(color="#0074D9", width=2)
        ))

    # Load Cell
    if "load_kg" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["load_kg"],
            mode="lines", name="Load (kg)",
            line=dict(color="#2ECC40", width=2)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Sample Index", showgrid=True, gridcolor="#333333"),
        yaxis=dict(title="Measured Value", showgrid=True, gridcolor="#333333"),
    )
    return fig


def create_risk_gauge_figure(fused_score: float) -> go.Figure:
    """Generate a radial risk indicator gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fused_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fused Risk Score", 'font': {'size': 16, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0.0, 1.0], 'tickwidth': 1, 'tickcolor': "#888888"},
            'bar': {'color': "#E0E0E0"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#444444",
            'steps': [
                {'range': [0.0, 0.62710], 'color': '#28a745'},   # Nominal (Green)
                {'range': [0.62710, 0.68850], 'color': '#ffc107'}, # Elevated (Yellow)
                {'range': [0.68850, 1.0], 'color': '#dc3545'},    # Critical (Red)
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': fused_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "#FFFFFF"},
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig
