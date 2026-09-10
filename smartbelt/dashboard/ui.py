"""
SmartBelt v2 — Streamlit Dashboard UI
Provides visual layout components for real-time monitoring and calibration.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import streamlit as st

from smartbelt.dashboard.charts import create_risk_gauge_figure, create_telemetry_history_figure
from smartbelt.fusion.risk_fusion import FusionState
from smartbelt.pipeline.orchestrator import SmartBeltOrchestrator, SmartBeltState


def render_header(state: SmartBeltState) -> None:
    """Render top title banner and hardware connectivity badges."""
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center;
                    padding: 8px 16px; background-color: #1a1c24; border-radius: 8px; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #ffffff; font-family: sans-serif;">SmartBelt v2 — Hardware Inspection System</h2>
                <span style="color: #8c9ba5; font-size: 13px;">Real Hardware Stream & Multimodal Anomaly Detection</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status Badges
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cam_status = "🟢 ACTIVE" if state.camera_fps > 0 else "🔴 OFFLINE"
        st.metric("Webcam (Logitech C270)", cam_status, f"{state.camera_fps:.1f} FPS")

    with c2:
        sensor_status = "🟢 HEALTHY" if state.sensor_healthy else "🔴 DISCONNECTED"
        st.metric("ESP32 IoT Bus", sensor_status, f"{state.serial_packets_total} pkts")

    with c3:
        inf_status = "🟢 READY" if state.inference_fps > 0 else "🟡 IDLE"
        st.metric("PatchCore CPU", inf_status, f"{state.inference_latency_s:.2f}s lat")

    with c4:
        calib_status = "🟢 CALIBRATED" if state.calibrated else "🟠 NEED CALIBRATION"
        st.metric("Baseline Profile", calib_status)


def render_alert_banner(state: SmartBeltState) -> None:
    """Displays color-coded operational risk banner."""
    f = state.fusion
    state_enum = f.state

    if state_enum == FusionState.MULTIMODAL_EMERGENCY:
        bg_color = "#721c24"
        border_color = "#f5c6cb"
        text_color = "#f8d7da"
        title = "CRITICAL ALERT: CONFIRMED MULTIMODAL EMERGENCY"
        desc = "Simultaneous visual surface damage and severe mechanical distress detected! Immediate inspection required."
    elif state_enum == FusionState.VISION_DOMINANT:
        bg_color = "#856404"
        border_color = "#ffeeba"
        text_color = "#fff3cd"
        title = "HIGH ALERT: VISUAL BELT DAMAGE DETECTED"
        desc = f"PatchCore identified surface rupture / tearing anomaly (Score: {f.visual_score:.4f})."
    elif state_enum == FusionState.SENSOR_DOMINANT:
        bg_color = "#856404"
        border_color = "#ffeeba"
        text_color = "#fff3cd"
        title = "HIGH ALERT: MECHANICAL DISTRESS DETECTED"
        desc = f"IoT sensors detected severe vibration, thermal, or load deviation (Score: {f.sensor_score:.4f})."
    elif state_enum == FusionState.ELEVATED:
        bg_color = "#533f03"
        border_color = "#ffe8a1"
        text_color = "#fff3cd"
        title = "WARNING: ELEVATED INSPECTION ZONE"
        desc = "System readings are trending above nominal baseline. Maintain operator observation."
    else:
        bg_color = "#155724"
        border_color = "#c3e6cb"
        text_color = "#d4edda"
        title = "NOMINAL OPERATION"
        desc = "Conveyor surface integrity and IoT telemetry are within healthy operating parameters."

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_color};
                    color: {text_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
            <strong style="font-size: 16px;">{title}</strong>
            <div style="font-size: 14px; margin-top: 4px;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_view(state: SmartBeltState) -> None:
    """Renders real-time video feed and sensor gauges in 2 columns."""
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Live Inspection Stream")
        if state.annotated_frame is not None:
            rgb_frame = cv2.cvtColor(state.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, channels="RGB", width="stretch")
        else:
            st.info("Awaiting camera frame from background capture thread...")

    with col_right:
        st.subheader("Anomaly Risk & Metrics")
        st.plotly_chart(create_risk_gauge_figure(state.fusion.fused_score), use_container_width=True)

        # Raw Telemetry Readouts
        pkt = state.sensor_packet
        m1, m2 = st.columns(2)
        with m1:
            temp_val = f"{pkt.temp_c:.1f} °C" if pkt else "—"
            st.metric("MLX90614 Temp", temp_val)

            load_val = f"{pkt.load_kg:.2f} kg" if pkt else "—"
            st.metric("HX711 Belt Load", load_val)

        with m2:
            vib_val = f"{pkt.vib_rms:.3f} g" if pkt else "—"
            st.metric("MPU-6050 Vib RMS", vib_val)

            speed_val = f"{pkt.belt_speed_mps:.2f} m/s" if pkt else "—"
            st.metric("E18 IR Speed", speed_val)


def render_telemetry_history(history: List[Dict[str, Any]]) -> None:
    """Render rolling historical sensor plots."""
    st.subheader("Physical Telemetry Trends (Last 60 Samples)")
    st.plotly_chart(create_telemetry_history_figure(history), use_container_width=True)


def render_sidebar_calibration(orchestrator: SmartBeltOrchestrator) -> None:
    """Sidebar controls for running live sensor baseline calibration."""
    st.sidebar.title("System Controls")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sensor Calibration")
    st.sidebar.caption(
        "Run a 30-60 second baseline collection while the conveyor is operating nominally. "
        "This computes machine-specific mean and std dev profiles for zero-mock scoring."
    )

    duration = st.sidebar.slider("Calibration Duration (seconds)", min_value=15, max_value=120, value=30, step=5)

    if st.sidebar.button("Run Baseline Calibration", type="primary", use_container_width=True):
        progress_bar = st.sidebar.progress(0, text="Calibrating physical sensors...")

        def on_prog(elapsed: float, total: float):
            pct = min(1.0, elapsed / total)
            progress_bar.progress(pct, text=f"Collecting samples ({int(elapsed)}s / {int(total)}s)...")

        success = orchestrator.run_calibration(duration_s=float(duration), on_progress=on_prog)
        if success:
            progress_bar.progress(1.0, text="Calibration Complete!")
            st.sidebar.success("Sensor baseline updated successfully.")
        else:
            st.sidebar.error("Calibration failed: insufficient ESP32 packets received.")
