"""
SmartBelt — Intelligent Monitoring & Prediction of Conveyor Belt Damage & Joint Rupture
Phase 5: Streamlit Operator Dashboard (app.py)

Features:
- Camera / Video Selector (Sample videos, uploaded video, webcam)
- Live Video Overlay with high-contrast colored borders and anomaly markers
- Real-time Visual Anomaly Score (PatchCore WideResNet-50)
- Physical Sensor Telemetry Panel (clearly tagged SIMULATED)
- XGBoost Predictive Classifier / Baseline Scoring Engine Interface
- Multi-Modal Conservative Safety Fusion Decision Layer
- Prominent Flashing/High-Visibility Alert Banner for CRITICAL and WARNING states
- Anomaly Score Over Time Trend Chart (Visual vs Sensor vs Fused)
- Telemetry Audit History Table (downloadable)
- Hardware Interface Controller Status (ESP32 Simulation Mode)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Set project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SmartBelt Core Imports
from smartbelt_software.phase4_integration.smartbelt_core import (
    SmartBeltApplication,
    DEFAULT_CHECKPOINT,
    THRESHOLD_1,
    THRESHOLD_2,
    fuse_risk,
    classify_score,
)
from smartbelt_software.phase4_integration.xgboost_sensor_module import (
    XGBoostSensorPredictor,
    simulate_sensor_reading,
    SENSOR_SPEC_BOUNDS,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartBelt — Conveyor Joint & Damage Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Industrial Theme CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1a1f2c;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3748;
        margin-bottom: 12px;
    }
    .status-badge-normal {
        background-color: #28a745;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #ff9800;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .status-badge-critical {
        background-color: #dc3545;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { opacity: 1.0; transform: scale(1.0); }
        100% { opacity: 0.85; transform: scale(1.02); }
    }
    .simulated-pill {
        background-color: #4a5568;
        color: #ffcc00;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid #718096;
    }
    .hardware-pill-sim {
        background-color: #2d3748;
        color: #63b3ed;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px dashed #4299e1;
    }
    .alert-banner-critical {
        background-color: #74151e;
        border: 2px solid #e53e3e;
        color: #fff;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 16px;
        box-shadow: 0 0 15px rgba(229, 62, 62, 0.4);
    }
    .alert-banner-warning {
        background-color: #664d03;
        border: 2px solid #ffc107;
        color: #fff;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 16px;
    }
    .alert-banner-normal {
        background-color: #0f5132;
        border: 2px solid #198754;
        color: #fff;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Application Backend Instance
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing SmartBelt PatchCore model and fusion engine...")
def get_smartbelt_app():
    """Load and cache the PatchCore and Fusion engine."""
    app = SmartBeltApplication(
        checkpoint_path=DEFAULT_CHECKPOINT,
        device="cpu",
        telemetry_log_path=PROJECT_ROOT / "smartbelt_software" / "phase5_dashboard" / "dashboard_telemetry.csv",
    )
    return app


# -----------------------------------------------------------------------------
# Sidebar: Controls & Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ SmartBelt Controls")
st.sidebar.caption("Intelligent Conveyor Joint & Damage Monitor")

# 1. Video Source Selection
st.sidebar.subheader("1. Video Stream Source")
sample_videos_dir = PROJECT_ROOT / "videos"
sample_options = {}
if sample_videos_dir.exists():
    for f in sample_videos_dir.glob("*.mp4"):
        sample_options[f.name] = str(f)

source_type = st.sidebar.selectbox(
    "Input Mode",
    options=["Sample Video", "Upload Video (MP4/AVI)", "Live Webcam"],
    index=0,
)

video_path = None
uploaded_file = None

if source_type == "Sample Video":
    if sample_options:
        selected_sample = st.sidebar.selectbox(
            "Select Sample Video",
            options=list(sample_options.keys()),
            index=0 if "conveyor_with_real_damage.mp4" in sample_options else 0,
        )
        video_path = sample_options[selected_sample]
    else:
        st.sidebar.warning("No sample videos found in videos/ directory.")
elif source_type == "Upload Video (MP4/AVI)":
    uploaded_file = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_dir = PROJECT_ROOT / "smartbelt_software" / "phase5_dashboard" / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_path = str(temp_path)
elif source_type == "Live Webcam":
    cam_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
    video_path = int(cam_idx)

# 2. Sensor Telemetry Scenario Selector
st.sidebar.subheader("2. Sensor Telemetry")
st.sidebar.markdown(
    '<span class="simulated-pill">SIMULATED — not a real measurement</span>',
    unsafe_allow_html=True,
)
sensor_mode = st.sidebar.selectbox(
    "Sensor Simulation Scenario",
    options=[
        "Nominal Baseline (Healthy System)",
        "Bearing Overheat (78.5°C)",
        "Splice Mechanical Vibration (8.45 mm/s)",
        "Drive Pulley Slip (28.0%)",
        "Catastrophic Multi-Modal Rupture",
        "Manual Slider Adjustment",
    ],
    index=0,
)

manual_sensors = None
if sensor_mode == "Manual Slider Adjustment":
    st.sidebar.markdown("**Manual Sensor Adjustments:**")
    m_temp = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=110.0, value=38.5, step=0.5)
    m_vib = st.sidebar.slider("Vibration RMS (mm/s)", min_value=0.1, max_value=15.0, value=1.45, step=0.05)
    m_slip = st.sidebar.slider("Belt Speed Slip (%)", min_value=0.0, max_value=45.0, value=1.8, step=0.2)
    m_tens = st.sidebar.slider("Belt Tension (kN)", min_value=5.0, max_value=40.0, value=24.5, step=0.5)
    manual_sensors = {
        "temperature_c": m_temp,
        "vibration_rms_mms": m_vib,
        "speed_slip_pct": m_slip,
        "tension_kn": m_tens,
    }

# 3. Processing Parameters
st.sidebar.subheader("3. Processing Controls")
frame_step = st.sidebar.slider("Frame Skip Interval (FPS pacing)", min_value=1, max_value=15, value=5, step=1)
max_frames_to_run = st.sidebar.slider("Max Frames to Process", min_value=10, max_value=300, value=60, step=10)

# 4. Hardware Interface Status
st.sidebar.subheader("4. Hardware Interface (ESP32)")
st.sidebar.markdown(
    """
    <div class="hardware-pill-sim">
        ● ESP32 SERIAL: SIMULATION MODE<br>
        Port: DISCONNECTED (Virtual Relay Active)<br>
        Baud: 115200 (Ready for Phase 6)
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶ Start Conveyor Inspection", type="primary", use_container_width=True)
stop_btn = st.sidebar.button("⏹ Stop / Reset", use_container_width=True)


# -----------------------------------------------------------------------------
# Main Dashboard UI Layout
# -----------------------------------------------------------------------------
st.title("⚙️ SmartBelt Industrial Conveyor Joint & Damage Monitor")
st.caption(
    "Multi-Modal Anomaly Detection: PatchCore WideResNet-50 Vision + XGBoost/ISO-10816 Sensor Fusion Engine"
)

# Placeholder containers for dynamic live updating
alert_placeholder = st.empty()
header_metrics_col1, header_metrics_col2, header_metrics_col3, header_metrics_col4 = st.columns(4)

col_video, col_telemetry = st.columns([1.3, 1.0])

with col_video:
    st.subheader("Live Visual Inspection Stream")
    video_placeholder = st.empty()
    video_caption_placeholder = st.empty()

with col_telemetry:
    st.subheader("Real-Time Telemetry & Fusion Analysis")
    telemetry_placeholder = st.empty()

chart_placeholder = st.empty()
table_placeholder = st.empty()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "history" not in st.session_state or stop_btn:
    st.session_state["history"] = []
if "running" not in st.session_state or stop_btn:
    st.session_state["running"] = False

if run_btn:
    st.session_state["running"] = True
    st.session_state["history"] = []


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_current_sensor_reading(mode_str: str, manual_dict: Optional[Dict[str, float]], step_idx: int):
    """Generate or retrieve sensor readings for the step."""
    if mode_str == "Manual Slider Adjustment" and manual_dict is not None:
        # Add slight natural jitter
        jitter = float(np.random.normal(0, 0.02))
        return {
            "temperature_c": round(manual_dict["temperature_c"] * (1.0 + jitter), 2),
            "vibration_rms_mms": round(manual_dict["vibration_rms_mms"] * (1.0 + jitter), 2),
            "speed_slip_pct": round(manual_dict["speed_slip_pct"] * (1.0 + jitter), 2),
            "tension_kn": round(manual_dict["tension_kn"] * (1.0 + jitter), 2),
        }, True

    scenario_map = {
        "Nominal Baseline (Healthy System)": "nominal",
        "Bearing Overheat (78.5°C)": "bearing_overheat",
        "Splice Mechanical Vibration (8.45 mm/s)": "splice_vibration",
        "Drive Pulley Slip (28.0%)": "belt_slip",
        "Catastrophic Multi-Modal Rupture": "catastrophic_snap",
    }
    scenario_key = scenario_map.get(mode_str, "nominal")
    return simulate_sensor_reading(scenario_key)


def render_alert_banner(risk: str, consensus: str, visual_s: float, sensor_s: float, fused_s: float):
    """Render high-contrast visual alert banner based on risk status."""
    if risk == "CRITICAL":
        alert_placeholder.markdown(
            f"""
            <div class="alert-banner-critical">
                🚨 <strong>CRITICAL EMERGENCY ALERT:</strong> {consensus} | Fused Score: {fused_s:.4f} &ge; {THRESHOLD_2:.4f}<br>
                <span style="font-size: 0.95rem; font-weight: normal;">
                Immediate Action: <strong>MOTOR RELAY TRIPPED</strong> | Audible Buzzer: <strong>PULSED EMERGENCY</strong> | Beacon: <strong>FLASHING RED</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif risk == "WARNING":
        alert_placeholder.markdown(
            f"""
            <div class="alert-banner-warning">
                ⚠️ <strong>MAINTENANCE WARNING:</strong> {consensus} | Fused Score: {fused_s:.4f} &ge; {THRESHOLD_1:.4f}<br>
                <span style="font-size: 0.95rem; font-weight: normal;">
                Action: <strong>SCHEDULE INSPECTION</strong> | Motor Relay: <strong>CLOSED (RUNNING)</strong> | Beacon: <strong>AMBER</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        alert_placeholder.markdown(
            f"""
            <div class="alert-banner-normal">
                ✅ <strong>SYSTEM NORMAL:</strong> {consensus} | Fused Score: {fused_s:.4f} &lt; {THRESHOLD_1:.4f}<br>
                <span style="font-size: 0.95rem; font-weight: normal;">
                Status: <strong>CONTINUOUS PRODUCTION NOMINAL</strong> | Motor Relay: <strong>CLOSED (RUNNING)</strong> | Beacon: <strong>GREEN</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_telemetry_panel(packet: Dict[str, Any]):
    """Render the rich telemetry inspection panel."""
    readings = packet["raw_sensor_readings"]
    risk = packet["risk_status"]
    
    badge_class = f"status-badge-{risk.lower()}"

    with telemetry_placeholder.container():
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700;">Overall Conveyor Health</span>
                    <span class="{badge_class}">{risk}</span>
                </div>
                <div style="font-size: 0.88rem; color: #a0aec0; margin-bottom: 12px;">
                    Consensus Logic: <strong>{packet['consensus_state']}</strong>
                </div>
                <div style="margin-bottom: 6px;">
                    <small>Fused Anomaly Score: <strong>{packet['final_risk_score']:.4f}</strong> (Norm &lt; {THRESHOLD_1:.3f}, Crit &ge; {THRESHOLD_2:.3f})</small>
                </div>
                <div style="background-color: #2d3748; border-radius: 4px; height: 10px; width: 100%;">
                    <div style="background-color: {'#dc3545' if risk=='CRITICAL' else ('#ff9800' if risk=='WARNING' else '#28a745')}; width: {min(100, int(packet['final_risk_score'] * 100))}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700;">Computer Vision (PatchCore)</span>
                    <span style="color: {'#e53e3e' if packet['visual_score'] >= THRESHOLD_2 else ('#ecc94b' if packet['visual_score'] >= THRESHOLD_1 else '#48bb78')}; font-weight: 700;">
                        Score: {packet['visual_score']:.4f}
                    </span>
                </div>
                <small style="color: #718096;">Model: PatchCore WideResNet-50 Memory Bank | AUROC: 1.000</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700;">Physical Sensor Telemetry</span>
                    <span class="simulated-pill">SIMULATED — not a real measurement</span>
                </div>
                <div style="font-size: 0.82rem; color: #a0aec0; margin-bottom: 10px;">
                    Engine: <code>{packet['sensor_model_type']}</code> | Dominant: <strong>{packet['dominant_sensor'].upper()}</strong>
                </div>
                <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #2d3748;">
                        <td style="padding: 4px 0;">Bearing Temp:</td>
                        <td style="text-align: right; font-weight: 600; color: {'#fc8181' if readings.get('temperature_c',0) > 70 else '#e2e8f0'};">{readings.get('temperature_c', 'N/A')} °C</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2d3748;">
                        <td style="padding: 4px 0;">Vibration RMS:</td>
                        <td style="text-align: right; font-weight: 600; color: {'#fc8181' if readings.get('vibration_rms_mms',0) > 4.5 else '#e2e8f0'};">{readings.get('vibration_rms_mms', 'N/A')} mm/s</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2d3748;">
                        <td style="padding: 4px 0;">Speed Slip:</td>
                        <td style="text-align: right; font-weight: 600; color: {'#fc8181' if readings.get('speed_slip_pct',0) > 10 else '#e2e8f0'};">{readings.get('speed_slip_pct', 'N/A')} %</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0;">Belt Tension:</td>
                        <td style="text-align: right; font-weight: 600; color: {'#fc8181' if readings.get('tension_kn',25) < 15 or readings.get('tension_kn',25) > 35 else '#e2e8f0'};">{readings.get('tension_kn', 'N/A')} kN</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-weight: 700;">Hardware Command Status</span>
                <div style="margin-top: 6px; font-size: 0.88rem;">
                    Relay: <strong>{packet['motor_relay_state']}</strong> | Buzzer: <strong>{packet['buzzer_state']}</strong> | Beacon: <strong>{packet['beacon_color']}</strong>
                </div>
                <div style="margin-top: 4px; font-size: 0.80rem; color: #718096;">
                    Protocol Dispatch: <code>{packet['hardware_command']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Video Inference & Streaming Execution Loop
# -----------------------------------------------------------------------------
if st.session_state["running"]:
    if video_path is None:
        st.error("No valid video input source selected. Please select or upload a video in the sidebar.")
        st.session_state["running"] = False
    else:
        app = get_smartbelt_app()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_path}")
            st.session_state["running"] = False
        else:
            frame_counter = 0
            processed_counter = 0

            status_bar = st.progress(0, text="Initializing conveyor inspection stream...")

            while cap.isOpened() and processed_counter < max_frames_to_run:
                ret, frame_bgr = cap.read()
                if not ret:
                    # Loop video if sample video reached end
                    if isinstance(video_path, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                    else:
                        break

                frame_counter += 1
                if frame_counter % frame_step != 0:
                    continue

                processed_counter += 1

                # 1. Obtain sensor reading (Simulated)
                sensor_reading, is_simulated = get_current_sensor_reading(
                    sensor_mode, manual_sensors, processed_counter
                )

                # 2. Run Unified Core Inference & Fusion
                t0 = time.time()
                packet = app.process_frame(
                    frame_bgr=frame_bgr,
                    sensor_readings=sensor_reading,
                    is_simulated_sensor=is_simulated,
                    frame_idx=frame_counter,
                )
                dt = time.time() - t0

                # 3. Render High-Contrast Annotated Overlay
                annotated_bgr = app.render_display_overlay(frame_bgr, packet)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # 4. Update Header Metrics
                header_metrics_col1.metric("Processed Steps", f"{processed_counter} / {max_frames_to_run}")
                header_metrics_col2.metric("PatchCore Score", f"{packet['visual_score']:.4f}", delta=f"{packet['visual_risk']}")
                header_metrics_col3.metric("Sensor Score", f"{packet['sensor_score']:.4f}", delta=f"{packet['sensor_risk']}")
                header_metrics_col4.metric("Fused Score", f"{packet['final_risk_score']:.4f}", delta=f"{packet['risk_status']}")

                # 5. Update Alert Banner & Telemetry Panel
                render_alert_banner(
                    packet["risk_status"],
                    packet["consensus_state"],
                    packet["visual_score"],
                    packet["sensor_score"],
                    packet["final_risk_score"],
                )
                render_telemetry_panel(packet)

                # 6. Update Video Display
                video_placeholder.image(annotated_rgb, channels="RGB", width="stretch")
                video_caption_placeholder.caption(
                    f"Frame {frame_counter:04d} | Step {processed_counter} | Inference Latency: {dt:.2f}s | "
                    f"Consensus: {packet['consensus_state']}"
                )

                # 7. Record History
                st.session_state["history"].append({
                    "Step": processed_counter,
                    "Frame": frame_counter,
                    "Visual Score": packet["visual_score"],
                    "Sensor Score": packet["sensor_score"],
                    "Fused Score": packet["final_risk_score"],
                    "Risk Status": packet["risk_status"],
                    "Consensus": packet["consensus_state"],
                    "Dominant Sensor": packet["dominant_sensor"],
                    "Bearing Temp (°C)": packet["raw_sensor_readings"].get("temperature_c", 0.0),
                    "Vib RMS (mm/s)": packet["raw_sensor_readings"].get("vibration_rms_mms", 0.0),
                    "Speed Slip (%)": packet["raw_sensor_readings"].get("speed_slip_pct", 0.0),
                    "Tension (kN)": packet["raw_sensor_readings"].get("tension_kn", 0.0),
                    "Relay State": packet["motor_relay_state"],
                })

                # Update progress bar
                pct = min(1.0, processed_counter / float(max_frames_to_run))
                status_bar.progress(pct, text=f"Inspecting Conveyor... ({processed_counter}/{max_frames_to_run})")

                time.sleep(0.01)

            cap.release()
            status_bar.progress(1.0, text="Inspection Run Complete.")
            st.session_state["running"] = False

# -----------------------------------------------------------------------------
# Bottom Section: Trends & Audit History Table
# -----------------------------------------------------------------------------
if st.session_state.get("history"):
    hist_df = pd.DataFrame(st.session_state["history"])

    with chart_placeholder.container():
        st.subheader("📈 Anomaly Trend Over Time (Multi-Modal Consensus)")
        chart_data = hist_df.set_index("Step")[["Visual Score", "Sensor Score", "Fused Score"]]
        st.line_chart(chart_data, color=["#3182ce", "#ed8936", "#e53e3e"])

        st.caption(
            f"Reference: Threshold 1 (Warning) = {THRESHOLD_1:.4f} | "
            f"Threshold 2 (Critical) = {THRESHOLD_2:.4f}"
        )

    with table_placeholder.container():
        st.subheader("📋 Step Audit History & Telemetry Log")
        st.dataframe(hist_df.tail(20), use_container_width=True)
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry Log (CSV)",
            data=csv_data,
            file_name="smartbelt_dashboard_telemetry.csv",
            mime="text/csv",
        )
else:
    if not st.session_state["running"]:
        alert_placeholder.info(
            "Ready to inspect conveyor belt. Click '▶ Start Conveyor Inspection' in the sidebar to stream frames, compute PatchCore visual anomaly distances, and fuse simulated telemetry."
        )
