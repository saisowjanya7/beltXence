"""
SmartBelt v2 — Live Operations Dashboard
Full physical hardware inspection: Logitech C270 USB + ESP32 Sensor Bus + PatchCore AI.
"""

import sys
import time
from pathlib import Path
import streamlit as st

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbelt.dashboard.ui import (
    render_alert_banner,
    render_header,
    render_live_view,
    render_sidebar_calibration,
    render_telemetry_history,
)
from smartbelt.pipeline.orchestrator import OrchestratorConfig, SmartBeltOrchestrator

st.set_page_config(
    page_title="SmartBelt v2 — Hardware Inspection System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_orchestrator() -> SmartBeltOrchestrator:
    """Initialize and start background acquisition and neural inference threads."""
    config = OrchestratorConfig()
    orch = SmartBeltOrchestrator(config)
    orch.start()
    return orch


# Initialize Session State
if "history" not in st.session_state:
    st.session_state["history"] = []
if "is_running" not in st.session_state:
    st.session_state["is_running"] = True

orchestrator = get_orchestrator()

# Render Sidebar
render_sidebar_calibration(orchestrator)

st.sidebar.markdown("---")
st.sidebar.subheader("Vision Input")
video_options = {
    "Live USB Camera (Logitech C270)": 0,
    "Test Video: Normal Belt (conveyorbelt.mp4)": str(PROJECT_ROOT / "videos" / "conveyorbelt.mp4"),
    "Test Video: Joint Damage (conveyor_with_real_damage.mp4)": str(PROJECT_ROOT / "videos" / "conveyor_with_real_damage.mp4"),
}
chosen_label = st.sidebar.selectbox("Select Vision Source", list(video_options.keys()), index=0)
chosen_source = video_options[chosen_label]

if "current_source" not in st.session_state or st.session_state["current_source"] != chosen_source:
    orchestrator.set_camera_source(chosen_source)
    st.session_state["current_source"] = chosen_source

st.sidebar.markdown("---")
st.sidebar.subheader("Live Stream Control")
toggle_btn = st.sidebar.button(
    "⏸ Pause Stream" if st.session_state["is_running"] else "▶ Resume Stream",
    use_container_width=True,
)
if toggle_btn:
    st.session_state["is_running"] = not st.session_state["is_running"]
    st.rerun()

refresh_hz = st.sidebar.slider("UI Refresh Interval (sec)", 0.2, 2.0, 0.5, step=0.1)

# Main UI Containers
header_box = st.empty()
banner_box = st.empty()
view_box = st.empty()
history_box = st.empty()

# Fetch current system telemetry snapshot
state = orchestrator.get_state()

# Update historical record
if state.sensor_packet and not state.sensor_packet.parse_error:
    st.session_state["history"].append({
        "temp_c": state.sensor_packet.temp_c,
        "vib_rms": state.sensor_packet.vib_rms,
        "load_kg": state.sensor_packet.load_kg,
        "speed_mps": state.sensor_packet.belt_speed_mps,
        "risk_score": state.fusion.fused_score,
    })
    # Keep rolling window of 60 records
    if len(st.session_state["history"]) > 60:
        st.session_state["history"].pop(0)

# Render layout
with header_box.container():
    render_header(state)

with banner_box.container():
    render_alert_banner(state)

with view_box.container():
    render_live_view(state)

with history_box.container():
    render_telemetry_history(st.session_state["history"])

# Auto-refresh loop when stream is active
if st.session_state["is_running"]:
         time.sleep(refresh_hz)
       st.return();
