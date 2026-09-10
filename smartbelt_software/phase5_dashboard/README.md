# Phase 5 — Streamlit Operator Dashboard

This directory contains the industrial operator monitoring dashboard for the **SmartBelt** conveyor inspection system.

## Features

1. **Multi-Input Video Ingestion**:
   - Sample recorded conveyor videos (`conveyor_with_real_damage.mp4`, `conveyorbelt.mp4`)
   - User video file upload (`.mp4`, `.avi`, `.mov`)
   - Live industrial USB webcam / RTSP camera feed (Index 0)

2. **Computer Vision Inference**:
   - Loads the existing trained `patchcore_joint_v1.ckpt` (WideResNet-50 backbone + 22,425 vector memory bank).
   - Real-time frame anomaly distance extraction and visual threshold classification.

3. **Physical Sensor Telemetry Integration**:
   - Monitored channels: **Temperature (°C)**, **Vibration RMS (mm/s)**, **Speed Slip (%)**, **Belt Tension (kN)**.
   - Scenario selector for interactive testing: *Nominal Baseline*, *Bearing Overheat*, *Splice Vibration*, *Drive Slip*, *Catastrophic Rupture*, and *Manual Sliders*.
   - Explicit industrial compliance indicator: `[SIMULATED — not a real measurement]`.
   - Prediction engine: ISO-10816 calibrated baseline engine (`PENDING_REAL_LABELED_SENSOR_DATA`).

4. **Multi-Modal Decision & Conservative Safety Fusion**:
   - Cross-modal consensus determination (`NOMINAL_OPERATION`, `CONFIRMED_MULTIMODAL_EMERGENCY`, `VISION_DOMINANT_FAILURE`, `SENSOR_DOMINANT_MECHANICAL_DISTRESS`, `ELEVATED_INSPECTION_ZONE`).
   - High-contrast visual overlays (Green, Amber, Red bounding frame).
   - Prominent flashing emergency banner when `CRITICAL` state is tripped.

5. **Telemetry Trends & Historical Audit Log**:
   - Real-time multi-line trend graph comparing Visual Score vs Sensor Score vs Fused Score against thresholds ($T_1=0.62710$, $T_2=0.68887$).
   - Live step audit table with one-click CSV export (`smartbelt_dashboard_telemetry.csv`).

6. **Hardware Interface Indicator**:
   - Visual status badge for ESP32 connection state (`SIMULATION MODE / Virtual Relay Active`).

---

## Quickstart & Execution

To launch the dashboard:

```bash
# From the project root (c:\Users\korra\OneDrive\Desktop\Conveyor):
.\anomalib_env\Scripts\streamlit.exe run smartbelt_software/phase5_dashboard/app.py
```

Or using the python module directly:
```bash
.\anomalib_env\Scripts\python.exe -m streamlit run smartbelt_software/phase5_dashboard/app.py
```

Once running, navigate to `http://localhost:8501` in any web browser.

---

## File Structure

- `app.py`: Streamlit frontend application integrating `SmartBeltApplication` backend.
- `README.md`: System documentation, usage instructions, and feature map.
- `uploads/`: Temporary storage for user-uploaded video files.
- `dashboard_telemetry.csv`: Live telemetry audit log generated during dashboard runs.
