# Phase 4 — Software Integration (PatchCore + XGBoost + Fusion)

## Overview
Phase 4 unifies all standalone analytical components into an importable, production-ready core application architecture:
- **PatchCore Joint Detector**: Direct inference using the untouched WideResNet-50 checkpoint (`results_joint/20260910_105659/patchcore_joint_v1.ckpt`).
- **XGBoost Sensor Prediction Module**: Standard feature schema, live/simulated prediction, offline CSV evaluation, and an ISO-10816 calibrated baseline scoring engine pending real labeled sensor data.
- **Fusion Decision Layer**: Multi-modal conservative risk engine balancing visual anomaly scores and mechanical sensor distress.
- **Telemetry & Alert Logger**: Structured CSV logging of every frame and decision with transparent data provenance.

---

## Directory Structure
```
smartbelt_software/phase4_integration/
├── smartbelt_core.py          # Unified application backend class (SmartBeltApplication)
├── xgboost_sensor_module.py    # XGBoost sensor inference & feature pipeline
├── test_pipeline_run.py        # End-to-end integration test runner
├── telemetry_log.csv           # Telemetry log created during runs
├── phase4_telemetry_run.csv    # Telemetry output from test run
├── phase4_demo_output.mp4      # Annotated video produced during test run
└── README.md                   # This documentation
```

---

## Sensor Module & Data Integrity Status
- **Current State**: No physical ESP32 or real conveyor sensor telemetry currently exists in this repository.
- **Scientific Integrity Rule**: No synthetic dataset was used to invent a "trained" model claiming real sensor accuracy.
- **XGBoost Interface**: 
  - Expects 4 features: `temperature_c`, `vibration_rms_mms`, `speed_slip_pct`, `tension_kn`.
  - Reports status: `PENDING_REAL_LABELED_SENSOR_DATA`.
  - When real labeled data is collected, the `.load_model()` method directly loads an XGBoost booster (`.json` or `.bin`).
  - Currently evaluates via an ISO-10816 calibrated industrial baseline scoring function to allow complete end-to-end testing.
  - Every sensor reading and output is strictly labeled as **`SIMULATED — not a real measurement`** in logs, CSVs, and API responses.

---

## Fusion / Decision Rule Justification
The function `fuse_risk(visual_score, sensor_score)` implements a **Conservative Multi-Modal Industrial Safety Rule**:

$$\text{Threshold 1 (Warning)} = 0.62710, \quad \text{Threshold 2 (Critical)} = 0.68887$$

1. **Cross-Modal Confirmation** ($V \ge T_2 \land S \ge T_2$):
   - Both camera and mechanical sensors observe severe distress (e.g. splice tearing + high vibration pounding).
   - Escalates score: $\min(1.0, \max(V, S) + 0.05) \rightarrow$ **`CRITICAL`** (`CONFIRMED_MULTIMODAL_EMERGENCY`).
2. **Visual Dominant** ($V \ge T_2, S < T_2$):
   - Belt joint tear or cord pullout detected optically before mechanical drive vibration occurs.
   - Score: $V \rightarrow$ **`CRITICAL`** (`VISION_DOMINANT_FAILURE`).
3. **Sensor Dominant** ($V < T_2, S \ge T_2$):
   - Mechanical bearing seizure, pulley slippage, or tension collapse occurring outside the joint camera field-of-view.
   - Score: $S \rightarrow$ **`CRITICAL`** (`SENSOR_DOMINANT_MECHANICAL_DISTRESS`).
   - Ensures blind-spot protection.
4. **Elevated Inspection Zone** ($V \ge T_1 \lor S \ge T_1$):
   - Score: $\max(V, S) \rightarrow$ **`WARNING`** (`ELEVATED_INSPECTION_ZONE`).
   - Alerts maintenance technicians via SCADA without halting production.
5. **Nominal Operation** ($V < T_1 \land S < T_1$):
   - Score: $0.60 V + 0.40 S \rightarrow$ **`NORMAL`** (`NOMINAL_OPERATION`).
   - Weighted average suppresses single-frame optical or sensor transient spikes.

---

## How to Run

### 1. Test the XGBoost Sensor Module:
```powershell
.\anomalib_env\Scripts\python.exe smartbelt_software\phase4_integration\xgboost_sensor_module.py
```

### 2. Test the Core Application Module:
```powershell
.\anomalib_env\Scripts\python.exe smartbelt_software\phase4_integration\smartbelt_core.py
```

### 3. Run the Video + Sensor Integration Test:
```powershell
.\anomalib_env\Scripts\python.exe smartbelt_software\phase4_integration\test_pipeline_run.py
```
Outputs:
- Annotated video: `phase4_demo_output.mp4`
- Detailed telemetry: `phase4_telemetry_run.csv`
