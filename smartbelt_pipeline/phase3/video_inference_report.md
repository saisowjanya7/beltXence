# SmartBelt — Phase 3: Real-Time Video Joint Detection Report

> **Module:** Video & Streaming Inference Engine  
> **Status:** Phase 3 Complete  
> **Output Directory:** `smartbelt_pipeline/phase3/`  
> **Date:** 2026-09-10  

---

## 1. Pipeline Architecture (`realtime_joint_detection.py`)

The real-time joint rupture detection pipeline provides an end-to-end continuous monitoring framework:

1. **Flexible Source Ingestion:**
   - Accepts recorded MP4/AVI industrial video files (`--input <path>`) or direct webcam streams (`--input 0`).
   - Graceful error handling for missing files, unreadable codecs, or disconnected camera feeds.
2. **Sampling Optimization:**
   - Processes frames at configurable intervals (`--frame-interval`, default: every 5th frame) to eliminate redundant compute on static conveyor speeds while maintaining sub-second defect response.
3. **Deep Anomaly Feature Extraction:**
   - Resizes and normalizes incoming frames matching PatchCore's WideResNet-50 feature extractor.
   - Computes chunked nearest-neighbor distances against the 22,425-vector memory bank.
4. **Statistical Risk Classification (from Phase 2):**
   - `< 0.62710`: **NORMAL**
   - `0.62710` to `0.68887`: **WARNING**
   - `≥ 0.68887`: **CRITICAL** (Emergency motor halt trigger)
5. **Real-Time Visual Overlays:**
   - Full-frame color-coded border (Green for NORMAL, Yellow for WARNING, Red for CRITICAL).
   - Semi-transparent top telemetry banner: Frame number, elapsed time, raw anomaly score, and risk status badge.
   - Prominent lower banner when defects occur: `CRITICAL: JOINT RUPTURE DETECTED - MOTOR HALT`.
6. **Continuous CSV Telemetry Stream:**
   - Logs `frame_number`, `timestamp_sec`, `anomaly_score`, `risk_status` for audit compliance and SCADA integration.

---

## 2. Test Execution & Sanity Check Results

We evaluated the pipeline across two test video scenarios:

### Scenario A: Baseline Normal Conveyor Run (`videos/conveyorbelt.mp4`)
* **Span Evaluated:** Frames 0 to 745 (0.0s to 24.8s) at interval 5.
* **Frames Processed:** 150 frames.
* **Risk Distribution:**
  - **NORMAL:** 150 frames (**100.0%**) — Score range: `0.40366` to `0.58898`
  - **WARNING:** 0 frames (0.0%)
  - **CRITICAL:** 0 frames (0.0%)
* **Sanity Check Outcome:** Confirms **0% false alarm rate** across continuous uninterrupted conveyor operation.

### Scenario B: Full Joint Damage Transition Video (`videos/conveyor_with_real_damage.mp4`)
* **Composition:**
  1. `0.00s – 1.93s` (Frames 0–58): Nominal belt operation
  2. `2.00s – 3.27s` (Frames 60–98): Real joint rupture passage (40 real damaged frames from `conveyor_original_damaged_joint_40/`)
  3. `3.33s – 5.27s` (Frames 100–158): Normal belt recovery
* **Frames Processed:** 80 frames (sampled at interval 2).
* **Risk Distribution:**
  - **NORMAL:** 60 frames (**75.0%**)
  - **WARNING:** 0 frames (0.0%)
  - **CRITICAL:** 20 frames (**25.0%**)

### Critical Detection Timeline:
* At **$t = 1.93\text{s}$ (Frame 58)**: Anomaly score = `0.48002` $\rightarrow$ **NORMAL** (Green border).
* At **$t = 2.00\text{s}$ (Frame 60)**: Damaged joint enters camera FOV $\rightarrow$ Score spikes to **`0.71305`** $\rightarrow$ **CRITICAL ALARM TRIGGERED** (Red border, Motor Halt message).
* Throughout the defect window ($t = 2.00\text{s}$ to $3.27\text{s}$, Frames 60–98): **100% of defect evaluations triggered CRITICAL** (scores between `0.69928` and `0.78790`).
* At **$t = 3.33\text{s}$ (Frame 100)**: Damaged joint exits FOV $\rightarrow$ Score immediately resets to **`0.47533`** $\rightarrow$ **NORMAL** (Green border).

---

## 3. Deliverables Index

All deliverables are located in [`smartbelt_pipeline/phase3/`](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase3/):

* [**`realtime_joint_detection.py`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase3/realtime_joint_detection.py) — Complete production video and camera inference CLI script.
* [**`sample_run_output.mp4`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase3/sample_run_output.mp4) — Fully annotated MP4 video with telemetry overlays and color-coded risk borders.
* [**`sample_run_log.csv`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase3/sample_run_log.csv) — Per-frame CSV telemetry audit log.
* [**`video_inference_report.md`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase3/video_inference_report.md) — Technical execution report.
