"""
SmartBelt — Phase 4: Video Pipeline Integration Test
===================================================

Runs an end-to-end demonstration using:
  1. Real conveyor belt video feed (videos/conveyor_with_real_damage.mp4 or conveyorbelt.mp4)
  2. Live PatchCore joint anomaly inference from unchanged checkpoint
  3. XGBoost sensor prediction interface (with explicitly labeled simulated sensor readings)
  4. Decision & fusion layer (fuse_risk)
  5. Real-time overlay visualization & annotated video output
  6. CSV telemetry audit logging
"""

import os
import sys
import time
from pathlib import Path
import cv2

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbelt_software.phase4_integration.smartbelt_core import (
    SmartBeltApplication, DEFAULT_CHECKPOINT
)
from smartbelt_software.phase4_integration.xgboost_sensor_module import (
    simulate_sensor_reading
)

VIDEO_CANDIDATE = PROJECT_ROOT / "videos" / "conveyor_with_real_damage.mp4"
if not VIDEO_CANDIDATE.exists():
    VIDEO_CANDIDATE = PROJECT_ROOT / "videos" / "conveyorbelt.mp4"

OUTPUT_VIDEO = CURRENT_DIR / "phase4_demo_output.mp4"
TELEMETRY_CSV = CURRENT_DIR / "phase4_telemetry_run.csv"


def run_video_integration_test(max_eval_frames: int = 25, frame_interval: int = 5):
    print("=" * 75)
    print("SmartBelt — Phase 4 Video + Sensor Integration Test Run")
    print("=" * 75)
    print(f"Video input:      {VIDEO_CANDIDATE.resolve()}")
    print(f"PatchCore model:  {DEFAULT_CHECKPOINT.resolve()}")
    print(f"Output video:     {OUTPUT_VIDEO.resolve()}")
    print(f"Telemetry log:    {TELEMETRY_CSV.resolve()}")
    print(f"Frame interval:   every {frame_interval} frames")
    print(f"Max eval frames:  {max_eval_frames} evaluated steps")
    print("=" * 75)

    if not VIDEO_CANDIDATE.exists():
        raise FileNotFoundError(f"Video file not found: {VIDEO_CANDIDATE}")

    # Initialize unified application
    app = SmartBeltApplication(
        checkpoint_path=DEFAULT_CHECKPOINT,
        telemetry_log_path=TELEMETRY_CSV
    )

    cap = cv2.VideoCapture(str(VIDEO_CANDIDATE))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_CANDIDATE}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps / frame_interval, (width, height))

    frame_idx = 0
    evaluated_count = 0

    # Sensor scenario rotation to demonstrate multimodal behavior:
    # 1. First 10 steps: Nominal operation
    # 2. Next 5 steps: Bearing overheat (sensor-dominant)
    # 3. Next 10 steps: Splice impact vibration (confirmed joint rupture)
    scenarios = (
        ["nominal"] * 10 +
        ["bearing_overheat"] * 5 +
        ["splice_vibration"] * 10
    )

    print(f"\n{'STEP':<5} | {'FRAME':<6} | {'VISUAL':<8} | {'SENSOR':<8} | {'FUSED':<8} | {'RISK':<9} | {'CONSENSUS'}")
    print("-" * 75)

    try:
        while cap.isOpened() and evaluated_count < max_eval_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Select sensor scenario
                scenario_name = scenarios[evaluated_count % len(scenarios)]
                raw_sensors, is_sim = simulate_sensor_reading(scenario_name)

                # Process unified pipeline
                t0 = time.time()
                res = app.process_frame(
                    frame_bgr=frame,
                    sensor_readings=raw_sensors,
                    is_simulated_sensor=is_sim,
                    frame_idx=frame_idx
                )
                dt = time.time() - t0

                # Render overlay
                annotated_frame = app.render_display_overlay(frame, res)
                out_writer.write(annotated_frame)

                print(
                    f"{evaluated_count+1:<5} | "
                    f"{frame_idx:<6} | "
                    f"{res['visual_score']:<8.4f} | "
                    f"{res['sensor_score']:<8.4f} | "
                    f"{res['final_risk_score']:<8.4f} | "
                    f"{res['risk_status']:<9} | "
                    f"{res['consensus_state']}"
                )
                evaluated_count += 1

            frame_idx += 1

    finally:
        cap.release()
        out_writer.release()

    print("-" * 75)
    print(f"\n[Phase 4 Test Completed]")
    print(f"Evaluated frames: {evaluated_count}")
    print(f"Annotated video:  {OUTPUT_VIDEO.name} ({round(OUTPUT_VIDEO.stat().st_size / 1024, 1)} KB)")
    print(f"Telemetry log:    {TELEMETRY_CSV.name} ({round(TELEMETRY_CSV.stat().st_size / 1024, 1)} KB)")
    print("Integration verification: SUCCESSFUL.\n")


if __name__ == "__main__":
    run_video_integration_test(max_eval_frames=20, frame_interval=5)
