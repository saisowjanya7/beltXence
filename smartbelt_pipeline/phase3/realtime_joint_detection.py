"""
SmartBelt — Phase 3: Real-Time / Video Joint Detection Pipeline

Processes live webcam feeds or recorded conveyor video files,
extracts frames at configurable intervals, runs PatchCore inference,
classifies risk (NORMAL, WARNING, CRITICAL), renders visual overlays,
and outputs annotated video alongside a structured CSV telemetry log.

Usage examples:
    # Process recorded conveyor belt video (default interval = every 5th frame):
    python realtime_joint_detection.py --input videos/conveyorbelt.mp4

    # Process live webcam (index 0):
    python realtime_joint_detection.py --input 0

    # Process first 600 frames at interval 5:
    python realtime_joint_detection.py --input videos/conveyorbelt.mp4 --max-frames 600 --frame-interval 5
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# ==============================================================================
# DEFAULT PATHS & CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

DEFAULT_CHECKPOINT = PROJECT_DIR / "results_joint" / "20260910_105659" / "patchcore_joint_v1.ckpt"
DEFAULT_VIDEO = PROJECT_DIR / "videos" / "conveyorbelt.mp4"
DEFAULT_OUTPUT_VIDEO = SCRIPT_DIR / "sample_run_output.mp4"
DEFAULT_OUTPUT_LOG = SCRIPT_DIR / "sample_run_log.csv"

# Risk Thresholds from Phase 2
THRESHOLD_1 = 0.62710  # NORMAL -> WARNING
THRESHOLD_2 = 0.68887  # WARNING -> CRITICAL

COLOR_MAP_BGR = {
    "NORMAL":   (40, 167, 69),    # Industrial Green
    "WARNING":  (0, 191, 255),    # Deep Amber / Yellow (BGR)
    "CRITICAL": (38, 38, 220),    # Bright Industrial Red (BGR)
}

IMAGE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def classify_risk(score: float) -> str:
    """Classify anomaly score into NORMAL, WARNING, or CRITICAL."""
    if score < THRESHOLD_1:
        return "NORMAL"
    elif score < THRESHOLD_2:
        return "WARNING"
    else:
        return "CRITICAL"


# ==============================================================================
# MODEL LOADER & INFERENCE ENGINE
# ==============================================================================

class PatchCoreDetector:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"[Detector] Loading PatchCore model from {checkpoint_path.name}...")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        from anomalib.models import Patchcore
        self.model = Patchcore.load_from_checkpoint(str(checkpoint_path), weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        mb = self.model.model.memory_bank.detach().to(self.device).float()
        self.mb_norm = torch.nn.functional.normalize(mb, p=2, dim=1)
        print(f"[Detector] Memory bank initialized: {tuple(self.mb_norm.shape)}")

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> float:
        """
        Run inference on a single BGR OpenCV frame.
        Returns:
            anomaly_score (float): PatchCore distance score.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(self.device)

        pc = self.model.model
        feats = pc.feature_extractor(tensor)
        feat_list = list(feats.values()) if isinstance(feats, dict) else list(feats)

        processed = []
        for f in feat_list:
            if f.ndim == 4:
                processed.append(pc.feature_pooler(f))

        th = max(x.shape[-2] for x in processed)
        tw = max(x.shape[-1] for x in processed)
        resized = []
        for f in processed:
            if f.shape[-2] != th or f.shape[-1] != tw:
                f = torch.nn.functional.interpolate(f, (th, tw), mode="bilinear", align_corners=False)
            resized.append(f)

        emb = torch.cat(resized, dim=1).permute(0, 2, 3, 1).reshape(-1, sum(x.shape[1] for x in resized))
        emb_n = torch.nn.functional.normalize(emb, p=2, dim=1)

        # Chunked nearest-neighbor search
        chunk = 512
        dists = []
        for s in range(0, emb_n.shape[0], chunk):
            d = torch.cdist(emb_n[s:s+chunk], self.mb_norm)
            dists.append(d.min(dim=1).values)
        dists = torch.cat(dists)

        score = float(dists.max().item())
        return score


# ==============================================================================
# VISUAL OVERLAY RENDERER
# ==============================================================================

def render_overlay(frame: np.ndarray, frame_num: int, timestamp_sec: float,
                   score: float, risk: str) -> np.ndarray:
    """Render color-coded border, telemetry banner, and emergency alerts."""
    h, w = frame.shape[:2]
    out = frame.copy()
    color = COLOR_MAP_BGR[risk]

    # 1. Color-coded border around the entire frame
    border_thick = max(6, int(min(w, h) * 0.015))
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness=border_thick)

    # 2. Semi-transparent telemetry header banner
    header_h = 70
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Header line divider
    cv2.line(out, (0, header_h), (w, header_h), color, thickness=2)

    # 3. Telemetry text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Title & Module
    cv2.putText(out, "SmartBelt | Joint Rupture Monitor", (15, 25), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Telemetry data string
    info_str = f"Frame: {frame_num:04d} | Time: {timestamp_sec:6.2f}s | Anomaly Score: {score:.4f}"
    cv2.putText(out, info_str, (15, 52), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Risk badge in top right
    badge_text = f"RISK: {risk}"
    (tw, th_box), _ = cv2.getTextSize(badge_text, font, 0.75, 2)
    badge_x = w - tw - 25
    cv2.rectangle(out, (badge_x - 10, 12), (w - 15, 56), color, -1)
    cv2.putText(out, badge_text, (badge_x, 42), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Critical Warning Banner across lower region
    if risk == "CRITICAL":
        alert_h = 60
        alert_y1 = h - alert_h - 20
        alert_y2 = h - 20
        alert_overlay = out.copy()
        cv2.rectangle(alert_overlay, (20, alert_y1), (w - 20, alert_y2), (0, 0, 180), -1)
        cv2.addWeighted(alert_overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (20, alert_y1), (w - 20, alert_y2), (255, 255, 255), 2)
        
        warn_msg = "CRITICAL: JOINT RUPTURE DETECTED - MOTOR HALT"
        (mw, mh), _ = cv2.getTextSize(warn_msg, font, 0.70, 2)
        cv2.putText(out, warn_msg, ((w - mw) // 2, alert_y1 + 38), font, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    elif risk == "WARNING":
        alert_h = 45
        alert_y1 = h - alert_h - 20
        alert_y2 = h - 20
        alert_overlay = out.copy()
        cv2.rectangle(alert_overlay, (20, alert_y1), (w - 20, alert_y2), (0, 140, 220), -1)
        cv2.addWeighted(alert_overlay, 0.80, out, 0.20, 0, out)
        cv2.rectangle(out, (20, alert_y1), (w - 20, alert_y2), (255, 255, 255), 2)

        warn_msg = "WARNING: JOINT ANOMALY ELEVATED - INSPECT SPLICE"
        (mw, mh), _ = cv2.getTextSize(warn_msg, font, 0.58, 2)
        cv2.putText(out, warn_msg, ((w - mw) // 2, alert_y1 + 30), font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ==============================================================================
# MAIN PROCESSING LOOP
# ==============================================================================

def run_detection(
    input_source: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_video_path: Path = DEFAULT_OUTPUT_VIDEO,
    output_log_path: Path = DEFAULT_OUTPUT_LOG,
    frame_interval: int = 5,
    max_frames: int = None,
    start_frame: int = 0
):
    print("=" * 70)
    print("SmartBelt — Real-Time Joint Damage Detection Pipeline")
    print("=" * 70)

    # 1. Determine input source (webcam index vs video file)
    is_webcam = False
    if input_source.isdigit():
        cap_arg = int(input_source)
        is_webcam = True
        print(f"[Input] Connecting to Webcam Index: {cap_arg}")
    else:
        video_file = Path(input_source)
        if not video_file.exists():
            print(f"[FATAL ERROR] Video file not found: {video_file.resolve()}", file=sys.stderr)
            sys.exit(1)
        cap_arg = str(video_file)
        print(f"[Input] Opening Video File: {video_file.resolve()}")

    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        source_desc = f"Webcam #{cap_arg}" if is_webcam else f"Video file '{cap_arg}'"
        print(f"[FATAL ERROR] Failed to open video source: {source_desc}.", file=sys.stderr)
        print("[Troubleshooting] Ensure file is not corrupted and video codecs are installed.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_in_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Source Stats] Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames_in_source}")
    print(f"[Processing Config] Interval: every {frame_interval} frames | Start: {start_frame} | Max: {max_frames or 'ALL'}")

    # 2. Load detector
    detector = PatchCoreDetector(checkpoint_path=checkpoint_path)

    # 3. Setup video writer
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output effective FPS: if we sample every N frames, playback at fps / N or nominal 6-10 fps
    out_fps = max(1.0, fps / frame_interval)
    writer = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        print(f"[WARN] Failed to open video writer for {output_video_path.name}. Trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (width, height))

    # 4. Setup CSV logger
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_sec", "anomaly_score", "risk_status"])

    # 5. Execution loop
    raw_frame_idx = 0
    processed_count = 0
    risk_summary = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
    critical_events = []

    print("\n--- Live Execution Stream ---")
    print(f"{'FRAME':>6} | {'TIME':>8} | {'ANOMALY SCORE':>14} | {'RISK STATUS':>11} | {'ACTION'}")
    print("-" * 65)

    if start_frame > 0 and not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        raw_frame_idx = start_frame

    t_start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check max frame limit
            if max_frames is not None and processed_count >= max_frames:
                break

            # Only process every Nth frame
            if (raw_frame_idx % frame_interval) == 0:
                timestamp_sec = raw_frame_idx / fps
                t_frame_start = time.time()

                # Anomaly Inference
                score = detector.predict_frame(frame)
                risk = classify_risk(score)
                risk_summary[risk] += 1

                if risk == "CRITICAL":
                    critical_events.append((raw_frame_idx, timestamp_sec, score))
                    action_tag = ">> MOTOR HALT TRIGGERED <<"
                elif risk == "WARNING":
                    action_tag = "Maintenance Flagged"
                else:
                    action_tag = "Nominal Operation"

                # Log to console
                print(f"{raw_frame_idx:6d} | {timestamp_sec:7.2f}s | {score:14.5f} | {risk:11s} | {action_tag}")

                # Log to CSV
                csv_writer.writerow([raw_frame_idx, round(timestamp_sec, 3), round(score, 5), risk])
                csv_file.flush()

                # Render overlay and write to output video
                annotated = render_overlay(frame, raw_frame_idx, timestamp_sec, score, risk)
                if writer.isOpened():
                    writer.write(annotated)

                processed_count += 1

            raw_frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted inference stream.")
    finally:
        cap.release()
        if writer.isOpened():
            writer.release()
        csv_file.close()

    total_time = time.time() - t_start
    print("-" * 65)
    print(f"\n[Processing Complete in {total_time:.1f}s]")
    print(f"  Total Frames Processed: {processed_count}")
    print(f"  Output Video Saved:     {output_video_path.resolve()}")
    print(f"  Output Log Saved:       {output_log_path.resolve()}")

    # Summary Statistics
    print("\n--- Risk Distribution Summary ---")
    for r_type in ["NORMAL", "WARNING", "CRITICAL"]:
        cnt = risk_summary[r_type]
        pct = (cnt / processed_count * 100) if processed_count > 0 else 0
        print(f"  {r_type:<10}: {cnt:4d} frames ({pct:5.1f}%)")

    print(f"\n--- Critical Joint Rupture Events ({len(critical_events)} total) ---")
    if critical_events:
        for f_num, t_sec, sc in critical_events[:15]:
            print(f"  CRITICAL at Frame {f_num:04d} ({t_sec:5.2f}s) — Score: {sc:.5f}")
        if len(critical_events) > 15:
            print(f"  ... and {len(critical_events) - 15} additional critical frames.")
    else:
        print("  None detected.")

    return {
        "processed_count": processed_count,
        "risk_summary": risk_summary,
        "critical_events": critical_events,
        "total_time": total_time
    }


# ==============================================================================
# ENTRY POINT & CLI PARSER
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBelt Real-Time Joint Anomaly Detection")
    parser.add_argument("--input", "-i", default=str(DEFAULT_VIDEO),
                        help="Path to video file or webcam index (default: videos/conveyorbelt.mp4)")
    parser.add_argument("--checkpoint", "-c", default=str(DEFAULT_CHECKPOINT),
                        help="Path to trained PatchCore checkpoint")
    parser.add_argument("--output-video", "-ov", default=str(DEFAULT_OUTPUT_VIDEO),
                        help="Path for output annotated MP4 video")
    parser.add_argument("--output-log", "-ol", default=str(DEFAULT_OUTPUT_LOG),
                        help="Path for CSV telemetry log")
    parser.add_argument("--frame-interval", "-f", type=int, default=5,
                        help="Frame skip interval (process every Nth frame, default: 5)")
    parser.add_argument("--start-frame", "-s", type=int, default=0,
                        help="Starting frame number in video (default: 0)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Maximum processed frames before stopping (default: None/all)")

    args = parser.parse_args()

    run_detection(
        input_source=args.input,
        checkpoint_path=Path(args.checkpoint),
        output_video_path=Path(args.output_video),
        output_log_path=Path(args.output_log),
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        start_frame=args.start_frame
    )
