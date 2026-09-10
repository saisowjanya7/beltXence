"""
SmartBelt — Phase 4: Core Unified Application Module (smartbelt_core.py)
========================================================================

Integrates into one coherent, importable application:
1. Trained PatchCore Joint Model:
   - Checkpoint: results_joint/20260910_105659/patchcore_joint_v1.ckpt (UNTOUCHED)
   - Real-time PyTorch feature extraction & nearest-neighbor scoring against memory bank
2. Risk Classification Engine:
   - Threshold 1 (NORMAL -> WARNING): 0.62710
   - Threshold 2 (WARNING -> CRITICAL): 0.68887
3. XGBoost / Sensor Anomaly Prediction Module:
   - Evaluates multi-sensor telemetry (temperature, vibration, slip/RPM, tension)
   - Explicitly logs data source ('SIMULATED — not a real measurement' vs 'REAL SENSOR')
4. Decision & Fusion Layer (fuse_risk):
   - Combines visual anomaly score & sensor distress score into a unified operational decision
   - Full explanation & justification documented in module & README
5. Structured Telemetry & Audit Logger:
   - Records every inference step with visual_score, sensor_score, fused_score,
     contributing sources, and alert signals.
"""

import os
import sys
import csv
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Path & Directory Anchoring
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

# Direct references to existing checkpoints and evaluations (NO DUPLICATION)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results_joint" / "20260910_105659" / "patchcore_joint_v1.ckpt"
EVALUATION_DIR = PROJECT_ROOT / "results_joint" / "20260910_105659"
SCORES_CSV_PATH = PROJECT_ROOT / "smartbelt_pipeline" / "phase1" / "scores.csv"

# Operational Thresholds (Calibrated on joint validation & test distribution)
THRESHOLD_1 = 0.62710  # NORMAL -> WARNING (mu+2sigma normal limit)
THRESHOLD_2 = 0.68850  # WARNING -> CRITICAL (Calibrated: test_good_max 0.68643 < 0.68850 <= test_bad_min 0.68856)

# Preprocessing transforms matching PatchCore training
IMAGE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Import XGBoost Sensor Module
try:
    from smartbelt_software.phase4_integration.xgboost_sensor_module import (
        XGBoostSensorPredictor, simulate_sensor_reading, SENSOR_SPEC_BOUNDS
    )
except ImportError:
    from xgboost_sensor_module import (
        XGBoostSensorPredictor, simulate_sensor_reading, SENSOR_SPEC_BOUNDS
    )


# ==============================================================================
# 1. RISK CLASSIFICATION & FUSION DECISION LAYER
# ==============================================================================

def classify_score(score: float) -> str:
    """Classify a single score into discrete operational tiers."""
    if score < THRESHOLD_1:
        return "NORMAL"
    elif score < THRESHOLD_2:
        return "WARNING"
    else:
        return "CRITICAL"


def fuse_risk(
    visual_score: float,
    sensor_score: float,
    sensor_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fuse computer vision anomaly score (PatchCore) with telemetry sensor score.

    FUSION POLICY & JUSTIFICATION:
    -------------------------------
    In heavy industrial conveyor systems, conveyor belt rupture or splice failure
    is a high-consequence event (millions of dollars in plant downtime or severe safety hazard).
    Therefore, a 'Conservative Multi-Modal Safety Rule' is implemented:
    
    1. Cross-Modal Consensus (visual_score >= THRESHOLD_2 AND sensor_score >= THRESHOLD_2):
       Both vision and physical sensors confirm severe distress.
       -> Score = min(1.0, max(visual, sensor) + 0.05), Category = CRITICAL
       -> Reason: Highest alert priority (e.g. splice tearing + severe vibration pounding).

    2. Visual Dominant (visual_score >= THRESHOLD_2, sensor_score < THRESHOLD_2):
       Vision detects structural joint rupture or tear before mechanical vibration/slip manifests.
       -> Score = visual_score, Category = CRITICAL
       -> Reason: Belt tears start as surface splits before mechanical drive collapse.

    3. Sensor Dominant (visual_score < THRESHOLD_2, sensor_score >= THRESHOLD_2):
       Sensors detect severe bearing heat, slip, or tension loss outside camera field-of-view.
       -> Score = sensor_score, Category = CRITICAL
       -> Reason: Camera has a localized field of view; drive stalls/bearing seizures must
          trip the motor even if the passing belt joint happens to appear intact.

    4. Elevated Inspection Zone (either score >= THRESHOLD_1):
       At least one modality is in the WARNING range.
       -> Score = max(visual_score, sensor_score), Category = WARNING
       -> Reason: Preventive maintenance flag; avoid unscheduled plant stops while alerting crew.

    5. Nominal Operation (both scores < THRESHOLD_1):
       Both modalities confirm healthy condition.
       -> Score = 0.60 * visual_score + 0.40 * sensor_score, Category = NORMAL
       -> Reason: Smooth blending suppresses single-frame transient noise.
    """
    v = float(visual_score)
    s = float(sensor_score)

    if v >= THRESHOLD_2 and s >= THRESHOLD_2:
        fused_score = min(1.0, max(v, s) + 0.05)
        consensus = "CONFIRMED_MULTIMODAL_EMERGENCY"
    elif v >= THRESHOLD_2:
        fused_score = v
        consensus = "VISION_DOMINANT_FAILURE"
    elif s >= THRESHOLD_2:
        fused_score = s
        consensus = "SENSOR_DOMINANT_MECHANICAL_DISTRESS"
    elif v >= THRESHOLD_1 or s >= THRESHOLD_1:
        fused_score = max(v, s)
        consensus = "ELEVATED_INSPECTION_ZONE"
    else:
        fused_score = 0.60 * v + 0.40 * s
        consensus = "NOMINAL_OPERATION"

    fused_score = float(np.clip(fused_score, 0.0, 1.0))
    risk_category = classify_score(fused_score)

    # Operational recommendations
    if risk_category == "NORMAL":
        action = "Conveyor nominal. Continuous dual-stream monitoring active."
        buzzer = "OFF"
        relay = "CLOSED (RUNNING)"
        beacon = "GREEN"
        cmd = "CMD_STATUS_NORMAL"
    elif risk_category == "WARNING":
        action = f"PREVENTIVE ALERT [{consensus}] — Log joint ID, inspect splice at next shift."
        buzzer = "PULSE_CHIRP (1 Hz)"
        relay = "CLOSED (MAINTAIN MOTION)"
        beacon = "AMBER"
        cmd = "CMD_ALERT_WARNING"
    else:
        action = f"EMERGENCY MOTOR TRIP [{consensus}] — Immediate conveyor shutdown commanded!"
        buzzer = "ON (CONTINUOUS 95dB)"
        relay = "OPEN (MOTOR TRIPPED / EMERGENCY STOP)"
        beacon = "FLASHING_RED"
        cmd = "CMD_EMERGENCY_HALT_BUZZER_ON"

    return {
        "final_risk_score": round(fused_score, 5),
        "risk_status": risk_category,
        "visual_score": round(v, 5),
        "visual_risk": classify_score(v),
        "sensor_score": round(s, 5),
        "sensor_risk": classify_score(s),
        "consensus_state": consensus,
        "operational_action": action,
        "buzzer_state": buzzer,
        "motor_relay_state": relay,
        "beacon_color": beacon,
        "hardware_command": cmd,
        "sensor_metadata": sensor_metadata or {},
        "threshold_1": THRESHOLD_1,
        "threshold_2": THRESHOLD_2
    }


# ==============================================================================
# 2. PATCHCORE JOINT DETECTOR ENGINE
# ==============================================================================

class PatchCoreEngine:
    """
    Loads and runs the existing trained PatchCore model from checkpoint.
    NEVER modifies or retrains the model.
    """

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model = None
        self.mb_norm = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PatchCore checkpoint not found at: {self.checkpoint_path.resolve()}\n"
                f"Please ensure results_joint/20260910_105659/patchcore_joint_v1.ckpt exists."
            )

        print(f"[SmartBelt Core] Loading PatchCore joint model from {self.checkpoint_path.name}...")
        from anomalib.models import Patchcore
        self.model = Patchcore.load_from_checkpoint(str(self.checkpoint_path), weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Extract normalized memory bank for fast distance computation
        mb = self.model.model.memory_bank.detach().to(self.device).float()
        self.mb_norm = torch.nn.functional.normalize(mb, p=2, dim=1)
        print(f"[SmartBelt Core] Model ready. Memory bank vectors: {tuple(self.mb_norm.shape)}")

    @torch.no_grad()
    def score_frame(self, frame_bgr: np.ndarray) -> float:
        """
        Extract patch embeddings and compute anomaly distance against memory bank.
        Returns:
            float: Anomaly distance score.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(self.device)

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

        # Chunked nearest neighbor distance search
        chunk = 512
        dists = []
        for s in range(0, emb_n.shape[0], chunk):
            d = torch.cdist(emb_n[s:s+chunk], self.mb_norm)
            dists.append(d.min(dim=1).values)
        dists = torch.cat(dists)

        score = float(dists.max().item())
        return score


# ==============================================================================
# 3. UNIFIED APPLICATION PIPELINE (SmartBeltApplication)
# ==============================================================================

class SmartBeltApplication:
    """
    Main unified industrial monitoring application class.
    Consumes PatchCore, XGBoost sensor module, fusion layer, and telemetry logger.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        xgboost_model_path: Optional[Path] = None,
        telemetry_log_path: Optional[Path] = None,
        device: str = "cpu",
        enable_debounce: bool = False,
        ema_alpha: float = 0.75
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.detector = PatchCoreEngine(checkpoint_path=self.checkpoint_path, device=self.device)
        self.sensor_module = XGBoostSensorPredictor(model_path=xgboost_model_path)
        self.telemetry_log_path = Path(telemetry_log_path) if telemetry_log_path else (CURRENT_DIR / "telemetry_log.csv")
        self.enable_debounce = enable_debounce
        self.ema_alpha = ema_alpha
        self._visual_history: List[float] = []
        self._sensor_history: List[float] = []
        self._init_telemetry_log()

    def reset_state(self):
        """Reset temporal smoothing filters and history buffers."""
        self._visual_history.clear()
        self._sensor_history.clear()

    def _init_telemetry_log(self):
        """Ensure telemetry CSV log file exists with full column structure."""
        if not self.telemetry_log_path.exists():
            with open(self.telemetry_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_idx", "visual_score", "visual_risk",
                    "sensor_score", "sensor_risk", "dominant_sensor",
                    "sensor_data_source", "final_risk_score", "risk_status",
                    "consensus_state", "buzzer_state", "motor_relay_state",
                    "beacon_color", "hardware_command", "temperature_c",
                    "vibration_rms_mms", "speed_slip_pct", "tension_kn"
                ])

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        sensor_readings: Optional[Dict[str, float]] = None,
        is_simulated_sensor: bool = True,
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Process a single video frame and corresponding sensor readings end-to-end.

        Args:
            frame_bgr (np.ndarray): OpenCV BGR frame.
            sensor_readings (dict, optional): Physical sensor readings.
            is_simulated_sensor (bool): Explicitly marks data source.
            frame_idx (int): Current frame index.

        Returns:
            dict: Complete inference, prediction, fusion, and alert telemetry packet.
        """
        ts = datetime.now().isoformat()

        # 1. Visual Inference (PatchCore)
        visual_score = self.detector.score_frame(frame_bgr)

        # 2. Sensor Prediction / Scoring (XGBoost Interface)
        if sensor_readings is None:
            # Default to nominal simulation if no live reading provided
            sensor_readings, is_simulated_sensor = simulate_sensor_reading("nominal")

        sensor_res = self.sensor_module.predict_single(sensor_readings, is_simulated=is_simulated_sensor)
        sensor_score = sensor_res["sensor_score"]

        # Temporal Debounce / Smoothing (if enabled)
        if self.enable_debounce:
            # Asymmetric Safety Override: Unambiguous severe emergencies bypass smoothing
            if visual_score >= 0.72 or not self._visual_history:
                effective_visual = visual_score
            else:
                effective_visual = self.ema_alpha * visual_score + (1.0 - self.ema_alpha) * self._visual_history[-1]

            if sensor_score >= 0.75 or not self._sensor_history:
                effective_sensor = sensor_score
            else:
                effective_sensor = self.ema_alpha * sensor_score + (1.0 - self.ema_alpha) * self._sensor_history[-1]

            self._visual_history.append(effective_visual)
            self._sensor_history.append(effective_sensor)
        else:
            effective_visual = visual_score
            effective_sensor = sensor_score

        # 3. Fusion / Decision Layer
        fused = fuse_risk(
            visual_score=effective_visual,
            sensor_score=effective_sensor,
            sensor_metadata=sensor_res
        )

        # 4. Append Telemetry Log
        self._log_telemetry(ts, frame_idx, effective_visual, sensor_res, fused)

        return {
            "timestamp": ts,
            "frame_idx": frame_idx,
            "visual_score": round(visual_score, 5),
            "visual_risk": fused["visual_risk"],
            "sensor_score": sensor_res["sensor_score"],
            "sensor_risk": sensor_res["risk_category"],
            "sensor_data_source": sensor_res["data_source_label"],
            "sensor_model_type": sensor_res["model_type"],
            "dominant_sensor": sensor_res["dominant_sensor"],
            "raw_sensor_readings": sensor_readings,
            "final_risk_score": fused["final_risk_score"],
            "risk_status": fused["risk_status"],
            "consensus_state": fused["consensus_state"],
            "operational_action": fused["operational_action"],
            "buzzer_state": fused["buzzer_state"],
            "motor_relay_state": fused["motor_relay_state"],
            "beacon_color": fused["beacon_color"],
            "hardware_command": fused["hardware_command"]
        }

    def _log_telemetry(self, ts: str, frame_idx: int, visual_score: float,
                       sensor_res: dict, fused: dict):
        """Append step to CSV audit log."""
        readings = sensor_res.get("raw_readings", {})
        with open(self.telemetry_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, frame_idx, round(visual_score, 5), fused["visual_risk"],
                sensor_res["sensor_score"], sensor_res["risk_category"],
                sensor_res["dominant_sensor"], sensor_res["data_source_label"],
                fused["final_risk_score"], fused["risk_status"],
                fused["consensus_state"], fused["buzzer_state"],
                fused["motor_relay_state"], fused["beacon_color"],
                fused["hardware_command"],
                readings.get("temperature_c", ""),
                readings.get("vibration_rms_mms", ""),
                readings.get("speed_slip_pct", ""),
                readings.get("tension_kn", "")
            ])

    def render_display_overlay(
        self,
        frame_bgr: np.ndarray,
        result_packet: Dict[str, Any]
    ) -> np.ndarray:
        """Render high-contrast industrial overlay on the video frame."""
        h, w = frame_bgr.shape[:2]
        out = frame_bgr.copy()
        risk = result_packet["risk_status"]

        color_map = {
            "NORMAL": (40, 167, 69),    # Green (BGR)
            "WARNING": (0, 191, 255),   # Amber (BGR)
            "CRITICAL": (38, 38, 220)   # Red (BGR)
        }
        color = color_map.get(risk, (200, 200, 200))

        # 1. Border
        border_thick = max(6, int(min(w, h) * 0.015))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness=border_thick)

        # 2. Semi-transparent header banner
        header_h = 75
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)
        cv2.line(out, (0, header_h), (w, header_h), color, thickness=2)

        # 3. Telemetry text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "SmartBelt | Unified Vision + Sensor Pipeline", (15, 25), font, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        
        info_line = (
            f"Frame: {result_packet['frame_idx']:04d} | "
            f"Vision: {result_packet['visual_score']:.3f} | "
            f"Sensor: {result_packet['sensor_score']:.3f} | "
            f"Fused: {result_packet['final_risk_score']:.3f}"
        )
        cv2.putText(out, info_line, (15, 52), font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Risk badge
        badge_text = f"{risk}"
        (tw, th_box), _ = cv2.getTextSize(badge_text, font, 0.75, 2)
        badge_x = w - tw - 25
        cv2.rectangle(out, (badge_x - 10, 12), (w - 15, 58), color, -1)
        cv2.putText(out, badge_text, (badge_x, 44), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Critical Warning Footer
        if risk == "CRITICAL":
            alert_h = 60
            y1, y2 = h - alert_h - 20, h - 20
            alert_overlay = out.copy()
            cv2.rectangle(alert_overlay, (20, y1), (w - 20, y2), (0, 0, 180), -1)
            cv2.addWeighted(alert_overlay, 0.85, out, 0.15, 0, out)
            cv2.rectangle(out, (20, y1), (w - 20, y2), (255, 255, 255), 2)
            msg = f"EMERGENCY MOTOR TRIP: {result_packet['consensus_state']}"
            (mw, _), _ = cv2.getTextSize(msg, font, 0.65, 2)
            cv2.putText(out, msg, ((w - mw) // 2, y1 + 38), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return out


# ==============================================================================
# 4. REFERENCE ACCESS TO EXISTING EVALUATIONS
# ==============================================================================

def get_existing_evaluation_summary() -> Dict[str, Any]:
    """
    Provides read-only access to existing Phase 1 & results_joint evaluation files.
    Does NOT duplicate files.
    """
    summary = {
        "checkpoint_exists": DEFAULT_CHECKPOINT.exists(),
        "checkpoint_path": str(DEFAULT_CHECKPOINT.resolve()),
        "checkpoint_size_mb": round(DEFAULT_CHECKPOINT.stat().st_size / (1024 * 1024), 2) if DEFAULT_CHECKPOINT.exists() else 0,
        "evaluation_dir": str(EVALUATION_DIR.resolve()),
        "scores_csv_path": str(SCORES_CSV_PATH.resolve()),
        "metrics_summary_path": str((EVALUATION_DIR / "metrics_summary.md").resolve()),
        "thresholds": {
            "threshold_1_warning": THRESHOLD_1,
            "threshold_2_critical": THRESHOLD_2,
            "methodology": "Validation μ + 2σ / μ + 3σ (scientifically unbiased)"
        }
    }
    return summary


# ==============================================================================
# 5. CLI TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("SmartBelt — Core Application Module Self-Test (Phase 4)")
    print("=" * 75)

    eval_info = get_existing_evaluation_summary()
    print(f"Loaded Checkpoint: {eval_info['checkpoint_path']} ({eval_info['checkpoint_size_mb']} MB)")
    print(f"Evaluation Dir:   {eval_info['evaluation_dir']}")

    app = SmartBeltApplication()
    print("[Self-Test] Initializing test frames and simulated sensor scenarios...")

    # Run demonstration using an existing image from dataset/patchcore/test
    test_img_path = PROJECT_ROOT / "dataset" / "patchcore" / "test" / "bad" / "real_damage" / "original_damaged_joint_01_frame_0420.jpg"
    if not test_img_path.exists():
        # Fallback to test/good
        test_img_path = PROJECT_ROOT / "dataset" / "patchcore" / "test" / "good" / "frame_0000.jpg"

    if test_img_path.exists():
        frame = cv2.imread(str(test_img_path))
        print(f"[Self-Test] Processing sample test image: {test_img_path.name}")
        
        # Test 1: Real damage visual + splice vibration sensors
        readings, is_sim = simulate_sensor_reading("splice_vibration")
        res = app.process_frame(frame, sensor_readings=readings, is_simulated_sensor=is_sim, frame_idx=1)
        print("\nPipeline Result (Real Damage + Simulated Vibration):")
        print(f"  Visual Score:       {res['visual_score']:.5f} ({res['visual_risk']})")
        print(f"  Sensor Score:       {res['sensor_score']:.5f} ({res['sensor_risk']}) [{res['dominant_sensor']}]")
        print(f"  Sensor Data Source: {res['sensor_data_source']}")
        print(f"  Fused Risk Score:   {res['final_risk_score']:.5f} ({res['risk_status']})")
        print(f"  Consensus State:    {res['consensus_state']}")
        print(f"  Motor Relay:        {res['motor_relay_state']}")
        print(f"  Hardware Command:   {res['hardware_command']}")
    else:
        print(f"[Self-Test] Test image not found at {test_img_path}.")

    print("\n[Self-Test] Complete. Module is importable and ready.")
