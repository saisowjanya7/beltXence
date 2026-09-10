"""
SmartBelt v2 — Pipeline Orchestrator
Coordinates background worker threads: CameraCapture, SerialReader, PatchCoreEngine,
and updates shared thread-safe system telemetry consumed by the Streamlit dashboard.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from smartbelt.camera.capture import CameraCapture
from smartbelt.fusion.risk_fusion import FusionResult, FusionState, fuse_risk
from smartbelt.inference.patchcore_engine import InferenceResult, PatchCoreEngine
from smartbelt.inference.sensor_scorer import SensorScorer
from smartbelt.serial_reader.reader import SensorPacket, SerialReader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results_joint" / "20260910_105659" / "patchcore_joint_v1.ckpt"
DEFAULT_BASELINE   = PROJECT_ROOT / "calibration" / "sensor_baseline.json"


@dataclass
class SmartBeltState:
    """Snapshot of system health, real-time telemetry, and inference outputs."""
    updated_at: float = 0.0

    # Vision
    visual_score: float = 0.0
    inference_latency_s: float = 0.0
    frame_id: int = 0
    annotated_frame: Optional[np.ndarray] = None

    # Sensor Telemetry
    sensor_score: float = 0.0
    sensor_packet: Optional[SensorPacket] = None
    sensor_healthy: bool = False

    # Risk Fusion
    fusion: FusionResult = field(default_factory=FusionResult)

    # Health & Diagnostics
    camera_fps: float = 0.0
    inference_fps: float = 0.0
    serial_packets_total: int = 0
    serial_parse_errors: int = 0
    calibrated: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    baseline_path: Path = DEFAULT_BASELINE
    camera_source: int | str | Path = 0
    serial_port: Optional[str] = None
    ema_alpha: float = 0.75
    fusion_interval_s: float = 0.25


class SmartBeltOrchestrator:
    """
    Coordinates hardware acquisition, neural inference, and risk fusion.
    Streamlit runs in the main thread and polls `get_state()` without blocking.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()

        self.camera = CameraCapture(source=self.config.camera_source)
        self.serial = SerialReader(port=self.config.serial_port)
        self.engine = PatchCoreEngine(self.config.checkpoint_path)
        self.scorer = SensorScorer(self.config.baseline_path)

        self._ema_v: List[Optional[float]] = [None]
        self._ema_s: List[Optional[float]] = [None]

        self._state_lock = threading.Lock()
        self._state = SmartBeltState()

        self._running = False
        self._fusion_thread: Optional[threading.Thread] = None
        self._frame_bridge_thread: Optional[threading.Thread] = None

        self._inference_count = 0
        self._last_frame_id = -1
        self._start_ts = 0.0

    def start(self) -> None:
        """Start hardware threads, load model, and launch processing loops."""
        if self._running:
            return

        logger.info("Initializing SmartBelt orchestrator...")
        self._start_ts = time.monotonic()
        errors = []

        # 1. Load PatchCore
        try:
            self.engine.load()
        except Exception as e:
            errors.append(f"Vision model load failed: {e}")
            logger.error(f"Vision model error: {e}")

        # 2. Check sensor baseline
        calibrated = self.scorer.load_baseline()

        # 3. Start Camera
        try:
            self.camera.start()
        except Exception as e:
            errors.append(f"Camera open error: {e}")
            logger.error(f"Camera error: {e}")

        # 4. Start Serial Reader
        self.serial.start()

        # 5. Start Inference Engine
        if not errors or all("Vision model" not in err for err in errors):
            self.engine.start()

        self._running = True

        # Frame submission loop
        self._frame_bridge_thread = threading.Thread(
            target=self._frame_bridge_loop, name="FrameBridge", daemon=True
        )
        self._frame_bridge_thread.start()

        # Fusion loop
        self._fusion_thread = threading.Thread(
            target=self._fusion_loop, name="FusionLoop", daemon=True
        )
        self._fusion_thread.start()

        with self._state_lock:
            self._state.calibrated = calibrated
            self._state.errors = errors

        logger.info("SmartBelt orchestrator active.")

    def stop(self) -> None:
        """Stop all background workers."""
        self._running = False
        if self._fusion_thread:
            self._fusion_thread.join(timeout=3.0)
        if self._frame_bridge_thread:
            self._frame_bridge_thread.join(timeout=3.0)

        self.engine.stop()
        self.serial.stop()
        self.camera.stop()
        logger.info("SmartBelt orchestrator shut down.")

    def set_camera_source(self, source: int | str | Path) -> bool:
        """Switch video source dynamically (e.g. between live webcam and demo video)."""
        if self.camera.source == source and self.camera.is_running:
            return True
        logger.info(f"Switching camera source to: {source}")
        self.camera.stop()
        self.camera = CameraCapture(source=source)
        try:
            self.camera.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start camera source {source}: {e}")
            with self._state_lock:
                self._state.errors.append(f"Camera source switch failed: {e}")
            return False

    def get_state(self) -> SmartBeltState:
        """Retrieve latest snapshot of system telemetry for UI display."""
        with self._state_lock:
            return SmartBeltState(
                updated_at=self._state.updated_at,
                visual_score=self._state.visual_score,
                inference_latency_s=self._state.inference_latency_s,
                frame_id=self._state.frame_id,
                annotated_frame=self._state.annotated_frame,
                sensor_score=self._state.sensor_score,
                sensor_packet=self._state.sensor_packet,
                sensor_healthy=self._state.sensor_healthy,
                fusion=self._state.fusion,
                camera_fps=self._state.camera_fps,
                inference_fps=self._state.inference_fps,
                serial_packets_total=self._state.serial_packets_total,
                serial_parse_errors=self._state.serial_parse_errors,
                calibrated=self._state.calibrated,
                errors=list(self._state.errors),
            )

    def run_calibration(self, duration_s: float = 60.0, on_progress: Optional[Callable[[float, float], None]] = None) -> bool:
        """Record nominal sensor readings to establish machine baseline."""
        logger.info(f"Initiating {duration_s}s baseline calibration...")
        packets = []
        t0 = time.monotonic()

        while (time.monotonic() - t0) < duration_s:
            pkt = self.serial.latest_packet
            if pkt and not pkt.parse_error:
                packets.append(pkt)
            elapsed = time.monotonic() - t0
            if on_progress:
                on_progress(elapsed, duration_s)
            time.sleep(0.2)

        if len(packets) < 5:
            logger.error(f"Calibration aborted: insufficient sensor data ({len(packets)} packets)")
            return False

        self.scorer.calibrate(packets)
        with self._state_lock:
            self._state.calibrated = True
        return True

    def _frame_bridge_loop(self) -> None:
        """Continuously feeds newly captured camera frames to inference queue."""
        last_grabbed = -1
        while self._running:
            frame = self.camera.get_latest_frame()
            if frame is not None and self.camera.frames_grabbed != last_grabbed:
                self.engine.submit_frame(frame)
                last_grabbed = self.camera.frames_grabbed
            time.sleep(0.02)

    def _fusion_loop(self) -> None:
        """Periodic fusion of latest camera result with live IoT sensor readings."""
        inf_fps = 0.0
        while self._running:
            t0 = time.monotonic()

            ir: Optional[InferenceResult] = self.engine.latest_result
            visual_score = ir.visual_score if ir else 0.0

            if ir and ir.frame_id != self._last_frame_id:
                self._last_frame_id = ir.frame_id
                self._inference_count += 1
                elapsed = max(time.monotonic() - self._start_ts, 1.0)
                inf_fps = self._inference_count / elapsed

            pkt: Optional[SensorPacket] = self.serial.latest_packet
            sensor_score = self.scorer.score(pkt) if pkt else 0.0

            fusion_res = fuse_risk(
                visual_score,
                sensor_score,
                ema_alpha=self.config.ema_alpha,
                _ema_v=self._ema_v,
                _ema_s=self._ema_s,
            )

            cam_elapsed = max(time.monotonic() - self._start_ts, 1.0)
            cam_fps = self.camera.frames_grabbed / cam_elapsed

            with self._state_lock:
                self._state.updated_at = time.monotonic()
                self._state.visual_score = visual_score
                self._state.inference_latency_s = ir.inference_latency_s if ir else 0.0
                self._state.frame_id = ir.frame_id if ir else 0
                self._state.annotated_frame = ir.annotated_frame if ir else None
                self._state.sensor_score = sensor_score
                self._state.sensor_packet = pkt
                self._state.sensor_healthy = self.serial.is_healthy
                self._state.fusion = fusion_res
                self._state.camera_fps = cam_fps
                self._state.inference_fps = inf_fps
                self._state.serial_packets_total = self.serial.packets_received
                self._state.serial_parse_errors = self.serial.parse_errors

            elapsed_loop = time.monotonic() - t0
            sleep_needed = max(0.01, self.config.fusion_interval_s - elapsed_loop)
            time.sleep(sleep_needed)
