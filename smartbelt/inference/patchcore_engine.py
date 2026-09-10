"""
SmartBelt v2 — PatchCore Inference Engine
Loads the existing PatchCore checkpoint and runs async inference in a background thread.
The main thread (Streamlit) is never blocked by inference latency.

Usage:
    from smartbelt.inference.patchcore_engine import PatchCoreEngine
    engine = PatchCoreEngine(checkpoint_path)
    engine.start()
    engine.submit_frame(bgr_frame)    # non-blocking
    result = engine.latest_result     # InferenceResult or None
    engine.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Thresholds (empirically calibrated on real conveyor joint data) ───────────
THRESHOLD_1 = 0.62710   # NORMAL -> ELEVATED  (mu + 2sigma baseline)
THRESHOLD_2 = 0.68850   # ELEVATED -> CRITICAL (separation boundary)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_SIZE = (256, 256)


@dataclass
class InferenceResult:
    """Output of one PatchCore inference cycle."""
    visual_score: float = 0.0                  # anomaly score in [0, 1]
    anomaly_map: Optional[np.ndarray] = None   # (H, W) float32 heatmap [0, 1]
    raw_frame: Optional[np.ndarray] = None     # original BGR frame
    annotated_frame: Optional[np.ndarray] = None  # frame with heatmap overlay & HUD
    inference_latency_s: float = 0.0
    frame_id: int = 0


class PatchCoreEngine:
    """
    Loads the PatchCore checkpoint and runs inference asynchronously.
    Inference is CPU-bound on ARM64 Snapdragon (~1.2-1.5s), so it runs in its own
    dedicated daemon thread without lagging the camera or UI threads.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._model = None
        self._transform = None
        self._device = torch.device("cpu")

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[InferenceResult] = None
        self._frame_id: int = 0
        self._frame_lock = threading.Lock()

    def load(self) -> None:
        """Load the PatchCore model from checkpoint. Call once before start()."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PatchCore checkpoint not found: {self.checkpoint_path}\n"
                "Ensure results_joint/20260910_105659/patchcore_joint_v1.ckpt exists."
            )
        logger.info(f"Loading PatchCore checkpoint: {self.checkpoint_path}")
        t0 = time.monotonic()

        # Ensure compatibility with PyTorch 2.6+ weights_only policy
        import functools
        _orig_load = torch.load
        @functools.wraps(_orig_load)
        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = _safe_load

        # Try anomalib loading
        try:
            from anomalib.models.image.patchcore import Patchcore
            self._model = Patchcore.load_from_checkpoint(
                str(self.checkpoint_path), map_location=self._device
            )
        except Exception as err:
            try:
                from anomalib.models import Patchcore
                self._model = Patchcore.load_from_checkpoint(
                    str(self.checkpoint_path), map_location=self._device
                )
            except Exception as err2:
                logger.error(f"Failed to load Patchcore model: {err2}")
                raise RuntimeError(f"Could not load checkpoint: {err2}") from None
        finally:
            torch.load = _orig_load

        self._model.eval()
        self._model.to(self._device)
        self._setup_transform()
        logger.info(f"PatchCore model ready in {time.monotonic() - t0:.2f}s")

    def start(self) -> None:
        """Start background inference worker thread. Call load() first."""
        if self._model is None:
            raise RuntimeError("Call load() before start().")
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, name="PatchCoreInference", daemon=True
        )
        self._thread.start()
        logger.info("PatchCore inference thread running.")

    def stop(self) -> None:
        """Stop inference worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("PatchCore inference stopped.")

    def submit_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Submit a new BGR frame for inference (non-blocking).
        Overwrites any uninspected frame so newest frame is always processed next.
        """
        with self._frame_lock:
            self._pending_frame = bgr_frame.copy()

    @property
    def latest_result(self) -> Optional[InferenceResult]:
        """Thread-safe access to the most recent inference result."""
        with self._lock:
            return self._latest_result

    def _setup_transform(self) -> None:
        """Configure input preprocessing pipeline."""
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _inference_loop(self) -> None:
        """Background thread: wait for frame, run forward pass, publish result."""
        while self._running:
            frame = None
            with self._frame_lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is None:
                time.sleep(0.04)
                continue

            self._frame_id += 1
            result = self._run_inference(frame, self._frame_id)
            with self._lock:
                self._latest_result = result

    def _run_inference(self, bgr_frame: np.ndarray, frame_id: int) -> InferenceResult:
        """Run PatchCore model on single frame and format output."""
        t0 = time.monotonic()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(tensor)

        # Handle various anomalib output structures (dict, Torch tensor, NamedTuple)
        score = 0.0
        amap = None
        if isinstance(output, dict):
            score = float(output.get("pred_score", output.get("anomaly_score", 0.0)))
            amap = output.get("anomaly_map", output.get("pred_mask", None))
        elif hasattr(output, "pred_score"):
            score = float(getattr(output, "pred_score", 0.0))
            amap = getattr(output, "anomaly_map", None)
        elif isinstance(output, (tuple, list)) and len(output) >= 1:
            score = float(output[0])
            if len(output) > 1:
                amap = output[1]

        # Process anomaly map
        anomaly_map_np = None
        if amap is not None:
            if isinstance(amap, torch.Tensor):
                amap_np = amap.squeeze().cpu().numpy().astype(np.float32)
            else:
                amap_np = np.asarray(amap, dtype=np.float32).squeeze()
            mn, mx = amap_np.min(), amap_np.max()
            if mx > mn:
                amap_np = (amap_np - mn) / (mx - mn)
            anomaly_map_np = amap_np

        annotated = self._render_overlay(bgr_frame, anomaly_map_np, score)
        latency = time.monotonic() - t0

        return InferenceResult(
            visual_score=score,
            anomaly_map=anomaly_map_np,
            raw_frame=bgr_frame,
            annotated_frame=annotated,
            inference_latency_s=latency,
            frame_id=frame_id,
        )

    @staticmethod
    def _render_overlay(
        bgr_frame: np.ndarray,
        anomaly_map: Optional[np.ndarray],
        score: float,
    ) -> np.ndarray:
        """Blend color-coded anomaly heatmap and telemetry text onto frame."""
        annotated = bgr_frame.copy()

        if anomaly_map is not None:
            h, w = bgr_frame.shape[:2]
            heatmap_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_uint8 = (heatmap_resized * 255).clip(0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            alpha = 0.35
            annotated = cv2.addWeighted(annotated, 1.0 - alpha, heatmap_color, alpha, 0)

        # Status text
        if score >= THRESHOLD_2:
            color = (0, 0, 255)      # Red
            label = f"CRITICAL DAMAGE  {score:.4f}"
        elif score >= THRESHOLD_1:
            color = (0, 165, 255)    # Orange
            label = f"ELEVATED RISK    {score:.4f}"
        else:
            color = (0, 200, 0)      # Green
            label = f"NOMINAL BELT     {score:.4f}"

        cv2.putText(annotated, label, (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return annotated
