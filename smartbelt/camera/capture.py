"""
SmartBelt v2 — Camera Capture Thread
Opens Logitech C270 HD (or specified camera index) via OpenCV and continuously grabs frames.
Runs in a daemon thread so it never blocks the main process or inference loop.

Usage:
    from smartbelt.camera.capture import CameraCapture
    cam = CameraCapture(device_index=0)
    cam.start()
    frame = cam.get_latest_frame()  # numpy array (H, W, 3) BGR, or None
    cam.stop()
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Continuously grabs frames from the camera device in a background thread.
    Only the most-recent frame is retained (latest frame wins) so the inference
    engine always gets the newest image, not a stale buffer frame.
    """

    def __init__(
        self,
        source: int | str | Path = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.is_video_file = isinstance(source, (str, Path)) or (isinstance(source, str) and not str(source).isdigit())

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Diagnostics
        self.frames_grabbed: int = 0
        self.frames_dropped: int = 0
        self.last_grab_ts: float = 0.0

    def start(self) -> None:
        """Open the camera or video file and start the background grab thread."""
        if self._running:
            return

        if self.is_video_file:
            path_str = str(self.source)
            self._cap = cv2.VideoCapture(path_str)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video file at '{path_str}'. Check file path.")
            logger.info(f"Video file opened: '{path_str}'")
        else:
            dev_idx = int(self.source)
            self._cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(dev_idx)

            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera at index {dev_idx}. "
                    "Check that the Logitech C270 is plugged in and not in use by another app."
                )

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                f"Camera opened: index={dev_idx}, "
                f"resolution={actual_w}x{actual_h} @ {actual_fps:.1f}fps"
            )

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, name="CameraCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal grab thread to stop and release the camera device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera capture stopped.")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the most recently grabbed frame as a BGR numpy array, or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def is_running(self) -> bool:
        return self._running

    def _grab_loop(self) -> None:
        """Background thread: continuously reads frames from camera or video file."""
        frame_interval = 1.0 / max(float(self.fps), 1.0)
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Camera/video source not open, stopping grab loop.")
                self._running = False
                break

            t_start = time.monotonic()
            ret, frame = self._cap.read()

            if not ret:
                if self.is_video_file:
                    # Auto-rewind video file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()

                if not ret:
                    self.frames_dropped += 1
                    time.sleep(0.02)
                    continue

            with self._lock:
                self._latest_frame = frame

            self.frames_grabbed += 1
            self.last_grab_ts = time.monotonic()

            # For video files, pace according to target FPS
            if self.is_video_file:
                elapsed = time.monotonic() - t_start
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)
