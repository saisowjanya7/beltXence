"""
SmartBelt v2 — Real Sensor Anomaly Scorer
Scores live physical sensor readings against an empirically calibrated baseline
collected directly from the running conveyor hardware. No mock data. No synthetic tables.

Workflow:
1. Run a 60-second baseline calibration with the conveyor running under nominal load.
   Computes mean (mu) and standard deviation (sigma) for temperature, vibration RMS, and load cell.
   Saves profile to calibration/sensor_baseline.json.
2. During live operation, incoming SensorPackets are scored via weighted Mahalanobis / z-score
   distance from nominal baseline, mapped smoothly to [0, 1] through a calibrated logistic curve.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from smartbelt.serial_reader.reader import SensorPacket

logger = logging.getLogger(__name__)

# Sensor channel weights (sums to 1.0)
CHANNEL_WEIGHTS = {
    "temp": 0.30,   # MLX90614 IR temperature
    "vib":  0.50,   # MPU-6050 vibration RMS
    "load": 0.20,   # HX711 load cell
}

SIGMOID_K = 1.5
MIN_STD = 0.001


@dataclass
class SensorBaseline:
    """Statistical summary of healthy conveyor baseline operation."""
    temp_mean: float = 0.0
    temp_std:  float = 1.0
    vib_mean:  float = 0.0
    vib_std:   float = 1.0
    load_mean: float = 0.0
    load_std:  float = 1.0
    n_samples:  int  = 0
    calibrated: bool = False


class SensorScorer:
    """
    Evaluates raw physical sensor telemetry against nominal machine baseline.
    Produces an anomaly score in range [0, 1].
    """

    def __init__(self, baseline_path: Path) -> None:
        self.baseline_path = Path(baseline_path)
        self.baseline: SensorBaseline = SensorBaseline()

    def load_baseline(self) -> bool:
        """Load baseline from disk. Returns True if successfully loaded."""
        if not self.baseline_path.exists():
            logger.info(f"No sensor baseline found at {self.baseline_path}. Run calibration before operation.")
            return False
        try:
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.baseline = SensorBaseline(**data)
            logger.info(
                f"Loaded baseline ({self.baseline.n_samples} samples): "
                f"temp={self.baseline.temp_mean:.1f}+-{self.baseline.temp_std:.2f}C, "
                f"vib={self.baseline.vib_mean:.3f}+-{self.baseline.vib_std:.3f}g, "
                f"load={self.baseline.load_mean:.2f}+-{self.baseline.load_std:.2f}kg"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to read baseline file: {e}")
            return False

    def calibrate(self, packets: List[SensorPacket]) -> SensorBaseline:
        """Calculate mean and standard deviation from nominal run telemetry."""
        valid_packets = [p for p in packets if not p.parse_error]
        if not valid_packets:
            raise ValueError("No valid sensor telemetry received for calibration.")

        temps = [p.temp_c for p in valid_packets]
        vibs  = [p.vib_rms for p in valid_packets]
        loads = [p.load_kg for p in valid_packets]

        def calc_mean(arr):
            return sum(arr) / len(arr)

        def calc_std(arr, m):
            if len(arr) <= 1:
                return 1.0
            variance = sum((x - m) ** 2 for x in arr) / (len(arr) - 1)
            return math.sqrt(variance)

        tm = calc_mean(temps)
        vm = calc_mean(vibs)
        lm = calc_mean(loads)

        ts = max(calc_std(temps, tm), MIN_STD)
        vs = max(calc_std(vibs, vm), MIN_STD)
        ls = max(calc_std(loads, lm), MIN_STD)

        self.baseline = SensorBaseline(
            temp_mean=tm, temp_std=ts,
            vib_mean=vm,  vib_std=vs,
            load_mean=lm, load_std=ls,
            n_samples=len(valid_packets),
            calibrated=True,
        )

        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.baseline), f, indent=2)

        logger.info(f"Saved new baseline profile to {self.baseline_path}")
        return self.baseline

    def score(self, packet: Optional[SensorPacket]) -> float:
        """Compute normalized anomaly score [0, 1] for live sensor packet."""
        if packet is None or packet.parse_error or not self.baseline.calibrated:
            return 0.0

        b = self.baseline
        w = CHANNEL_WEIGHTS

        z_temp = abs(packet.temp_c - b.temp_mean) / b.temp_std
        z_vib  = abs(packet.vib_rms - b.vib_mean) / b.vib_std
        z_load = abs(packet.load_kg - b.load_mean) / b.load_std

        z_combined = (w["temp"] * z_temp) + (w["vib"] * z_vib) + (w["load"] * z_load)

        # Map to [0, 1] using logistic function centered at z=2.0 standard deviations
        score = 1.0 / (1.0 + math.exp(-SIGMOID_K * (z_combined - 2.0)))
        return float(score)
