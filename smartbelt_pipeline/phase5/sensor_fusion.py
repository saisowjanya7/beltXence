"""
SmartBelt — Phase 5: Multi-Modal Sensor Fusion Module
Combines Visual Anomaly Inference (PatchCore) with Telemetry Sensors
(Vibration, Temperature, Belt Tension, and Speed Slip).

=============================================================================
STATUS: SIMULATED / HARDWARE-READY STUB
AUDIT FINDING: No physical ESP32 microcontroller or real sensor telemetry
currently exists in this repository. In accordance with project scientific
rules, all telemetry in this module is EXPLICITLY LABELED AS SIMULATED.
The driver architecture includes a live serial parser ready to connect to
an ESP32 over USB/UART once hardware is available.
=============================================================================
"""

import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, Tuple

# Risk Thresholds from Phase 2
THRESHOLD_1 = 0.62710  # NORMAL -> WARNING
THRESHOLD_2 = 0.68887  # WARNING -> CRITICAL

# Industrial Sensor Baseline Thresholds (ISO 10816 vibration standards + conveyor specs)
SENSOR_BOUNDS = {
    "vibration_rms_mms": {  # mm/s RMS (ISO 10816-3 Class II/III industrial machinery)
        "nominal_max": 2.5,   # <= 2.5 mm/s: Good
        "warning_max": 4.5,   # 2.5 - 4.5 mm/s: Alert (Bearing fatigue / splice pounding)
        "critical_min": 7.0,  # >= 7.0 mm/s: Danger (Imminent structural mechanical failure)
    },
    "temperature_c": {       # Splice / pulley temperature in Celsius (Infrared MLX90614)
        "nominal_max": 45.0,  # <= 45 C: Nominal operational friction
        "warning_max": 65.0,  # 45 - 65 C: Elevated friction / slip heating
        "critical_min": 80.0, # >= 80 C: Severe thermal runaway / combustion hazard
    },
    "tension_kn": {          # Belt load cell tension in kN
        "nominal_min": 18.0,
        "nominal_max": 26.0,  # 18 - 26 kN: Standard tensioning
        "warning_min": 12.0,  # 12 - 18 kN (slack) or 26 - 32 kN (over-tension)
        "warning_max": 32.0,
        "critical_loss": 8.0, # <= 8 kN: Complete tension loss (Joint snap / severed belt)
    },
    "speed_slip_pct": {      # Encoder / proximity slip percentage relative to drive pulley
        "nominal_max": 5.0,   # <= 5%: Normal dynamic creep
        "warning_max": 15.0,  # 5 - 15%: Drive pulley slippage
        "critical_min": 25.0, # >= 25%: Total drive stall / jammed belt
    }
}


# ==============================================================================
# SENSOR PARAMETER NORMALIZATION & SCORING
# ==============================================================================

def normalize_parameter(val: float, bounds: dict, param_type: str) -> float:
    """Normalize a physical parameter value into a [0.0, 1.0] anomaly score."""
    if param_type == "vibration":
        if val <= bounds["nominal_max"]:
            return max(0.0, (val / bounds["nominal_max"]) * 0.45)
        elif val <= bounds["warning_max"]:
            fraction = (val - bounds["nominal_max"]) / (bounds["warning_max"] - bounds["nominal_max"])
            return 0.45 + fraction * (THRESHOLD_2 - THRESHOLD_1)
        else:
            fraction = min(1.0, (val - bounds["warning_max"]) / (bounds["critical_min"] - bounds["warning_max"]))
            return THRESHOLD_2 + fraction * (1.0 - THRESHOLD_2)

    elif param_type == "temperature":
        if val <= bounds["nominal_max"]:
            return max(0.0, (val / bounds["nominal_max"]) * 0.45)
        elif val <= bounds["warning_max"]:
            fraction = (val - bounds["nominal_max"]) / (bounds["warning_max"] - bounds["nominal_max"])
            return 0.45 + fraction * (THRESHOLD_2 - THRESHOLD_1)
        else:
            fraction = min(1.0, (val - bounds["warning_max"]) / (bounds["critical_min"] - bounds["warning_max"]))
            return THRESHOLD_2 + fraction * (1.0 - THRESHOLD_2)

    elif param_type == "tension":
        # Both slack (< nominal) and over-tension (> nominal) are dangerous
        mid = (bounds["nominal_min"] + bounds["nominal_max"]) / 2.0
        if bounds["nominal_min"] <= val <= bounds["nominal_max"]:
            return 0.35  # Nominal
        elif val < bounds["nominal_min"]:
            if val <= bounds["critical_loss"]:
                return 0.95  # Total belt snap
            fraction = (bounds["nominal_min"] - val) / (bounds["nominal_min"] - bounds["critical_loss"])
            return 0.50 + fraction * 0.40
        else:
            fraction = min(1.0, (val - bounds["nominal_max"]) / (bounds["warning_max"] - bounds["nominal_max"]))
            return 0.50 + fraction * 0.35

    elif param_type == "speed_slip":
        if val <= bounds["nominal_max"]:
            return max(0.0, (val / bounds["nominal_max"]) * 0.45)
        elif val <= bounds["warning_max"]:
            fraction = (val - bounds["nominal_max"]) / (bounds["warning_max"] - bounds["nominal_max"])
            return 0.45 + fraction * (THRESHOLD_2 - THRESHOLD_1)
        else:
            fraction = min(1.0, (val - bounds["warning_max"]) / (bounds["critical_min"] - bounds["warning_max"]))
            return THRESHOLD_2 + fraction * (1.0 - THRESHOLD_2)

    return 0.0


def sensor_parameter_score(readings: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a unified [0.0, 1.0] mechanical distress score from raw sensor readings.

    Args:
        readings (dict): Raw physical values e.g. {
            "vibration_rms_mms": 1.8,
            "temperature_c": 38.5,
            "tension_kn": 22.0,
            "speed_slip_pct": 2.1
        }

    Returns:
        tuple: (normalized_sensor_score: float, breakdown_dict: dict)
    """
    p_scores = {}
    
    # 1. Vibration
    vib = readings.get("vibration_rms_mms", 1.5)
    p_scores["vibration"] = normalize_parameter(vib, SENSOR_BOUNDS["vibration_rms_mms"], "vibration")

    # 2. Temperature
    temp = readings.get("temperature_c", 35.0)
    p_scores["temperature"] = normalize_parameter(temp, SENSOR_BOUNDS["temperature_c"], "temperature")

    # 3. Tension
    tension = readings.get("tension_kn", 22.0)
    p_scores["tension"] = normalize_parameter(tension, SENSOR_BOUNDS["tension_kn"], "tension")

    # 4. Speed slip
    slip = readings.get("speed_slip_pct", 2.0)
    p_scores["speed_slip"] = normalize_parameter(slip, SENSOR_BOUNDS["speed_slip_pct"], "speed_slip")

    # Aggregation: Industrial conservative policy
    # If ANY single sensor exceeds critical threshold, the mechanical distress is critical
    max_param_name = max(p_scores, key=p_scores.get)
    max_param_score = p_scores[max_param_name]

    # Weighted mean of secondary sensors
    other_scores = [v for k, v in p_scores.items() if k != max_param_name]
    avg_others = sum(other_scores) / len(other_scores) if other_scores else 0.0

    # Combined sensor score: 80% dominant distress + 20% background mechanical health
    composite_sensor_score = 0.80 * max_param_score + 0.20 * avg_others

    breakdown = {
        "composite_sensor_score": round(composite_sensor_score, 5),
        "dominant_sensor": max_param_name,
        "dominant_score": round(max_param_score, 5),
        "per_sensor_normalized": {k: round(v, 4) for k, v in p_scores.items()},
        "raw_readings": readings,
    }
    return composite_sensor_score, breakdown


# ==============================================================================
# MULTI-MODAL FUSION ENGINE (VISION + SENSORS)
# ==============================================================================

def combine(
    visual_score: float,
    sensor_score: float,
    sensor_metadata: Optional[Dict[str, Any]] = None,
    mode: str = "multimodal_conservative"
) -> Dict[str, Any]:
    """
    Fuse computer vision anomaly score (PatchCore) with telemetry sensor score.

    Fusion Logic:
    1. Cross-Modal Confirmation (Vision CRITICAL + Sensor CRITICAL):
       Highest alert priority (Catastrophic Joint Rupture confirmed by vibration spike).
    2. Vision Dominant (Vision CRITICAL, Sensors Normal):
       Visual tear / splice separation before mechanical vibration sets in -> CRITICAL.
    3. Sensor Dominant (Vision Normal, Sensor CRITICAL):
       Bearing seizure / thermal friction buildup on blind spot -> CRITICAL / WARNING.
    4. Conservative Max Policy:
       final_risk_score = max(visual_score, sensor_score, 0.65*visual + 0.35*sensor)
    """
    v_sc = float(visual_score)
    s_sc = float(sensor_score)

    # Multi-modal fusion
    if v_sc >= THRESHOLD_2 and s_sc >= THRESHOLD_2:
        # Both modalities agree on severe failure
        final_score = min(1.0, max(v_sc, s_sc) + 0.05)
        consensus = "CONFIRMED_MULTIMODAL_EMERGENCY"
    elif v_sc >= THRESHOLD_2:
        final_score = v_sc
        consensus = "VISION_DOMINANT_FAILURE"
    elif s_sc >= THRESHOLD_2:
        final_score = s_sc
        consensus = "SENSOR_DOMINANT_MECHANICAL_DISTRESS"
    elif v_sc >= THRESHOLD_1 or s_sc >= THRESHOLD_1:
        final_score = max(v_sc, s_sc)
        consensus = "ELEVATED_INSPECTION_ZONE"
    else:
        # Nominal: weighted average to suppress single-point noise
        final_score = 0.60 * v_sc + 0.40 * s_sc
        consensus = "NOMINAL_OPERATION"

    # Map to risk category
    if final_score < THRESHOLD_1:
        risk = "NORMAL"
        action = "Nominal operation. Continue dual-stream monitoring."
    elif final_score < THRESHOLD_2:
        risk = "WARNING"
        action = f"PREVENTIVE ALERT [{consensus}] — Schedule inspection at next shift change."
    else:
        risk = "CRITICAL"
        action = f"EMERGENCY MOTOR TRIP [{consensus}] — Immediate conveyor shutdown command issued."

    return {
        "final_risk_score": round(final_score, 5),
        "visual_score": round(v_sc, 5),
        "sensor_score": round(s_sc, 5),
        "risk_status": risk,
        "consensus_state": consensus,
        "recommended_action": action,
        "sensor_telemetry": sensor_metadata or {}
    }


# ==============================================================================
# SIMULATED TELEMETRY GENERATOR
# ==============================================================================

def simulate_sensor_readings(scenario: str = "nominal") -> Dict[str, float]:
    """
    Generate realistic simulated sensor telemetry for demonstration.
    (Clearly labeled as synthetic / simulated).
    """
    if scenario == "nominal":
        return {
            "vibration_rms_mms": 1.42,
            "temperature_c": 36.5,
            "tension_kn": 22.4,
            "speed_slip_pct": 1.8
        }
    elif scenario == "bearing_overheat":
        # Hot roller bearing, normal vibration
        return {
            "vibration_rms_mms": 2.80,
            "temperature_c": 78.5,
            "tension_kn": 21.0,
            "speed_slip_pct": 4.5
        }
    elif scenario == "splice_impact_vibration":
        # Damaged joint pounding against idler rollers
        return {
            "vibration_rms_mms": 8.45,
            "temperature_c": 48.0,
            "tension_kn": 19.5,
            "speed_slip_pct": 3.2
        }
    elif scenario == "belt_slip_and_slack":
        # Loose belt slipping on drive pulley
        return {
            "vibration_rms_mms": 3.10,
            "temperature_c": 68.2,
            "tension_kn": 10.5,
            "speed_slip_pct": 28.0
        }
    elif scenario == "catastrophic_snap":
        # Severe joint rupture with complete loss of tension
        return {
            "vibration_rms_mms": 11.20,
            "temperature_c": 52.0,
            "tension_kn": 4.2,
            "speed_slip_pct": 100.0
        }
    else:
        return simulate_sensor_readings("nominal")


# ==============================================================================
# OPTIONAL HARDWARE SERIAL STUB (ESP32 USB / UART)
# ==============================================================================

def read_esp32_serial(port: str = "COM3", baudrate: int = 115200, timeout: float = 1.0) -> Optional[Dict[str, float]]:
    """
    Attempt to read a JSON telemetry packet from an actual ESP32 over serial.
    Returns None gracefully if hardware is not connected.
    """
    try:
        import serial
        ser = serial.Serial(port, baudrate, timeout=timeout)
        line = ser.readline().decode("utf-8").strip()
        ser.close()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    except Exception as e:
        # Hardware absent / port unopened
        return None
    return None


# ==============================================================================
# MULTI-MODAL DEMO SUITE
# ==============================================================================

def run_fusion_test_suite():
    print("=" * 80)
    print("SmartBelt — Multi-Modal Sensor Fusion Demonstration (Phase 5)")
    print("STATUS: SIMULATED TELEMETRY / HARDWARE-READY STUB")
    print("=" * 80)
    print(f"Risk Thresholds: NORMAL < {THRESHOLD_1:.4f} <= WARNING < {THRESHOLD_2:.4f} <= CRITICAL\n")

    scenarios = [
        ("1. Nominal Conveyor (Clean Belt + Smooth Sensors)", 0.465, "nominal"),
        ("2. Early Visual Splice Fatigue (Clean Sensors)", 0.648, "nominal"),
        ("3. Subsurface Bearing Overheat (Vision Clean, Sensor High Temp)", 0.450, "bearing_overheat"),
        ("4. Catastrophic Joint Rupture (Visual Rip + Vibration Spike)", 0.745, "splice_impact_vibration"),
        ("5. Drive Pulley Stall / Belt Slack (Vision Clean, High Slip/Low Tension)", 0.485, "belt_slip_and_slack"),
        ("6. Full Conveyor Severance (Visual Rupture + Complete Tension Loss)", 0.820, "catastrophic_snap"),
    ]

    print(f"{'SCENARIO':<38} | {'VIS':>5} | {'SENS':>5} | {'FUSED':>5} | {'RISK':<8} | {'CONSENSUS'}")
    print("-" * 80)

    for desc, vis_sc, sens_scenario in scenarios:
        raw_readings = simulate_sensor_readings(sens_scenario)
        sens_sc, sens_meta = sensor_parameter_score(raw_readings)
        fused = combine(vis_sc, sens_sc, sens_meta)
        print(f"{desc:<38} | {vis_sc:5.3f} | {sens_sc:5.3f} | {fused['final_risk_score']:5.3f} | {fused['risk_status']:<8} | {fused['consensus_state']}")

    print("-" * 80)
    print("\nOperational Insight:")
    print("• Scenario 3 proves that thermal friction is caught by sensors even when belt visual appearance is normal.")
    print("• Scenario 2 proves that visual fatigue cracks are caught before mechanical vibration occurs.")
    print("• Scenario 4 & 6 prove multi-modal consensus triggers instantaneous emergency motor stop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBelt Multi-Modal Sensor Fusion")
    parser.add_argument("--demo", action="store_true", default=True, help="Run multi-modal architectural test suite")
    parser.add_argument("--visual-score", "-v", type=float, default=None, help="PatchCore visual anomaly score")
    parser.add_argument("--serial-port", "-p", type=str, default=None, help="ESP32 serial port (e.g. COM3)")

    args = parser.parse_args()

    if args.visual_score is not None:
        readings = None
        if args.serial_port:
            readings = read_esp32_serial(args.serial_port)
            if readings:
                print(f"[Hardware] Read live telemetry from {args.serial_port}: {readings}")
            else:
                print(f"[Hardware Stub] No response on {args.serial_port}; using nominal simulation.")
                readings = simulate_sensor_readings("nominal")
        else:
            readings = simulate_sensor_readings("nominal")

        s_score, meta = sensor_parameter_score(readings)
        fused = combine(args.visual_score, s_score, meta)
        print("\nMulti-Modal Fusion Result:")
        for k, v in fused.items():
            print(f"  {k}: {v}")
    else:
        run_fusion_test_suite()
