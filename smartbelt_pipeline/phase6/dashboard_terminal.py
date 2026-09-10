"""
SmartBelt — Phase 6: Live Terminal Dashboard (SCADA Industrial UI)

Renders an updating ANSI terminal dashboard displaying:
  - Belt Status & Joint Status
  - Visual Anomaly Score (PatchCore)
  - Multi-Sensor Telemetry (Vibration, Temperature, Tension, Slip)
  - Fused Risk Score & Industrial Gauge
  - Camera Feed & Frame Viewport
  - Alert Subsystem & Trip Relay State

Run with:
    python dashboard_terminal.py
"""

import os
import sys
import time
import math
import random
from pathlib import Path

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from smartbelt_pipeline.phase2.risk_classification import THRESHOLD_1, THRESHOLD_2
from smartbelt_pipeline.phase5.sensor_fusion import sensor_parameter_score, combine, simulate_sensor_readings
from smartbelt_pipeline.phase6.alert_system import alert_trigger

# ANSI Terminal Colors
RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"
BG_RED  = "\033[41m"
BG_WARN = "\033[43m"
BG_NORM = "\033[42m"


def make_gauge_bar(score: float, width: int = 30) -> str:
    """Generate a colorized ASCII gauge meter for risk score."""
    filled = int(round(score * width))
    bar = ""
    for i in range(width):
        frac = i / width
        if i < filled:
            if frac < (THRESHOLD_1):
                bar += f"{GREEN}#{RESET}"
            elif frac < (THRESHOLD_2):
                bar += f"{YELLOW}#{RESET}"
            else:
                bar += f"{RED}#{RESET}"
        else:
            bar += f"{GRAY}-{RESET}"
    return bar


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def run_terminal_dashboard(duration_sec: int = 30, scenario: str = "cycle"):
    """
    Run an interactive terminal dashboard.
    Simulates real-time telemetry updates and video feed stream.
    """
    scenarios = ["nominal", "nominal", "nominal", "early_fatigue", "bearing_hot", "rupture", "nominal"]
    frame_idx = 100
    t_start = time.time()
    idx = 0

    print("Starting SmartBelt Terminal Dashboard (Press Ctrl+C to exit)...")
    time.sleep(1)

    try:
        while True:
            elapsed = time.time() - t_start
            if duration_sec and elapsed > duration_sec:
                break

            current_scenario = scenarios[idx % len(scenarios)]
            idx += 1
            frame_idx += 5

            # Telemetry logic based on scenario
            if current_scenario == "nominal":
                v_score = 0.450 + random.uniform(-0.03, 0.04)
                sens_readings = simulate_sensor_readings("nominal")
                joint_status = "NORMAL (Splice Intact)"
                belt_status = "NORMAL (Surface Clean)"
            elif current_scenario == "early_fatigue":
                v_score = 0.655 + random.uniform(-0.01, 0.02)
                sens_readings = simulate_sensor_readings("nominal")
                joint_status = "WARNING (Splice Cord Fatigue)"
                belt_status = "NORMAL (Surface Clean)"
            elif current_scenario == "bearing_hot":
                v_score = 0.480 + random.uniform(-0.02, 0.02)
                sens_readings = simulate_sensor_readings("bearing_overheat")
                joint_status = "NORMAL (Splice Intact)"
                belt_status = "WARNING (Idler Friction Heat)"
            else:  # rupture
                v_score = 0.748 + random.uniform(-0.01, 0.04)
                sens_readings = simulate_sensor_readings("splice_impact_vibration")
                joint_status = "CRITICAL (JOINT RUPTURE DETECTED)"
                belt_status = "NORMAL"

            # Compute scores and alerts
            s_score, s_meta = sensor_parameter_score(sens_readings)
            fused = combine(v_score, s_score, s_meta)
            risk = fused["risk_status"]
            alert_info = alert_trigger(fused["final_risk_score"])

            # Color coding
            if risk == "CRITICAL":
                badge = f"{BG_RED}{WHITE}{BOLD} CRITICAL RUPTURE {RESET}"
                banner_color = RED
            elif risk == "WARNING":
                badge = f"{BG_WARN}{WHITE}{BOLD} WARNING ALERT {RESET}"
                banner_color = YELLOW
            else:
                badge = f"{BG_NORM}{WHITE}{BOLD} NOMINAL RUN {RESET}"
                banner_color = GREEN

            clear_screen()
            print(f"{banner_color}+==============================================================================+{RESET}")
            print(f"{banner_color}|           SmartBelt(TM) - INDUSTRIAL MONITORING & SCADA DASHBOARD            |{RESET}")
            print(f"{banner_color}|   Conveyor Belt Damage & Joint Rupture Prevention System | v2.0-Live         |{RESET}")
            print(f"{banner_color}+==============================================================================+{RESET}")
            print(f" Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  Elapsed: {elapsed:5.1f}s  |  Overall State: {badge}\n")

            # PANEL 1: VISUAL & STRUCTURAL INTEGRITY
            print(f"{BOLD}[1] VISUAL INSPECTION SUBSYSTEM (PatchCore WideResNet-50){RESET}")
            print(f"  * Camera Feed:     [LIVE] Camera 01 (Joint Overhead Optical)  Frame: #{frame_idx:05d}")
            print(f"  * Joint Status:    {banner_color}{joint_status}{RESET}")
            print(f"  * Belt Surface:    {belt_status}")
            print(f"  * Anomaly Score:   {v_score:.5f}  (Thresholds: Warn {THRESHOLD_1:.4f} | Crit {THRESHOLD_2:.4f})\n")

            # PANEL 2: MULTI-MODAL PHYSICAL TELEMETRY
            print(f"{BOLD}[2] SENSOR TELEMETRY SUBSYSTEM (ESP32 Serial Stream / Calibrated ISO-10816){RESET}")
            vib = sens_readings['vibration_rms_mms']
            temp = sens_readings['temperature_c']
            tens = sens_readings['tension_kn']
            slip = sens_readings['speed_slip_pct']
            vib_color = RED if vib >= 7.0 else (YELLOW if vib >= 4.5 else GREEN)
            temp_color = RED if temp >= 80.0 else (YELLOW if temp >= 65.0 else GREEN)
            print(f"  * Vibration:       {vib_color}{vib:5.2f} mm/s RMS{RESET}  (ISO Alert > 4.5 mm/s | Trip > 7.0 mm/s)")
            print(f"  * Temperature:     {temp_color}{temp:5.1f} deg C{RESET}        (Friction Alert > 65 C | Fire Risk > 80 C)")
            print(f"  * Belt Tension:    {tens:5.1f} kN          (Nominal span: 18 - 26 kN)")
            print(f"  * Pulley Slip:     {slip:5.1f} %           (Nominal creep: < 5%)")
            print(f"  * Composite Score: {s_score:.5f}\n")

            # PANEL 3: FUSED RISK & INDUSTRIAL GAUGES
            print(f"{BOLD}[3] FUSED MULTI-MODAL RISK ENGINE{RESET}")
            gauge_str = make_gauge_bar(fused["final_risk_score"], width=36)
            print(f"  * Risk Metric:     [{gauge_str}] {fused['final_risk_score']:.5f}")
            print(f"  * Risk Class:      {badge}")
            print(f"  * Consensus Mode:  {fused['consensus_state']}\n")

            # PANEL 4: EMERGENCY ALERT & TRIP STATE
            print(f"{BOLD}[4] AUTOMATED SAFETY RESPONSE & SCADA CONTROL{RESET}")
            print(f"  * Buzzer Siren:    {banner_color}{alert_info['buzzer_state']}{RESET}")
            print(f"  * Motor Trip Relay:{banner_color}{alert_info['motor_relay_state']}{RESET}")
            print(f"  * Plant Beacon:    {alert_info['beacon_color']}")
            print(f"  * Safety Action:   {alert_info['action_description']}")
            print(f"{GRAY}------------------------------------------------------------------------------{RESET}")
            print(f" (Cycle demo streaming live telemetry... Ctrl+C to halt)")

            time.sleep(1.8)

    except KeyboardInterrupt:
        print("\n[Dashboard] Operator stopped monitor stream.")


if __name__ == "__main__":
    run_terminal_dashboard(duration_sec=15)
