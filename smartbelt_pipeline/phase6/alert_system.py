"""
SmartBelt — Phase 6: Industrial Alert & Trip Subsystem
Manages automated plant responses, emergency buzzer signals, motor trip relays,
and telemetry logging based on final multi-modal risk scores.

=============================================================================
STATUS: HARDWARE-READY INTERFACE WITH SIMULATED IO FALLBACK
If physical ESP32 or relay hardware is connected via USB/Serial, commands are
transmitted directly to microcontroller GPIO pins. Otherwise, simulated hardware
calls are logged and printed with clear industrial tagging.
=============================================================================
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ALERT_LOG_PATH = SCRIPT_DIR / "alert_events.csv"

# Risk Thresholds from Phase 2
THRESHOLD_1 = 0.62710  # NORMAL -> WARNING
THRESHOLD_2 = 0.68887  # WARNING -> CRITICAL


class AlertSubsystem:
    def __init__(self, serial_port: Optional[str] = None, baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_conn = None
        self.hardware_active = False

        # Attempt to connect to ESP32 / Industrial Relay if port specified
        if self.serial_port:
            try:
                import serial
                self.serial_conn = serial.Serial(self.serial_port, self.baudrate, timeout=0.5)
                self.hardware_active = True
                print(f"[Alert System] Connected to physical hardware on {self.serial_port}")
            except Exception as e:
                print(f"[Alert System] Hardware port {self.serial_port} unavailable ({e}). Using SIMULATED mode.")
                self.hardware_active = False
        else:
            self.hardware_active = False

        # Initialize CSV log file with headers if it doesn't exist
        if not ALERT_LOG_PATH.exists():
            with open(ALERT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "final_risk_score", "risk_status",
                    "buzzer_state", "motor_relay_state", "beacon_color",
                    "hardware_mode", "details"
                ])

    def dispatch_hardware_signal(self, command: str) -> str:
        """Send command to physical microcontroller or return simulated response."""
        if self.hardware_active and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode("utf-8"))
                return f"[PHYSICAL ESP32] Sent: {command}"
            except Exception as e:
                return f"[PHYSICAL ESP32 ERROR] {e}"
        else:
            # Simulated GPIO / Relay Call
            return f"[SIMULATED HARDWARE GPIO] Triggered: {command}"

    def alert_trigger(
        self,
        final_risk_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate final risk score and trigger industrial alarm/safety response.

        Args:
            final_risk_score (float): Fused multi-modal risk score [0.0 - 1.0].
            context (dict, optional): Auxiliary telemetry (visual score, sensor readings).

        Returns:
            dict: Structured alert status and physical hardware actions.
        """
        ts_str = datetime.now().isoformat()
        score = float(final_risk_score)
        ctx = context or {}

        if score < THRESHOLD_1:
            # State 1: NORMAL
            status = "NORMAL"
            buzzer = "OFF"
            motor_relay = "CLOSED (RUNNING)"
            beacon = "GREEN"
            hardware_cmd = "CMD_STATUS_NORMAL"
            action_desc = "Conveyor operates at full speed. All systems nominal."

        elif score < THRESHOLD_2:
            # State 2: WARNING
            status = "WARNING"
            buzzer = "PULSE_CHIRP (1 Hz)"
            motor_relay = "CLOSED (MAINTAIN MOTION)"
            beacon = "AMBER"
            hardware_cmd = "CMD_ALERT_WARNING"
            action_desc = "Preventive inspection flagged. Maintenance team notified via SCADA."

        else:
            # State 3: CRITICAL
            status = "CRITICAL"
            buzzer = "ON (CONTINUOUS 95dB)"
            motor_relay = "OPEN (MOTOR TRIPPED / EMERGENCY STOP)"
            beacon = "FLASHING_RED"
            hardware_cmd = "CMD_EMERGENCY_HALT_BUZZER_ON"
            action_desc = "CRITICAL JOINT RUPTURE HAZARD! Conveyor motor tripped immediately."

        # Dispatch command to hardware (real or simulated)
        hw_log = self.dispatch_hardware_signal(hardware_cmd)

        # Print prominent industrial console log
        if status == "CRITICAL":
            print(f"\n{'!' * 75}")
            print(f"!!! [EMERGENCY SAFETY ACTION] RISK SCORE: {score:.5f} (CRITICAL) !!!")
            print(f"!!! HARDWARE OUTPUT: BUZZER ON | MOTOR TRIP RELAY OPEN          !!!")
            print(f"!!! {hw_log} !!!")
            print(f"{'!' * 75}\n")
        elif status == "WARNING":
            print(f"[*] [MAINTENANCE ALERT] Score: {score:.5f} (WARNING) | BEACON: AMBER | {hw_log}")

        # Append to persistent audit log
        with open(ALERT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_str, round(score, 5), status, buzzer, motor_relay, beacon,
                "PHYSICAL" if self.hardware_active else "SIMULATED",
                action_desc
            ])

        return {
            "timestamp": ts_str,
            "final_risk_score": round(score, 5),
            "risk_status": status,
            "buzzer_state": buzzer,
            "motor_relay_state": motor_relay,
            "beacon_color": beacon,
            "hardware_mode": "PHYSICAL" if self.hardware_active else "SIMULATED",
            "hardware_dispatch_log": hw_log,
            "action_description": action_desc,
            "context": ctx
        }


# Global singleton instance for easy import
default_alert_system = AlertSubsystem()

def alert_trigger(final_risk_score: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Top-level convenience function."""
    return default_alert_system.alert_trigger(final_risk_score, context)


def run_alert_demo():
    """Test and demonstrate all 3 alert states."""
    print("=" * 75)
    print("SmartBelt — Alert Logic & Trip Response Verification")
    print("=" * 75)
    print(f"Thresholds: NORMAL < {THRESHOLD_1:.4f} <= WARNING < {THRESHOLD_2:.4f} <= CRITICAL\n")

    test_cases = [
        (0.485, "Nominal Joint Run"),
        (0.652, "Elevated Splice Vibration (Pre-Rupture Fatigue)"),
        (0.745, "Severe Joint Tear / Cord Pullout"),
    ]

    for sc, label in test_cases:
        print(f"\n--- Testing Scenario: {label} (Score: {sc:.4f}) ---")
        res = alert_trigger(sc, context={"test_scenario": label})
        print(f"  Status:          {res['risk_status']}")
        print(f"  Buzzer:          {res['buzzer_state']}")
        print(f"  Motor Relay:     {res['motor_relay_state']}")
        print(f"  Beacon Color:    {res['beacon_color']}")
        print(f"  Hardware Signal: {res['hardware_dispatch_log']}")
        print(f"  Action Taken:    {res['action_description']}")

    print(f"\nAll alert events appended to: {ALERT_LOG_PATH.resolve()}")


if __name__ == "__main__":
    run_alert_demo()
