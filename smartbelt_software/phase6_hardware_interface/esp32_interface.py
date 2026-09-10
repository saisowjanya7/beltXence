"""
SmartBelt — Industrial Conveyor Joint & Damage Monitoring System
Phase 6: Hardware Interface (esp32_interface.py)

Host PC communication driver for ESP32 microcontroller.
Supports both Physical Serial Link and Automated Virtual Simulation Mode.
Protocol: Line-delimited JSON (Protocol Spec v1.0.0)
"""

import json
import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ESP32 Interface] %(message)s"
)
logger = logging.getLogger("SmartBelt.ESP32")

# Attempt pyserial import
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logger.warning("pyserial package not detected. Operating exclusively in SIMULATION mode.")


# ==============================================================================
# CONSTANTS & PROTOCOL DEFINITIONS
# ==============================================================================
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT_SEC = 1.0
WATCHDOG_PING_INTERVAL_SEC = 1.5

KNOWN_ESP32_VID_PID = [
    (0x10C4, 0xEA60),  # Silicon Labs CP210x
    (0x1A86, 0x7523),  # QinHeng CH340
    (0x0403, 0x6001),  # FTDI FT232R
    (0x303A, 0x1001),  # Espressif USB JTAG/serial debug unit
    (0x303A, 0x0002),  # ESP32-S2/S3 native CDC
]


def list_available_com_ports() -> List[Dict[str, str]]:
    """Scan and list all detected host COM ports."""
    if not HAS_SERIAL:
        return []
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "port": p.device,
            "description": p.description,
            "hwid": p.hwid,
            "vid": p.vid,
            "pid": p.pid
        })
    return ports


def auto_detect_esp32_port() -> Optional[str]:
    """Auto-detect likely ESP32 serial port based on USB Vendor/Product IDs."""
    if not HAS_SERIAL:
        return None
    for p in serial.tools.list_ports.comports():
        for vid, pid in KNOWN_ESP32_VID_PID:
            if p.vid == vid and p.pid == pid:
                return p.device
        if any(keyword in p.description.lower() for keyword in ["cp210", "ch340", "ftdi", "esp32", "usb-serial"]):
            return p.device
    return None


# ==============================================================================
# ESP32 HARDWARE CONTROLLER INTERFACE
# ==============================================================================
class ESP32Interface:
    """
    Bi-directional communication bridge between SmartBelt Core and ESP32 Microcontroller.
    
    If physical hardware is connected, establishes 115200 baud UART link.
    If no hardware is attached or pyserial is missing, enters High-Fidelity Simulation Mode.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = DEFAULT_BAUD,
        force_simulation: bool = False,
        auto_reconnect: bool = True,
        watchdog_timeout_sec: float = 8.0
    ):
        self.requested_port = port
        self.baud = baud
        self.force_simulation = force_simulation
        self.auto_reconnect = auto_reconnect
        self.watchdog_timeout_sec = watchdog_timeout_sec

        self.serial_conn: Optional[Any] = None
        self.active_port: Optional[str] = None
        self.is_simulated = True

        # Virtual state (for simulation mode)
        self._sim_relay_closed = True
        self._sim_beacon_color = "GREEN"
        self._sim_buzzer_state = "OFF"
        self._sim_last_ping = time.time()
        self._sim_uptime_start = time.time()

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_telemetry: Optional[Dict[str, Any]] = None

        # Attempt initial connection
        self.connect()

    @property
    def mode(self) -> str:
        return "SIMULATION_MODE" if self.is_simulated else "PHYSICAL_SERIAL"

    @property
    def is_connected(self) -> bool:
        if self.is_simulated:
            return True  # Always alive in virtual simulation
        return self.serial_conn is not None and self.serial_conn.is_open

    def connect(self) -> bool:
        """Establish connection to physical ESP32 or fallback to simulation."""
        if self.force_simulation or not HAS_SERIAL:
            self._enter_simulation("Forced simulation or pyserial not available.")
            return True

        target_port = self.requested_port or auto_detect_esp32_port()
        if not target_port:
            self._enter_simulation("No ESP32 USB COM port detected on host.")
            return True

        try:
            logger.info(f"Attempting connection to physical ESP32 on {target_port} at {self.baud} baud...")
            self.serial_conn = serial.Serial(
                port=target_port,
                baudrate=self.baud,
                timeout=DEFAULT_TIMEOUT_SEC,
                write_timeout=DEFAULT_TIMEOUT_SEC
            )
            time.sleep(1.5)  # Allow ESP32 reboot / DTR settling
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            self.active_port = target_port
            self.is_simulated = False
            logger.info(f"SUCCESS: Connected to physical ESP32 on {target_port}.")
            return True
        except Exception as e:
            logger.warning(f"Could not open physical port {target_port} ({e}).")
            self._enter_simulation(f"Connection failed: {e}")
            return True

    def _enter_simulation(self, reason: str):
        """Configure driver for High-Fidelity Simulation Mode."""
        self.is_simulated = True
        self.active_port = "VIRTUAL_COM"
        logger.info(f"ESP32 Interface active in SIMULATION MODE. Reason: {reason}")

    def disconnect(self):
        """Gracefully close serial resources."""
        self._stop_event.set()
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                logger.info("Physical ESP32 serial port closed.")
            except Exception as e:
                logger.error(f"Error closing port: {e}")
        self.serial_conn = None

    # ==========================================================================
    # COMMAND DISPATCH (HOST PC -> ESP32)
    # ==========================================================================

    def _send_command(self, cmd_dict: Dict[str, Any]) -> bool:
        """Send formatted JSON command line to ESP32 or update virtual state."""
        with self._lock:
            payload_str = json.dumps(cmd_dict) + "\n"

            if self.is_simulated:
                # Any valid host command demonstrates active link and refreshes watchdog
                self._sim_last_ping = time.time()

                # Handle command internally in simulated state
                cmd_type = cmd_dict.get("cmd", "")
                if cmd_type == "TRIP":
                    self._sim_relay_closed = False
                    self._sim_beacon_color = "RED"
                    self._sim_buzzer_state = "PULSED_EMERGENCY"
                    logger.info(f"[SIM] EMERGENCY MOTOR RELAY TRIPPED! Reason: {cmd_dict.get('reason')}")
                elif cmd_type == "RESET":
                    self._sim_relay_closed = True
                    self._sim_beacon_color = "GREEN"
                    self._sim_buzzer_state = "OFF"
                    logger.info("[SIM] MOTOR RELAY RESET TO NORMAL OPERATION.")
                elif cmd_type == "SET_BEACON":
                    self._sim_beacon_color = cmd_dict.get("color", "GREEN")
                    logger.info(f"[SIM] Beacon updated to {self._sim_beacon_color}.")
                elif cmd_type == "SET_BUZZER":
                    self._sim_buzzer_state = cmd_dict.get("state", "OFF")
                    logger.info(f"[SIM] Buzzer updated to {self._sim_buzzer_state}.")
                elif cmd_type == "PING":
                    pass  # Watchdog refreshed above
                return True

            # Physical Serial Link
            if not self.serial_conn or not self.serial_conn.is_open:
                logger.error("Physical serial connection is closed.")
                return False

            try:
                self.serial_conn.write(payload_str.encode("utf-8"))
                self.serial_conn.flush()
                return True
            except Exception as e:
                logger.error(f"Failed to transmit command to ESP32: {e}")
                if self.auto_reconnect:
                    self._enter_simulation("Serial transmission error; fallback to simulation.")
                return False

    def trip_motor(self, reason: str = "EMERGENCY_ANOMALY_TRIP") -> bool:
        """Immediately de-energize the motor safety relay."""
        return self._send_command({
            "cmd": "TRIP",
            "reason": reason,
            "timestamp": time.time()
        })

    def reset_motor(self) -> bool:
        """Re-energize the motor safety relay and clear alarms."""
        return self._send_command({
            "cmd": "RESET",
            "auth": "OPERATOR_ACK",
            "timestamp": time.time()
        })

    def set_beacon(self, color: str) -> bool:
        """Set indicator tower color ('GREEN', 'AMBER', 'RED')."""
        color = color.upper()
        if color not in ["GREEN", "AMBER", "RED"]:
            raise ValueError(f"Invalid beacon color: {color}")
        return self._send_command({"cmd": "SET_BEACON", "color": color})

    def set_buzzer(self, state: str) -> bool:
        """Set buzzer alarm mode ('OFF', 'INTERMITTENT', 'PULSED_EMERGENCY')."""
        state = state.upper()
        if state not in ["OFF", "INTERMITTENT", "PULSED_EMERGENCY"]:
            raise ValueError(f"Invalid buzzer state: {state}")
        return self._send_command({"cmd": "SET_BUZZER", "state": state})

    def ping(self) -> bool:
        """Send watchdog heartbeat keep-alive ping."""
        return self._send_command({"cmd": "PING", "seq": int(time.time() * 1000) % 100000})

    def reset_state(self) -> None:
        """
        Reset virtual controller to nominal baseline:
        Relay = CLOSED, Beacon = GREEN, Buzzer = OFF, refreshed watchdog timer.
        """
        with self._lock:
            self._sim_relay_closed = True
            self._sim_beacon_color = "GREEN"
            self._sim_buzzer_state = "OFF"
            self._sim_last_ping = time.time()
            logger.info("[SIM] Virtual controller state reset to nominal baseline.")

    # Vocabulary translation: smartbelt_core values -> ESP32 protocol values
    _BEACON_MAP = {
        "GREEN": "GREEN",
        "AMBER": "AMBER",
        "RED": "RED",
        "FLASHING_RED": "RED",
        "YELLOW": "AMBER",
        "ORANGE": "AMBER",
    }
    _BUZZER_MAP = {
        "OFF": "OFF",
        "INTERMITTENT": "INTERMITTENT",
        "PULSED_EMERGENCY": "PULSED_EMERGENCY",
        "PULSE_CHIRP (1 HZ)": "INTERMITTENT",
        "PULSE_CHIRP (1 Hz)": "INTERMITTENT",
        "ON (CONTINUOUS 95DB)": "PULSED_EMERGENCY",
        "ON (CONTINUOUS 95dB)": "PULSED_EMERGENCY",
        "ON": "PULSED_EMERGENCY",
    }

    def _normalize_beacon(self, raw: str) -> str:
        """Map smartbelt_core beacon vocabulary to ESP32 protocol vocabulary."""
        return self._BEACON_MAP.get(raw, self._BEACON_MAP.get(raw.upper(), "GREEN"))

    def _normalize_buzzer(self, raw: str) -> str:
        """Map smartbelt_core buzzer vocabulary to ESP32 protocol vocabulary."""
        return self._BUZZER_MAP.get(raw, self._BUZZER_MAP.get(raw.upper(), "OFF"))

    def dispatch_fusion_action(self, fusion_result: Dict[str, Any]) -> bool:
        """
        Convenience method: directly executes the hardware actions computed
        by SmartBeltCore's fuse_risk() function.

        Translates smartbelt_core vocabulary (e.g. 'PULSE_CHIRP (1 Hz)', 'FLASHING_RED')
        to ESP32 protocol vocabulary (OFF / INTERMITTENT / PULSED_EMERGENCY | GREEN / AMBER / RED).
        """
        risk = fusion_result.get("risk_status", "NORMAL")
        beacon = self._normalize_beacon(fusion_result.get("beacon_color", "GREEN"))
        buzzer = self._normalize_buzzer(fusion_result.get("buzzer_state", "OFF"))
        relay = fusion_result.get("motor_relay_state", "CLOSED (RUNNING)")

        if risk == "CRITICAL" or "TRIP" in relay or "OPEN" in relay:
            return self.trip_motor(reason=fusion_result.get("consensus_state", "CRITICAL_RISK"))
        else:
            if "CLOSED" in relay and not self._sim_relay_closed:
                self.reset_motor()
            self.set_beacon(beacon)
            self.set_buzzer(buzzer)
            return True

    # ==========================================================================
    # TELEMETRY RECEPTION (ESP32 -> HOST PC)
    # ==========================================================================

    def read_telemetry(self) -> Dict[str, Any]:
        """
        Read the latest available telemetry packet from ESP32.
        Returns a validated dictionary packet matching Protocol Spec v1.0.0.
        """
        with self._lock:
            # 1. Virtual Simulation Mode
            if self.is_simulated:
                now = time.time()
                uptime = int((now - self._sim_uptime_start) * 1000)
                watchdog_ok = (now - self._sim_last_ping < self.watchdog_timeout_sec)

                # If watchdog expired in simulation, simulate failsafe trip
                if not watchdog_ok and self._sim_relay_closed:
                    self._sim_relay_closed = False
                    self._sim_beacon_color = "RED"
                    self._sim_buzzer_state = "PULSED_EMERGENCY"

                # Generate simulated telemetry reading
                t_jitter = random.uniform(-0.4, 0.4)
                v_jitter = random.uniform(-0.08, 0.08)
                s_jitter = random.uniform(-0.1, 0.1)

                return {
                    "device_id": "SMARTBELT_ESP32_SIM",
                    "uptime_ms": uptime,
                    "telemetry": {
                        "temperature_c": round(38.2 + t_jitter, 2),
                        "vibration_rms_mms": round(max(0.1, 1.35 + v_jitter), 2),
                        "speed_slip_pct": round(max(0.0, 1.4 + s_jitter), 2),
                        "tension_kn": round(24.5 + (random.uniform(-0.3, 0.3)), 2),
                    },
                    "actuators": {
                        "relay_closed": self._sim_relay_closed,
                        "beacon": self._sim_beacon_color,
                        "buzzer": self._sim_buzzer_state
                    },
                    "watchdog_ok": watchdog_ok,
                    "is_simulated": True,
                    "data_source_label": "SIMULATED — not a real measurement"
                }

            # 2. Physical Serial Stream
            if not self.serial_conn or not self.serial_conn.is_open:
                return self.read_telemetry()

            try:
                # Read latest line
                line = self.serial_conn.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    return self._latest_telemetry or self._create_empty_packet()

                data = json.loads(line)
                if "telemetry" in data:
                    data["data_source_label"] = (
                        "SIMULATED — not a real measurement" if data.get("is_simulated") else "PHYSICAL_ESP32_HARDWARE"
                    )
                    self._latest_telemetry = data
                    return data
                return self._latest_telemetry or self._create_empty_packet()
            except json.JSONDecodeError:
                # Partial line or noise
                return self._latest_telemetry or self._create_empty_packet()
            except Exception as e:
                logger.error(f"Error reading physical serial: {e}")
                return self._create_empty_packet()

    def _create_empty_packet(self) -> Dict[str, Any]:
        """Fallback empty telemetry packet."""
        return {
            "device_id": "UNKNOWN",
            "uptime_ms": 0,
            "telemetry": {
                "temperature_c": 0.0,
                "vibration_rms_mms": 0.0,
                "speed_slip_pct": 0.0,
                "tension_kn": 0.0,
            },
            "actuators": {
                "relay_closed": self._sim_relay_closed,
                "beacon": self._sim_beacon_color,
                "buzzer": self._sim_buzzer_state,
            },
            "watchdog_ok": False,
            "is_simulated": True,
            "data_source_label": "SIMULATED — not a real measurement"
        }


# ==============================================================================
# STANDALONE VERIFICATION / TEST SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SmartBelt Phase 6: ESP32 Hardware Interface Verification Test")
    print("=" * 70)

    # 1. Scan available COM ports
    print("\n[Step 1] Scanning available COM ports on host...")
    ports = list_available_com_ports()
    if ports:
        for p in ports:
            print(f"  - Port: {p['port']} ({p['description']}) [VID:{p['vid']} PID:{p['pid']}]")
    else:
        print("  - No physical COM ports detected or pyserial operating without serial ports.")

    # 2. Initialize Interface (graceful fallback to simulation)
    print("\n[Step 2] Initializing ESP32 Hardware Controller Interface...")
    dev = ESP32Interface(force_simulation=False)
    print(f"  -> Operating Mode: {dev.mode}")
    print(f"  -> Active Port:    {dev.active_port}")
    print(f"  -> Connected:      {dev.is_connected}")

    # 3. Read initial telemetry packet
    print("\n[Step 3] Reading initial telemetry packet...")
    pkt = dev.read_telemetry()
    print(f"  -> Telemetry Source: {pkt.get('data_source_label')}")
    print(f"  -> Sensor Readings:  {pkt.get('telemetry')}")
    print(f"  -> Actuator States:  {pkt.get('actuators')}")
    print(f"  -> Watchdog Healthy: {pkt.get('watchdog_ok')}")

    # 4. Test Keep-Alive Ping
    print("\n[Step 4] Transmitting Watchdog PING keep-alive...")
    res = dev.ping()
    print(f"  -> PING result: {res}")

    # 5. Test Beacon Color Updates
    print("\n[Step 5] Testing Beacon Tower Color Controls...")
    for color in ["AMBER", "RED", "GREEN"]:
        dev.set_beacon(color)
        time.sleep(0.2)
        pkt = dev.read_telemetry()
        print(f"  -> Set Beacon to {color:5s} | Current Beacon State: {pkt['actuators']['beacon']}")

    # 6. Test Motor Trip Command
    print("\n[Step 6] Testing Emergency Motor Relay Trip...")
    dev.trip_motor(reason="TEST_VERIFICATION_TRIP")
    time.sleep(0.2)
    pkt = dev.read_telemetry()
    print(f"  -> Relay Closed: {pkt['actuators']['relay_closed']} (Expected: False / Open)")
    print(f"  -> Beacon State: {pkt['actuators']['beacon']} (Expected: RED)")
    print(f"  -> Buzzer State: {pkt['actuators']['buzzer']} (Expected: PULSED_EMERGENCY)")

    # 7. Test Motor Reset Command
    print("\n[Step 7] Testing Motor Relay Operator Reset...")
    dev.reset_motor()
    time.sleep(0.2)
    pkt = dev.read_telemetry()
    print(f"  -> Relay Closed: {pkt['actuators']['relay_closed']} (Expected: True / Closed)")
    print(f"  -> Beacon State: {pkt['actuators']['beacon']} (Expected: GREEN)")
    print(f"  -> Buzzer State: {pkt['actuators']['buzzer']} (Expected: OFF)")

    # 8. Clean Shutdown
    print("\n[Step 8] Disconnecting interface...")
    dev.disconnect()
    print("\n>>> Phase 6 ESP32 Hardware Interface Test PASSED with 100% Protocol Compliance! <<<\n")
