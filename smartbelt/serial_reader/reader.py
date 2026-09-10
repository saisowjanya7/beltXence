"""
SmartBelt v2 — ESP32 Serial Reader Thread
Auto-detects the ESP32 COM port, reads JSON packets at 5 Hz,
and exposes the latest parsed SensorPacket via a thread-safe property.

Usage:
    from smartbelt.serial_reader.reader import SerialReader
    reader = SerialReader()            # auto-detects port
    reader.start()
    pkt = reader.latest_packet        # SensorPacket dataclass or None
    reader.stop()

Manual port override:
    reader = SerialReader(port='COM5')
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# USB-Serial chip descriptions found on ESP32 development boards
_ESP32_USB_DESCRIPTORS = [
    "CH340",
    "CP2102",
    "CP210X",
    "USB-SERIAL",
    "SILICON LABS",
    "WCH.CN",
    "USB SERIAL",
]


@dataclass
class SensorPacket:
    """Parsed sensor reading from the ESP32 JSON packet."""
    timestamp_ms: int = 0          # ESP32 millis()
    received_at: float = 0.0      # time.monotonic() on host machine
    temp_c: float = 0.0           # MLX90614 object temperature (°C)
    vib_x: float = 0.0            # MPU-6050 accel X (g)
    vib_y: float = 0.0            # MPU-6050 accel Y (g)
    vib_z: float = 0.0            # MPU-6050 accel Z (g, gravity subtracted)
    vib_rms: float = 0.0          # combined vibration RMS (g)
    load_kg: float = 0.0          # HX711 load cell reading (kg)
    ir_state: int = 1             # E18-D80NK: 0=object, 1=clear
    belt_speed_mps: float = 0.0   # derived from IR pulse count (m/s)
    parse_error: bool = False     # True if this packet had a JSON decode error


class SerialReader:
    """
    Reads JSON packets from the ESP32 over USB serial in a daemon thread.
    The latest successfully parsed packet is always available via `latest_packet`.
    Automatically reconnects if the serial port disconnects.
    """

    WATCHDOG_TIMEOUT_S = 5.0   # Warn if no packet received for this long
    RECONNECT_DELAY_S  = 2.0   # Wait between reconnect attempts

    def __init__(self, port: Optional[str] = None, baud: int = 115200) -> None:
        self._port = port       # None = auto-detect
        self._baud = baud
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[SensorPacket] = None

        # Diagnostics
        self.packets_received: int = 0
        self.parse_errors: int = 0
        self.last_packet_ts: float = 0.0

    def start(self) -> None:
        """Start the serial reader background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="SerialReader", daemon=True)
        self._thread.start()
        logger.info("SerialReader thread started.")

    def stop(self) -> None:
        """Stop the background thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._close_serial()
        logger.info("SerialReader stopped.")

    @property
    def latest_packet(self) -> Optional[SensorPacket]:
        """Thread-safe access to the most recently parsed sensor packet."""
        with self._lock:
            return self._latest

    @property
    def is_healthy(self) -> bool:
        """True if a packet was received within the watchdog window."""
        if self._latest is None:
            return False
        return (time.monotonic() - self.last_packet_ts) < self.WATCHDOG_TIMEOUT_S

    @staticmethod
    def auto_detect_port() -> Optional[str]:
        """Scan COM ports and return the first one matching an ESP32 USB descriptor."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").upper()
            mfg = (p.manufacturer or "").upper()
            combined = f"{desc} {mfg}"
            for keyword in _ESP32_USB_DESCRIPTORS:
                if keyword in combined:
                    logger.info(f"Auto-detected ESP32 on {p.device} ({p.description})")
                    return p.device
        if ports:
            logger.info(f"No explicit ESP32 match found; available COM ports: {[p.device for p in ports]}")
        return None

    def _open_serial(self) -> bool:
        """Attempt to open the serial port. Returns True on success."""
        port = self._port or self.auto_detect_port()
        if port is None:
            return False
        try:
            self._serial = serial.Serial(port, self._baud, timeout=2.0)
            logger.info(f"Serial port opened: {port} @ {self._baud} baud")
            return True
        except (serial.SerialException, OSError) as e:
            logger.debug(f"Failed to open serial port {port}: {e}")
            return False

    def _close_serial(self) -> None:
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    def _read_loop(self) -> None:
        """Main loop: open port, read lines, parse JSON, store latest packet."""
        while self._running:
            if self._serial is None or not self._serial.is_open:
                if not self._open_serial():
                    time.sleep(self.RECONNECT_DELAY_S)
                    continue

            try:
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                pkt = self._parse_line(line)
                with self._lock:
                    self._latest = pkt
                self.last_packet_ts = time.monotonic()
                if not pkt.parse_error:
                    self.packets_received += 1
                else:
                    self.parse_errors += 1

            except (serial.SerialException, OSError) as e:
                logger.error(f"Serial read error: {e}. Attempting reconnect...")
                self._close_serial()
                time.sleep(self.RECONNECT_DELAY_S)

    @staticmethod
    def _parse_line(line: str) -> SensorPacket:
        """Parse a single JSON line into a SensorPacket."""
        pkt = SensorPacket(received_at=time.monotonic())
        try:
            data = json.loads(line)
            pkt.timestamp_ms    = int(data.get("ts", 0))
            pkt.temp_c          = float(data.get("temp_c", 0.0))
            pkt.vib_x           = float(data.get("vib_x", 0.0))
            pkt.vib_y           = float(data.get("vib_y", 0.0))
            pkt.vib_z           = float(data.get("vib_z", 0.0))
            pkt.vib_rms         = float(data.get("vib_rms", 0.0))
            pkt.load_kg         = float(data.get("load_kg", 0.0))
            pkt.ir_state        = int(data.get("ir_state", 1))
            pkt.belt_speed_mps  = float(data.get("belt_speed_mps", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError):
            pkt.parse_error = True
        return pkt
