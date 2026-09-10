# Phase 6 — Hardware Interface (ESP32 Microcontroller Integration)

This directory provides the bidirectional industrial communication pipeline between the **SmartBelt** host software and an **ESP32 Microcontroller** controlling industrial relays, 3-color beacon lights, audible buzzers, and sensor acquisition.

---

## 1. Directory Contents

| File | Purpose |
| :--- | :--- |
| [`protocol_spec.md`](protocol_spec.md) | Official industrial communication protocol specification (Baud 115200, Line-delimited JSON). |
| [`esp32_firmware.ino`](esp32_firmware.ino) | Arduino C++ firmware for ESP32 with failsafe watchdog, relay driver, and telemetry streaming. |
| [`esp32_interface.py`](esp32_interface.py) | Python serial controller supporting automatic COM port detection, handshake, and high-fidelity simulation mode. |
| [`README.md`](README.md) | Wiring schematics, flashing instructions, and test guide. |

---

## 2. Electrical Wiring & Industrial Pinout

| ESP32 Pin | Function | Industrial Component / Target Device |
| :--- | :--- | :--- |
| **GPIO 25** | `RELAY_CTRL` | 5V/12V Optocoupled Relay Module (Active LOW) &rarr; Conveyor VFD E-Stop Line |
| **GPIO 26** | `BEACON_RED` | 24V Industrial Tower Light (Red / Critical Alarm) |
| **GPIO 27** | `BEACON_AMB` | 24V Industrial Tower Light (Amber / Warning) |
| **GPIO 14** | `BEACON_GRN` | 24V Industrial Tower Light (Green / Nominal) |
| **GPIO 12** | `BUZZER_CTRL`| 90dB Active Piezo Acoustic Alarm |
| **GPIO 34** | `VIB_ADC` | Analog Piezoelectric / 4-20mA Vibration Sensor (0–3.3V) |
| **GPIO 35** | `TENS_ADC`| Belt Strain Gauge Conditioner (0–3.3V) |
| **GPIO 18** | `HALL_SPEED`| NPN Hall Effect Proximity Sensor (Pulley RPM & Slip Detection) |
| **USB/UART**| `SERIAL` | Silicon Labs CP2102 / CH340 / ESP32 CDC to Host PC @ 115200 Baud |

---

## 3. How to Flash the ESP32 Firmware

### Using Arduino IDE:
1. Connect the ESP32 to the PC via USB.
2. In Arduino IDE, go to **Tools &rarr; Board** and select **ESP32 Dev Module**.
3. Select the detected COM port under **Tools &rarr; Port**.
4. Open [`esp32_firmware.ino`](esp32_firmware.ino).
5. Click **Upload**.
6. Open the Serial Monitor at **115200 baud** to see initial telemetry packets.

---

## 4. Running the Python Interface

### Standalone Self-Test Mode:
You can run `esp32_interface.py` directly to verify port scanning, keep-alive handshakes, beacon actuation, and emergency trip commands:

```bash
# Run verification test using the project virtual environment
.\anomalib_env\Scripts\python.exe smartbelt_software/phase6_hardware_interface/esp32_interface.py
```

### Auto-Fallback to High-Fidelity Simulation:
If no physical ESP32 is plugged in, `esp32_interface.py` automatically initializes in **Simulation Mode**:
- Commands update virtual relay, beacon, and buzzer states.
- Telemetry packets are generated matching Protocol Spec v1.0.0.
- All telemetry readings are explicitly labeled: `"SIMULATED — not a real measurement"`.

---

## 5. Software Integration with SmartBelt Core

To route decisions from `smartbelt_core.py` directly to the ESP32 controller:

```python
from smartbelt_software.phase6_hardware_interface.esp32_interface import ESP32Interface
from smartbelt_software.phase4_integration.smartbelt_core import SmartBeltApplication

# Initialize ESP32 controller
esp = ESP32Interface()

# In your frame processing loop:
packet = app.process_frame(frame_bgr)

# Dispatch computed hardware safety action
esp.dispatch_fusion_action(packet)
```
