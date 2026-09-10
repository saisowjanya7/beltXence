# SmartBelt Industrial Microcontroller Interface: Hardware Communication Protocol Specification

**Document Version:** 1.0.0  
**Target Hardware:** Espressif ESP32-WROOM-32 / ESP32-S3  
**Host System:** SmartBelt Edge PC (Python 3.11 / Anomalib / PatchCore Backend)  
**Interface:** USB Serial (CDC) / RS-485 via UART2  
**Default Baud Rate:** 115200 bps (8-N-1, Flow Control: None)

---

## 1. Protocol Architecture & Overview

The SmartBelt hardware architecture employs a **bidirectional master-slave control and telemetry stream**:

```
+-------------------------------------------------------------+
|               Host PC (SmartBelt Core Engine)               |
|  - PatchCore Vision Anomaly Distance Evaluation             |
|  - XGBoost Sensor Anomaly Prediction Engine                 |
|  - Conservative Multi-Modal Decision Fusion                 |
+-------------------------------------------------------------+
          ▲                                       │
          │ Telemetry Stream                      │ Actuator & Trip Commands
          │ (JSON @ 5-10 Hz)                      │ (Async Events & Heartbeat)
          │                                       ▼
+-------------------------------------------------------------+
|             ESP32 Microcontroller (Firmware)                |
|  - Physical Sensor ADC / I2C / 1-Wire Acquisition           |
|  - Relay Emergency Motor Trip Driver                        |
|  - 3-Color Industrial Beacon Tower (Green, Amber, Red)      |
|  - High-Decibel Piezo Alarm Buzzer                          |
|  - Hardware Watchdog Timer (3000 ms failsafe trip)          |
+-------------------------------------------------------------+
```

---

## 2. Electrical Pinout & Hardware Specification

| ESP32 Pin | Direction | Function | Description / Industrial Component |
| :--- | :--- | :--- | :--- |
| **GPIO 25** | Output | `RELAY_CTRL` | Optocoupled 10A Relay Module (Active LOW / NC for Fail-Safe Conveyor VFD Trip Circuit) |
| **GPIO 26** | Output | `BEACON_RED` | Red High-Intensity LED / 24V Beacon Tower Transistor Driver |
| **GPIO 27** | Output | `BEACON_AMB` | Amber Warning LED / 24V Beacon Tower Transistor Driver |
| **GPIO 14** | Output | `BEACON_GRN` | Green Normal Operation LED / 24V Beacon Tower Transistor Driver |
| **GPIO 12** | Output | `BUZZER_CTRL`| Active Piezo Buzzer (Pulse-width / Strobe Warning Sound) |
| **GPIO 34** | Input (ADC) | `VIB_PIN` | Analog Accelerometer / Vibration Sensor (0–3.3V, ISO-10816 calibrated) |
| **GPIO 35** | Input (ADC) | `TENS_PIN` | Load Cell Belt Tension Conditioner (0–3.3V) |
| **GPIO 4**  | Bidirectional| `TEMP_1WIRE` | Dallas DS18B20 Digital Temperature Sensor (Bearing Housing) |
| **GPIO 18** | Input (ExtInt)| `HALL_SPEED`| NPN Hall Effect Proximity Sensor (Pulley RPM & Slip Detection) |
| **GPIO 1**  | Output (TX0) | `UART_TX` | USB-UART Bridge / CP2102 / CH340 TX to Host PC |
| **GPIO 3**  | Input (RX0)  | `UART_RX` | USB-UART Bridge / CP2102 / CH340 RX from Host PC |

---

## 3. Message Framing & Transport Rules

1. **Delimiter:** All messages must be a single line terminated with newline (`\n` or `\r\n`).
2. **Encoding:** UTF-8 JSON.
3. **Pacing:** Telemetry packets are streamed from the ESP32 to the PC at **5 Hz (every 200 ms)**.
4. **Buffering & Flushing:** The Host PC flushes its input buffer before reading to ensure zero latency.

---

## 4. Host-to-ESP32 Commands (Downlink)

Commands sent from Host PC to ESP32:

### 4.1 Emergency Motor Trip (`CMD_TRIP`)
Fires when the fusion engine decides `CRITICAL` risk (e.g. splice separation or severe bearing seizure).
```json
{"cmd": "TRIP", "reason": "CONFIRMED_MULTIMODAL_EMERGENCY", "timestamp": 1725960000}
```
- **Action:** Relay is de-energized immediately (opening the motor safety circuit), Red Beacon turned ON, Buzzer set to emergency pulsed mode.

### 4.2 Motor Circuit Reset (`CMD_RESET`)
Operator manually acknowledges the alarm and resets the conveyor circuit.
```json
{"cmd": "RESET", "auth": "OPERATOR_ACK"}
```
- **Action:** Relay is re-energized (contacts closed), Red Beacon turned OFF, Green Beacon turned ON, Buzzer silenced.

### 4.3 Set Beacon State (`CMD_SET_BEACON`)
Changes visual indicator tower color:
```json
{"cmd": "SET_BEACON", "color": "GREEN"}
{"cmd": "SET_BEACON", "color": "AMBER"}
{"cmd": "SET_BEACON", "color": "RED"}
```

### 4.4 Set Buzzer State (`CMD_SET_BUZZER`)
```json
{"cmd": "SET_BUZZER", "state": "OFF"}
{"cmd": "SET_BUZZER", "state": "INTERMITTENT"}
{"cmd": "SET_BUZZER", "state": "PULSED_EMERGENCY"}
```

### 4.5 Heartbeat / Watchdog Keep-Alive (`CMD_PING`)
Must be received by ESP32 at least once every 3000 ms while the conveyor is in production mode.
```json
{"cmd": "PING", "seq": 1042}
```
- **ESP32 Response:**
```json
{"ack": "PONG", "seq": 1042, "status": "OK"}
```

---

## 5. ESP32-to-Host Telemetry Packets (Uplink)

Streamed periodically at 5 Hz:

```json
{
  "device_id": "SMARTBELT_ESP32_01",
  "uptime_ms": 142380,
  "telemetry": {
    "temperature_c": 38.4,
    "vibration_rms_mms": 1.42,
    "speed_slip_pct": 1.6,
    "tension_kn": 24.8
  },
  "actuators": {
    "relay_closed": true,
    "beacon": "GREEN",
    "buzzer": "OFF"
  },
  "watchdog_ok": true,
  "is_simulated": false
}
```

---

## 6. Safety Failsafe & Watchdog Specification

- **Host Watchdog (Hardware Level):**  
  The ESP32 firmware initializes a 3000 ms software timer. Every valid command or `PING` from the host resets this timer. If communication with the host PC is lost for $\ge 3000\text{ ms}$, the ESP32 assumes Host PC failure or cable severance and executes a **FAILSAFE TRIP**:
  1. `RELAY_CTRL` de-energized (Motor tripped).
  2. `BEACON_RED` activated.
  3. `BUZZER_CTRL` sounds continuous intermittent alert.
  4. Telemetry packet logs `"watchdog_ok": false`.

- **Safe Recovery:**  
  Conveyor can only be re-energized when a valid `{"cmd": "RESET"}` command is received after host connection is restored.
