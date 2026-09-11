# SIH Smart Conveyor Belt Detection & Tri-Node Telemetry Hub

An industrial-grade IoT Conveyor Belt Monitoring System built for real-time velocity tracking, white line marker station identification, digital load cell weighing, and frame vibration kinematics with automated safety alerts.

---

## 🚀 System Architecture

```
+---------------------------------------------------------------------------------------------------------+
|                                      CONVEYOR SYSTEM HARDWARE                                            |
|                                                                                                         |
|   +---------------------------------+   +-----------------------------+   +---------------------------+ |
|   |      NODE 1: ESP32 (COM19)      |   |   NODE 2: ARDUINO (COM23)   |   |   NODE 3: ESP32 (COM21)   | |
|   |  (Conveyor Frame Motion / Vib)  |   | (Conveyor Belt Tracker & V) |   | (Weigh Station Scale+IMU) | |
|   |                                 |   |                             |   |                           | |
|   | • MPU-6050 ONLY (pure Wire.h)   |   | • E18-D80NK Digital Optical |   | • HX711 24-bit ADC Scale  | |
|   | • I2C: SDA=GPIO 21, SCL=GPIO 22 |   | • Pin 2 (INPUT_PULLUP)      |   |   DOUT=GPIO 18, SCK=GPIO19| |
|   | • 115200 Baud                   |   | • 9600 Baud                 |   | • MPU-6050 6-Axis IMU     | |
|   | • Telemetry:                    |   | • Telemetry:                |   |   I2C: SDA=21, SCL=22     | |
|   |   - 3D Attitude (Pitch/Roll/Yaw)|   |   - Microsecond Velocity    |   | • 115200 Baud             | |
|   |   - Linear Accel (Ax, Ay, Az)   |   |   - 4-Second Hold Delay     |   | • Telemetry:              | |
|   |   - Frame Vibration Index (G)   |   |   - Station (1->2->3->4)    |   |   - Net Weight (g / kg)   | |
|   |   - MPU-6050 Temp               |   |   - Live Pin 2 Monitor      |   |   - 3D Attitude & Tilt    | |
|   +---------------------------------+   +-----------------------------+   +---------------------------+ |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
                   +-------------------------------------------------------------------+
                   |         UNIFIED WEB TELEMETRY HUB (http://localhost:8080)         |
                   |                                                                   |
                   |   [NODE 1: COM19 (Cyan)]  [NODE 2: COM23 (Purple)]  [NODE 3: COM21 (Gold)]|
                   |   • 3D Frame Attitude     • Animated Belt Stage     • HX711 Weight Scale  |
                   |   • Attitude Horizon      • 4 Station Zones         • Tare Scale & Offset |
                   |   • Vibration Index Badge • Real-Time Speedometer   • 3D Board Visualizer |
                   |   • Accel & Gyro Charts   • 4s Delay Countdown      • Attitude Horizon    |
                   |   • Live COM19 Console    • Live COM23 Console      • Accel & Gyro Charts |
                   |                                                     • Live COM21 Console  |
                   +-------------------------------------------------------------------+
```

---

## ⚡ Core Features

1. **Conveyor Belt Tracking & Microsecond Velocity (Node 2 - Arduino Uno COM23)**:
   - Identifies conveyor stations (Location 1 &rarr; Location 2 &rarr; Location 3 &rarr; Location 4) using white line optical detection on Pin 2.
   - Computes real-time belt speed: $v = \frac{\text{Distance}}{\Delta t_{\text{transit}}}$.
   - Automatic 4-second hold delay at each station with countdown timer.
   - Interactive line spacing calibration on the dashboard.

2. **Frame Vibration & 3D Kinematics (Node 1 - ESP32 COM19)**:
   - High-speed 50 Hz MPU-6050 6-Axis IMU stream (Pure `Wire.h` fast I2C).
   - Real-time 3D board visualizer and aerospace artificial horizon overlay.
   - Vibration index calculation: $G_{\text{total}} = \sqrt{a_x^2 + a_y^2 + a_z^2}$.

3. **Digital Weigh Station & Tilt (Node 3 - ESP32 COM21)**:
   - HX711 24-bit ADC with non-blocking timeout protection.
   - Real-time weight readouts in grams and kilograms with peak weight retention.
   - Remote software tare support.

4. **Real-Time Safety Alarms & Warnings**:
   - **Load Cell Overload Alert (`> 50g`)**: Pulsing neon red card highlight, `⚠️ OVERLOAD` badge, drop-down global banner, and Web Audio API synthesized siren.
   - **MPU-6050 Sudden Disturbance**: Detects shock/impact ($G_{\text{total}} > 1.45\text{g}$, $\Delta G > 0.40\text{g}$, or angular rate $> 90^\circ/\text{s}$), flashing amber warnings and chirp alarms.
   - **Thresholds Modal**: Configurable alarm limits and sound mute toggle.

5. **Data Export & Realistic Simulation**:
   - **Demo Mode**: Full physics-driven simulation of all 3 nodes running simultaneously.
   - **CSV Export**: Single-click export of 19 telemetry channels with timestamps.

---

## 🔌 Hardware Pinouts

### Node 1: ESP32 Conveyor Frame IMU (COM19)
| Component | ESP32 Pin | Note |
| :--- | :--- | :--- |
| **MPU-6050 SDA** | GPIO 21 | I2C Data (Pure Wire.h, 400kHz) |
| **MPU-6050 SCL** | GPIO 22 | I2C Clock |
| **VCC / GND** | 3.3V / GND | Power |

### Node 2: Arduino Uno Conveyor Belt Tracker (COM23)
| Component | Arduino Pin | Note |
| :--- | :--- | :--- |
| **E18-D80NK Signal (Black)** | Pin 2 | Digital Input (`INPUT_PULLUP`) |
| **Status LED** | Pin 13 | Onboard Builtin LED |
| **VCC (Brown) / GND (Blue)** | 5V / GND | Power |

### Node 3: ESP32 Weigh Station & IMU (COM21)
| Component | ESP32 Pin | Note |
| :--- | :--- | :--- |
| **HX711 DOUT** | GPIO 18 | Digital Data |
| **HX711 SCK** | GPIO 19 | Clock |
| **MPU-6050 SDA** | GPIO 21 | I2C Data |
| **MPU-6050 SCL** | GPIO 22 | I2C Clock |
| **VCC / GND** | 3.3V / GND | Power |

---

## 💻 Getting Started

### 1. Run the Web Dashboard
```bash
cd web_dashboard
python -m http.server 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in Google Chrome, Microsoft Edge, or Brave (Web Serial API compatible).

### 2. Connect Hardware
- Click **Connect COM19** for Node 1 (115200 baud).
- Click **Connect COM23** for Node 2 (9600 baud).
- Click **Connect COM21** for Node 3 (115200 baud).

Or simply click **Demo / Sim Mode** to test the entire telemetry suite and alert system in software simulation.
