# SmartBelt v2 — Ground-Up Physical Inspection & IoT Fusion

A real-hardware conveyor belt inspection system with multi-modal anomaly detection:
- **Vision:** Logitech C270 HD USB camera + PatchCore WideResNet-50 Anomaly Detection.
- **Physical Sensors:** ESP32 reading MLX90614 (IR Non-contact Temp), MPU-6050 (Vibration RMS), HX711 (Tension Load Cell), and E18-D80NK (IR Belt Speed).
- **Inference:** Non-blocking asynchronous inference running on CPU (Snapdragon ARM64).
- **Dashboard:** Real-time Streamlit dashboard with live heatmaps, dynamic Plotly telemetry trends, and operational risk consensus.

---

## 1. Hardware Pinout & Wiring

| Sensor / Module | Sensor Pin | ESP32 GPIO | Notes |
|---|---|---|---|
| **MLX90614 (IR Temp)** | SDA | GPIO 21 | Shared I²C bus (Addr: 0x5A) |
| | SCL | GPIO 22 | Shared I²C bus |
| | VIN | 3.3V | |
| | GND | GND | |
| **MPU-6050 (Vibration)** | SDA | GPIO 21 | Shared I²C bus (Addr: 0x68) |
| | SCL | GPIO 22 | Shared I²C bus |
| | VCC | 3.3V | |
| | GND | GND | |
| | AD0 | GND | Sets address to 0x68 |
| **HX711 (Load Cell)** | DOUT | GPIO 16 | Serial data out |
| | SCK | GPIO 17 | Serial clock |
| | VCC | 3.3V or 5V | Depending on breakout |
| | GND | GND | |
| **E18-D80NK (IR Speed)** | Signal | GPIO 34 | NPN-NO (Input Pull-up, active LOW) |
| | VCC | 5V | Requires 5V supply |
| | GND | GND | Common ground with ESP32 |

---

## 2. Flashing the ESP32 Firmware

1. Open Arduino IDE.
2. Open [`firmware/esp32_smartbelt/esp32_smartbelt.ino`](file:///firmware/esp32_smartbelt/esp32_smartbelt.ino).
3. Install the required Arduino libraries via Library Manager:
   - `Adafruit MLX90614 Library`
   - `MPU6050` (by Electronic Cats or Jeff Rowberg)
   - `HX711 Arduino Library` (by Bogdan Necula)
   - `ArduinoJson` (v6 or v7)
4. Connect your ESP32 via USB and select **DOIT ESP32 DEVKIT V1** (or matching board).
5. Click **Upload**.
6. Open Serial Monitor at **115200 baud** to verify JSON streaming:
   ```json
   {"ts":10240,"temp_c":27.4,"vib_x":0.012,"vib_y":-0.005,"vib_z":0.021,"vib_rms":0.025,"load_kg":0.00,"ir_state":1,"belt_speed_mps":0.000}
   ```

---

## 3. Running the Live Inspection Dashboard

1. Connect the **Logitech C270 USB webcam** to the laptop.
2. Connect the **ESP32** via USB.
3. Launch the dashboard from the project root:
   ```bash
   streamlit run app.py
   ```
4. Access the UI in your browser at `http://localhost:8501`.

---

## 4. Establishing Real Sensor Calibration

1. Start the conveyor belt running under nominal, healthy conditions.
2. In the Streamlit sidebar, locate **Sensor Calibration**.
3. Choose duration (e.g. 30 to 60 seconds) and click **Run Baseline Calibration**.
4. The system collects actual sensor readings, computes mean ($\mu$) and standard deviation ($\sigma$) per channel, and saves the calibrated profile to [`calibration/sensor_baseline.json`](file:///calibration/sensor_baseline.json).
5. All live sensor readings are now evaluated as true statistical z-score anomalies against your physical machine.
