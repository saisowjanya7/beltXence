# SmartBelt — Windows Deployment & Setup Guide

This guide provides step-by-step instructions for setting up and running the **SmartBelt** Conveyor Joint & Damage Monitoring System on a fresh Windows laptop (Windows 10 / 11, 64-bit).

---

## 1. System Requirements & Prerequisites

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 10 or Windows 11 (64-bit) |
| **Python Version** | **Python 3.11.x (64-bit)** recommended (Python 3.10 – 3.12 supported) |
| **Processor (CPU)** | Modern Intel Core i5/i7/i9 or AMD Ryzen 5/7/9 (x86_64) |
| **System Memory (RAM)** | 8 GB minimum (16 GB recommended) |
| **Disk Space** | ~2 GB free storage (includes dependencies and 226 MB model weights) |
| **GPU Acceleration** | Optional. The inference pipeline is CPU-optimized by default. |
| **Hardware Link** | USB port for physical ESP32 (High-fidelity virtual simulation fallback automatically engages if no hardware is connected) |

---

## 2. Directory Structure & Required Files

Ensure the project directory contains the following critical folders and files:

```text
Conveyor/
│
├── app.py                                   <-- Root Streamlit dashboard entrypoint
├── requirements.txt                         <-- Clean Python package dependencies
├── README_DEPLOYMENT.md                     <-- This deployment guide
│
├── results_joint/                           <-- REQUIRED MODEL WEIGHTS
│   └── 20260910_105659/
│       ├── patchcore_joint_v1.ckpt         <-- Validated PatchCore model (226.5 MB)
│       └── ...
│
├── smartbelt_software/                      <-- Unified Software Stack
│   ├── phase4_integration/                  <-- Core backend & fusion engine
│   │   ├── smartbelt_core.py
│   │   └── xgboost_sensor_module.py
│   ├── phase5_dashboard/                    <-- Dashboard components
│   │   └── app.py
│   ├── phase6_hardware_interface/           <-- ESP32 driver & Arduino firmware
│   │   ├── esp32_interface.py
│   │   └── esp32_firmware.ino
│   └── phase7_testing/                      <-- 50-trial verification suite
│       └── test_suite.py
│
├── videos/                                  <-- Sample Video Demonstration Files
│   ├── conveyor_with_real_damage.mp4        <-- Joint rupture test video
│   └── conveyorbelt.mp4
│
└── dataset/                                 <-- Test Splits (for verification)
    └── patchcore/test/
        ├── good/                            <-- 37 normal joint images
        └── bad/real_damage/                 <-- 40 real damaged joint images
```

> [!IMPORTANT]
> **Model Checkpoint Integrity:** The file `results_joint/20260910_105659/patchcore_joint_v1.ckpt` contains the full WideResNet-50 backbone weights and the pre-computed memory bank (22,425 feature vectors). It must **not** be retrained, renamed, or modified.

---

## 3. Step-by-Step Installation on a Fresh Windows Laptop

### Step 1: Open PowerShell / Terminal in the Project Root
Open Windows Terminal or PowerShell and navigate to the project directory:
```powershell
cd C:\path\to\Conveyor
```

*(If PowerShell scripts are restricted on your system, enable execution for the current session:)*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### Step 2: Create a Fresh Python Virtual Environment
Create an isolated virtual environment using Python 3.11:
```powershell
python -m venv venv
```

---

### Step 3: Activate the Virtual Environment
Activate the environment:
- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

Your prompt will now display `(venv)`.

---

### Step 4: Upgrade Package Installer
```powershell
python -m pip install --upgrade pip
```

---

### Step 5: Install Project Dependencies
Install all required libraries using the clean specification:
```powershell
pip install -r requirements.txt
```

*(Optional: For direct CPU-only PyTorch optimization on systems without a dedicated NVIDIA GPU, you can alternatively install PyTorch via:*
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
*)*

---

## 4. Running the SmartBelt Dashboard

### Exact Launch Command
From the project root directory, run:
```powershell
streamlit run app.py
```

Streamlit will start a local web server and automatically open your default browser:
```text
  Local URL: http://localhost:8501
  Network URL: http://<your-ip>:8501
```

### Dashboard Usage Instructions
1. **Video Stream Source (Sidebar):**
   - **Sample Video:** Select `conveyor_with_real_damage.mp4` to inspect verified conveyor splice damage.
   - **Upload Video:** Upload custom MP4, AVI, or MOV conveyor recordings.
   - **Live Webcam:** Select device index (`0`, `1`, etc.) for real-time camera inspection.
2. **Sensor Telemetry Scenario (Sidebar):**
   - Choose an operational scenario: `Nominal Baseline`, `Bearing Overheat (78.5°C)`, `Splice Mechanical Vibration`, `Drive Pulley Slip`, `Catastrophic Multi-Modal Rupture`, or `Manual Slider Adjustment`.
   - All simulated telemetry is clearly labeled `[SIMULATED — not a real measurement]`.
3. **Execute Inspection:**
   - Click **`▶ Start Conveyor Inspection`**.
   - Watch the live annotated frame stream with real-time anomaly scores, multi-modal consensus badges, and HUD alert banners.
4. **Audit History & Telemetry Export:**
   - Scroll to the bottom of the dashboard to view the line-chart trend.
   - Click **`📥 Download Telemetry Log (CSV)`** to export the inspection log.

---

## 5. Running the Complete Verification Test Suite

To verify the entire vision, fusion, and ESP32 actuator pipeline:
```powershell
python smartbelt_software/phase7_testing/test_suite.py
```

This runs 50 automated verification trials (10 across each of the 5 operational scenarios) and outputs:
- Real-time scenario accuracy metrics
- Confusion matrix plot: `smartbelt_software/phase7_testing/confusion_matrix.png`
- Performance comparison plot: `smartbelt_software/phase7_testing/scenarios_performance.png`
- Audit report: `smartbelt_software/phase7_testing/test_report.md`

---

## 6. Hardware Connection (ESP32 Controller)

- **Physical ESP32 Link:** Plug the ESP32 microcontroller into any USB port. The host interface auto-scans COM ports (e.g. `COM3`, `COM4`) and establishes a 115200 baud UART link.
- **Simulation Fallback:** If no ESP32 is attached, the driver automatically engages **Virtual Simulation Mode**. All relay trips, beacon colors, and buzzer states are simulated in software with zero crashes.
- **Firmware Upload (Optional):** If programming a new ESP32 board, open `smartbelt_software/phase6_hardware_interface/esp32_firmware.ino` in the Arduino IDE and flash to the board.

---

## 7. Troubleshooting & FAQs

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `File does not exist: app.py` | Command run outside project root | Ensure your terminal is in the project root directory where `app.py` resides. |
| `Script execution is disabled` | PowerShell restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell. |
| `Port 8501 is already in use` | Another Streamlit instance running | Run `streamlit run app.py --server.port 8502` |
| `No module named torch` | Virtual environment not activated | Activate your environment: `.\venv\Scripts\Activate.ps1` |
| Video does not play | Missing codec | Ensure `opencv-python` is installed via `pip install -r requirements.txt`. |
