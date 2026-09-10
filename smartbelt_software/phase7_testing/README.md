# Phase 7 — End-to-End System Testing & Validation

This directory contains the automated end-to-end verification harness for the **SmartBelt** conveyor joint and damage monitoring system.

---

## Overview of Test Scenarios

The suite evaluates 5 operational industrial scenarios using real test images from `dataset/patchcore/test/` combined with simulated sensor telemetry:

1. **Scenario 1: Nominal Conveyor Operation**
   - *Input:* Healthy joint frame (`dataset/patchcore/test/good/`) + Nominal baseline sensors.
   - *Expected Outcome:* Risk = `NORMAL`, Relay = `CLOSED (RUNNING)`, Beacon = `GREEN`.
   - *Validates:* Zero false alarms under normal factory production.

2. **Scenario 2: Bearing Overheat Emergency (Blind-Spot Protection)**
   - *Input:* Healthy joint frame + Bearing temperature 78.5°C.
   - *Expected Outcome:* Risk = `CRITICAL` (`SENSOR_DOMINANT_MECHANICAL_DISTRESS`), Relay = `TRIPPED`, Beacon = `RED`.
   - *Validates:* Safe tripping even when visual joint camera is in a blind-spot.

3. **Scenario 3: Splice Degradation / Vibration Warning**
   - *Input:* Healthy joint frame + Warning-tier vibration (4.85 mm/s).
   - *Expected Outcome:* Risk = `WARNING` (`ELEVATED_INSPECTION_ZONE`), Relay = `CLOSED`, Beacon = `AMBER`.
   - *Validates:* Preventive maintenance alerting without unscheduled downtime.

4. **Scenario 4: Visual Joint Rupture Emergency**
   - *Input:* Ruptured joint frame (`dataset/patchcore/test/bad/real_damage/`) + Nominal sensors.
   - *Expected Outcome:* Risk = `CRITICAL` (`VISION_DOMINANT_FAILURE`), Relay = `TRIPPED`, Beacon = `RED`.
   - *Validates:* Instantaneous tripping upon visual joint tear before mechanical vibration propagates.

5. **Scenario 5: Catastrophic Multi-Modal Rupture**
   - *Input:* Ruptured joint frame + Severe vibration (8.45 mm/s) + Pulley slip (28%).
   - *Expected Outcome:* Risk = `CRITICAL` (`CONFIRMED_MULTIMODAL_EMERGENCY`), Relay = `TRIPPED`, Beacon = `RED`.
   - *Validates:* Cross-modal consensus handling with boosted risk scoring.

---

## How to Run the Test Suite

Execute the test suite using the project virtual environment:

```bash
# From the project root (c:\Users\korra\OneDrive\Desktop\Conveyor):
.\anomalib_env\Scripts\python.exe smartbelt_software/phase7_testing/test_suite.py
```

---

## Generated Artifacts

Upon completion, the suite automatically outputs:
- [`phase7_detailed_eval.csv`](phase7_detailed_eval.csv): Full trial-by-trial logs of inputs, scores, latencies, and decisions.
- [`confusion_matrix.png`](confusion_matrix.png): High-resolution plot of the emergency trip decision confusion matrix.
- [`scenarios_performance.png`](scenarios_performance.png): Comparative bar chart of Visual Score vs. Sensor Score vs. Fused Score across all 5 operational scenarios.
- [`test_report.md`](test_report.md): Formal technical verification report with metric tables and architectural analysis.
