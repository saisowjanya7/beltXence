# SmartBelt -- Phase 7: Integration Test Results

**Run timestamp:** 2026-09-10T12:08:12.849020  
**Overall result:** 5/5 scenarios PASSED  
**Thresholds:** NORMAL < 0.62710 <= WARNING < 0.68887 <= CRITICAL  

## Data Provenance

| Source | Status |
|--------|--------|
| Visual anomaly scores | **REAL** -- PatchCore inference from Phase 1 scores.csv |
| Sensor telemetry | **SIMULATED** -- Phase 5 simulate_sensor_readings() |
| Model weights | UNTOUCHED -- results_joint/20260910_105659/ |
| New inference | None -- cached scores used |

---

## Scenario Summary

| # | Scenario | Expected | Actual | Visual | Sensor | Fused | Result |
|---|----------|----------|--------|--------|--------|-------|--------|
| 1 | Normal Conveyor Operation | NORMAL | NORMAL | 0.45280 | 0.34317 | 0.40895 | **PASS** |
| 2 | Early Joint Fatigue (Pre-Rupture Warning) | WARNING | WARNING | 0.65991 | 0.34317 | 0.65991 | **PASS** |
| 3 | Severe Joint Rupture (Visual CRITICAL) | CRITICAL | CRITICAL | 0.79867 | 0.87315 | 0.92315 | **PASS** |
| 4 | General Belt Damage -- Sensor Dominant (No Joint Visible) | CRITICAL | CRITICAL | 0.43544 | 0.93492 | 0.93492 | **PASS** |
| 5 | Sensor Anomaly on Normal Belt (Bearing Overheat) | CRITICAL | CRITICAL | 0.51587 | 0.85606 | 0.85606 | **PASS** |

---

## Scenario 1: Normal Conveyor Operation  [PASS]

**Description:** Belt joint passes through camera zone in pristine condition. All sensors nominal. Expect NORMAL classification at every layer.

### Input Data
- **Visual score:** `0.45280` -- Real PatchCore score -- tag: good (test set, Phase 1 scores.csv)
- **Sensor data:** SIMULATED via simulate_sensor_readings('nominal')
  - Vibration: `1.42 mm/s RMS`
  - Temperature: `36.5 C`
  - Tension: `22.4 kN`
  - Speed Slip: `1.8%`

### Pipeline Trace
| Layer | Output |
|-------|--------|
| Phase 2 -- Visual Risk | `0.45280` --> **NORMAL** |
| Phase 5 -- Sensor Score | `0.34317` --> **NORMAL** (dominant: temperature @ 0.3650) |
| Phase 5 -- Fusion | `0.40895` --> **NORMAL** (NOMINAL_OPERATION) |
| Phase 6 -- Alert | Buzzer: OFF \| Relay: CLOSED (RUNNING) \| Beacon: GREEN |
| Phase 6 -- GPIO Dispatch | `[SIMULATED HARDWARE GPIO] Triggered: CMD_STATUS_NORMAL` |

### Pass/Fail Evaluation
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Risk Classification | `NORMAL` | `NORMAL` | **PASS** |
| Alert GPIO Command | `CMD_STATUS_NORMAL` | (in dispatch log) | **PASS** |
| **Overall** | | | **PASS** |

---

## Scenario 2: Early Joint Fatigue (Pre-Rupture Warning)  [PASS]

**Description:** Splice shows early fatigue indicators visible to camera. Score sits in WARNING zone. Sensors nominal. System must escalate to WARNING without triggering motor stop.

### Input Data
- **Visual score:** `0.65991` -- Real PatchCore score -- frame_0230.jpg (tag: good, Phase 1 scores.csv)
- **Sensor data:** SIMULATED via simulate_sensor_readings('nominal')
  - Vibration: `1.42 mm/s RMS`
  - Temperature: `36.5 C`
  - Tension: `22.4 kN`
  - Speed Slip: `1.8%`

### Pipeline Trace
| Layer | Output |
|-------|--------|
| Phase 2 -- Visual Risk | `0.65991` --> **WARNING** |
| Phase 5 -- Sensor Score | `0.34317` --> **NORMAL** (dominant: temperature @ 0.3650) |
| Phase 5 -- Fusion | `0.65991` --> **WARNING** (ELEVATED_INSPECTION_ZONE) |
| Phase 6 -- Alert | Buzzer: PULSE_CHIRP (1 Hz) \| Relay: CLOSED (MAINTAIN MOTION) \| Beacon: AMBER |
| Phase 6 -- GPIO Dispatch | `[SIMULATED HARDWARE GPIO] Triggered: CMD_ALERT_WARNING` |

### Pass/Fail Evaluation
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Risk Classification | `WARNING` | `WARNING` | **PASS** |
| Alert GPIO Command | `CMD_ALERT_WARNING` | (in dispatch log) | **PASS** |
| **Overall** | | | **PASS** |

---

## Scenario 3: Severe Joint Rupture (Visual CRITICAL)  [PASS]

**Description:** Full joint separation visible in camera frame. PatchCore detects high anomaly. Vibration spike from pounding. Both modalities agree: CONFIRMED_MULTIMODAL_EMERGENCY. Emergency motor trip must be commanded.

### Input Data
- **Visual score:** `0.79867` -- Real PatchCore score -- highest real_damage image (Phase 1 scores.csv)
- **Sensor data:** SIMULATED via simulate_sensor_readings('splice_impact_vibration')
  - Vibration: `8.45 mm/s RMS`
  - Temperature: `48.0 C`
  - Tension: `19.5 kN`
  - Speed Slip: `3.2%`

### Pipeline Trace
| Layer | Output |
|-------|--------|
| Phase 2 -- Visual Risk | `0.79867` --> **CRITICAL** |
| Phase 5 -- Sensor Score | `0.87315` --> **CRITICAL** (dominant: vibration @ 1.0000) |
| Phase 5 -- Fusion | `0.92315` --> **CRITICAL** (CONFIRMED_MULTIMODAL_EMERGENCY) |
| Phase 6 -- Alert | Buzzer: ON (CONTINUOUS 95dB) \| Relay: OPEN (MOTOR TRIPPED / EMERGENCY STOP) \| Beacon: FLASHING_RED |
| Phase 6 -- GPIO Dispatch | `[SIMULATED HARDWARE GPIO] Triggered: CMD_EMERGENCY_HALT_BUZZER_ON` |

### Pass/Fail Evaluation
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Risk Classification | `CRITICAL` | `CRITICAL` | **PASS** |
| Alert GPIO Command | `CMD_EMERGENCY_HALT_BUZZER_ON` | (in dispatch log) | **PASS** |
| **Overall** | | | **PASS** |

---

## Scenario 4: General Belt Damage -- Sensor Dominant (No Joint Visible)  [PASS]

**Description:** Belt damage occurs in a zone not visible to the joint camera (blind spot scenario). Visual score is normal. However, severe belt slip and low tension are detected by sensors. Sensor-dominant path must escalate to CRITICAL independently.

### Input Data
- **Visual score:** `0.43544` -- Real PatchCore score -- normal test image (Phase 1 scores.csv)
- **Sensor data:** SIMULATED via simulate_sensor_readings('belt_slip_and_slack')
  - Vibration: `3.1 mm/s RMS`
  - Temperature: `68.2 C`
  - Tension: `10.5 kN`
  - Speed Slip: `28.0%`

### Pipeline Trace
| Layer | Output |
|-------|--------|
| Phase 2 -- Visual Risk | `0.43544` --> **NORMAL** |
| Phase 5 -- Sensor Score | `0.93492` --> **CRITICAL** (dominant: speed_slip @ 1.0000) |
| Phase 5 -- Fusion | `0.93492` --> **CRITICAL** (SENSOR_DOMINANT_MECHANICAL_DISTRESS) |
| Phase 6 -- Alert | Buzzer: ON (CONTINUOUS 95dB) \| Relay: OPEN (MOTOR TRIPPED / EMERGENCY STOP) \| Beacon: FLASHING_RED |
| Phase 6 -- GPIO Dispatch | `[SIMULATED HARDWARE GPIO] Triggered: CMD_EMERGENCY_HALT_BUZZER_ON` |

### Pass/Fail Evaluation
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Risk Classification | `CRITICAL` | `CRITICAL` | **PASS** |
| Alert GPIO Command | `CMD_EMERGENCY_HALT_BUZZER_ON` | (in dispatch log) | **PASS** |
| **Overall** | | | **PASS** |

---

## Scenario 5: Sensor Anomaly on Normal Belt (Bearing Overheat)  [PASS]

**Description:** Joint visual appears healthy. Bearing overheat detected by temperature sensor (78.5 C >> 65 C warning threshold). Sensor-only path must catch this thermal fault as CRITICAL.

### Input Data
- **Visual score:** `0.51587` -- Real PatchCore score -- median test/good image (Phase 1 scores.csv)
- **Sensor data:** SIMULATED via simulate_sensor_readings('bearing_overheat')
  - Vibration: `2.8 mm/s RMS`
  - Temperature: `78.5 C`
  - Tension: `21.0 kN`
  - Speed Slip: `4.5%`

### Pipeline Trace
| Layer | Output |
|-------|--------|
| Phase 2 -- Visual Risk | `0.51587` --> **NORMAL** |
| Phase 5 -- Sensor Score | `0.85606` --> **CRITICAL** (dominant: temperature @ 0.9689) |
| Phase 5 -- Fusion | `0.85606` --> **CRITICAL** (SENSOR_DOMINANT_MECHANICAL_DISTRESS) |
| Phase 6 -- Alert | Buzzer: ON (CONTINUOUS 95dB) \| Relay: OPEN (MOTOR TRIPPED / EMERGENCY STOP) \| Beacon: FLASHING_RED |
| Phase 6 -- GPIO Dispatch | `[SIMULATED HARDWARE GPIO] Triggered: CMD_EMERGENCY_HALT_BUZZER_ON` |

### Pass/Fail Evaluation
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Risk Classification | `CRITICAL` | `CRITICAL` | **PASS** |
| Alert GPIO Command | `CMD_EMERGENCY_HALT_BUZZER_ON` | (in dispatch log) | **PASS** |
| **Overall** | | | **PASS** |

---

## Upgrade Requirements

The following limitations exist due to hardware/data constraints at test time:

| Limitation | Impact | What is Needed |
|------------|--------|----------------|
| No physical ESP32 hardware | GPIO dispatch is simulated; buzzer/relay not physically verified | Connect ESP32 on USB, pass --serial-port COM3 to alert_system |
| Sensor telemetry simulated | Sensor fusion validated architecturally, not with real readings | Deploy 4-sensor ESP32 sketch, pipe JSON over UART |
| No belt-surface camera images | General belt damage (Scenario 4) uses only sensor path | Collect 150+ normal + 40+ damaged belt-surface images |
| Single-joint dataset | PatchCore trained on one joint type / one belt | Re-train with multiple belt joints and belt widths |
| WARNING zone borderline case | Only 1 real image truly in WARNING zone (0.65991-0.68643) | Collect more partially damaged joint images at early failure stage |

---

*Report generated by `smartbelt_pipeline/phase7/run_integration_tests.py` at 2026-09-10T12:08:12.849020*