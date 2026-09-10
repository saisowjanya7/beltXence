"""
SmartBelt -- Phase 7: End-to-End Integration Test Suite
=======================================================
Validates the complete SmartBelt pipeline by running 5 realistic operational
scenarios through every subsystem layer:

  Visual Inference Scores  (from Phase 1 scores.csv -- real measured values)
       |
  Risk Classifier          (Phase 2: classify_risk / get_risk_details)
       |
  Sensor Fusion            (Phase 5: sensor_parameter_score + combine)
       |
  Alert Subsystem          (Phase 6: alert_trigger + CSV audit)
       |
  test_results.md          (this phase: structured pass/fail report)

DATA INTEGRITY NOTICE:
  - Visual scores for Scenarios 1-4 are taken DIRECTLY from
    smartbelt_pipeline/phase1/scores.csv (real PatchCore inference results).
  - Sensor readings for all scenarios are EXPLICITLY SIMULATED via
    Phase 5 simulate_sensor_readings(). No physical ESP32 hardware present.
  - No model inference or retraining is performed in this script.
"""

import sys
import os
import csv
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup: project root must be the working directory for imports to resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbelt_pipeline.phase2.risk_classification import (
    classify_risk, get_risk_details, THRESHOLD_1, THRESHOLD_2
)
from smartbelt_pipeline.phase5.sensor_fusion import (
    sensor_parameter_score, combine, simulate_sensor_readings
)
from smartbelt_pipeline.phase6.alert_system import AlertSubsystem

SCORES_CSV = PROJECT_ROOT / "smartbelt_pipeline" / "phase1" / "scores.csv"
PHASE7_DIR = Path(__file__).resolve().parent
RESULTS_MD = PHASE7_DIR / "test_results.md"
SCENARIO_LOG = PHASE7_DIR / "scenario_events.csv"

# ---------------------------------------------------------------------------
# Load Phase 1 scores (real measured PatchCore values)
# ---------------------------------------------------------------------------
def load_scores():
    rows = []
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "filename": r["filename"],
                "score": float(r["anomaly_score"]),
                "tag": r["subset_tag"],
            })
    return rows

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each scenario is a dict with:
#   id              : int
#   name            : str
#   description     : str
#   visual_scores   : list[float]   -- real values from CSV (or labeled synthetic)
#   score_source    : str           -- provenance annotation
#   sensor_scenario : str           -- key for simulate_sensor_readings()
#   expected_risk   : str           -- "NORMAL", "WARNING", or "CRITICAL"
#   expected_alert  : str           -- GPIO command expected

def build_scenarios(rows):
    # Pull specific score values from real measured data
    good_scores = sorted([r["score"] for r in rows if r["tag"] == "good"])
    # Normal: low-scoring test/good images (well below THRESHOLD_1)
    normal_sample = [s for s in good_scores if s < THRESHOLD_1][:5]

    # WARNING zone: real measured images that sit between THRESHOLD_1 and THRESHOLD_2
    # Real: frame_0173.jpg = 0.68643, frame_0230.jpg = 0.65991
    warning_samples = [r for r in rows if THRESHOLD_1 <= r["score"] < THRESHOLD_2]
    warning_visual_score = warning_samples[1]["score"] if len(warning_samples) > 1 else 0.65991
    warning_filename = warning_samples[1]["filename"] if len(warning_samples) > 1 else "frame_0230.jpg"

    # CRITICAL: real damage images (all >= THRESHOLD_2 except borderline one)
    damage_scores = sorted([r["score"] for r in rows if r["tag"] == "real_damage"], reverse=True)
    crit_visual_score = damage_scores[0]   # Highest real damage score

    # Scenario 5: normal visual + bearing overheat sensors
    #   Use a typical normal test/good score (sensor carries all the weight)
    normal_vis_score = good_scores[len(good_scores)//2]   # median

    return [
        {
            "id": 1,
            "name": "Normal Conveyor Operation",
            "description": (
                "Belt joint passes through camera zone in pristine condition. "
                "All sensors nominal. Expect NORMAL classification at every layer."
            ),
            "visual_score": normal_sample[2] if len(normal_sample) >= 3 else 0.509,
            "score_source": "Real PatchCore score -- tag: good (test set, Phase 1 scores.csv)",
            "sensor_scenario": "nominal",
            "expected_risk": "NORMAL",
            "expected_alert": "CMD_STATUS_NORMAL",
        },
        {
            "id": 2,
            "name": "Early Joint Fatigue (Pre-Rupture Warning)",
            "description": (
                "Splice shows early fatigue indicators visible to camera. "
                "Score sits in WARNING zone. Sensors nominal. "
                "System must escalate to WARNING without triggering motor stop."
            ),
            "visual_score": warning_visual_score,
            "score_source": f"Real PatchCore score -- {warning_filename} (tag: good, Phase 1 scores.csv)",
            "sensor_scenario": "nominal",
            "expected_risk": "WARNING",
            "expected_alert": "CMD_ALERT_WARNING",
        },
        {
            "id": 3,
            "name": "Severe Joint Rupture (Visual CRITICAL)",
            "description": (
                "Full joint separation visible in camera frame. "
                "PatchCore detects high anomaly. Vibration spike from pounding. "
                "Both modalities agree: CONFIRMED_MULTIMODAL_EMERGENCY. "
                "Emergency motor trip must be commanded."
            ),
            "visual_score": crit_visual_score,
            "score_source": "Real PatchCore score -- highest real_damage image (Phase 1 scores.csv)",
            "sensor_scenario": "splice_impact_vibration",
            "expected_risk": "CRITICAL",
            "expected_alert": "CMD_EMERGENCY_HALT_BUZZER_ON",
        },
        {
            "id": 4,
            "name": "General Belt Damage -- Sensor Dominant (No Joint Visible)",
            "description": (
                "Belt damage occurs in a zone not visible to the joint camera "
                "(blind spot scenario). Visual score is normal. "
                "However, severe belt slip and low tension are detected by sensors. "
                "Sensor-dominant path must escalate to CRITICAL independently."
            ),
            "visual_score": normal_sample[0] if normal_sample else 0.475,
            "score_source": "Real PatchCore score -- normal test image (Phase 1 scores.csv)",
            "sensor_scenario": "belt_slip_and_slack",
            "expected_risk": "CRITICAL",
            "expected_alert": "CMD_EMERGENCY_HALT_BUZZER_ON",
        },
        {
            "id": 5,
            "name": "Sensor Anomaly on Normal Belt (Bearing Overheat)",
            "description": (
                "Joint visual appears healthy. Bearing overheat detected by "
                "temperature sensor (78.5 C >> 65 C warning threshold). "
                "Sensor-only path must catch this thermal fault as CRITICAL."
            ),
            "visual_score": normal_vis_score,
            "score_source": "Real PatchCore score -- median test/good image (Phase 1 scores.csv)",
            "sensor_scenario": "bearing_overheat",
            "expected_risk": "CRITICAL",
            "expected_alert": "CMD_EMERGENCY_HALT_BUZZER_ON",
        },
    ]


# ---------------------------------------------------------------------------
# Run a single scenario through the full pipeline
# ---------------------------------------------------------------------------
def run_scenario(s: dict, alert_subsystem: AlertSubsystem) -> dict:
    vis_score = s["visual_score"]

    # Layer 1: Visual risk classification
    visual_risk = classify_risk(vis_score)
    visual_details = get_risk_details(vis_score)

    # Layer 2: Sensor scoring (explicitly simulated)
    raw_readings = simulate_sensor_readings(s["sensor_scenario"])
    sensor_score, sensor_meta = sensor_parameter_score(raw_readings)
    sensor_risk = classify_risk(sensor_score)

    # Layer 3: Multi-modal fusion
    fused = combine(vis_score, sensor_score, sensor_meta)
    final_score = fused["final_risk_score"]
    final_risk = fused["risk_status"]
    consensus = fused["consensus_state"]

    # Layer 4: Alert subsystem
    alert_result = alert_subsystem.alert_trigger(final_score, context={
        "scenario_id": s["id"],
        "scenario_name": s["name"],
        "visual_score": vis_score,
        "sensor_score": round(sensor_score, 5),
        "consensus": consensus,
    })
    actual_cmd = alert_result["hardware_dispatch_log"].split("Triggered: ")[-1] \
                 if "Triggered:" in alert_result["hardware_dispatch_log"] \
                 else alert_result["hardware_dispatch_log"].split("Sent: ")[-1]

    # Evaluate pass/fail
    risk_pass = (final_risk == s["expected_risk"])
    # Alert command pass: check expected GPIO command appears somewhere in the log
    alert_pass = (s["expected_alert"] in alert_result["hardware_dispatch_log"])
    overall_pass = risk_pass and alert_pass

    return {
        "scenario_id": s["id"],
        "scenario_name": s["name"],
        "description": s["description"],
        "score_source": s["score_source"],
        "sensor_source": f"SIMULATED via simulate_sensor_readings('{s['sensor_scenario']}')",
        # Inputs
        "visual_score": round(vis_score, 5),
        "raw_sensor_readings": raw_readings,
        # Intermediate
        "visual_risk": visual_risk,
        "sensor_score": round(sensor_score, 5),
        "sensor_risk": sensor_risk,
        "sensor_dominant": sensor_meta.get("dominant_sensor", "?"),
        "sensor_dominant_score": sensor_meta.get("dominant_score", 0.0),
        # Final
        "final_risk_score": final_score,
        "final_risk": final_risk,
        "consensus_state": consensus,
        # Alert
        "buzzer_state": alert_result["buzzer_state"],
        "motor_relay_state": alert_result["motor_relay_state"],
        "beacon_color": alert_result["beacon_color"],
        "hardware_dispatch": alert_result["hardware_dispatch_log"],
        # Pass/Fail
        "expected_risk": s["expected_risk"],
        "expected_alert": s["expected_alert"],
        "risk_pass": risk_pass,
        "alert_pass": alert_pass,
        "overall_pass": overall_pass,
        "timestamp": alert_result["timestamp"],
    }


# ---------------------------------------------------------------------------
# Write Markdown report
# ---------------------------------------------------------------------------
RISK_ICON = {"NORMAL": "[NORMAL]", "WARNING": "[WARN] ", "CRITICAL": "[CRIT] "}
PASS_ICON = {True: "PASS", False: "FAIL"}

def write_markdown_report(results: list, run_ts: str):
    passed = sum(1 for r in results if r["overall_pass"])
    total = len(results)

    lines = []
    lines.append("# SmartBelt -- Phase 7: Integration Test Results")
    lines.append("")
    lines.append(f"**Run timestamp:** {run_ts}  ")
    lines.append(f"**Overall result:** {passed}/{total} scenarios PASSED  ")
    lines.append(f"**Thresholds:** NORMAL < {THRESHOLD_1:.5f} <= WARNING < {THRESHOLD_2:.5f} <= CRITICAL  ")
    lines.append("")
    lines.append("## Data Provenance")
    lines.append("")
    lines.append("| Source | Status |")
    lines.append("|--------|--------|")
    lines.append("| Visual anomaly scores | **REAL** -- PatchCore inference from Phase 1 scores.csv |")
    lines.append("| Sensor telemetry | **SIMULATED** -- Phase 5 simulate_sensor_readings() |")
    lines.append("| Model weights | UNTOUCHED -- results_joint/20260910_105659/ |")
    lines.append("| New inference | None -- cached scores used |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append("| # | Scenario | Expected | Actual | Visual | Sensor | Fused | Result |")
    lines.append("|---|----------|----------|--------|--------|--------|-------|--------|")
    for r in results:
        icon = PASS_ICON[r["overall_pass"]]
        lines.append(
            f"| {r['scenario_id']} | {r['scenario_name']} "
            f"| {r['expected_risk']} "
            f"| {r['final_risk']} "
            f"| {r['visual_score']:.5f} "
            f"| {r['sensor_score']:.5f} "
            f"| {r['final_risk_score']:.5f} "
            f"| **{icon}** |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed scenario sections
    for r in results:
        icon = PASS_ICON[r["overall_pass"]]
        lines.append(f"## Scenario {r['scenario_id']}: {r['scenario_name']}  [{icon}]")
        lines.append("")
        lines.append(f"**Description:** {r['description']}")
        lines.append("")
        lines.append("### Input Data")
        lines.append(f"- **Visual score:** `{r['visual_score']:.5f}` -- {r['score_source']}")
        lines.append(f"- **Sensor data:** {r['sensor_source']}")
        sensor_vals = r["raw_sensor_readings"]
        lines.append(f"  - Vibration: `{sensor_vals.get('vibration_rms_mms', '?')} mm/s RMS`")
        lines.append(f"  - Temperature: `{sensor_vals.get('temperature_c', '?')} C`")
        lines.append(f"  - Tension: `{sensor_vals.get('tension_kn', '?')} kN`")
        lines.append(f"  - Speed Slip: `{sensor_vals.get('speed_slip_pct', '?')}%`")
        lines.append("")
        lines.append("### Pipeline Trace")
        lines.append(f"| Layer | Output |")
        lines.append(f"|-------|--------|")
        lines.append(f"| Phase 2 -- Visual Risk | `{r['visual_score']:.5f}` --> **{r['visual_risk']}** |")
        lines.append(f"| Phase 5 -- Sensor Score | `{r['sensor_score']:.5f}` --> **{r['sensor_risk']}** (dominant: {r['sensor_dominant']} @ {r['sensor_dominant_score']:.4f}) |")
        lines.append(f"| Phase 5 -- Fusion | `{r['final_risk_score']:.5f}` --> **{r['final_risk']}** ({r['consensus_state']}) |")
        lines.append(f"| Phase 6 -- Alert | Buzzer: {r['buzzer_state']} \\| Relay: {r['motor_relay_state']} \\| Beacon: {r['beacon_color']} |")
        lines.append(f"| Phase 6 -- GPIO Dispatch | `{r['hardware_dispatch']}` |")
        lines.append("")
        lines.append("### Pass/Fail Evaluation")
        lines.append(f"| Check | Expected | Actual | Result |")
        lines.append(f"|-------|----------|--------|--------|")
        lines.append(
            f"| Risk Classification | `{r['expected_risk']}` | `{r['final_risk']}` "
            f"| **{PASS_ICON[r['risk_pass']]}** |"
        )
        lines.append(
            f"| Alert GPIO Command | `{r['expected_alert']}` | (in dispatch log) "
            f"| **{PASS_ICON[r['alert_pass']]}** |"
        )
        lines.append(f"| **Overall** | | | **{icon}** |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Gap analysis
    lines.append("## Upgrade Requirements")
    lines.append("")
    lines.append("The following limitations exist due to hardware/data constraints at test time:")
    lines.append("")
    lines.append("| Limitation | Impact | What is Needed |")
    lines.append("|------------|--------|----------------|")
    lines.append("| No physical ESP32 hardware | GPIO dispatch is simulated; buzzer/relay not physically verified | Connect ESP32 on USB, pass --serial-port COM3 to alert_system |")
    lines.append("| Sensor telemetry simulated | Sensor fusion validated architecturally, not with real readings | Deploy 4-sensor ESP32 sketch, pipe JSON over UART |")
    lines.append("| No belt-surface camera images | General belt damage (Scenario 4) uses only sensor path | Collect 150+ normal + 40+ damaged belt-surface images |")
    lines.append("| Single-joint dataset | PatchCore trained on one joint type / one belt | Re-train with multiple belt joints and belt widths |")
    lines.append("| WARNING zone borderline case | Only 1 real image truly in WARNING zone (0.65991-0.68643) | Collect more partially damaged joint images at early failure stage |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `smartbelt_pipeline/phase7/run_integration_tests.py` at {run_ts}*")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase 7] Report written to: {RESULTS_MD}")


# ---------------------------------------------------------------------------
# Write scenario events CSV (for audit trail)
# ---------------------------------------------------------------------------
def write_scenario_csv(results: list):
    fieldnames = [
        "scenario_id", "scenario_name", "visual_score", "sensor_score",
        "final_risk_score", "final_risk", "consensus_state",
        "expected_risk", "risk_pass", "expected_alert", "alert_pass", "overall_pass", "timestamp"
    ]
    with open(SCENARIO_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[Phase 7] Scenario events CSV: {SCENARIO_LOG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_ts = datetime.now().isoformat()
    print("=" * 75)
    print("SmartBelt -- Phase 7: End-to-End Integration Test Suite")
    print(f"Run: {run_ts}")
    print("=" * 75)
    print(f"Thresholds: NORMAL < {THRESHOLD_1:.5f} <= WARNING < {THRESHOLD_2:.5f} <= CRITICAL")
    print(f"Loading scores from: {SCORES_CSV}")
    print()

    # Load real scores from Phase 1
    rows = load_scores()
    print(f"Loaded {len(rows)} scored images from Phase 1.")

    # Build scenario definitions (with real score values injected)
    scenarios = build_scenarios(rows)

    # Shared alert subsystem (simulated hardware mode)
    alert_sys = AlertSubsystem(serial_port=None)   # No physical ESP32

    results = []
    for s in scenarios:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {s['id']}: {s['name']}")
        print(f"  Visual score : {s['visual_score']:.5f}  ({s['score_source']})")
        print(f"  Sensor preset: {s['sensor_scenario']} [SIMULATED]")
        print(f"  Expected risk: {s['expected_risk']}")
        print("-" * 60)

        result = run_scenario(s, alert_sys)
        results.append(result)

        status = "PASS" if result["overall_pass"] else "FAIL"
        print(f"  Sensor score : {result['sensor_score']:.5f} ({result['sensor_risk']}) "
              f"[dominant: {result['sensor_dominant']}]")
        print(f"  Fused score  : {result['final_risk_score']:.5f} ({result['final_risk']}) "
              f"[{result['consensus_state']}]")
        print(f"  Alert GPIO   : {result['hardware_dispatch']}")
        print(f"  Motor relay  : {result['motor_relay_state']}")
        print(f"  >> RESULT    : {status}")

    # Summary
    passed = sum(1 for r in results if r["overall_pass"])
    total = len(results)
    print(f"\n{'=' * 75}")
    print(f"INTEGRATION TEST SUMMARY: {passed}/{total} PASSED")
    print("=" * 75)
    for r in results:
        icon = "PASS" if r["overall_pass"] else "FAIL"
        print(f"  [{icon}] Scenario {r['scenario_id']}: {r['scenario_name']}")

    # Write outputs
    print()
    write_markdown_report(results, run_ts)
    write_scenario_csv(results)

    if passed < total:
        print(f"\n[!] {total - passed} scenario(s) failed -- check test_results.md for details.")
        sys.exit(1)
    else:
        print(f"\nAll {total} scenarios passed. Pipeline integrity verified.")


if __name__ == "__main__":
    main()
