"""
SmartBelt — Phase 4: Dual-Branch Visual Score Fusion
Combines the Joint-Specific Anomaly Model with the General Belt Surface Model.

Since conveyor failures can occur at either the joint splice (rupture risk)
or across the continuous belt surface (rip, puncture, abrasion), SmartBelt
employs a conservative dual-branch visual fusion architecture.

Fusion Principle:
    Industrial conveyor integrity operates on an OR-condition: severe damage
    to EITHER the joint OR the continuous belt surface demands immediate
    preventive response. Thus, conservative_max is the primary fusion logic:
        final_visual_score = max(joint_score, belt_score)
"""

import sys
import argparse
from typing import Dict, Any, Optional

# Risk Thresholds from Phase 2
THRESHOLD_1 = 0.62710  # NORMAL -> WARNING
THRESHOLD_2 = 0.68887  # WARNING -> CRITICAL


def classify_risk(score: float) -> str:
    """Classify anomaly score into NORMAL, WARNING, or CRITICAL."""
    if score < THRESHOLD_1:
        return "NORMAL"
    elif score < THRESHOLD_2:
        return "WARNING"
    else:
        return "CRITICAL"


def combined_visual_score(
    joint_score: float,
    belt_score: Optional[float] = None,
    fusion_mode: str = "conservative_max"
) -> Dict[str, Any]:
    """
    Fuse the visual anomaly scores from the Joint Model and Belt Surface Model.

    Args:
        joint_score (float): Anomaly score from patchcore_joint_v1 (WideResNet-50).
        belt_score (float, optional): Anomaly score from general belt surface model.
                                     If None, branch is logged as unpopulated.
        fusion_mode (str): Fusion policy. Options:
                           - "conservative_max" (default): max(joint, belt)
                           - "weighted_priority": 0.65 * joint + 0.35 * belt (if joint rupture is highest criticality)

    Returns:
        dict: Detailed fusion telemetry containing:
            - final_visual_score (float)
            - joint_score (float)
            - belt_score (float or None)
            - dominant_failure_mode (str: "JOINT_RUPTURE", "BELT_SURFACE", or "NOMINAL")
            - risk_status (str: "NORMAL", "WARNING", "CRITICAL")
            - alert_action (str)
    """
    if belt_score is None:
        # Belt branch is not yet connected or data pending
        final_score = float(joint_score)
        dominant = "JOINT_RUPTURE" if joint_score >= THRESHOLD_1 else "NOMINAL"
        belt_status = "UNPOPULATED (Awaiting Belt Surface Training Data)"
    else:
        belt_status = f"{belt_score:.5f}"
        if fusion_mode == "conservative_max":
            final_score = max(float(joint_score), float(belt_score))
        elif fusion_mode == "weighted_priority":
            # Joint rupture is structurally catastrophic, given higher weight
            final_score = max(float(joint_score), 0.65 * float(joint_score) + 0.35 * float(belt_score))
        else:
            final_score = max(float(joint_score), float(belt_score))

        if final_score < THRESHOLD_1:
            dominant = "NOMINAL"
        elif joint_score >= belt_score:
            dominant = "JOINT_RUPTURE"
        else:
            dominant = "BELT_SURFACE_DAMAGE"

    risk = classify_risk(final_score)

    if risk == "CRITICAL":
        action = f"EMERGENCY MOTOR TRIP — Critical {dominant} detected!"
    elif risk == "WARNING":
        action = f"PREVENTIVE MAINTENANCE ALERT — Inspect {dominant} at scheduled shift change."
    else:
        action = "CONTINUE NOMINAL OPERATION — Belt surface and joint verified intact."

    return {
        "final_visual_score": round(final_score, 5),
        "joint_score": round(joint_score, 5),
        "belt_score": round(belt_score, 5) if belt_score is not None else None,
        "belt_branch_status": belt_status,
        "dominant_failure_mode": dominant,
        "risk_status": risk,
        "alert_action": action
    }


def run_demo_scenarios():
    """Demonstrate the dual-branch fusion logic across synthetic test scenarios."""
    print("=" * 75)
    print("SmartBelt — Dual-Branch Visual Fusion Scenarios")
    print("=" * 75)
    print(f"Thresholds: NORMAL < {THRESHOLD_1:.4f} <= WARNING < {THRESHOLD_2:.4f} <= CRITICAL\n")

    scenarios = [
        ("Scenario 1: Nominal Belt & Splice", 0.465, 0.420),
        ("Scenario 2: Incipient Splice Fatigue (Joint Warning, Belt Clean)", 0.645, 0.440),
        ("Scenario 3: Longitudinal Belt Rip (Belt Critical, Joint Clean)", 0.480, 0.760),
        ("Scenario 4: Catastrophic Joint Rupture (Joint Critical, Belt Clean)", 0.742, 0.510),
        ("Scenario 5: Multi-Point Failure (Joint Critical + Belt Surface Tears)", 0.785, 0.810),
        ("Scenario 6: Belt Model Unpopulated (Stand-Alone Joint Monitoring)", 0.725, None),
    ]

    print(f"{'SCENARIO':<40} | {'JOINT':>6} | {'BELT':>6} | {'FUSED':>6} | {'RISK':<9} | {'DOMINANT'}")
    print("-" * 75)

    for desc, j_sc, b_sc in scenarios:
        res = combined_visual_score(j_sc, b_sc)
        b_str = f"{b_sc:.3f}" if b_sc is not None else " N/A "
        print(f"{desc:<40} | {j_sc:6.3f} | {b_str:>6} | {res['final_visual_score']:6.3f} | {res['risk_status']:<9} | {res['dominant_failure_mode']}")

    print("-" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBelt Visual Score Fusion")
    parser.add_argument("--joint-score", "-j", type=float, default=None, help="PatchCore joint anomaly score")
    parser.add_argument("--belt-score", "-b", type=float, default=None, help="Belt surface anomaly score (optional)")
    parser.add_argument("--demo", action="store_true", help="Run multi-scenario architectural test suite")

    args = parser.parse_args()

    if args.demo or (args.joint_score is None and args.belt_score is None):
        run_demo_scenarios()
    else:
        j = args.joint_score if args.joint_score is not None else 0.50
        res = combined_visual_score(j, args.belt_score)
        print("\nFused Visual Assessment:")
        for k, v in res.items():
            print(f"  {k}: {v}")
