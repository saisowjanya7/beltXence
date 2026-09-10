"""
SmartBelt — Risk Classification Module
Maps raw PatchCore joint anomaly scores into operational 3-tier risk states:
  - NORMAL   (score < 0.62710)
  - WARNING  (0.62710 <= score < 0.68887)
  - CRITICAL (score >= 0.68887)
"""

THRESHOLD_1 = 0.62710  # NORMAL -> WARNING boundary (Validation mean + 2*std)
THRESHOLD_2 = 0.68887  # WARNING -> CRITICAL boundary (Validation mean + 3*std)

RISK_COLORS = {
    "NORMAL":   (0, 255, 0),     # Green (BGR for OpenCV)
    "WARNING":  (0, 255, 255),   # Yellow
    "CRITICAL": (0, 0, 255),     # Red
}

def classify_risk(score: float) -> str:
    """
    Classify a continuous PatchCore Euclidean distance anomaly score
    into a discrete 3-tier industrial risk category.

    Args:
        score (float): PatchCore raw Euclidean distance score.

    Returns:
        str: "NORMAL", "WARNING", or "CRITICAL".
    """
    if score < THRESHOLD_1:
        return "NORMAL"
    elif score < THRESHOLD_2:
        return "WARNING"
    else:
        return "CRITICAL"

def get_risk_details(score: float) -> dict:
    """
    Return comprehensive classification details including risk status,
    color, confidence delta, and recommended operational response.
    """
    risk = classify_risk(score)
    if risk == "NORMAL":
        action = "Nominal operation. Continue automated monitoring."
        color_bgr = (0, 255, 0)
        color_hex = "#28a745"
    elif risk == "WARNING":
        action = "Elevated anomaly. Log joint for technician inspection. Increase sampling rate."
        color_bgr = (0, 255, 255)
        color_hex = "#ffc107"
    else:
        action = "CRITICAL RUPTURE RISK. Trigger emergency alert and halt conveyor."
        color_bgr = (0, 0, 255)
        color_hex = "#dc3545"

    return {
        "score": float(score),
        "risk_status": risk,
        "threshold_1": THRESHOLD_1,
        "threshold_2": THRESHOLD_2,
        "color_bgr": color_bgr,
        "color_hex": color_hex,
        "operational_action": action
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            val = float(sys.argv[1])
            res = get_risk_details(val)
            print(f"Score: {res['score']:.5f} -> Risk: {res['risk_status']}")
            print(f"Action: {res['operational_action']}")
        except ValueError:
            print("Usage: python risk_classification.py <anomaly_score>")
    else:
        print("SmartBelt Risk Classifier Loaded.")
        print(f"Threshold 1 (NORMAL -> WARNING):   {THRESHOLD_1:.5f}")
        print(f"Threshold 2 (WARNING -> CRITICAL): {THRESHOLD_2:.5f}")
