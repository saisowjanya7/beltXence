"""
SmartBelt v2 — Multi-Modal Risk Fusion Layer
Combines real visual anomaly score (PatchCore) and physical sensor score
into a conservative fused risk score and operational decision state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from smartbelt.inference.patchcore_engine import THRESHOLD_1, THRESHOLD_2


class FusionState(str, Enum):
    NOMINAL                         = "NOMINAL_OPERATION"
    ELEVATED                        = "ELEVATED_INSPECTION_ZONE"
    VISION_DOMINANT                 = "VISION_DOMINANT_FAILURE"
    SENSOR_DOMINANT                 = "SENSOR_DOMINANT_MECHANICAL_DISTRESS"
    MULTIMODAL_EMERGENCY            = "CONFIRMED_MULTIMODAL_EMERGENCY"


ALERT_LEVEL = {
    FusionState.NOMINAL:             0,   # Nominal / Green
    FusionState.ELEVATED:            1,   # Warning / Yellow
    FusionState.VISION_DOMINANT:     2,   # High / Orange
    FusionState.SENSOR_DOMINANT:     2,   # High / Orange
    FusionState.MULTIMODAL_EMERGENCY: 3,  # Critical / Red
}


@dataclass
class FusionResult:
    """Output of multimodal risk fusion."""
    visual_score:   float = 0.0
    sensor_score:   float = 0.0
    fused_score:    float = 0.0
    state:          FusionState = FusionState.NOMINAL
    alert_level:    int = 0            # 0=Nominal, 1=Elevated, 2=High, 3=Emergency
    dominant:       str = "none"       # 'vision' | 'sensor' | 'both' | 'none'


def fuse_risk(
    visual_score: float,
    sensor_score: float,
    ema_alpha: float = 0.75,
    _ema_v: Optional[list] = None,
    _ema_s: Optional[list] = None,
) -> FusionResult:
    """
    Fuse visual and sensor anomaly scores.
    Applies exponential moving average (EMA) if state lists are provided.
    """
    if _ema_v is not None:
        _ema_v[0] = visual_score if _ema_v[0] is None else \
                    (ema_alpha * _ema_v[0] + (1.0 - ema_alpha) * visual_score)
        visual_score = _ema_v[0]
    if _ema_s is not None:
        _ema_s[0] = sensor_score if _ema_s[0] is None else \
                    (ema_alpha * _ema_s[0] + (1.0 - ema_alpha) * sensor_score)
        sensor_score = _ema_s[0]

    v_crit = visual_score >= THRESHOLD_2
    s_crit = sensor_score >= THRESHOLD_2
    v_elev = visual_score >= THRESHOLD_1
    s_elev = sensor_score >= THRESHOLD_1

    if v_crit and s_crit:
        state = FusionState.MULTIMODAL_EMERGENCY
        fused = min(1.0, max(visual_score, sensor_score) + 0.05)
        dominant = "both"
    elif v_crit:
        state = FusionState.VISION_DOMINANT
        fused = visual_score
        dominant = "vision"
    elif s_crit:
        state = FusionState.SENSOR_DOMINANT
        fused = sensor_score
        dominant = "sensor"
    elif v_elev or s_elev:
        state = FusionState.ELEVATED
        fused = max(visual_score, sensor_score)
        dominant = "vision" if visual_score >= sensor_score else "sensor"
    else:
        state = FusionState.NOMINAL
        fused = 0.6 * visual_score + 0.4 * sensor_score
        dominant = "none"

    return FusionResult(
        visual_score=visual_score,
        sensor_score=sensor_score,
        fused_score=fused,
        state=state,
        alert_level=ALERT_LEVEL[state],
        dominant=dominant,
    )
