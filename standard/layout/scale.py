"""Scale clamping (spec §4: "Scale이 필요해도 0.97~1.03 범위만 허용")."""
from __future__ import annotations

from standard.objects.food_object import SCALE_MAX, SCALE_MIN


def clamp_scale(desired_scale: float) -> float:
    return max(SCALE_MIN, min(SCALE_MAX, desired_scale))
