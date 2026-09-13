"""
Fallback strategies (spec §7). Each one reuses existing Color Grade /
Layout Engine logic rather than reimplementing anything - only the call
arguments differ.
"""
from __future__ import annotations

from dataclasses import replace
from enum import Enum

from standard.color.presets import STANDARD, ColorGradeParams, food_params_for
from standard.layout.translate import recenter_group
from standard.objects.food_object import FoodObject


class Strategy(str, Enum):
    CONSERVATIVE_COLOR = "conservative_color"
    NEUTRAL_COLOR = "neutral_color"
    RECOMPUTE_TRANSLATION = "recompute_translation"
    FORCE_SCALE_1_0 = "force_scale_1_0"


# Both reuse standard.color.presets.food_params_for - the same function
# Color Grade itself uses to derive the food pass from a preset - just
# called with a smaller conservatism factor than STANDARD's own food pass
# (whatever STANDARD.food_params_for(FOOD_CONSERVATISM) normally is).
# NEUTRAL_COLOR_FACTOR = 0.0 collapses it to the exact identity transform:
# "stop touching color" is the literal, final fallback once even a
# conservative regrade hasn't worked.
CONSERVATIVE_COLOR_FACTOR = 0.25
NEUTRAL_COLOR_FACTOR = 0.0


def conservative_color_params() -> ColorGradeParams:
    return food_params_for(STANDARD, conservatism=CONSERVATIVE_COLOR_FACTOR)


def neutral_color_params() -> ColorGradeParams:
    return food_params_for(STANDARD, conservatism=NEUTRAL_COLOR_FACTOR)


def force_scale_1_0(objects: list[FoodObject]) -> list[FoodObject]:
    return [replace(obj, scale=1.0) for obj in objects]


def recompute_translation(
    objects: list[FoodObject], canvas_size: tuple[int, int], attempt_index: int
) -> list[FoodObject]:
    """
    Reruns Layout Engine's own `recenter_group`, then nudges every object
    away from the group centroid by a spacing factor that grows with
    `attempt_index` - each call is measurably less tightly packed than the
    last (spec §4's "spacing" DOF), which is the direction that reduces
    occlusion drift. `attempt_index` is the caller's own retry counter, not
    tracked here, so repeated calls never silently collapse to the same
    output.
    """
    recentered = recenter_group(objects, canvas_size)
    if not recentered:
        return recentered

    spacing_factor = 1.0 + 0.05 * (attempt_index + 1)  # 1.05, 1.10, 1.15, ...
    cx = sum(obj.final_center[0] for obj in recentered) / len(recentered)
    cy = sum(obj.final_center[1] for obj in recentered) / len(recentered)

    return [
        replace(
            obj,
            final_center=(
                cx + (obj.final_center[0] - cx) * spacing_factor,
                cy + (obj.final_center[1] - cy) * spacing_factor,
            ),
        )
        for obj in recentered
    ]
