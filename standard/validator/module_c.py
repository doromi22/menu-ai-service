"""
Module C - Appearance Integrity (spec §6): saturation/luminance *gain*
(increase only, per the policy field names) of the food region, comparing
the actual original photo against the final output - unlike Module B,
this one legitimately compares against the true original, since it's
measuring how much the visible appearance changed overall, not verifying
a specific known transform.

"Red/orange emphasis, highlight" are named in §6's module description but
have no corresponding policy threshold in §6's yaml - not scored in V1,
same as Module A's contour/alpha gap.
"""
from __future__ import annotations

import numpy as np

from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.severity import classify_violation


def compute_mean_saturation(rgb: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    saturation = np.where(max_c > 0, (max_c - min_c) / np.where(max_c > 0, max_c, 1.0), 0.0)
    return float(saturation[mask].mean())


def compute_mean_luminance(rgb: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(luminance[mask].mean())


def validate_appearance_integrity(
    original_food_rgb: np.ndarray, final_food_rgb: np.ndarray, food_mask: np.ndarray, policy: Policy
) -> list[tuple[GateResult, ReasonCode]]:
    violations: list[tuple[GateResult, ReasonCode]] = []

    saturation_gain = max(
        0.0, compute_mean_saturation(final_food_rgb, food_mask) - compute_mean_saturation(original_food_rgb, food_mask)
    )
    verdict = classify_violation(
        saturation_gain,
        policy.threshold("food", "max_saturation_gain"),
        "max",
        policy.reject_multiplier("food", "max_saturation_gain"),
    )
    if verdict is not None:
        violations.append((verdict, ReasonCode.FOOD_SATURATION_EXCEEDED))

    luminance_gain = max(
        0.0, compute_mean_luminance(final_food_rgb, food_mask) - compute_mean_luminance(original_food_rgb, food_mask)
    )
    verdict = classify_violation(
        luminance_gain,
        policy.threshold("food", "max_luminance_gain"),
        "max",
        policy.reject_multiplier("food", "max_luminance_gain"),
    )
    if verdict is not None:
        violations.append((verdict, ReasonCode.FOOD_LUMINANCE_EXCEEDED))

    return violations
