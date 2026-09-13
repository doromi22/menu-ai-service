"""
Module A - Implementation Integrity (spec §6): mask IoU/area - catches
pipeline bugs (e.g. a future renderer accidentally resampling a food
mask), not intentional edits. In the current pipeline (Layout Engine only
ever sets final_center/scale/rotation, never touches `mask` itself),
original_mask and final_mask are the same array, so this always passes -
it exists to catch regressions once something downstream (the §9 renderer)
starts actually resampling pixels.

Contour/alpha checks are named in §6's module description but have no
corresponding policy threshold in §6's yaml - not scored in V1, same as
Module C's red/orange emphasis/highlight (see module_c.py).
"""
from __future__ import annotations

import numpy as np

from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.severity import classify_violation


def compute_mask_iou(original: np.ndarray, final: np.ndarray) -> float:
    union = np.logical_or(original, final).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(original, final).sum()
    return float(intersection / union)


def compute_mask_area_delta(original: np.ndarray, final: np.ndarray) -> float:
    original_area = int(original.sum())
    final_area = int(final.sum())
    if original_area == 0:
        return 0.0 if final_area == 0 else 1.0
    return abs(final_area - original_area) / original_area


def validate_implementation_integrity(
    original_mask: np.ndarray, final_mask: np.ndarray, policy: Policy
) -> list[tuple[GateResult, ReasonCode]]:
    violations: list[tuple[GateResult, ReasonCode]] = []

    iou = compute_mask_iou(original_mask, final_mask)
    verdict = classify_violation(iou, policy.threshold("mask", "min_iou"), "min", policy.reject_multiplier("mask", "min_iou"))
    if verdict is not None:
        violations.append((verdict, ReasonCode.MASK_CHANGED))

    area_delta = compute_mask_area_delta(original_mask, final_mask)
    verdict = classify_violation(
        area_delta, policy.threshold("mask", "max_area_delta"), "max", policy.reject_multiplier("mask", "max_area_delta")
    )
    if verdict is not None:
        violations.append((verdict, ReasonCode.MASK_CHANGED))

    return violations
