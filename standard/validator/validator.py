"""
Integrity Validator (spec §6): combines all 4 modules into one PASS /
REVIEW / REJECT verdict + reason codes, the same aggregation rule as the
Segmentation Gate (any REJECT wins, else any violation means REVIEW,
else PASS) - one severity model for the whole pipeline (standard.severity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from standard.objects.food_object import FoodObject
from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.validator.module_a import validate_implementation_integrity
from standard.validator.module_b import validate_content_integrity
from standard.validator.module_c import validate_appearance_integrity
from standard.validator.module_d import validate_layout_integrity


@dataclass
class ValidatorResult:
    result: GateResult
    reasons: list[ReasonCode]


def run_validator(
    *,
    original_mask: np.ndarray,
    final_mask: np.ndarray,
    expected_food_rgb: np.ndarray,
    final_food_rgb: np.ndarray,
    food_mask: np.ndarray,
    original_food_rgb: np.ndarray,
    original_objects: list[FoodObject],
    final_objects: list[FoodObject],
    policy: Policy,
    used_grid_fallback: bool = False,
) -> ValidatorResult:
    violations: list[tuple[GateResult, ReasonCode]] = []
    violations += validate_implementation_integrity(original_mask, final_mask, policy)
    violations += validate_content_integrity(expected_food_rgb, final_food_rgb, food_mask, policy)
    violations += validate_appearance_integrity(original_food_rgb, final_food_rgb, food_mask, policy)
    violations += validate_layout_integrity(original_objects, final_objects, policy, used_grid_fallback=used_grid_fallback)

    if any(verdict == GateResult.REJECT for verdict, _ in violations):
        result = GateResult.REJECT
    elif violations:
        result = GateResult.REVIEW
    else:
        result = GateResult.PASS

    return ValidatorResult(result=result, reasons=[reason for _, reason in violations])
