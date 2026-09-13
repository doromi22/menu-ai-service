"""
Module D - Layout Integrity (spec §6): reuses the Layout Engine's own
rank/occlusion *measurement* functions as-is (spec §4's rank/occlusion
rules were built there first, §11 step 2), and applies this module's own
Policy-driven severity classification on top - the Layout Engine's own
`occlusion_change_violations` is a separate, simpler >threshold flag it
uses for its own early heads-up (see standard/layout/occlusion.py's
docstring); this is the one authoritative, Policy-governed verdict.

Object-area-rank reversal, and the grid-fallback flag from the Layout
Engine (`used_grid_fallback`, spec §2: "unknown 비율 > 50%인 프레임은 ...
자동 REVIEW"), both have no severity tag in policy.yaml by design (§2/§4
fix their outcome to REVIEW outright - see standard/policy/policy.yaml's
comment), so both are appended as a fixed REVIEW rather than run through
`classify_violation`.

`preserve_front_back_order` (§4) is folded into OCCLUSION_CHANGED here
rather than scored separately: §4 states both in the same sentence with
the same REVIEW consequence, and §6's enum has no dedicated code for
front/back order alone. Nothing built so far (Layout Engine never
reorders z_index) can actually trigger this branch yet - not silently
skipped, just nothing live to check.
"""
from __future__ import annotations

from standard.layout.occlusion import compute_occlusion_changes
from standard.layout.rank import is_area_rank_preserved
from standard.objects.food_object import OCCLUSION_BASELINE, FoodObject
from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.severity import classify_violation


def compute_scale_deltas(objects: list[FoodObject]) -> dict[str, float]:
    return {obj.id: abs(obj.scale - 1.0) for obj in objects}


def compute_max_relative_scale_ratio_change(objects: list[FoodObject]) -> float:
    """
    Largest pairwise deviation from a 1:1 scale ratio between any two
    objects. Not given an exact formula by spec §6 (only the
    `max_relative_scale_ratio_change` threshold value is given) -
    provisional, like Module B's HF-error formula.
    """
    scales = [obj.scale for obj in objects]
    max_diff = 0.0
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            if scales[j] == 0:
                continue
            ratio = scales[i] / scales[j]
            max_diff = max(max_diff, abs(ratio - 1.0))
    return max_diff


def validate_layout_integrity(
    original_objects: list[FoodObject],
    final_objects: list[FoodObject],
    policy: Policy,
    *,
    used_grid_fallback: bool = False,
) -> list[tuple[GateResult, ReasonCode]]:
    violations: list[tuple[GateResult, ReasonCode]] = []

    if used_grid_fallback:
        violations.append((GateResult.REVIEW, ReasonCode.LAYOUT_UNKNOWN_ROLE_MAJORITY))

    if not is_area_rank_preserved(original_objects, final_objects):
        violations.append((GateResult.REVIEW, ReasonCode.OBJECT_AREA_RANK_REVERSED))

    occlusion_changes = compute_occlusion_changes(final_objects, OCCLUSION_BASELINE)
    occlusion_threshold = policy.threshold("layout", "max_occlusion_change")
    occlusion_multiplier = policy.reject_multiplier("layout", "max_occlusion_change")
    for change in occlusion_changes.values():
        verdict = classify_violation(change, occlusion_threshold, "max", occlusion_multiplier)
        if verdict is not None:
            violations.append((verdict, ReasonCode.OCCLUSION_CHANGED))

    scale_threshold = policy.threshold("layout", "max_scale_delta_per_object")
    scale_multiplier = policy.reject_multiplier("layout", "max_scale_delta_per_object")
    for delta in compute_scale_deltas(final_objects).values():
        verdict = classify_violation(delta, scale_threshold, "max", scale_multiplier)
        if verdict is not None:
            violations.append((verdict, ReasonCode.OBJECT_SCALE_CHANGED))

    relative_scale_change = compute_max_relative_scale_ratio_change(final_objects)
    verdict = classify_violation(
        relative_scale_change,
        policy.threshold("layout", "max_relative_scale_ratio_change"),
        "max",
        policy.reject_multiplier("layout", "max_relative_scale_ratio_change"),
    )
    if verdict is not None:
        violations.append((verdict, ReasonCode.RELATIVE_SCALE_CHANGED))

    return violations
