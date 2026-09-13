"""
Occlusion-change measurement (spec §4: "Occlusion은 ... 원본 대비 가림 비율
변화가 크면 REVIEW").

Split into a raw measurement function (`compute_occlusion_changes`) and a
simple threshold flag (`occlusion_change_violations`) used by the Layout
Engine itself as an early heads-up. The Integrity Validator's Layout
module (§6 module D, §11 step 4) reuses the raw measurement and applies
the real Policy-driven severity classification
(`standard.severity.classify_violation`) instead of this module's flag -
so there is exactly one place occlusion geometry gets computed, and
exactly one place REJECT/REVIEW severity gets decided.

Convention: FoodObject.z_index - higher is front (occludes lower z_index
objects). Occlusion ratio for an ordered pair (front, back) is the
fraction of `back`'s area covered by `front`, matching spec §2:
`occlusion[(A,B)] = "A가 B를 가리는 비율"`.
"""
from __future__ import annotations

from standard.geometry import translate_mask_by_center_delta
from standard.objects.food_object import OCCLUSION_BASELINE, FoodObject, SemanticRole


def _placed_mask(obj: FoodObject):
    return translate_mask_by_center_delta(obj.mask, obj.original_center, obj.final_center)


def occlusion_ratio(front: FoodObject, back: FoodObject) -> float:
    front_mask = _placed_mask(front)
    back_mask = _placed_mask(back)
    back_area = back_mask.sum()
    if back_area == 0:
        return 0.0
    overlap = (front_mask & back_mask).sum()
    return float(overlap / back_area)


def _role_representative(objects: list[FoodObject], role: str) -> FoodObject | None:
    """
    First object with the given role. Occlusion pairs are role-level
    (§2's OCCLUSION_BASELINE is keyed by role, not object id); when more
    than one object shares a role this picks one representative rather
    than unioning masks across objects, which is a known simplification
    for V1 - documented rather than silently assumed.
    """
    for obj in objects:
        if obj.semantic_role == role:
            return obj
    return None


def compute_role_pair_occlusions(objects: list[FoodObject]) -> dict[tuple[str, str], float]:
    """Occlusion ratio for every (front_role, back_role) pair actually present, keyed by role."""
    roles_present = {obj.semantic_role for obj in objects if obj.semantic_role != SemanticRole.UNKNOWN.value}
    results: dict[tuple[str, str], float] = {}
    for front_role in roles_present:
        for back_role in roles_present:
            if front_role == back_role:
                continue
            front_obj = _role_representative(objects, front_role)
            back_obj = _role_representative(objects, back_role)
            if front_obj.z_index <= back_obj.z_index:
                continue  # front_obj isn't actually in front of back_obj
            results[(front_role, back_role)] = occlusion_ratio(front_obj, back_obj)
    return results


def compute_occlusion_changes(
    objects: list[FoodObject], baseline: dict[tuple[str, str], float] = OCCLUSION_BASELINE
) -> dict[tuple[str, str], float]:
    """|actual - baseline| for every role pair present in both `objects` and `baseline`. No severity judgement."""
    pair_ratios = compute_role_pair_occlusions(objects)
    return {pair: abs(ratio - baseline[pair]) for pair, ratio in pair_ratios.items() if pair in baseline}


def occlusion_change_violations(
    objects: list[FoodObject],
    baseline: dict[tuple[str, str], float] = OCCLUSION_BASELINE,
    max_change: float = 0.15,
) -> dict[tuple[str, str], float]:
    """Simple >threshold flag for the Layout Engine's own use - not severity-classified."""
    changes = compute_occlusion_changes(objects, baseline)
    return {pair: change for pair, change in changes.items() if change > max_change}
