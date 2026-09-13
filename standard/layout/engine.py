"""
Layout Engine (spec §4): translation-first, deterministic. Wires together
the pure functions in this package and decides between the default
(recenter) and grid-fallback arrangement.

Note on what this engine can actually violate: `recenter_group` is a rigid
group translation, so it can never change area rank (masks aren't
resized) or occlusion (a shared shift doesn't change relative overlap).
`arrange_grid` also can't change area rank, but its per-object fit-scaling
*can* change occlusion between packed objects. The rank/occlusion/scale
*check* functions are intentionally still pure and independently callable
with any (original, final) object pair - not just what this engine
produces - because the Integrity Validator's Layout module (§6 module D,
§11 step 4) reuses them against whatever a future, less trivial layout
algorithm eventually does.
"""
from __future__ import annotations

from dataclasses import dataclass

from standard.layout.grid import arrange_grid
from standard.layout.occlusion import occlusion_change_violations
from standard.layout.rank import is_area_rank_preserved
from standard.layout.translate import recenter_group
from standard.objects.food_object import OCCLUSION_BASELINE, FoodObject, SemanticRole
from standard.reason_codes import ReasonCode

DEFAULT_UNKNOWN_RATIO_THRESHOLD = 0.5
DEFAULT_MAX_OCCLUSION_CHANGE = 0.15


def unknown_ratio(objects: list[FoodObject]) -> float:
    if not objects:
        return 0.0
    unknown_count = sum(1 for obj in objects if obj.semantic_role == SemanticRole.UNKNOWN.value)
    return unknown_count / len(objects)


@dataclass
class LayoutResult:
    objects: list[FoodObject]
    used_grid_fallback: bool
    unknown_ratio: float
    rank_preserved: bool
    occlusion_violations: dict[tuple[str, str], float]
    review_reasons: list[ReasonCode]


class LayoutEngine:
    def __init__(
        self,
        unknown_ratio_threshold: float = DEFAULT_UNKNOWN_RATIO_THRESHOLD,
        occlusion_baseline: dict[tuple[str, str], float] = OCCLUSION_BASELINE,
        max_occlusion_change: float = DEFAULT_MAX_OCCLUSION_CHANGE,
    ) -> None:
        self._unknown_ratio_threshold = unknown_ratio_threshold
        self._occlusion_baseline = occlusion_baseline
        self._max_occlusion_change = max_occlusion_change

    def run(self, objects: list[FoodObject], canvas_size: tuple[int, int]) -> LayoutResult:
        ratio = unknown_ratio(objects)
        used_grid_fallback = ratio > self._unknown_ratio_threshold

        final_objects = arrange_grid(objects, canvas_size) if used_grid_fallback else recenter_group(objects, canvas_size)

        rank_preserved = is_area_rank_preserved(objects, final_objects)
        occlusion_violations = occlusion_change_violations(
            final_objects, self._occlusion_baseline, self._max_occlusion_change
        )

        review_reasons: list[ReasonCode] = []
        if used_grid_fallback:
            review_reasons.append(ReasonCode.LAYOUT_UNKNOWN_ROLE_MAJORITY)
        if not rank_preserved:
            review_reasons.append(ReasonCode.OBJECT_AREA_RANK_REVERSED)
        if occlusion_violations:
            review_reasons.append(ReasonCode.OCCLUSION_CHANGED)

        return LayoutResult(
            objects=final_objects,
            used_grid_fallback=used_grid_fallback,
            unknown_ratio=ratio,
            rank_preserved=rank_preserved,
            occlusion_violations=occlusion_violations,
            review_reasons=review_reasons,
        )
