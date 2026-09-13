"""
Object-area-rank preservation (spec §4: "Object 간 상대 면적 순위(main >
rice > soup > side) 보존 - 순위 역전 시 REVIEW").

Pure functions, reused as-is by the Integrity Validator's Layout module
(spec §6 module D, §11 step 4) - no policy/severity involved here, since
§4 already fixes the outcome to REVIEW rather than leaving it tunable.
"""
from __future__ import annotations

from standard.objects.food_object import FoodObject, SemanticRole


def compute_role_area_rank(objects: list[FoodObject]) -> list[str]:
    """
    Distinct semantic roles present, ranked by total area descending (ties
    broken alphabetically for determinism). `unknown` objects are excluded
    - their role isn't trusted (§2), so they can't participate in a rank
    comparison that's supposed to be about *known* roles.
    """
    role_areas: dict[str, int] = {}
    for obj in objects:
        if obj.semantic_role == SemanticRole.UNKNOWN.value:
            continue
        role_areas[obj.semantic_role] = role_areas.get(obj.semantic_role, 0) + obj.area
    return sorted(role_areas, key=lambda role: (-role_areas[role], role))


def is_area_rank_preserved(original: list[FoodObject], final: list[FoodObject]) -> bool:
    return compute_role_area_rank(original) == compute_role_area_rank(final)
