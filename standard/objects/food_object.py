"""
FoodObject data model (spec §2).

Two invariants from the spec live on the dataclass itself, not on callers,
so every producer/consumer gets them for free:

- role_confidence < 0.70 forces semantic_role back to "unknown" ("정확히
  맞히는 것보다 모르면 unknown으로 빠지는 것이 안전").
- scale is only ever *checked* against the 0.97-1.03 band, never clamped -
  the Layout Engine (§4) decides what to do with an out-of-band scale
  (e.g. raise OBJECT_SCALE_CHANGED via the Integrity Validator, §6).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

SCALE_MIN = 0.97
SCALE_MAX = 1.03
ROLE_CONFIDENCE_FLOOR = 0.70


class SemanticRole(str, Enum):
    MAIN = "main"
    RICE = "rice"
    SOUP = "soup"
    SIDE = "side"
    GARNISH = "garnish"
    UNKNOWN = "unknown"


# Expected occlusion ratio (share of A's area covered by B) between role
# pairs in a well-formed original photo. Consumed later by the Layout
# Engine / Layout Integrity validator (§4, §6) to detect occlusion drift;
# it lives here because it is keyed on FoodObject.semantic_role.
OCCLUSION_BASELINE: dict[tuple[str, str], float] = {
    (SemanticRole.MAIN.value, SemanticRole.SIDE.value): 0.18,
    (SemanticRole.RICE.value, SemanticRole.SOUP.value): 0.05,
}


@dataclass
class FoodObject:
    id: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    original_center: tuple[float, float]
    final_center: tuple[float, float]

    scale: float = 1.0
    rotation: float = 0.0
    z_index: int = 0

    semantic_role: str = SemanticRole.UNKNOWN.value
    role_confidence: float = 0.0

    shadow_profile: str = "default"

    def __post_init__(self) -> None:
        if self.role_confidence < ROLE_CONFIDENCE_FLOOR:
            self.semantic_role = SemanticRole.UNKNOWN.value

    @property
    def is_scale_within_bounds(self) -> bool:
        return SCALE_MIN <= self.scale <= SCALE_MAX

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))
