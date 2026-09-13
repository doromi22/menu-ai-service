import numpy as np

from standard.objects.food_object import FoodObject, SemanticRole


def _mask(h: int = 4, w: int = 4) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[1:3, 1:3] = True
    return m


def _make(**overrides) -> FoodObject:
    defaults = dict(
        id="f1",
        mask=_mask(),
        bbox=(1, 1, 2, 2),
        original_center=(1.5, 1.5),
        final_center=(1.5, 1.5),
    )
    defaults.update(overrides)
    return FoodObject(**defaults)


def test_low_confidence_role_falls_back_to_unknown():
    obj = _make(semantic_role=SemanticRole.MAIN.value, role_confidence=0.5)
    assert obj.semantic_role == SemanticRole.UNKNOWN.value


def test_high_confidence_role_is_kept():
    obj = _make(semantic_role=SemanticRole.MAIN.value, role_confidence=0.9)
    assert obj.semantic_role == SemanticRole.MAIN.value


def test_confidence_exactly_at_floor_is_kept():
    obj = _make(semantic_role=SemanticRole.SIDE.value, role_confidence=0.70)
    assert obj.semantic_role == SemanticRole.SIDE.value


def test_scale_within_bounds():
    assert _make(scale=1.0).is_scale_within_bounds is True
    assert _make(scale=1.03).is_scale_within_bounds is True
    assert _make(scale=1.05).is_scale_within_bounds is False
    assert _make(scale=0.90).is_scale_within_bounds is False


def test_area_counts_mask_pixels():
    assert _make().area == 4
