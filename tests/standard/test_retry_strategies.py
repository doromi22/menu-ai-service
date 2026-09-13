from standard.color.presets import STANDARD
from standard.retry.strategies import (
    conservative_color_params,
    force_scale_1_0,
    neutral_color_params,
    recompute_translation,
)
from tests.standard.helpers import make_food_object


def test_neutral_color_is_the_exact_identity_transform():
    params = neutral_color_params()
    assert params.brightness == 1.0
    assert params.contrast == 1.0
    assert params.saturation == 1.0
    assert params.temperature_strength == 0.0


def test_conservative_color_is_milder_than_standard_but_not_identity():
    conservative = conservative_color_params()
    assert conservative != neutral_color_params()
    assert abs(conservative.brightness - 1.0) < abs(STANDARD.brightness - 1.0)
    assert abs(conservative.contrast - 1.0) < abs(STANDARD.contrast - 1.0)
    assert conservative.brightness != 1.0  # still some correction, unlike neutral


def test_force_scale_1_0_resets_every_object():
    a = make_food_object("a", (0, 0, 9, 9), scale=1.06)
    b = make_food_object("b", (20, 20, 29, 29), scale=0.90)

    result = force_scale_1_0([a, b])

    assert all(obj.scale == 1.0 for obj in result)
    assert all(obj.is_scale_within_bounds for obj in result)


def test_recompute_translation_recenters_the_group():
    a = make_food_object("a", (0, 0, 19, 19), canvas_size=(100, 100))
    b = make_food_object("b", (20, 20, 39, 39), canvas_size=(100, 100))

    result = recompute_translation([a, b], canvas_size=(100, 100), attempt_index=0)

    group_cx = sum(o.final_center[0] for o in result) / 2
    group_cy = sum(o.final_center[1] for o in result) / 2
    assert abs(group_cx - 50) < 5
    assert abs(group_cy - 50) < 5


def test_recompute_translation_varies_by_attempt_index():
    a = make_food_object("a", (0, 0, 19, 19), canvas_size=(100, 100))
    b = make_food_object("b", (20, 20, 39, 39), canvas_size=(100, 100))

    first = recompute_translation([a, b], canvas_size=(100, 100), attempt_index=0)
    second = recompute_translation([a, b], canvas_size=(100, 100), attempt_index=1)

    first_a = next(o for o in first if o.id == "a")
    second_a = next(o for o in second if o.id == "a")
    assert first_a.final_center != second_a.final_center


def test_recompute_translation_empty_list():
    assert recompute_translation([], canvas_size=(100, 100), attempt_index=0) == []
