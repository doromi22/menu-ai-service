from standard.layout.scale import clamp_scale
from standard.objects.food_object import SCALE_MAX, SCALE_MIN


def test_scale_within_bounds_is_unchanged():
    assert clamp_scale(1.0) == 1.0
    assert clamp_scale(0.99) == 0.99


def test_scale_exactly_at_bounds_is_unchanged():
    assert clamp_scale(SCALE_MIN) == SCALE_MIN  # 0.97
    assert clamp_scale(SCALE_MAX) == SCALE_MAX  # 1.03


def test_scale_below_minimum_is_clamped_up():
    assert clamp_scale(0.90) == SCALE_MIN


def test_scale_above_maximum_is_clamped_down():
    assert clamp_scale(1.10) == SCALE_MAX
