import numpy as np

from standard.templates.background import _fixed_texture, _gradient_field, render_background
from standard.templates.definitions import TEMPLATES, ShadowSpec, Template


def test_gradient_field_range_and_anchor():
    field = _gradient_field((50, 50), "top_left")
    assert field.shape == (50, 50)
    assert field.max() <= 1.0 and field.min() >= 0.0
    assert field[0, 0] == 1.0  # exactly at the anchor
    assert field[-1, -1] < 0.05  # far corner, near zero


def test_fixed_texture_is_bounded_and_deterministic():
    first = _fixed_texture((40, 40))
    second = _fixed_texture((40, 40))
    assert np.array_equal(first, second)
    assert first.max() <= 1.0 and first.min() >= -1.0


def test_render_background_shape_and_dtype():
    template = TEMPLATES["T01_warm_ivory"]
    bg = render_background(template, canvas_size=(64, 48))
    assert bg.shape == (48, 64, 3)
    assert bg.dtype == np.uint8


def test_render_background_is_deterministic():
    template = TEMPLATES["T01_warm_ivory"]
    first = render_background(template, canvas_size=(64, 48))
    second = render_background(template, canvas_size=(64, 48))
    assert np.array_equal(first, second)


def test_different_templates_render_differently():
    bg1 = render_background(TEMPLATES["T01_warm_ivory"], (64, 64))
    bg2 = render_background(TEMPLATES["T04_dark_premium"], (64, 64))
    assert not np.array_equal(bg1, bg2)


def test_gradient_direction_makes_anchor_corner_brighter_on_average():
    # Amplified strength + block-averaging so the signal is well clear of
    # the small, deterministic fixed-texture pattern layered on top.
    template = Template(
        id="test",
        base_color="#606060",
        gradient_direction="top_left",
        gradient_strength=0.5,
        texture_strength=0.01,
        shadow=ShadowSpec(opacity=0.1, blur=10, offset_y=5),
    )
    bg = render_background(template, canvas_size=(60, 60)).astype(np.float32)

    near_anchor = bg[0:5, 0:5].mean()
    far_corner = bg[-5:, -5:].mean()
    assert near_anchor > far_corner
