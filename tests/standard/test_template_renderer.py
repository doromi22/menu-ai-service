import numpy as np

from standard.templates.definitions import TEMPLATES
from standard.templates.renderer import TemplateRenderer
from tests.standard.helpers import make_food_object

CANVAS = (60, 60)  # (width, height)


def test_food_pixels_placed_at_correct_location():
    obj = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS, z_index=0)
    source = np.zeros((60, 60, 3), dtype=np.uint8)
    source[10:30, 10:30] = (200, 50, 50)

    scene = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [obj], CANVAS, source)

    assert scene.shape == (60, 60, 3)
    assert tuple(scene[20, 20]) == (200, 50, 50)  # well inside the object, unaffected by any shadow


def test_rendering_is_deterministic():
    obj = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS)
    source = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)

    first = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [obj], CANVAS, source)
    second = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [obj], CANVAS, source)

    assert np.array_equal(first, second)


def test_scale_affects_placed_footprint_area():
    source = np.full((60, 60, 3), 255, dtype=np.uint8)
    full_scale = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS, scale=1.0)
    half_scale = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS, scale=0.5)

    scene_full = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [full_scale], CANVAS, source)
    scene_half = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [half_scale], CANVAS, source)

    # white (255,255,255) food pixels are distinguishable from the (darker) template background
    footprint_full = np.all(scene_full == 255, axis=-1).sum()
    footprint_half = np.all(scene_half == 255, axis=-1).sum()

    assert footprint_half < footprint_full
    ratio = footprint_half / footprint_full
    assert 0.15 < ratio < 0.35  # expect ~0.25 (0.5**2), generous tolerance for resize rounding


def test_higher_z_index_wins_in_full_overlap():
    source = np.zeros((60, 60, 3), dtype=np.uint8)
    source[0:20, 0:20] = (10, 10, 200)  # blue - object "a"'s source crop
    source[0:20, 20:40] = (200, 10, 10)  # red - object "b"'s source crop

    obj_a = make_food_object("a", (0, 0, 19, 19), canvas_size=CANVAS, z_index=0, final_center=(30, 30))
    obj_b = make_food_object("b", (20, 0, 39, 19), canvas_size=CANVAS, z_index=1, final_center=(30, 30))

    scene = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [obj_a, obj_b], CANVAS, source)

    assert tuple(scene[30, 30]) == (200, 10, 10)  # b (higher z_index) wins


def test_empty_objects_renders_just_the_background():
    from standard.templates.background import render_background

    scene = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [], CANVAS, np.zeros((60, 60, 3), dtype=np.uint8))
    bg_only = render_background(TEMPLATES["T01_warm_ivory"], CANVAS)

    assert np.array_equal(scene, bg_only)  # no objects -> zero shadow density -> pure background


def _with_soft_rim(obj, rim_value: float):
    """Same object, plus a 1px outer rim of `rim_value` coverage around its bbox."""
    from dataclasses import replace

    x0, y0, x1, y1 = obj.bbox
    alpha = np.full((y1 - y0 + 3, x1 - x0 + 3), rim_value, dtype=np.float32)
    alpha[1:-1, 1:-1] = 1.0
    return replace(obj, alpha=alpha, alpha_origin=(x0 - 1, y0 - 1))


def test_soft_alpha_blends_rim_pixels_between_food_and_background():
    source = np.zeros((60, 60, 3), dtype=np.uint8)
    source[9:31, 9:31] = (200, 50, 50)
    hard = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS)
    soft = _with_soft_rim(hard, 0.5)

    hard_scene = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [hard], CANVAS, source).astype(float)
    soft_scene = TemplateRenderer().render(TEMPLATES["T01_warm_ivory"], [soft], CANVAS, source).astype(float)

    expected_rim = 0.5 * hard_scene[20, 9] + 0.5 * np.array([200, 50, 50])
    assert np.allclose(soft_scene[20, 9], expected_rim, atol=1.0)  # column 9 is just outside the bbox
    assert np.array_equal(soft_scene[20, 20], hard_scene[20, 20])  # interior identical


def test_soft_alpha_does_not_shift_placement_when_scaled():
    from standard.templates.renderer import _place_object

    source = np.full((60, 60, 3), 255, dtype=np.uint8)
    hard = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS, scale=0.5, final_center=(30, 30))
    soft = _with_soft_rim(hard, 0.5)

    hard_mask, _, _ = _place_object(hard, source, (60, 60))
    soft_mask, _, _ = _place_object(soft, source, (60, 60))

    ys_h, xs_h = np.nonzero(hard_mask)
    ys_s, xs_s = np.nonzero(soft_mask)
    assert abs(xs_h.mean() - xs_s.mean()) <= 1.0
    assert abs(ys_h.mean() - ys_s.mean()) <= 1.0
