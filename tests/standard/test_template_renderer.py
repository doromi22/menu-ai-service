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
