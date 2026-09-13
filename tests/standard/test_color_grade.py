import numpy as np
from PIL import Image

from standard.color.grade import (
    apply_fixed_tone_curve,
    apply_grade,
    apply_white_balance,
    grade_food_and_background,
)
from standard.color.presets import HERO, ColorGradeParams

IDENTITY = ColorGradeParams(brightness=1.0, contrast=1.0, saturation=1.0, temperature_strength=0.0)


def _solid_image(size=(20, 20), color=(120, 90, 60)):
    return Image.new("RGB", size, color)


def test_identity_params_leave_image_unchanged():
    image = _solid_image()
    graded = apply_grade(image, IDENTITY)
    assert np.array_equal(np.asarray(image), np.asarray(graded))


def test_white_balance_zero_strength_is_a_noop():
    image = _solid_image()
    assert apply_white_balance(image, 0.0) is image


def test_white_balance_warms_the_image():
    image = _solid_image(color=(100, 100, 100))
    warmed = apply_white_balance(image, strength=1.0)
    arr = np.asarray(warmed).astype(np.float32)
    assert arr[..., 0].mean() > 100  # red boosted
    assert arr[..., 2].mean() < 100  # blue reduced


def test_fixed_tone_curve_identity_at_zero_strength():
    image = _solid_image()
    curved = apply_fixed_tone_curve(image, 0.0)
    assert np.array_equal(np.asarray(image), np.asarray(curved))


def test_fixed_tone_curve_is_deterministic():
    image = _solid_image(color=(80, 150, 200))
    first = apply_fixed_tone_curve(image, 0.2)
    second = apply_fixed_tone_curve(image, 0.2)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_fixed_tone_curve_does_not_depend_on_image_content():
    # "fixed" means the LUT depends only on `strength`, never on this
    # image's own histogram - two very different images at the same
    # strength must be transformed by literally the same function.
    flat = _solid_image(color=(50, 50, 50))
    varied = Image.fromarray(np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8))

    from standard.color.grade import _fixed_tone_curve_lut

    lut = _fixed_tone_curve_lut(0.3)
    expected_flat = lut[np.asarray(flat)]
    expected_varied = lut[np.asarray(varied)]

    assert np.array_equal(np.asarray(apply_fixed_tone_curve(flat, 0.3)), expected_flat)
    assert np.array_equal(np.asarray(apply_fixed_tone_curve(varied, 0.3)), expected_varied)


def test_grade_food_and_background_composites_by_mask():
    image = _solid_image(size=(10, 10), color=(120, 90, 60))
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True  # food region

    result = grade_food_and_background(image, mask, HERO)

    background_graded = apply_grade(image, result.background_params)
    food_graded = apply_grade(image, result.food_params)
    composed_arr = np.asarray(result.image)

    assert np.array_equal(composed_arr[mask], np.asarray(food_graded)[mask])
    assert np.array_equal(composed_arr[~mask], np.asarray(background_graded)[~mask])
    assert result.food_params != result.background_params  # food grading really is a different pass


def test_forbidden_operations_are_not_implemented():
    import standard.color.grade as grade_module

    forbidden = {
        "apply_clahe",
        "adaptive_tone_map",
        "local_contrast",
        "object_aware_enhance",
        "edge_aware_sharpen",
    }
    assert not (forbidden & set(dir(grade_module)))
