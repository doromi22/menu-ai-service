import numpy as np

from standard.reason_codes import GateResult, ReasonCode
from standard.validator.module_c import compute_mean_luminance, compute_mean_saturation, validate_appearance_integrity
from tests.standard.helpers import make_policy


def _flat_image(color, shape=(10, 10)):
    arr = np.zeros((*shape, 3), dtype=np.float32)
    arr[..., 0], arr[..., 1], arr[..., 2] = color
    return arr


def test_gray_image_has_zero_saturation():
    mask = np.ones((10, 10), dtype=bool)
    assert compute_mean_saturation(_flat_image((100, 100, 100)), mask) == 0.0


def test_pure_red_has_full_saturation():
    mask = np.ones((10, 10), dtype=bool)
    assert compute_mean_saturation(_flat_image((255, 0, 0)), mask) == 1.0


def test_luminance_matches_perceptual_weights():
    mask = np.ones((10, 10), dtype=bool)
    expected = (0.299 * 255) / 255.0
    assert abs(compute_mean_luminance(_flat_image((255, 0, 0)), mask) - expected) < 1e-6


def test_no_violation_for_identical_images():
    image = _flat_image((120, 90, 60))
    mask = np.ones((10, 10), dtype=bool)
    policy = make_policy()

    assert validate_appearance_integrity(image, image, mask, policy) == []


def test_saturation_gain_violation_isolated():
    original = _flat_image((100, 100, 100))  # saturation 0
    final = _flat_image((200, 100, 50))  # saturation 0.75
    mask = np.ones((10, 10), dtype=bool)
    policy = make_policy(overrides={"food": {"max_luminance_gain": 10.0}})  # disable luminance check

    violations = validate_appearance_integrity(original, final, mask, policy)

    assert (GateResult.REVIEW, ReasonCode.FOOD_SATURATION_EXCEEDED) in violations
    assert not any(code == ReasonCode.FOOD_LUMINANCE_EXCEEDED for _, code in violations)


def test_luminance_gain_violation_isolated():
    original = _flat_image((50, 50, 50))
    final = _flat_image((200, 200, 200))  # much brighter, still gray -> saturation gain 0
    mask = np.ones((10, 10), dtype=bool)
    policy = make_policy()

    violations = validate_appearance_integrity(original, final, mask, policy)

    assert (GateResult.REVIEW, ReasonCode.FOOD_LUMINANCE_EXCEEDED) in violations
    assert not any(code == ReasonCode.FOOD_SATURATION_EXCEEDED for _, code in violations)


def test_saturation_or_luminance_decrease_is_not_a_violation():
    # "gain" per the policy field name is an increase - a decrease shouldn't count.
    original = _flat_image((200, 100, 50))
    final = _flat_image((100, 100, 100))  # less saturated and darker than original
    mask = np.ones((10, 10), dtype=bool)
    policy = make_policy()

    assert validate_appearance_integrity(original, final, mask, policy) == []
