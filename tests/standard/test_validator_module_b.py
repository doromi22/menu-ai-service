import numpy as np
from scipy import ndimage

from standard.reason_codes import GateResult, ReasonCode
from standard.validator.module_b import compute_hf_error, validate_content_integrity
from tests.standard.helpers import make_policy

RNG = np.random.default_rng(42)


def _textured_image(shape=(40, 40)):
    gray = RNG.integers(0, 255, shape, dtype=np.uint8).astype(np.float32)
    return np.stack([gray, gray, gray], axis=-1)


def test_identical_images_have_zero_hf_error():
    image = _textured_image()
    mask = np.ones(image.shape[:2], dtype=bool)

    error = compute_hf_error(image, image, mask, contrast_weight=0.35, sharpness_weight=0.50)

    assert error == 0.0


def test_blurring_away_detail_raises_hf_error():
    expected = _textured_image()
    final = np.stack([ndimage.gaussian_filter(expected[..., c], sigma=3.0) for c in range(3)], axis=-1)
    mask = np.ones(expected.shape[:2], dtype=bool)

    error = compute_hf_error(expected, final, mask, contrast_weight=0.35, sharpness_weight=0.50)

    assert error > 0.0


def test_no_violation_when_output_matches_expected_transform():
    image = _textured_image()
    mask = np.ones(image.shape[:2], dtype=bool)
    policy = make_policy()

    assert validate_content_integrity(image, image, mask, policy) == []


def test_violation_when_output_diverges_from_expected_transform():
    expected = _textured_image()
    final = np.stack([ndimage.gaussian_filter(expected[..., c], sigma=5.0) for c in range(3)], axis=-1)
    mask = np.ones(expected.shape[:2], dtype=bool)
    policy = make_policy()

    violations = validate_content_integrity(expected, final, mask, policy)

    assert (GateResult.REVIEW, ReasonCode.CONTENT_HF_ERROR) in violations
