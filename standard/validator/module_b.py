"""
Module B - Content Integrity (spec §6): transform-aware high-frequency
error. The core rule, verbatim from spec:

    Original Food -> Known Global Transform -> Expected Food
    Pipeline -> Actual Final Food
    compare: Expected Food <-> Final Food   (never Original <-> Final directly)

"Known Global Transform" is whatever `standard.color.grade.apply_grade`
actually ran with (its `food_params` - see §6 module B's caller, which
must build `expected` via `apply_grade(original, food_params)` itself;
this module only compares two already-graded images, it doesn't re-derive
the transform).

HF-error formula: no exact combination of `contrast_weight` /
`sharpness_weight` is given in §6 beyond the two weights themselves - this
build defines it as a weighted blend of a local-contrast-difference map
and a Laplacian-sharpness-difference map, normalized to roughly the same
0..1 scale as `base_threshold` (0.04). Provisional, like the segmentation
severity split - retune from real §10 data if needed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.severity import classify_violation


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float32).mean(axis=-1)


def _local_contrast_map(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    blurred = ndimage.gaussian_filter(gray, sigma=sigma)
    return np.abs(gray - blurred)


def _sharpness_map(gray: np.ndarray) -> np.ndarray:
    return np.abs(ndimage.laplace(gray))


def compute_hf_error(
    expected_rgb: np.ndarray,
    final_rgb: np.ndarray,
    mask: np.ndarray,
    contrast_weight: float,
    sharpness_weight: float,
) -> float:
    expected_gray = _to_gray(expected_rgb)
    final_gray = _to_gray(final_rgb)

    contrast_diff = np.abs(_local_contrast_map(expected_gray) - _local_contrast_map(final_gray))
    sharpness_diff = np.abs(_sharpness_map(expected_gray) - _sharpness_map(final_gray))

    combined = contrast_weight * contrast_diff + sharpness_weight * sharpness_diff
    masked = combined[mask]
    if masked.size == 0:
        return 0.0
    return float(masked.mean() / 255.0)


def validate_content_integrity(
    expected_food_rgb: np.ndarray, final_food_rgb: np.ndarray, food_mask: np.ndarray, policy: Policy
) -> list[tuple[GateResult, ReasonCode]]:
    error = compute_hf_error(
        expected_food_rgb,
        final_food_rgb,
        food_mask,
        policy.threshold("high_frequency", "contrast_weight"),
        policy.threshold("high_frequency", "sharpness_weight"),
    )
    verdict = classify_violation(
        error,
        policy.threshold("high_frequency", "base_threshold"),
        "max",
        policy.reject_multiplier("high_frequency", "base_threshold"),
    )
    return [(verdict, ReasonCode.CONTENT_HF_ERROR)] if verdict is not None else []
