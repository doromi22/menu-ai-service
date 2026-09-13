import time

import numpy as np

from standard.reason_codes import GateResult, ReasonCode
from standard.segmentation.gate import SegmentationGate

IMAGE_SIZE = 100  # image_area = 10_000, so ratio math stays exact


class FakeBackend:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask.astype(np.float32)

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._mask


class _ExplodingBackend:
    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated model load / inference failure")


class _FlakyBackend:
    """Fails `fail_count` times, then returns `mask` on the next call."""

    def __init__(self, mask: np.ndarray, fail_count: int) -> None:
        self._mask = mask.astype(np.float32)
        self._fail_count = fail_count
        self.calls = 0

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise RuntimeError(f"simulated transient failure #{self.calls}")
        return self._mask


def _blank_mask() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)


def _blank_image() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def _rect(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    mask[y0 : y1 + 1, x0 : x1 + 1] = True


def _run(mask: np.ndarray, **config_overrides):
    gate = SegmentationGate(FakeBackend(mask), config=config_overrides or None)
    return gate.run(_blank_image())


def test_clean_centered_blob_passes():
    mask = _blank_mask()
    _rect(mask, 30, 30, 69, 69)  # 40x40 = 1600px -> ratio 0.16, no edge/hole issues

    result = _run(mask)

    assert result.result == GateResult.PASS
    assert result.reasons == []
    assert len(result.food_objects) == 1
    assert result.food_objects[0].area == 1600


def test_empty_mask_rejects():
    result = _run(_blank_mask())

    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_NO_FOOD_DETECTED]
    assert result.food_objects == []


def test_backend_failure_rejects_with_its_own_reason_code_after_exhausting_retries():
    gate = SegmentationGate(_ExplodingBackend(), max_backend_retries=2, backend_retry_backoff_seconds=0)

    result = gate.run(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_BACKEND_ERROR]
    assert result.food_objects == []
    assert result.is_infra_error is True
    assert "RuntimeError" in result.metrics["backend_error"]
    assert result.metrics["backend_attempts"] == 3  # 1 initial + 2 retries


def test_infra_retry_stops_on_time_budget_before_exhausting_the_count_cap():
    # High count cap (10) so it would never be the limiting factor here;
    # a tight time budget (0.05s) with a real (if tiny) backoff (0.05s)
    # must still cut retrying short well before 10 attempts - proving the
    # time budget, not just the count, actually bounds retry duration.
    gate = SegmentationGate(
        _ExplodingBackend(),
        max_backend_retries=10,
        backend_retry_backoff_seconds=0.05,
        max_backend_retry_seconds=0.05,
    )

    start = time.perf_counter()
    result = gate.run(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    elapsed = time.perf_counter() - start

    assert result.result == GateResult.REJECT
    assert result.is_infra_error is True
    assert result.metrics["backend_attempts"] < 11  # stopped well short of the count cap (10 retries + 1)
    assert elapsed < 2.0  # bounded by the time budget, not left to spin toward the count cap


def test_backend_recovers_on_a_later_infra_retry():
    mask = _blank_mask()
    _rect(mask, 30, 30, 69, 69)  # a clean, otherwise-PASS-eligible blob
    backend = _FlakyBackend(mask, fail_count=1)  # fails once, succeeds on attempt 2
    gate = SegmentationGate(backend, max_backend_retries=2, backend_retry_backoff_seconds=0)

    result = gate.run(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

    assert backend.calls == 2
    assert result.result == GateResult.PASS
    assert result.is_infra_error is False
    assert len(result.food_objects) == 1


def test_infra_reject_and_data_quality_reject_are_distinguishable():
    infra_gate = SegmentationGate(_ExplodingBackend(), max_backend_retries=0, backend_retry_backoff_seconds=0)
    data_quality_gate = SegmentationGate(FakeBackend(_blank_mask()))  # empty mask -> SEGMENTATION_NO_FOOD_DETECTED

    infra_result = infra_gate.run(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    data_quality_result = data_quality_gate.run(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

    assert infra_result.result == GateResult.REJECT
    assert data_quality_result.result == GateResult.REJECT
    # both REJECT, but only one is an infra fault - this is exactly the
    # distinction §11-4's API/Laravel layer needs for different user messages.
    assert infra_result.is_infra_error is True
    assert data_quality_result.is_infra_error is False


def test_area_below_minimum_but_above_reject_line_reviews():
    mask = _blank_mask()
    _rect(mask, 10, 10, 29, 19)  # 20x10 = 200px -> ratio 0.02 (below 0.03, above reject line 0.015)

    result = _run(mask)

    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_AREA_TOO_SMALL]


def test_area_far_below_minimum_rejects():
    mask = _blank_mask()
    _rect(mask, 10, 10, 19, 19)  # 10x10 = 100px -> ratio 0.01, below reject line 0.015

    result = _run(mask)

    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_AREA_TOO_SMALL]
    # unlike the "no food at all" REJECT, a real (if too-small) object was
    # still found, so it stays in the result for debugging/analytics.
    assert len(result.food_objects) == 1


def test_area_above_maximum_rejects_immediately():
    mask = _blank_mask()
    _rect(mask, 0, 0, 99, 94)  # 100x95 = 9500px -> ratio 0.95 boundary is PASS; add 1 more row
    mask[95, 0] = True  # 9501px -> ratio 0.9501, just over the ceiling

    result = _run(mask, max_boundary_touch_ratio=10.0)  # isolate the area check from boundary-touch

    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_AREA_TOO_LARGE]


def test_boundary_touch_exceeded_reviews():
    mask = _blank_mask()
    _rect(mask, 0, 0, 34, 99)  # spans full height at the left edge

    result = _run(mask)

    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_BOUNDARY_TOUCH_EXCEEDED]


def test_hole_ratio_exceeded_reviews():
    mask = _blank_mask()
    _rect(mask, 30, 30, 69, 69)  # outer 40x40 = 1600px
    mask[40:60, 40:60] = False  # carve a 20x20 = 400px hole -> hole_ratio 0.25

    result = _run(mask)

    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_HOLE_RATIO_EXCEEDED]


def test_multi_object_detection_true_splits_objects():
    mask = _blank_mask()
    _rect(mask, 10, 10, 39, 39)  # 30x30, away from edges
    _rect(mask, 60, 60, 89, 89)  # 30x30, disjoint from the first

    result = _run(mask, multi_object_detection=True)

    assert result.result == GateResult.PASS
    assert len(result.food_objects) == 2


def test_multi_object_detection_false_merges_objects():
    mask = _blank_mask()
    _rect(mask, 10, 10, 39, 39)
    _rect(mask, 60, 60, 89, 89)

    result = _run(mask, multi_object_detection=False)

    assert result.result == GateResult.PASS
    assert len(result.food_objects) == 1
    assert result.food_objects[0].area == 900 + 900


def test_noise_speck_is_filtered_before_gating():
    mask = _blank_mask()
    _rect(mask, 30, 30, 69, 69)  # real 1600px object
    mask[5, 5] = True  # 1px speck, well under min_component_area_ratio * 10_000 = 20

    result = _run(mask)

    assert result.result == GateResult.PASS
    assert len(result.food_objects) == 1


# --- Boundary-value coverage (task: verify every yaml threshold's exact
# crossing point, plus the new REJECT/REVIEW severity split from §1). ---


def test_component_exactly_at_noise_floor_survives():
    mask = _blank_mask()
    _rect(mask, 10, 10, 13, 14)  # 4x5 = 20px, exactly min_component_area_ratio * 10_000

    result = _run(mask)

    # survives the noise filter, so it's judged as a real (if far too
    # small) object -> REJECT via area severity, not "no food detected".
    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_AREA_TOO_SMALL]
    assert len(result.food_objects) == 1


def test_component_one_pixel_below_noise_floor_is_filtered():
    mask = _blank_mask()
    _rect(mask, 10, 10, 28, 10)  # 19x1 = 19px, one below the 20px noise floor

    result = _run(mask)

    assert result.result == GateResult.REJECT
    assert result.reasons == [ReasonCode.SEGMENTATION_NO_FOOD_DETECTED]
    assert result.food_objects == []


def test_area_ratio_exactly_at_minimum_passes():
    mask = _blank_mask()
    _rect(mask, 10, 10, 39, 19)  # 30x10 = 300px -> ratio exactly 0.03

    result = _run(mask)

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_area_ratio_exactly_at_reject_line_reviews_not_rejects():
    mask = _blank_mask()
    _rect(mask, 10, 10, 24, 19)  # 15x10 = 150px -> ratio exactly 0.015 (= 0.03 * 0.5)

    result = _run(mask)

    # the reject line is an exclusive "<", so landing exactly on it is
    # still the (less severe) REVIEW side.
    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_AREA_TOO_SMALL]


def test_area_ratio_exactly_at_maximum_passes():
    mask = _blank_mask()
    _rect(mask, 0, 0, 99, 94)  # 100x95 = 9500px -> ratio exactly 0.95

    result = _run(mask, max_boundary_touch_ratio=10.0)  # isolate from boundary-touch

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_boundary_touch_ratio_just_under_threshold_passes():
    mask = _blank_mask()
    _rect(mask, 1, 0, 79, 4)  # 79 of the 396 perimeter px touched -> ratio 0.1995 < 0.20

    result = _run(mask)

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_boundary_touch_ratio_just_over_threshold_reviews():
    mask = _blank_mask()
    _rect(mask, 1, 0, 80, 4)  # 80 touched px -> ratio 0.2020 > 0.20, never escalates (reject_multiplier: null)

    result = _run(mask)

    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_BOUNDARY_TOUCH_EXCEEDED]


def test_hole_ratio_exactly_at_threshold_passes():
    mask = _blank_mask()
    _rect(mask, 20, 20, 59, 44)  # 40x25 = 1000px outer, away from every edge
    mask[30:35, 35:45] = False  # 5x10 = 50px hole -> hole_ratio exactly 0.05

    result = _run(mask)

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_hole_ratio_just_over_threshold_reviews():
    mask = _blank_mask()
    _rect(mask, 20, 20, 59, 44)  # 1000px outer
    mask[30:35, 35:45] = False  # 50px hole
    mask[35, 35] = False  # +1px, still interior -> 51px, ratio 0.051 > 0.05

    result = _run(mask)

    assert result.result == GateResult.REVIEW
    assert result.reasons == [ReasonCode.SEGMENTATION_HOLE_RATIO_EXCEEDED]
