import numpy as np

from standard.reason_codes import GateResult, ReasonCode
from standard.validator.module_a import compute_mask_area_delta, compute_mask_iou, validate_implementation_integrity
from tests.standard.helpers import make_policy


def _mask(bbox, shape=(50, 50)):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = bbox
    m[y0 : y1 + 1, x0 : x1 + 1] = True
    return m


def test_identical_masks_have_iou_one_and_no_area_delta():
    mask = _mask((10, 10, 29, 29))
    assert compute_mask_iou(mask, mask) == 1.0
    assert compute_mask_area_delta(mask, mask) == 0.0


def test_iou_drops_and_area_delta_rises_when_masks_diverge():
    original = _mask((10, 10, 29, 29))  # 400px
    final = _mask((15, 15, 34, 34))  # 400px, shifted -> partial overlap

    assert compute_mask_iou(original, final) < 1.0
    assert compute_mask_area_delta(original, final) == 0.0  # same area, just moved


def test_no_violations_for_identical_masks():
    mask = _mask((10, 10, 29, 29))
    policy = make_policy()

    assert validate_implementation_integrity(mask, mask, policy) == []


def test_low_iou_triggers_mask_changed():
    original = _mask((10, 10, 29, 29))
    final = _mask((25, 25, 44, 44))  # heavily shifted, low overlap
    policy = make_policy()

    violations = validate_implementation_integrity(original, final, policy)

    assert (GateResult.REVIEW, ReasonCode.MASK_CHANGED) in violations
