"""
Exact boundary-value coverage for SegmentationGate's severity classifier,
independent of any pixel geometry. Complements the pixel-based boundary
tests in test_segmentation_gate.py, which confirm the metrics computed
from a mask actually reach this function correctly.
"""
from standard.reason_codes import GateResult
from standard.severity import classify_violation


def test_min_direction_no_violation_at_or_above_threshold():
    assert classify_violation(0.03, 0.03, "min", 0.5) is None
    assert classify_violation(0.05, 0.03, "min", 0.5) is None


def test_min_direction_review_zone():
    assert classify_violation(0.0299, 0.03, "min", 0.5) == GateResult.REVIEW
    assert classify_violation(0.015, 0.03, "min", 0.5) == GateResult.REVIEW  # exactly on the reject line


def test_min_direction_reject_zone():
    assert classify_violation(0.0149, 0.03, "min", 0.5) == GateResult.REJECT
    assert classify_violation(0.0, 0.03, "min", 0.5) == GateResult.REJECT


def test_min_direction_without_reject_multiplier_never_escalates():
    assert classify_violation(0.0, 0.03, "min", None) == GateResult.REVIEW


def test_max_direction_no_violation_at_or_below_threshold():
    assert classify_violation(0.20, 0.20, "max", None) is None
    assert classify_violation(0.10, 0.20, "max", None) is None


def test_max_direction_review_zone_with_no_reject_multiplier():
    assert classify_violation(0.2001, 0.20, "max", None) == GateResult.REVIEW
    assert classify_violation(0.99, 0.20, "max", None) == GateResult.REVIEW  # never escalates, however far over


def test_max_direction_reject_multiplier_one_collapses_review_zone():
    # max_food_area_ratio's configured tier: any exceedance is REJECT immediately.
    assert classify_violation(0.9501, 0.95, "max", 1.0) == GateResult.REJECT
    assert classify_violation(0.999, 0.95, "max", 1.0) == GateResult.REJECT


def test_max_direction_reject_multiplier_above_one_leaves_a_review_zone():
    assert classify_violation(0.25, 0.20, "max", 1.5) == GateResult.REVIEW  # 0.20 < 0.25 <= 0.30
    assert classify_violation(0.30, 0.20, "max", 1.5) == GateResult.REVIEW  # exactly on the reject line
    assert classify_violation(0.31, 0.20, "max", 1.5) == GateResult.REJECT
