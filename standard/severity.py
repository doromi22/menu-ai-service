"""
Shared REVIEW/REJECT severity classifier.

Originally lived in standard/segmentation/gate.py; pulled out so the
Policy loader (§6, §11 step 3) and, later, the Integrity Validator that
reads it can use the exact same escalation rule the Segmentation Gate
uses (§3) - one severity model for the whole pipeline, not a re-derived
one per stage.
"""
from __future__ import annotations

from typing import Literal

from standard.reason_codes import GateResult


def classify_violation(
    value: float,
    base_threshold: float,
    direction: Literal["min", "max"],
    reject_multiplier: float | None,
) -> GateResult | None:
    """
    direction="min": violated when value < base_threshold.
    direction="max": violated when value > base_threshold.

    `reject_multiplier`, when set, is a multiplier of base_threshold marking
    the point where a REVIEW-worthy violation escalates to REJECT (< 1 for
    "min" thresholds, >= 1 for "max" thresholds; exactly at the boundary of
    base_threshold for a "max" threshold, i.e. multiplier 1.0, means any
    exceedance is REJECT immediately, no REVIEW buffer). `None` means the
    check can only ever produce REVIEW.

    Returns None when there's no violation at all.
    """
    if direction == "min":
        if value >= base_threshold:
            return None
        if reject_multiplier is not None and value < base_threshold * reject_multiplier:
            return GateResult.REJECT
        return GateResult.REVIEW
    if direction == "max":
        if value <= base_threshold:
            return None
        if reject_multiplier is not None and value > base_threshold * reject_multiplier:
            return GateResult.REJECT
        return GateResult.REVIEW
    raise ValueError(f"unknown direction: {direction!r}")
