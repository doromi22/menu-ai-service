"""
Full pipeline integration test (§11 step 2 completion): a single synthetic
photo run through Segmentation Gate -> Layout Engine -> Shadow Engine ->
Color Grade -> Integrity Validator, once for a PASS outcome and once for a
REVIEW outcome.

Role assignment is simulated by hand after the Segmentation Gate - no role
classifier exists yet (explicitly out of scope for this build step), so
this stands in for one, clearly marked as such rather than silently
assumed away.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from standard.color.grade import apply_grade, grade_food_and_background
from standard.color.presets import HERO, STANDARD
from standard.layout.engine import LayoutEngine
from standard.reason_codes import GateResult, ReasonCode
from standard.segmentation.gate import SegmentationGate
from standard.shadow.engine import AmbientShadowParams, ShadowEngine
from standard.validator.validator import run_validator
from tests.standard.helpers import make_policy

CANVAS = 120


class _FakeBackend:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask.astype(np.float32)

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._mask


def _synthetic_photo_and_mask():
    rng = np.random.default_rng(123)
    image = rng.integers(60, 200, (CANVAS, CANVAS, 3), dtype=np.uint8)
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[30:90, 30:90] = True  # 60x60 = 3600px, ratio 0.25 - comfortably within segmentation bounds
    return image, mask


def _run_through_segmentation_and_layout():
    image, mask = _synthetic_photo_and_mask()
    gate = SegmentationGate(_FakeBackend(mask))
    seg_result = gate.run(image)
    assert seg_result.result == GateResult.PASS  # sanity: the synthetic mask is clean by construction

    # Role-classifier stand-in (not part of this build step): a single
    # confident "main" object so Layout Engine takes its normal path
    # instead of the unknown-majority grid fallback.
    original_objects = [replace(obj, semantic_role="main", role_confidence=0.95) for obj in seg_result.food_objects]

    layout = LayoutEngine()
    layout_result = layout.run(original_objects, canvas_size=(CANVAS, CANVAS))
    return image, mask, original_objects, layout_result


def test_full_pipeline_pass_case():
    image, mask, original_objects, layout_result = _run_through_segmentation_and_layout()
    assert layout_result.used_grid_fallback is False
    assert layout_result.review_reasons == []

    shadow = ShadowEngine().render(
        layout_result.objects, (CANVAS, CANVAS), AmbientShadowParams(opacity=0.12, blur=42, offset_y=18)
    )
    assert shadow.combined.shape == (CANVAS, CANVAS)

    pil_image = Image.fromarray(image)
    grade_result = grade_food_and_background(pil_image, mask, STANDARD)

    # "Known Global Transform" applied to the true original - Module B's
    # comparison target, never the original itself (spec §6).
    expected_food = np.asarray(apply_grade(pil_image, grade_result.food_params))
    final_food = np.asarray(grade_result.image)  # the pipeline's actual output

    result = run_validator(
        original_mask=mask,
        final_mask=mask,  # Layout Engine never touches the mask itself
        expected_food_rgb=expected_food,
        final_food_rgb=final_food,
        food_mask=mask,
        original_food_rgb=np.asarray(pil_image),
        original_objects=original_objects,
        final_objects=layout_result.objects,
        policy=make_policy(),
    )

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_full_pipeline_review_case():
    image, mask, original_objects, layout_result = _run_through_segmentation_and_layout()

    pil_image = Image.fromarray(image)
    grade_result = grade_food_and_background(pil_image, mask, STANDARD)
    expected_food = np.asarray(apply_grade(pil_image, grade_result.food_params))

    # Simulate a compositing bug: the food region actually got the much
    # stronger *background*-strength (HERO) treatment instead of its own
    # conservative food_params - exactly the drift the Integrity Validator
    # exists to catch (compare against `expected_food`/original, never
    # against this buggy value being mistaken for "expected").
    final_food = np.asarray(apply_grade(pil_image, HERO))

    result = run_validator(
        original_mask=mask,
        final_mask=mask,
        expected_food_rgb=expected_food,
        final_food_rgb=final_food,
        food_mask=mask,
        original_food_rgb=np.asarray(pil_image),
        original_objects=original_objects,
        final_objects=layout_result.objects,
        policy=make_policy(),
    )

    assert result.result in (GateResult.REVIEW, GateResult.REJECT)
    assert len(result.reasons) > 0


def test_full_pipeline_unknown_majority_forces_review():
    # 3 objects, only 1 gets a confident role - the other 2 stay "unknown",
    # the realistic default until a role classifier exists (out of scope).
    rng = np.random.default_rng(99)
    image = rng.integers(60, 200, (CANVAS, CANVAS, 3), dtype=np.uint8)
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[10:30, 10:30] = True
    mask[50:70, 50:70] = True
    mask[90:110, 10:30] = True

    seg_result = SegmentationGate(_FakeBackend(mask)).run(image)
    assert seg_result.result == GateResult.PASS
    assert len(seg_result.food_objects) == 3

    original_objects = list(seg_result.food_objects)
    original_objects[0] = replace(original_objects[0], semantic_role="main", role_confidence=0.95)

    layout_result = LayoutEngine().run(original_objects, canvas_size=(CANVAS, CANVAS))
    assert layout_result.used_grid_fallback is True
    assert layout_result.unknown_ratio > 0.5

    pil_image = Image.fromarray(image)
    grade_result = grade_food_and_background(pil_image, mask, STANDARD)
    expected_food = np.asarray(apply_grade(pil_image, grade_result.food_params))
    final_food = np.asarray(grade_result.image)  # a correctly-graded output - nothing else is wrong

    result = run_validator(
        original_mask=mask,
        final_mask=mask,
        expected_food_rgb=expected_food,
        final_food_rgb=final_food,
        food_mask=mask,
        original_food_rgb=np.asarray(pil_image),
        original_objects=original_objects,
        final_objects=layout_result.objects,
        policy=make_policy(),
        used_grid_fallback=layout_result.used_grid_fallback,
    )

    assert result.result == GateResult.REVIEW
    assert ReasonCode.LAYOUT_UNKNOWN_ROLE_MAJORITY in result.reasons
