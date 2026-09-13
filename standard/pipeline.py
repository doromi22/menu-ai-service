"""
The full Standard pipeline orchestrator (§11 step 4): Segmentation ->
Layout -> Shadow (inside TemplateRenderer) -> Color -> Validator -> Retry,
wired together end to end. This is the single place that logic exists -
both `mvp_harness/runner.py` (file-based, batch, writes images to disk)
and `standard/api/main.py` (one HTTP request at a time) call `run_pipeline`
rather than each re-implementing the wiring.

Role assignment is hand-simulated after segmentation: no role classifier
exists yet (explicitly out of scope, see standard/README.md). The largest
object is treated as a confident "main"; the rest stay "unknown".
"""
from __future__ import annotations

import io
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from standard.color.grade import apply_grade
from standard.color.presets import STANDARD, food_params_for
from standard.layout.engine import LayoutEngine
from standard.objects.food_object import FoodObject
from standard.policy.schema import Policy
from standard.reason_codes import GateResult, ReasonCode
from standard.retry.config import RetryPolicyConfig
from standard.retry.state_machine import RetryParams, RetryStateMachine
from standard.segmentation.backend import SegmentationBackend
from standard.segmentation.gate import SegmentationGate
from standard.templates.definitions import Template
from standard.templates.renderer import TemplateRenderer, place_food_layer
from standard.validator.validator import ValidatorResult, run_validator

# DECIDED - see mvp_harness/README.md's "JPEG quality" section for the
# measured size/HF-error/PSNR comparison (scripts/compare_jpeg_quality.py)
# this is based on. 92 gives a ~3.5x safety margin under
# standard/policy/policy.yaml's high_frequency.base_threshold (0.04) from
# compression alone, while 85 (the most aggressive level tested) still
# only used up less than half that budget - so this is not a tight
# tradeoff, just a reasonable default. Revisit together with
# base_threshold if real §10 photos show meaningfully different numbers
# than the synthetic textured test image this was calibrated against.
JPEG_QUALITY = 92


def _encode_decode_jpeg(image_array: np.ndarray, quality: int) -> np.ndarray:
    """Round-trips an array through the exact format it gets saved as, so
    "Actual Final Food" (spec §6 module B) means the pixels a merchant
    actually receives - lossy JPEG compression included - not an
    in-memory value that's never what gets delivered."""
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def assign_stand_in_roles(objects: list[FoodObject]) -> list[FoodObject]:
    if not objects:
        return []
    largest = max(range(len(objects)), key=lambda i: objects[i].area)
    return [
        replace(obj, semantic_role="main", role_confidence=0.95) if i == largest else obj
        for i, obj in enumerate(objects)
    ]


@dataclass
class PipelineResult:
    segmentation_status: GateResult
    segmentation_reasons: list[ReasonCode]
    is_infra_error: bool
    validator_status: GateResult | None  # None: segmentation REJECTed before the Validator ever ran
    validator_reasons: list[ReasonCode]
    retry_attempts_used: int
    rendered_image: np.ndarray | None  # None: segmentation REJECTed, nothing to render (§7: "결과물 미사용")
    overall_status: GateResult  # REVIEW if either stage isn't a clean PASS; REJECT only from segmentation
    overall_reasons: list[ReasonCode]
    metrics: dict


def run_pipeline(
    image_rgb: np.ndarray,
    template: Template,
    *,
    segmentation_backend: SegmentationBackend,
    policy: Policy,
    retry_config: RetryPolicyConfig,
    image_id: str = "food",
    max_backend_retries: int = 2,
    backend_retry_backoff_seconds: float = 1.5,
    max_backend_retry_seconds: float = 20.0,  # leaves headroom under the 60s API timeout (docs/standard-api-contract.md) for the rest of the pipeline after segmentation
    jpeg_quality: int = JPEG_QUALITY,
) -> PipelineResult:
    height, width = image_rgb.shape[:2]
    canvas_size = (width, height)
    canvas_shape = (height, width)
    image = Image.fromarray(image_rgb)

    gate = SegmentationGate(
        segmentation_backend,
        max_backend_retries=max_backend_retries,
        backend_retry_backoff_seconds=backend_retry_backoff_seconds,
        max_backend_retry_seconds=max_backend_retry_seconds,
    )
    seg_result = gate.run(image_rgb, image_id=image_id)

    if seg_result.result == GateResult.REJECT:
        # §7: REJECT = "결과물 미사용" - nothing to render, and the retry
        # policy has no strategy for any SEGMENTATION_* reason anyway
        # (see standard/retry/retry_policy.yaml).
        return PipelineResult(
            segmentation_status=seg_result.result,
            segmentation_reasons=seg_result.reasons,
            is_infra_error=seg_result.is_infra_error,
            validator_status=None,
            validator_reasons=[],
            retry_attempts_used=0,
            rendered_image=None,
            overall_status=GateResult.REJECT,
            overall_reasons=seg_result.reasons,
            metrics=seg_result.metrics,
        )

    objects = assign_stand_in_roles(seg_result.food_objects)
    union_mask = np.logical_or.reduce([obj.mask for obj in objects])

    layout_result = LayoutEngine().run(objects, canvas_size)
    food_params = food_params_for(STANDARD)

    last_rendered: dict[str, np.ndarray] = {}

    def render_and_validate(color_params, layout_objects) -> ValidatorResult:
        graded_food = np.asarray(apply_grade(image, color_params))
        rendered = TemplateRenderer().render(template, layout_objects, canvas_size, graded_food)
        last_rendered["image"] = rendered

        # "Expected Food" (spec §6 module B): the known transform applied to
        # the original, placed at the SAME final positions the real render
        # used - place_food_layer redoes the actual per-object placement
        # rather than assuming one uniform shift, so this stays correct even
        # after grid-fallback or a retry's non-uniform recompute_translation.
        placed_mask, expected_food = place_food_layer(layout_objects, canvas_shape, graded_food)
        # Module C's "true original" reference needs to be aligned the same
        # way, since it's compared through the same food_mask.
        _, original_placed = place_food_layer(layout_objects, canvas_shape, image_rgb)
        # "Actual Final Food": the pixels a merchant actually receives,
        # lossy JPEG compression included - not the ideal in-memory array.
        final_food = _encode_decode_jpeg(rendered, jpeg_quality)

        return run_validator(
            original_mask=union_mask,
            final_mask=union_mask,
            expected_food_rgb=expected_food,
            final_food_rgb=final_food,
            food_mask=placed_mask,
            original_food_rgb=original_placed,
            original_objects=objects,
            final_objects=layout_objects,
            policy=policy,
            used_grid_fallback=layout_result.used_grid_fallback,
        )

    result = render_and_validate(food_params, layout_result.objects)
    retry_attempts_used = 0

    if result.result == GateResult.REJECT:

        def evaluate(params: RetryParams) -> ValidatorResult:
            color_params = params.color_params if params.color_params is not None else food_params
            layout_objects = params.layout_objects if params.layout_objects is not None else layout_result.objects
            return render_and_validate(color_params, layout_objects)

        outcome = RetryStateMachine(retry_config).run(result.reasons, layout_result.objects, canvas_size, evaluate)
        retry_attempts_used = len(outcome.attempts)
        final_status = outcome.result
        final_reasons = outcome.final_reasons
    else:
        final_status = result.result
        final_reasons = result.reasons

    overall_status = (
        GateResult.REVIEW if seg_result.result == GateResult.REVIEW or final_status != GateResult.PASS else GateResult.PASS
    )
    overall_reasons = list(seg_result.reasons) + list(final_reasons)

    return PipelineResult(
        segmentation_status=seg_result.result,
        segmentation_reasons=seg_result.reasons,
        is_infra_error=False,
        validator_status=final_status,
        validator_reasons=final_reasons,
        retry_attempts_used=retry_attempts_used,
        rendered_image=last_rendered["image"],
        overall_status=overall_status,
        overall_reasons=overall_reasons,
        metrics=seg_result.metrics,
    )
