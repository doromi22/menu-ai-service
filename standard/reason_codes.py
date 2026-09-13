"""
Unified reason-code registry (spec §6, §8).

Every gate and validator across the Standard pipeline reports through this
one enum instead of module-local error strings, so metadata (§8) stays
comparable across stages and later bottleneck analysis - "62% of REVIEWs
were segmentation problems" - doesn't require reconciling per-module
vocabularies after the fact.
"""
from __future__ import annotations

from enum import Enum


class GateResult(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class ReasonCode(str, Enum):
    # --- Segmentation Gate (spec §3) - produced by standard.segmentation.gate
    SEGMENTATION_NO_FOOD_DETECTED = "SEGMENTATION_NO_FOOD_DETECTED"
    SEGMENTATION_AREA_TOO_SMALL = "SEGMENTATION_AREA_TOO_SMALL"
    SEGMENTATION_AREA_TOO_LARGE = "SEGMENTATION_AREA_TOO_LARGE"
    SEGMENTATION_BOUNDARY_TOUCH_EXCEEDED = "SEGMENTATION_BOUNDARY_TOUCH_EXCEEDED"
    SEGMENTATION_HOLE_RATIO_EXCEEDED = "SEGMENTATION_HOLE_RATIO_EXCEEDED"

    # Reserved - spec calls these out (§3, §6) but they need a model this
    # build step doesn't add yet (a per-mask confidence score, and a
    # non-food object classifier for chopsticks/etc overlap detection).
    SEGMENTATION_LOW_CONFIDENCE = "SEGMENTATION_LOW_CONFIDENCE"
    SEGMENTATION_NONFOOD_OVERLAP = "SEGMENTATION_NONFOOD_OVERLAP"

    # Infrastructure failure (model load error, inference OOM/crash), NOT a
    # data-quality signal about the photo itself - deliberately absent from
    # standard/retry/retry_policy.yaml's strategy_by_reason. A different
    # color/layout parameter can't fix a backend that failed to load or ran
    # out of memory, so this is never retried; it always REJECTs immediately
    # (see standard/segmentation/gate.py's exception handling).
    SEGMENTATION_BACKEND_ERROR = "SEGMENTATION_BACKEND_ERROR"

    # --- Integrity Validator, modules A-D (spec §6) - producers land in
    # later build steps (§11-3~5); defined now so the enum stays single-sourced.
    MASK_CHANGED = "MASK_CHANGED"
    CONTENT_HF_ERROR = "CONTENT_HF_ERROR"
    FOOD_SATURATION_EXCEEDED = "FOOD_SATURATION_EXCEEDED"
    FOOD_LUMINANCE_EXCEEDED = "FOOD_LUMINANCE_EXCEEDED"
    OBJECT_SCALE_CHANGED = "OBJECT_SCALE_CHANGED"
    RELATIVE_SCALE_CHANGED = "RELATIVE_SCALE_CHANGED"
    OCCLUSION_CHANGED = "OCCLUSION_CHANGED"

    # Added beyond §6's literal enum listing: §4 states object-area-rank
    # preservation ("main > rice > soup > side") as its own REVIEW trigger,
    # in a separate sentence from occlusion/front-back order ("순위 역전 시
    # REVIEW"), but §6's enum has no dedicated code for it - without one,
    # a rank-reversal REVIEW would be unreportable/uncountable, defeating
    # the enum's stated purpose (§6: bottleneck analysis by reason code).
    OBJECT_AREA_RANK_REVERSED = "OBJECT_AREA_RANK_REVERSED"

    # --- Layout Engine (spec §4/§2) - producer: standard.layout.engine
    # §2: "unknown 비율 > 50%인 프레임은 layout engine을 단순 grid/원본 배치로
    # 낮추고 자동 REVIEW". Not in §6's enum (§6 predates the Layout Engine
    # build step) but needed so this specific fallback is countable.
    LAYOUT_UNKNOWN_ROLE_MAJORITY = "LAYOUT_UNKNOWN_ROLE_MAJORITY"
