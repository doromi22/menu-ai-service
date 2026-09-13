"""
Segmentation Gate (spec §3) - the first checkpoint in the Standard
pipeline. Runs a SegmentationBackend, reduces the raw alpha mask to
geometry/edge metrics, and turns those into PASS / REVIEW / REJECT plus
the reason codes that earned the verdict. Nothing downstream (layout,
color, validator) should trust a mask that hasn't passed through here.

Severity model (PASS/REVIEW/REJECT mapping): §3 only pins one case
explicitly (non-food overlap with low confidence -> REVIEW, "자동 처리
금지"); it does not say which of the five yaml thresholds should REJECT
vs REVIEW outright. This build tags each threshold in
standard/config/segmentation.yaml's `segmentation.severity` block with an
optional `reject_multiplier`:

- `reject_multiplier` present -> crossing `base_threshold * reject_multiplier`
  (further in the "worse" direction than the base threshold itself) escalates
  the violation from REVIEW to REJECT. For a "min_" style threshold the
  multiplier is < 1 (a lower floor); for a "max_" style threshold it is >= 1
  (a higher ceiling, or exactly 1.0 to mean "no REVIEW buffer at all - any
  exceedance is REJECT immediately").
- `reject_multiplier: null` -> that check can only ever produce REVIEW; §3
  gives no basis for auto-rejecting on that signal alone.

Concretely: food_area_ratio below half of min_food_area_ratio, or any
excess over max_food_area_ratio, is REJECT ("아무도 crop으로 못 살림");
boundary-touch and hole-ratio violations stay REVIEW-only ("사람이
crop·보정" - §3). `min_component_area_ratio` isn't a REJECT/REVIEW
threshold on a metric; it's the noise floor used when extracting
components below, and lives on the REJECT side implicitly - if every
component gets filtered out by it, nothing survives and that IS the
"no usable food mask" REJECT case (see `_reject_no_food`).

This is a provisional split for the Phase-1 MVP, not derived from real
data. Retune the multipliers once the §10 360-image test harness produces
actual REVIEW-queue volume; that's exactly what the reason-code enum (§6,
§8) exists to make possible.

Backend infra failures (model load error, inference crash/OOM) are a
different kind of REJECT from all of the above - the photo may be
perfectly fine; the backend itself is unavailable. `run()` retries the
SAME call, bounded primarily by `max_backend_retry_seconds` of total
elapsed time (not just a fixed attempt count - a single call can itself
take up to ~35s on CPU, so counting attempts alone doesn't bound how long
retrying can take) plus a secondary `max_backend_retries` attempt-count
safety cap, before giving up; `SegmentationGateResult.is_infra_error` lets a caller tell this
case apart from a real data-quality REJECT without a new `GateResult`
value - see standard/segmentation/README.md for the full rationale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from standard.objects.food_object import FoodObject
from standard.reason_codes import GateResult, ReasonCode
from standard.segmentation.backend import SegmentationBackend
from standard.severity import classify_violation

DEFAULT_CONFIG = {
    "min_food_area_ratio": 0.03,
    "max_food_area_ratio": 0.95,
    "max_boundary_touch_ratio": 0.20,
    "min_component_area_ratio": 0.002,
    "max_hole_ratio": 0.05,
    "multi_object_detection": True,
    "mask_binarize_threshold": 0.5,
    "severity": {
        "min_food_area_ratio": {"reject_multiplier": 0.5},
        "max_food_area_ratio": {"reject_multiplier": 1.0},
        "max_boundary_touch_ratio": {"reject_multiplier": None},
        "max_hole_ratio": {"reject_multiplier": None},
    },
}


@dataclass
class ComponentMetrics:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area_ratio: float
    hole_ratio: float


@dataclass
class SegmentationGateResult:
    result: GateResult
    reasons: list[ReasonCode]
    food_objects: list[FoodObject]
    metrics: dict

    @property
    def is_infra_error(self) -> bool:
        """
        True when `result == REJECT` because the backend itself failed
        (model load error, inference crash/OOM after exhausting infra
        retries) rather than because the photo is unsupported. Derived
        from `reasons` instead of a separately-settable field so it can
        never drift out of sync with them, and deliberately NOT a new
        `GateResult` enum member - that enum is shared by the Validator
        and Retry state machine too, and neither of those need to
        represent "the segmentation backend was down" as one of their own
        states (segmentation short-circuits before they ever run). See
        standard/segmentation/README.md for the full rationale - this
        property exists so a caller (§11-4's API/Laravel) can show a
        different message for "일시적 오류, 잠시 후 다시 시도" vs "이 사진은
        지원되지 않습니다" without the pipeline's internal REJECT handling
        needing to change at all.
        """
        return ReasonCode.SEGMENTATION_BACKEND_ERROR in self.reasons


class SegmentationGate:
    def __init__(
        self,
        backend: SegmentationBackend,
        config: dict | None = None,
        *,
        max_backend_retries: int = 2,
        backend_retry_backoff_seconds: float = 1.5,
        max_backend_retry_seconds: float = 20.0,
    ) -> None:
        self._backend = backend
        self._config = self._merge_config(DEFAULT_CONFIG, config or {})
        self._max_backend_retries = max_backend_retries
        self._backend_retry_backoff_seconds = backend_retry_backoff_seconds
        # Primary control on retry duration - see _infer_mask_with_infra_retry.
        # max_backend_retries stays as a secondary safety cap for a
        # pathological fast-failing backend (e.g. instant connection
        # refused) that would otherwise spin many times within budget.
        self._max_backend_retry_seconds = max_backend_retry_seconds

    @staticmethod
    def _merge_config(default: dict, override: dict) -> dict:
        merged = dict(default)
        for key, value in override.items():
            if key == "severity" and isinstance(value, dict):
                merged_severity = {k: dict(v) for k, v in default.get("severity", {}).items()}
                for metric, tags in value.items():
                    merged_severity[metric] = {**merged_severity.get(metric, {}), **tags}
                merged["severity"] = merged_severity
            else:
                merged[key] = value
        return merged

    def run(self, image_rgb: np.ndarray, image_id: str = "food") -> SegmentationGateResult:
        cfg = self._config
        h, w = image_rgb.shape[:2]
        image_area = h * w

        alpha, backend_error, attempts_made = self._infer_mask_with_infra_retry(image_rgb)
        if alpha is None:
            # Infra fault (model load error, inference crash/OOM), not a
            # data-quality judgment about this photo - the retry above is a
            # plain same-parameters infra retry, separate from and blind to
            # standard.retry.state_machine (which varies color/layout
            # parameters to fix REAL data-quality REJECTs; neither a
            # different color nor a different layout can fix a backend
            # that's down, so this reason is never handed to it - see
            # reason_codes.py and standard/segmentation/README.md).
            return SegmentationGateResult(
                result=GateResult.REJECT,
                reasons=[ReasonCode.SEGMENTATION_BACKEND_ERROR],
                food_objects=[],
                metrics={
                    "image_area": image_area,
                    "food_area_ratio": 0.0,
                    "backend_error": backend_error,
                    "backend_attempts": attempts_made,
                },
            )
        binary = alpha >= cfg["mask_binarize_threshold"]

        if not binary.any():
            return self._reject_no_food(image_area, raw_component_count=0)

        labeled, raw_component_count = ndimage.label(binary)
        min_component_area = cfg["min_component_area_ratio"] * image_area

        significant = self._extract_significant_components(labeled, raw_component_count, min_component_area, image_area)

        if not significant:
            return self._reject_no_food(image_area, raw_component_count)

        active = significant if cfg["multi_object_detection"] else [self._merge(significant, image_area)]

        total_food_area_ratio = sum(c.area_ratio for c in significant)
        boundary_touch_ratio = self._boundary_touch_ratio(np.logical_or.reduce([c.mask for c in significant]))
        max_hole_ratio = max(c.hole_ratio for c in active)

        severity = cfg["severity"]
        violations: list[tuple[GateResult, ReasonCode]] = []

        def check(value: float, key: str, direction: Literal["min", "max"], reason: ReasonCode) -> None:
            verdict = classify_violation(value, cfg[key], direction, severity[key]["reject_multiplier"])
            if verdict is not None:
                violations.append((verdict, reason))

        check(total_food_area_ratio, "min_food_area_ratio", "min", ReasonCode.SEGMENTATION_AREA_TOO_SMALL)
        check(total_food_area_ratio, "max_food_area_ratio", "max", ReasonCode.SEGMENTATION_AREA_TOO_LARGE)
        check(boundary_touch_ratio, "max_boundary_touch_ratio", "max", ReasonCode.SEGMENTATION_BOUNDARY_TOUCH_EXCEEDED)
        check(max_hole_ratio, "max_hole_ratio", "max", ReasonCode.SEGMENTATION_HOLE_RATIO_EXCEEDED)

        if any(verdict == GateResult.REJECT for verdict, _ in violations):
            gate_result = GateResult.REJECT
        elif violations:
            gate_result = GateResult.REVIEW
        else:
            gate_result = GateResult.PASS

        food_objects = [
            FoodObject(
                id=f"{image_id}_{i}",
                mask=comp.mask,
                bbox=comp.bbox,
                original_center=self._centroid(comp.mask),
                final_center=self._centroid(comp.mask),
                z_index=i,
            )
            for i, comp in enumerate(active)
        ]

        return SegmentationGateResult(
            result=gate_result,
            reasons=[reason for _, reason in violations],
            food_objects=food_objects,
            metrics={
                "image_area": image_area,
                "food_area_ratio": total_food_area_ratio,
                "boundary_touch_ratio": boundary_touch_ratio,
                "max_hole_ratio": max_hole_ratio,
                "component_count": len(active),
                "raw_component_count": raw_component_count,
            },
        )

    def _infer_mask_with_infra_retry(self, image_rgb: np.ndarray) -> tuple[np.ndarray | None, str | None, int]:
        """
        Same-parameters infra retry, distinct from
        `standard.retry.state_machine` (which varies color/layout
        parameters to work around a data-quality REJECT - no such
        parameter can ever fix a backend that's down). Bounded by TWO
        independent limits, whichever is hit first:

        - `max_backend_retry_seconds` (primary): total elapsed time since
          the first attempt. A single BiRefNet call can itself take up to
          ~35s on CPU (standard/segmentation/README.md's measured worst
          case), so a fixed *count* of retries has no bound on total time
          - two retries could each take 35s and blow well past an API
          caller's own timeout (60s in ai-service/docs/standard-api-contract.md)
          on retries alone. Checked *before* starting each additional
          attempt (an in-flight synchronous call can't be cancelled
          mid-inference without added thread/process machinery, which is
          out of scope here) - so the very first attempt is never cut
          short, but no further attempt starts once the budget is spent.
        - `max_backend_retries` (secondary safety cap): guards the
          opposite failure mode - a backend that fails near-instantly
          (e.g. connection refused) would otherwise retry very many times
          within the time budget for no benefit.

        Returns (alpha, None, attempts) on success, or
        (None, error_message, attempts) once retrying stops.
        """
        last_error: str | None = None
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._backend.infer_mask(image_rgb), None, attempt
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any backend failure counts
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt > self._max_backend_retries:
                    break
                if time.monotonic() - start >= self._max_backend_retry_seconds:
                    break
                time.sleep(self._backend_retry_backoff_seconds)
        return None, last_error, attempt

    @staticmethod
    def _extract_significant_components(
        labeled: np.ndarray, num_labels: int, min_component_area: float, image_area: int
    ) -> list[ComponentMetrics]:
        components = []
        for label in range(1, num_labels + 1):
            comp_mask = labeled == label
            comp_area = int(comp_mask.sum())
            if comp_area < min_component_area:
                continue  # noise speck, not a real object - spec's "isolated components" check

            ys, xs = np.nonzero(comp_mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            filled = ndimage.binary_fill_holes(comp_mask)
            filled_area = int(filled.sum())
            hole_ratio = (filled_area - comp_area) / filled_area if filled_area else 0.0

            components.append(
                ComponentMetrics(mask=comp_mask, bbox=bbox, area_ratio=comp_area / image_area, hole_ratio=hole_ratio)
            )
        return components

    @staticmethod
    def _merge(components: list[ComponentMetrics], image_area: int) -> ComponentMetrics:
        union_mask = np.logical_or.reduce([c.mask for c in components])
        ys, xs = np.nonzero(union_mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        filled = ndimage.binary_fill_holes(union_mask)
        comp_area = int(union_mask.sum())
        filled_area = int(filled.sum())
        hole_ratio = (filled_area - comp_area) / filled_area if filled_area else 0.0
        return ComponentMetrics(mask=union_mask, bbox=bbox, area_ratio=comp_area / image_area, hole_ratio=hole_ratio)

    @staticmethod
    def _reject_no_food(image_area: int, raw_component_count: int) -> SegmentationGateResult:
        return SegmentationGateResult(
            result=GateResult.REJECT,
            reasons=[ReasonCode.SEGMENTATION_NO_FOOD_DETECTED],
            food_objects=[],
            metrics={"image_area": image_area, "food_area_ratio": 0.0, "raw_component_count": raw_component_count},
        )

    @staticmethod
    def _boundary_touch_ratio(mask: np.ndarray) -> float:
        h, w = mask.shape
        perimeter = 2 * (h + w) - 4
        if perimeter <= 0:
            return 0.0
        touching = mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
        return float(touching) / perimeter

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())
