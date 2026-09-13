"""
Retry Policy state machine (spec §7). Decides *what to try next and when
to give up* - it does not itself re-render an image. Actually re-running
the pipeline with new parameters (regrading colors, recompositing,
re-running the Integrity Validator) needs the Graphic template renderer
(§9, still pending) and a pipeline orchestrator that doesn't exist yet;
until then, callers supply an `evaluate` callback that takes this
attempt's `RetryParams` and returns the resulting `ValidatorResult` -
in production that callback re-runs the real pipeline, in tests it's a
fake. The strategy *parameters themselves* are produced for real, by
the real Color Grade / Layout Engine functions - only the "did this
attempt actually pass" scoring step is injected.

Priority when several reasons are active at once: layout-category
reasons (OCCLUSION_CHANGED, OBJECT_SCALE_CHANGED,
RELATIVE_SCALE_CHANGED, OBJECT_AREA_RANK_REVERSED) are processed before
color-category reasons (CONTENT_HF_ERROR, FOOD_SATURATION_EXCEEDED,
FOOD_LUMINANCE_EXCEEDED). Both still land in the *same* attempt's
RetryParams (only one evaluate() call per attempt regardless of how many
reasons it addresses), so this ordering doesn't delay anything in the
current architecture - color grading here doesn't depend on object
positions, and vice versa. It's kept anyway as a deliberate, stable
convention (fix structural/geometric problems before surface/color
polish) that would become load-bearing once a real renderer makes color
grading depend on the composited scene; it also gives deterministic
logging order in `RetryAttemptRecord.reasons_addressed`.

Anti-infinite-loop: a reason is retried at most len(strategy list) times
- once that list is exhausted (checked by array index, so it's exact by
construction, not a fingerprint heuristic), the reason is dropped from
the retryable set instead of being handed a repeated strategy step. If
that empties the retryable set, the loop stops immediately rather than
burning the rest of `max_attempts` on evaluate() calls that cannot
change (spec §7: "단순 재실행 금지").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from standard.color.presets import ColorGradeParams
from standard.objects.food_object import FoodObject
from standard.reason_codes import GateResult, ReasonCode
from standard.retry.config import RetryPolicyConfig
from standard.retry.strategies import (
    Strategy,
    conservative_color_params,
    force_scale_1_0,
    neutral_color_params,
    recompute_translation,
)
from standard.validator.validator import ValidatorResult

LAYOUT_REASONS = frozenset(
    {
        ReasonCode.OCCLUSION_CHANGED,
        ReasonCode.OBJECT_SCALE_CHANGED,
        ReasonCode.RELATIVE_SCALE_CHANGED,
        ReasonCode.OBJECT_AREA_RANK_REVERSED,
    }
)
COLOR_REASONS = frozenset(
    {
        ReasonCode.CONTENT_HF_ERROR,
        ReasonCode.FOOD_SATURATION_EXCEEDED,
        ReasonCode.FOOD_LUMINANCE_EXCEEDED,
    }
)


def _priority(reason: ReasonCode) -> int:
    if reason in LAYOUT_REASONS:
        return 0
    if reason in COLOR_REASONS:
        return 1
    return 2


@dataclass(frozen=True)
class RetryParams:
    color_params: ColorGradeParams | None = None
    layout_objects: list[FoodObject] | None = None


@dataclass
class RetryAttemptRecord:
    attempt_index: int
    reasons_addressed: list[ReasonCode]
    strategies_applied: dict[ReasonCode, Strategy]
    params: RetryParams
    result: ValidatorResult


@dataclass
class RetryOutcome:
    result: GateResult  # PASS, or REVIEW per final_action - never REJECT (§7's whole point of retrying)
    attempts: list[RetryAttemptRecord] = field(default_factory=list)
    final_reasons: list[ReasonCode] = field(default_factory=list)


class RetryStateMachine:
    def __init__(self, config: RetryPolicyConfig) -> None:
        self._config = config

    def run(
        self,
        reasons: list[ReasonCode],
        objects: list[FoodObject],
        canvas_size: tuple[int, int],
        evaluate: Callable[[RetryParams], ValidatorResult],
    ) -> RetryOutcome:
        strategy_by_reason = self._config.strategy_by_reason
        tries_used: dict[ReasonCode, int] = {}
        current_reasons = list(reasons)
        current_objects = objects
        attempts: list[RetryAttemptRecord] = []

        for attempt_index in range(self._config.max_attempts):
            retryable = [
                r
                for r in current_reasons
                if r in strategy_by_reason and tries_used.get(r, 0) < len(strategy_by_reason[r])
            ]
            if not retryable:
                break  # nothing left we know how to fix - don't burn remaining attempts

            ordered = sorted(retryable, key=lambda r: (_priority(r), r.value))

            strategies_applied: dict[ReasonCode, Strategy] = {}
            needed: set[Strategy] = set()
            for reason in ordered:
                strategies = strategy_by_reason[reason]
                step = tries_used.get(reason, 0)
                strategy = strategies[step]
                strategies_applied[reason] = strategy
                needed.add(strategy)
                tries_used[reason] = step + 1

            layout_objects = current_objects
            if Strategy.RECOMPUTE_TRANSLATION in needed:
                layout_objects = recompute_translation(layout_objects, canvas_size, attempt_index)
            if Strategy.FORCE_SCALE_1_0 in needed:
                layout_objects = force_scale_1_0(layout_objects)

            color_params = None
            if Strategy.NEUTRAL_COLOR in needed:
                color_params = neutral_color_params()
            elif Strategy.CONSERVATIVE_COLOR in needed:
                color_params = conservative_color_params()

            current_objects = layout_objects
            params = RetryParams(color_params=color_params, layout_objects=layout_objects)
            result = evaluate(params)
            attempts.append(
                RetryAttemptRecord(
                    attempt_index=attempt_index,
                    reasons_addressed=ordered,
                    strategies_applied=strategies_applied,
                    params=params,
                    result=result,
                )
            )

            if result.result == GateResult.PASS:
                return RetryOutcome(result=GateResult.PASS, attempts=attempts, final_reasons=[])

            current_reasons = list(result.reasons)

        return RetryOutcome(result=GateResult.REVIEW, attempts=attempts, final_reasons=current_reasons)
