from standard.reason_codes import GateResult, ReasonCode
from standard.retry.config import RetryPolicyConfig
from standard.retry.state_machine import RetryStateMachine
from standard.retry.strategies import Strategy, conservative_color_params, neutral_color_params
from standard.validator.validator import ValidatorResult
from tests.standard.helpers import make_food_object

CANVAS = (100, 100)


def test_single_reason_succeeds_on_first_attempt():
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={ReasonCode.CONTENT_HF_ERROR: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR]},
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9))]
    calls = []

    def evaluate(params):
        calls.append(params)
        return ValidatorResult(result=GateResult.PASS, reasons=[])

    outcome = machine.run([ReasonCode.CONTENT_HF_ERROR], objects, CANVAS, evaluate)

    assert outcome.result == GateResult.PASS
    assert len(outcome.attempts) == 1
    assert len(calls) == 1
    assert outcome.attempts[0].params.color_params == conservative_color_params()


def test_single_reason_needs_second_strategy_and_params_actually_differ():
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={ReasonCode.CONTENT_HF_ERROR: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR]},
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9))]
    responses = iter(
        [
            ValidatorResult(result=GateResult.REJECT, reasons=[ReasonCode.CONTENT_HF_ERROR]),
            ValidatorResult(result=GateResult.PASS, reasons=[]),
        ]
    )

    outcome = machine.run([ReasonCode.CONTENT_HF_ERROR], objects, CANVAS, lambda params: next(responses))

    assert outcome.result == GateResult.PASS
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].params.color_params == conservative_color_params()
    assert outcome.attempts[1].params.color_params == neutral_color_params()
    assert outcome.attempts[0].params.color_params != outcome.attempts[1].params.color_params


def test_simultaneous_reasons_bundle_into_one_attempt_layout_first():
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={
            ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION],
            ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        },
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9)), make_food_object("b", (20, 20, 29, 29))]

    outcome = machine.run(
        [ReasonCode.FOOD_SATURATION_EXCEEDED, ReasonCode.OCCLUSION_CHANGED],  # color listed first in input
        objects,
        CANVAS,
        lambda params: ValidatorResult(result=GateResult.PASS, reasons=[]),
    )

    assert len(outcome.attempts) == 1
    attempt = outcome.attempts[0]
    assert attempt.reasons_addressed == [ReasonCode.OCCLUSION_CHANGED, ReasonCode.FOOD_SATURATION_EXCEEDED]
    assert attempt.params.color_params is not None
    assert attempt.params.layout_objects is not None


def test_exhausting_all_max_attempts_forces_review_with_reasons():
    # Synthetic 3-entry strategy list (the real shipped config maxes out at
    # 2) purely to exercise the "max_attempts truly consumed" path - see
    # test_reason_with_single_strategy_stops_early below for the more
    # realistic early-stop case.
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={
            ReasonCode.FOOD_SATURATION_EXCEEDED: [
                Strategy.CONSERVATIVE_COLOR,
                Strategy.NEUTRAL_COLOR,
                Strategy.NEUTRAL_COLOR,
            ]
        },
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9))]

    outcome = machine.run(
        [ReasonCode.FOOD_SATURATION_EXCEEDED],
        objects,
        CANVAS,
        lambda params: ValidatorResult(result=GateResult.REJECT, reasons=[ReasonCode.FOOD_SATURATION_EXCEEDED]),
    )

    assert outcome.result == GateResult.REVIEW
    assert len(outcome.attempts) == 3
    assert outcome.final_reasons == [ReasonCode.FOOD_SATURATION_EXCEEDED]


def test_reason_with_single_strategy_stops_early_instead_of_looping():
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION]},
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9)), make_food_object("b", (20, 20, 29, 29))]
    call_count = 0

    def evaluate(params):
        nonlocal call_count
        call_count += 1
        return ValidatorResult(result=GateResult.REJECT, reasons=[ReasonCode.OCCLUSION_CHANGED])

    outcome = machine.run([ReasonCode.OCCLUSION_CHANGED], objects, CANVAS, evaluate)

    assert outcome.result == GateResult.REVIEW
    assert call_count == 1  # never retries with an identical, already-tried strategy step
    assert outcome.final_reasons == [ReasonCode.OCCLUSION_CHANGED]


def test_unmapped_reason_is_never_retried_but_persists_to_final_reasons():
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR]},
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9))]

    outcome = machine.run(
        [ReasonCode.MASK_CHANGED, ReasonCode.FOOD_SATURATION_EXCEEDED],
        objects,
        CANVAS,
        # the color issue "gets fixed" by the pipeline, but MASK_CHANGED
        # (no strategy exists for it - it's a bug signal, not fixable by
        # retrying color/layout) persists.
        lambda params: ValidatorResult(result=GateResult.REVIEW, reasons=[ReasonCode.MASK_CHANGED]),
    )

    assert outcome.result == GateResult.REVIEW
    assert outcome.final_reasons == [ReasonCode.MASK_CHANGED]
    assert len(outcome.attempts) == 1  # round 2 has nothing retryable left, so it never runs


def test_simultaneous_reasons_from_different_categories_use_attempts_efficiently():
    """
    Confirms bundling (every currently-active reason's strategy is applied
    together in ONE attempt), not sequential per-category consumption:
    with OCCLUSION_CHANGED (1 strategy) and FOOD_SATURATION_EXCEEDED (2
    strategies) both active, if attempts were consumed one category at a
    time, exhausting both could cost 3 attempts on its own, leaving no
    budget for anything else. Bundled, it costs 2: attempt 0 fixes
    occlusion AND tries the first color strategy together; attempt 1 only
    needs to advance the still-failing color strategy.
    """
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={
            ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION],
            ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        },
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9)), make_food_object("b", (20, 20, 29, 29))]

    responses = iter(
        [
            # attempt 0: occlusion fix + first color strategy applied together;
            # occlusion resolves, saturation doesn't yet.
            ValidatorResult(result=GateResult.REJECT, reasons=[ReasonCode.FOOD_SATURATION_EXCEEDED]),
            ValidatorResult(result=GateResult.PASS, reasons=[]),
        ]
    )

    outcome = machine.run(
        [ReasonCode.OCCLUSION_CHANGED, ReasonCode.FOOD_SATURATION_EXCEEDED],
        objects,
        CANVAS,
        lambda params: next(responses),
    )

    assert outcome.result == GateResult.PASS
    assert len(outcome.attempts) == 2  # not 3 - both categories were bundled into attempt 0

    first = outcome.attempts[0]
    assert first.reasons_addressed == [ReasonCode.OCCLUSION_CHANGED, ReasonCode.FOOD_SATURATION_EXCEEDED]
    assert first.params.layout_objects is not None  # occlusion's fix...
    assert first.params.color_params == conservative_color_params()  # ...applied in the SAME attempt as color's

    second = outcome.attempts[1]
    assert second.reasons_addressed == [ReasonCode.FOOD_SATURATION_EXCEEDED]  # occlusion no longer active
    assert second.params.color_params == neutral_color_params()  # progressed to the next color strategy


def test_a_brand_new_reason_appearing_after_a_fix_is_detected_and_retried():
    """
    Distinguishes "did the old reason disappear" from "re-validate
    everything from scratch": here, addressing FOOD_SATURATION_EXCEEDED
    causes an entirely different reason (CONTENT_HF_ERROR) to appear -
    never present in the original input. The state machine must pick it
    up from evaluate()'s full reasons list (not a diff against what it
    already knew) and retry it starting from its OWN first strategy.
    """
    config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={
            ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
            ReasonCode.CONTENT_HF_ERROR: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        },
        final_action="review",
    )
    machine = RetryStateMachine(config)
    objects = [make_food_object("a", (0, 0, 9, 9))]

    responses = iter(
        [
            ValidatorResult(result=GateResult.REJECT, reasons=[ReasonCode.CONTENT_HF_ERROR]),  # brand new reason
            ValidatorResult(result=GateResult.PASS, reasons=[]),
        ]
    )

    outcome = machine.run(
        [ReasonCode.FOOD_SATURATION_EXCEEDED],
        objects,
        CANVAS,
        lambda params: next(responses),
    )

    assert outcome.result == GateResult.PASS
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].reasons_addressed == [ReasonCode.FOOD_SATURATION_EXCEEDED]
    # the new reason starts its own strategy list from step 0, not wherever
    # FOOD_SATURATION_EXCEEDED's progress happened to be.
    assert outcome.attempts[1].reasons_addressed == [ReasonCode.CONTENT_HF_ERROR]
    assert outcome.attempts[1].strategies_applied[ReasonCode.CONTENT_HF_ERROR] == Strategy.CONSERVATIVE_COLOR


def test_no_fixable_reasons_makes_zero_attempts():
    config = RetryPolicyConfig(max_attempts=3, strategy_by_reason={}, final_action="review")
    machine = RetryStateMachine(config)

    def evaluate(params):
        raise AssertionError("evaluate() should never be called when nothing is retryable")

    outcome = machine.run([ReasonCode.MASK_CHANGED], [], CANVAS, evaluate)

    assert outcome.result == GateResult.REVIEW
    assert outcome.attempts == []
    assert outcome.final_reasons == [ReasonCode.MASK_CHANGED]
