import numpy as np

from standard.reason_codes import GateResult, ReasonCode
from standard.validator.validator import run_validator
from tests.standard.helpers import make_food_object, make_policy

RNG = np.random.default_rng(7)


def _textured_image(shape=(40, 40)):
    gray = RNG.integers(0, 255, shape, dtype=np.uint8).astype(np.float32)
    return np.stack([gray, gray, gray], axis=-1)


def _base_kwargs():
    image = _textured_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[10:30, 10:30] = True
    objects = [
        make_food_object("main", (10, 10, 29, 29), canvas_size=(40, 40), semantic_role="main"),
    ]
    return dict(
        original_mask=mask,
        final_mask=mask,
        expected_food_rgb=image,
        final_food_rgb=image,
        food_mask=mask,
        original_food_rgb=image,
        original_objects=objects,
        final_objects=objects,
    )


def test_pass_when_nothing_changed():
    kwargs = _base_kwargs()
    policy = make_policy()

    result = run_validator(**kwargs, policy=policy)

    assert result.result == GateResult.PASS
    assert result.reasons == []


def test_review_when_one_module_finds_a_violation():
    kwargs = _base_kwargs()
    # brighten the food region only, well past max_luminance_gain
    brighter = kwargs["final_food_rgb"].copy()
    brighter[kwargs["food_mask"]] = np.clip(brighter[kwargs["food_mask"]] + 150, 0, 255)
    kwargs["final_food_rgb"] = brighter
    policy = make_policy()

    result = run_validator(**kwargs, policy=policy)

    assert result.result == GateResult.REVIEW
    assert ReasonCode.FOOD_LUMINANCE_EXCEEDED in result.reasons


def test_reject_when_a_module_escalates_via_policy_severity():
    kwargs = _base_kwargs()
    brighter = kwargs["final_food_rgb"].copy()
    brighter[kwargs["food_mask"]] = np.clip(brighter[kwargs["food_mask"]] + 150, 0, 255)
    kwargs["final_food_rgb"] = brighter
    policy = make_policy(overrides={"severity": {"food": {
        "max_saturation_gain": {"reject_multiplier": None},
        "max_luminance_gain": {"reject_multiplier": 1.0},
    }}})

    result = run_validator(**kwargs, policy=policy)

    assert result.result == GateResult.REJECT
    assert ReasonCode.FOOD_LUMINANCE_EXCEEDED in result.reasons


def test_reasons_accumulate_across_multiple_modules():
    kwargs = _base_kwargs()
    # Module C violation (luminance)
    brighter = kwargs["final_food_rgb"].copy()
    brighter[kwargs["food_mask"]] = np.clip(brighter[kwargs["food_mask"]] + 150, 0, 255)
    kwargs["final_food_rgb"] = brighter
    # Module D violation (scale)
    scaled_obj = make_food_object("main", (10, 10, 29, 29), canvas_size=(40, 40), semantic_role="main", scale=1.10)
    kwargs["final_objects"] = [scaled_obj]
    policy = make_policy()

    result = run_validator(**kwargs, policy=policy)

    assert ReasonCode.FOOD_LUMINANCE_EXCEEDED in result.reasons
    assert ReasonCode.OBJECT_SCALE_CHANGED in result.reasons
