from standard.reason_codes import GateResult, ReasonCode
from standard.validator.module_d import (
    compute_max_relative_scale_ratio_change,
    compute_scale_deltas,
    validate_layout_integrity,
)
from tests.standard.helpers import make_food_object, make_policy


def test_compute_scale_deltas():
    a = make_food_object("a", (0, 0, 9, 9), scale=1.0)
    b = make_food_object("b", (20, 20, 29, 29), scale=1.06)

    deltas = compute_scale_deltas([a, b])
    assert deltas["a"] == 0.0
    assert abs(deltas["b"] - 0.06) < 1e-9


def test_compute_max_relative_scale_ratio_change():
    a = make_food_object("a", (0, 0, 9, 9), scale=1.0)
    b = make_food_object("b", (20, 20, 29, 29), scale=0.97)

    change = compute_max_relative_scale_ratio_change([a, b])

    assert abs(change - (1.0 / 0.97 - 1.0)) < 1e-9


def test_grid_fallback_forces_review_even_with_no_other_violations():
    a = make_food_object("a", (0, 0, 9, 9), semantic_role="garnish", z_index=1)
    b = make_food_object("b", (50, 50, 59, 59), semantic_role="side", z_index=0)
    policy = make_policy()

    violations = validate_layout_integrity([a, b], [a, b], policy, used_grid_fallback=True)

    assert (GateResult.REVIEW, ReasonCode.LAYOUT_UNKNOWN_ROLE_MAJORITY) in violations
    assert len(violations) == 1  # nothing else about [a, b] is actually wrong


def test_no_grid_fallback_flag_when_not_used():
    a = make_food_object("a", (0, 0, 9, 9), semantic_role="garnish", z_index=1)
    b = make_food_object("b", (50, 50, 59, 59), semantic_role="side", z_index=0)
    policy = make_policy()

    violations = validate_layout_integrity([a, b], [a, b], policy, used_grid_fallback=False)

    assert violations == []


def test_no_violations_when_nothing_changed():
    a = make_food_object("a", (0, 0, 9, 9), semantic_role="garnish", z_index=1)
    b = make_food_object("b", (50, 50, 59, 59), semantic_role="side", z_index=0)
    policy = make_policy()

    assert validate_layout_integrity([a, b], [a, b], policy) == []


def test_rank_reversal_is_flagged_as_review_unconditionally():
    original = [
        make_food_object("main", (0, 0, 39, 39), semantic_role="main"),
        make_food_object("side", (50, 0, 59, 9), semantic_role="side"),
    ]
    final = [
        make_food_object("main", (0, 0, 9, 9), semantic_role="main"),
        make_food_object("side", (50, 0, 89, 39), semantic_role="side"),
    ]
    policy = make_policy()

    violations = validate_layout_integrity(original, final, policy)

    assert (GateResult.REVIEW, ReasonCode.OBJECT_AREA_RANK_REVERSED) in violations


def test_occlusion_change_violation_via_policy_severity():
    main = make_food_object("main", (0, 0, 69, 69), z_index=1, semantic_role="main")  # fully covers side
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")
    policy = make_policy()

    violations = validate_layout_integrity([main, side], [main, side], policy)

    assert (GateResult.REVIEW, ReasonCode.OCCLUSION_CHANGED) in violations


def test_occlusion_violation_escalates_to_reject_with_a_reject_multiplier():
    main = make_food_object("main", (0, 0, 69, 69), z_index=1, semantic_role="main")
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")
    policy = make_policy(
        overrides={
            "severity": {
                "layout": {
                    "max_scale_delta_per_object": {"reject_multiplier": None},
                    "max_relative_scale_ratio_change": {"reject_multiplier": None},
                    "max_occlusion_change": {"reject_multiplier": 1.0},
                }
            }
        }
    )

    violations = validate_layout_integrity([main, side], [main, side], policy)

    assert (GateResult.REJECT, ReasonCode.OCCLUSION_CHANGED) in violations


def test_scale_delta_violation():
    big = make_food_object("main", (0, 0, 9, 9), semantic_role="garnish", scale=1.06)
    other = make_food_object("side", (50, 50, 59, 59), semantic_role="side", scale=1.0)
    policy = make_policy()

    violations = validate_layout_integrity([big, other], [big, other], policy)

    assert (GateResult.REVIEW, ReasonCode.OBJECT_SCALE_CHANGED) in violations


def test_relative_scale_ratio_violation():
    a = make_food_object("a", (0, 0, 9, 9), semantic_role="garnish", scale=1.0)
    b = make_food_object("b", (50, 50, 59, 59), semantic_role="side", scale=0.97)
    policy = make_policy()

    violations = validate_layout_integrity([a, b], [a, b], policy)

    assert (GateResult.REVIEW, ReasonCode.RELATIVE_SCALE_CHANGED) in violations
