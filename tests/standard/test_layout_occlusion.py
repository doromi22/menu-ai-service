from standard.layout.occlusion import (
    compute_occlusion_changes,
    compute_role_pair_occlusions,
    occlusion_change_violations,
    occlusion_ratio,
)
from tests.standard.helpers import make_food_object


def test_occlusion_ratio_is_fraction_of_back_covered_by_front():
    main = make_food_object("main", (0, 0, 49, 49), z_index=1)  # 50x50 = 2500px
    side = make_food_object("side", (30, 30, 69, 69), z_index=0)  # 40x40 = 1600px, overlap [30,49]x[30,49]=400px

    assert occlusion_ratio(main, side) == 400 / 1600


def test_occlusion_ratio_uses_final_center_not_original_position():
    # main starts far from side; only its *final* position overlaps.
    main = make_food_object("main", (0, 0, 19, 19), z_index=1, final_center=(60.5, 60.5))
    side = make_food_object("side", (50, 50, 69, 69), z_index=0)

    ratio = occlusion_ratio(main, side)
    assert ratio > 0.8  # shifted mask now covers most of side


def test_compute_role_pair_occlusions_respects_z_index_front_back_convention():
    main = make_food_object("main", (0, 0, 49, 49), z_index=1, semantic_role="main")
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    pairs = compute_role_pair_occlusions([main, side])

    assert ("main", "side") in pairs
    assert ("side", "main") not in pairs  # side has the lower z_index, so it isn't "in front"


def test_compute_occlusion_changes_against_baseline():
    main = make_food_object("main", (0, 0, 49, 49), z_index=1, semantic_role="main")  # covers side fully
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    changes = compute_occlusion_changes([main, side], baseline={("main", "side"): 0.18})

    assert changes[("main", "side")] == abs(400 / 1600 - 0.18)


def test_occlusion_change_violations_flags_only_large_changes():
    main = make_food_object("main", (0, 0, 69, 69), z_index=1, semantic_role="main")  # fully covers side
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    violations = occlusion_change_violations([main, side], baseline={("main", "side"): 0.18}, max_change=0.15)

    assert ("main", "side") in violations
    assert violations[("main", "side")] == abs(1.0 - 0.18)


def test_occlusion_change_violations_empty_when_within_tolerance():
    main = make_food_object("main", (0, 0, 49, 49), z_index=1, semantic_role="main")  # ratio 0.25 vs baseline 0.18
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    violations = occlusion_change_violations([main, side], baseline={("main", "side"): 0.18}, max_change=0.15)

    assert violations == {}
