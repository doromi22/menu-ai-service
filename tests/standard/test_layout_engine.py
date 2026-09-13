from standard.layout.engine import LayoutEngine, unknown_ratio
from standard.reason_codes import ReasonCode
from tests.standard.helpers import make_food_object


def test_unknown_ratio_computation():
    known = make_food_object("main", (0, 0, 9, 9), semantic_role="main")
    unknown = make_food_object("mystery", (20, 20, 29, 29), semantic_role="unknown")

    assert unknown_ratio([known, unknown]) == 0.5
    assert unknown_ratio([known]) == 0.0
    assert unknown_ratio([]) == 0.0


def test_normal_case_uses_recenter_and_reports_no_review_reasons():
    main = make_food_object("main", (0, 0, 49, 49), z_index=1, semantic_role="main")  # occlusion ratio 0.25 vs baseline 0.18
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    engine = LayoutEngine()
    result = engine.run([main, side], canvas_size=(100, 100))

    assert result.used_grid_fallback is False
    assert result.unknown_ratio == 0.0
    assert result.rank_preserved is True
    assert result.occlusion_violations == {}
    assert result.review_reasons == []
    # recenter_group actually ran: final_center should differ from a raw pass-through
    # whenever the group isn't already canvas-centered.
    assert any(o.final_center != o.original_center for o in result.objects)


def test_unknown_majority_triggers_grid_fallback():
    main = make_food_object("main", (0, 0, 9, 9), semantic_role="main")
    mystery_a = make_food_object("m1", (20, 20, 29, 29), semantic_role="unknown")
    mystery_b = make_food_object("m2", (40, 40, 49, 49), semantic_role="unknown")

    engine = LayoutEngine()
    result = engine.run([main, mystery_a, mystery_b], canvas_size=(100, 100))

    assert result.used_grid_fallback is True
    assert result.unknown_ratio == 2 / 3
    assert ReasonCode.LAYOUT_UNKNOWN_ROLE_MAJORITY in result.review_reasons
    # grid fallback places 3 objects on a 2x2 grid (ceil(sqrt(3)) = 2 cols)
    assert len(result.objects) == 3


def test_large_occlusion_change_is_flagged():
    main = make_food_object("main", (0, 0, 69, 69), z_index=1, semantic_role="main")  # fully covers side
    side = make_food_object("side", (30, 30, 69, 69), z_index=0, semantic_role="side")

    engine = LayoutEngine()
    result = engine.run([main, side], canvas_size=(100, 100))

    assert ("main", "side") in result.occlusion_violations
    assert ReasonCode.OCCLUSION_CHANGED in result.review_reasons
