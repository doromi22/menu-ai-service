from standard.layout.translate import recenter_group
from tests.standard.helpers import make_food_object


def test_recenter_centers_group_bbox_on_canvas():
    a = make_food_object("a", (0, 0, 19, 19))  # group bbox spans (0,0)-(39,39) with b
    b = make_food_object("b", (20, 20, 39, 39))

    result = recenter_group([a, b], canvas_size=(100, 100))

    # group bbox center is (19.5, 19.5); canvas center is (50, 50) -> shift (30.5, 30.5)
    a_final = next(o for o in result if o.id == "a")
    b_final = next(o for o in result if o.id == "b")
    assert a_final.final_center == (9.5 + 30.5, 9.5 + 30.5)
    assert b_final.final_center == (29.5 + 30.5, 29.5 + 30.5)


def test_recenter_is_a_rigid_shift_preserving_relative_distance():
    a = make_food_object("a", (0, 0, 9, 9))
    b = make_food_object("b", (50, 50, 59, 59))

    result = recenter_group([a, b], canvas_size=(200, 200))
    a_final = next(o for o in result if o.id == "a")
    b_final = next(o for o in result if o.id == "b")

    original_dx = b.original_center[0] - a.original_center[0]
    final_dx = b_final.final_center[0] - a_final.final_center[0]
    assert final_dx == original_dx


def test_recenter_empty_list_returns_empty():
    assert recenter_group([], canvas_size=(100, 100)) == []


def test_recenter_never_changes_scale():
    # Locks in the invariant the Shadow Engine / Template Renderer's
    # known scale-mismatch limitation depends on being REVIEW-only
    # (README): the normal, non-grid-fallback layout path must never be
    # able to produce a non-1.0 scale on its own, since that would let
    # the mismatch reach a silently-PASSing frame no one reviews.
    a = make_food_object("a", (0, 0, 9, 9), scale=1.0)
    b = make_food_object("b", (20, 20, 29, 29), scale=1.0)

    result = recenter_group([a, b], canvas_size=(100, 100))

    assert all(obj.scale == 1.0 for obj in result)
