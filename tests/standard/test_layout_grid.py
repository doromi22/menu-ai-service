from standard.layout.grid import arrange_grid
from standard.objects.food_object import SCALE_MIN
from tests.standard.helpers import make_food_object


def test_four_objects_placed_on_a_2x2_grid():
    objects = [make_food_object(f"o{i}", (40, 40, 49, 49)) for i in range(4)]  # small, centered bboxes

    placed = arrange_grid(objects, canvas_size=(100, 100))

    assert [o.final_center for o in placed] == [(25, 25), (75, 25), (25, 75), (75, 75)]


def test_oversized_object_scale_is_clamped_to_fit_its_cell():
    oversized = make_food_object("big", (0, 0, 79, 9))  # 80 wide x 10 tall
    filler = [make_food_object(f"o{i}", (40, 40, 49, 49)) for i in range(1, 4)]

    placed = arrange_grid([oversized] + filler, canvas_size=(100, 100))  # 2x2 grid, 50x50 cells

    big_final = next(o for o in placed if o.id == "big")
    # naive fit would be 50/80 = 0.625, but that's below SCALE_MIN - must clamp, not use the raw ratio
    assert big_final.scale == SCALE_MIN
    assert big_final.is_scale_within_bounds


def test_arrange_grid_empty_list_returns_empty():
    assert arrange_grid([], canvas_size=(100, 100)) == []
