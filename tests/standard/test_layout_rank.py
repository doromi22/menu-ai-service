from standard.layout.rank import compute_role_area_rank, is_area_rank_preserved
from tests.standard.helpers import make_food_object


def test_rank_orders_by_total_area_descending():
    main = make_food_object("main", (0, 0, 39, 39), semantic_role="main")  # 1600px
    rice = make_food_object("rice", (50, 0, 69, 19), semantic_role="rice")  # 400px
    side = make_food_object("side", (50, 50, 59, 59), semantic_role="side")  # 100px

    assert compute_role_area_rank([main, rice, side]) == ["main", "rice", "side"]


def test_unknown_objects_excluded_from_rank():
    main = make_food_object("main", (0, 0, 39, 39), semantic_role="main")
    mystery = make_food_object("mystery", (50, 50, 90, 90), semantic_role="unknown")  # bigger, but untrusted

    assert compute_role_area_rank([main, mystery]) == ["main"]


def test_ties_broken_alphabetically_for_determinism():
    a = make_food_object("a", (0, 0, 9, 9), semantic_role="side")  # 100px
    b = make_food_object("b", (50, 50, 59, 59), semantic_role="garnish")  # 100px, same area

    assert compute_role_area_rank([a, b]) == ["garnish", "side"]


def test_rank_preserved_when_areas_unchanged():
    original = [
        make_food_object("main", (0, 0, 39, 39), semantic_role="main"),
        make_food_object("side", (50, 50, 59, 59), semantic_role="side"),
    ]
    # same objects, only final_center would differ in a real layout pass - area rank uses mask area, not position
    assert is_area_rank_preserved(original, original) is True


def test_rank_reversed_between_original_and_final():
    original = [
        make_food_object("main", (0, 0, 39, 39), semantic_role="main"),  # 1600px
        make_food_object("side", (50, 0, 59, 9), semantic_role="side"),  # 100px
    ]
    # simulate a final state where side somehow became the bigger role
    final = [
        make_food_object("main", (0, 0, 9, 9), semantic_role="main"),  # 100px
        make_food_object("side", (50, 0, 89, 39), semantic_role="side"),  # 1600px
    ]
    assert is_area_rank_preserved(original, final) is False
