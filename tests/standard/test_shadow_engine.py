import numpy as np

from standard.shadow.engine import AmbientShadowParams, ShadowEngine, render_ambient_shadow, render_contact_shadow
from tests.standard.helpers import make_food_object

CANVAS_SHAPE = (100, 100)  # (height, width)
CANVAS_SIZE = (100, 100)  # (width, height) - square here so both read the same


def test_contact_shadow_shape_and_range():
    obj = make_food_object("main", (30, 30, 69, 69), canvas_size=CANVAS_SIZE, shadow_profile="default")

    density = render_contact_shadow(obj, CANVAS_SHAPE)

    assert density.shape == CANVAS_SHAPE
    assert density.min() >= 0.0
    assert density.max() <= 1.0
    assert density.max() > 0.0  # something was actually rendered


def test_contact_shadow_is_denser_under_a_heavy_profile_than_glass():
    heavy = make_food_object("bowl", (30, 30, 69, 69), canvas_size=CANVAS_SIZE, shadow_profile="heavy")
    glass = make_food_object("cup", (30, 30, 69, 69), canvas_size=CANVAS_SIZE, shadow_profile="glass")

    heavy_density = render_contact_shadow(heavy, CANVAS_SHAPE)
    glass_density = render_contact_shadow(glass, CANVAS_SHAPE)

    assert heavy_density.max() > glass_density.max()


def test_contact_shadow_is_zero_far_from_the_object():
    obj = make_food_object("main", (30, 30, 39, 39), canvas_size=CANVAS_SIZE, shadow_profile="default")

    density = render_contact_shadow(obj, CANVAS_SHAPE)

    assert density[0:5, 0:5].max() == 0.0  # far corner, well outside any blur radius


def test_ambient_shadow_empty_objects_is_all_zero():
    params = AmbientShadowParams(opacity=0.12, blur=42, offset_y=18)
    density = render_ambient_shadow([], CANVAS_SHAPE, params)
    assert np.all(density == 0.0)


def test_ambient_shadow_nonzero_near_object_union():
    a = make_food_object("a", (10, 10, 29, 29), canvas_size=CANVAS_SIZE)
    b = make_food_object("b", (60, 60, 79, 79), canvas_size=CANVAS_SIZE)
    params = AmbientShadowParams(opacity=0.12, blur=10, offset_y=5)

    density = render_ambient_shadow([a, b], CANVAS_SHAPE, params)

    assert density.max() > 0.0
    assert density.max() <= 1.0


def test_shadow_engine_combined_is_elementwise_max_of_ambient_and_contacts():
    a = make_food_object("a", (30, 30, 69, 69), canvas_size=CANVAS_SIZE, shadow_profile="heavy")
    params = AmbientShadowParams(opacity=0.12, blur=42, offset_y=18)

    layer = ShadowEngine().render([a], CANVAS_SHAPE, params)

    assert np.all(layer.combined >= layer.ambient)
    assert np.all(layer.combined >= layer.contacts["a"])
    assert np.array_equal(layer.combined, np.maximum(layer.ambient, layer.contacts["a"]))


def test_teishoku_multiple_objects_each_keep_their_own_contact_shadow():
    bowl = make_food_object("bowl", (10, 10, 39, 39), canvas_size=CANVAS_SIZE, shadow_profile="heavy")
    cup = make_food_object("cup", (60, 60, 79, 79), canvas_size=CANVAS_SIZE, shadow_profile="glass")
    params = AmbientShadowParams(opacity=0.12, blur=20, offset_y=8)

    layer = ShadowEngine().render([bowl, cup], CANVAS_SHAPE, params)

    assert set(layer.contacts.keys()) == {"bowl", "cup"}
    assert layer.contacts["bowl"].max() > layer.contacts["cup"].max()  # heavy > glass, as profiles


def test_rendering_is_deterministic():
    a = make_food_object("a", (30, 30, 69, 69), canvas_size=CANVAS_SIZE)
    params = AmbientShadowParams(opacity=0.12, blur=42, offset_y=18)

    first = ShadowEngine().render([a], CANVAS_SHAPE, params)
    second = ShadowEngine().render([a], CANVAS_SHAPE, params)

    assert np.array_equal(first.combined, second.combined)
