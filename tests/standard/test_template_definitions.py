from standard.templates.definitions import TEMPLATES, ShadowSpec, Template


def test_warm_ivory_matches_spec_exactly():
    t = TEMPLATES["T01_warm_ivory"]
    assert t.base_color == "#F1E8D8"
    assert t.gradient_direction == "top_left"
    assert t.gradient_strength == 0.08
    assert t.texture_strength == 0.015
    assert t.shadow == ShadowSpec(opacity=0.12, blur=42, offset_y=18)


def test_six_templates_are_defined():
    assert len(TEMPLATES) == 6


def test_every_template_has_a_valid_hex_color():
    for template in TEMPLATES.values():
        assert template.base_color.startswith("#")
        assert len(template.base_color) == 7
        int(template.base_color[1:], 16)  # must not raise


def test_templates_do_not_reference_a_food_category():
    # structural check: Template has no category-shaped field at all -
    # spec §9's "Template <-> Food category 완전 분리".
    assert not hasattr(Template, "category")
    assert not hasattr(Template, "food_category")
