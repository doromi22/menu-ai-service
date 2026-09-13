from standard.color.presets import (
    FOOD_SATURATION_CAP,
    HERO,
    PREMIUM,
    STANDARD,
    ColorGradeParams,
    food_params_for,
)


def test_preset_values_match_spec_exactly():
    assert STANDARD == ColorGradeParams(1.05, 1.04, 1.03, 0.08)
    assert PREMIUM == ColorGradeParams(1.08, 1.08, 1.06, 0.15)
    assert HERO == ColorGradeParams(1.12, 1.12, 1.10, 0.22)


def test_food_params_are_more_conservative_than_background():
    for preset in (STANDARD, PREMIUM, HERO):
        food = food_params_for(preset)
        assert abs(food.brightness - 1.0) < abs(preset.brightness - 1.0)
        assert abs(food.contrast - 1.0) < abs(preset.contrast - 1.0)
        assert food.temperature_strength < preset.temperature_strength


def test_food_saturation_never_exceeds_cap():
    for preset in (STANDARD, PREMIUM, HERO):
        assert food_params_for(preset).saturation <= FOOD_SATURATION_CAP


def test_hero_food_saturation_lands_exactly_on_the_cap():
    # 1.0 + (1.10 - 1.0) * 0.5 = 1.05, which happens to equal the cap exactly.
    assert food_params_for(HERO).saturation == FOOD_SATURATION_CAP


def test_cap_actually_clamps_when_scaled_value_would_exceed_it():
    hot_preset = ColorGradeParams(brightness=1.0, contrast=1.0, saturation=1.20, temperature_strength=0.0)
    # scaled: 1.0 + 0.20 * 0.5 = 1.10, above the 1.05 cap - must clamp, not pass through.
    assert food_params_for(hot_preset, conservatism=0.5).saturation == FOOD_SATURATION_CAP
