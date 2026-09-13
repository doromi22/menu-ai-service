"""Color grade presets (spec §5, numbers exact)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorGradeParams:
    brightness: float
    contrast: float
    saturation: float
    temperature_strength: float


STANDARD = ColorGradeParams(brightness=1.05, contrast=1.04, saturation=1.03, temperature_strength=0.08)
PREMIUM = ColorGradeParams(brightness=1.08, contrast=1.08, saturation=1.06, temperature_strength=0.15)
HERO = ColorGradeParams(brightness=1.12, contrast=1.12, saturation=1.10, temperature_strength=0.22)

# §5: "Food의 saturation multiplier는 초기값 1.05 상한으로 시작, A/B 데이터로 조정."
FOOD_SATURATION_CAP = 1.05

# §5: "Global(배경) vs Food 보정 강도를 분리한다 (Food는 항상 더 보수적)." No
# exact ratio is given in the spec - 0.5 (half the background's delta from
# neutral) is this build's provisional choice, tunable later from A/B data
# same as the saturation cap above.
FOOD_CONSERVATISM = 0.5


def food_params_for(preset: ColorGradeParams, conservatism: float = FOOD_CONSERVATISM) -> ColorGradeParams:
    """The same preset, scaled toward neutral (1.0) by `conservatism`, with saturation additionally capped."""

    def scale(value: float) -> float:
        return 1.0 + (value - 1.0) * conservatism

    return ColorGradeParams(
        brightness=scale(preset.brightness),
        contrast=scale(preset.contrast),
        saturation=min(scale(preset.saturation), FOOD_SATURATION_CAP),
        temperature_strength=preset.temperature_strength * conservatism,
    )
