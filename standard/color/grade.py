"""
Color Grade (spec §5): Standard-tier operations only.

**Deliberately not implemented in this module** - not stubbed, not
disabled by a flag, simply never written, so nobody can wire one in by
accident later: CLAHE, local contrast, object-aware enhancement,
edge-aware sharpening, adaptive tone mapping. Spec §5: "이 제약이 있어야
뒤의 transform-aware validator가 자기모순 없이 작동한다" - the Integrity
Validator's Content module (§6 module B, §11 step 4) compares the actual
output against `apply_grade(original, these_same_params)`; if grading
ever became image-dependent (adaptive), that comparison would no longer
mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance

from standard.color.presets import ColorGradeParams, FOOD_CONSERVATISM, food_params_for


def apply_brightness_contrast_saturation(image: Image.Image, params: ColorGradeParams) -> Image.Image:
    out = ImageEnhance.Brightness(image).enhance(params.brightness)
    out = ImageEnhance.Contrast(out).enhance(params.contrast)
    out = ImageEnhance.Color(out).enhance(params.saturation)  # PIL's "Color" enhancer = saturation
    return out


def apply_white_balance(image: Image.Image, strength: float) -> Image.Image:
    """Global warm/cool shift: nudges R up and B down uniformly by `strength`. Not adaptive - the
    same multiplier applies everywhere regardless of this image's own color statistics."""
    if strength == 0:
        return image
    arr = np.asarray(image).astype(np.float32)
    arr[..., 0] *= 1 + strength * 0.06
    arr[..., 2] *= 1 - strength * 0.06
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _fixed_tone_curve_lut(strength: float) -> np.ndarray:
    """
    A fixed (non-adaptive) S-curve LUT: the same smoothstep formula every
    time, blended toward identity by `strength` alone - never derived from
    this image's own histogram, which is exactly what distinguishes it
    from the adaptive tone mapping spec §5 excludes.
    """
    x = np.linspace(0.0, 1.0, 256)
    smoothstep = x * x * (3 - 2 * x)
    curve = x + strength * (smoothstep - x)
    return np.clip(np.round(curve * 255.0), 0, 255).astype(np.uint8)


def apply_fixed_tone_curve(image: Image.Image, strength: float) -> Image.Image:
    lut = _fixed_tone_curve_lut(strength).tolist()
    return image.point(lut * len(image.getbands()))


def apply_grade(image: Image.Image, params: ColorGradeParams) -> Image.Image:
    out = apply_brightness_contrast_saturation(image, params)
    out = apply_white_balance(out, params.temperature_strength)
    out = apply_fixed_tone_curve(out, params.temperature_strength)
    return out


@dataclass
class ColorGradeResult:
    image: Image.Image
    background_params: ColorGradeParams
    food_params: ColorGradeParams


def grade_food_and_background(
    image: Image.Image,
    food_mask: np.ndarray,
    preset: ColorGradeParams,
    food_conservatism: float = FOOD_CONSERVATISM,
) -> ColorGradeResult:
    """
    Grades the whole image twice - once at full preset strength (for the
    background), once at the more conservative food strength (§5) - and
    composites: background pixels come from the full-strength pass, food
    pixels from the conservative pass. Returns the params actually used
    for each, needed by the Integrity Validator's transform-aware Content
    check (§6 module B): `Expected Food = apply_grade(Original, food_params)`.
    """
    food_params = food_params_for(preset, food_conservatism)
    background_graded = apply_grade(image, preset)
    food_graded = apply_grade(image, food_params)

    mask_image = Image.fromarray((food_mask.astype(np.uint8)) * 255, mode="L")
    composed = Image.composite(food_graded, background_graded, mask_image)

    return ColorGradeResult(image=composed, background_params=preset, food_params=food_params)
