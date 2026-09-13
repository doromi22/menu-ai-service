"""
Graphic template definitions (spec §9): background is managed as
PARAMETERS, not images - PIL/numpy only, no GPU, ~zero marginal cost per
render. A template never references a food category; per §9, that
separation is complete ("Template ↔ Food category 완전 분리 (Template은
카테고리를 몰라도 됨)") - only the Layout Engine's single/multi-object
split (already built) affects composition.

Only `T01_warm_ivory`'s numbers come from the spec text itself (§9's exact
example). The spec names five more templates - Cool White, Beige, Dark
Premium, Japanese Editorial, Fresh Natural (§9, §10) - but gives no
parameter values for them. Their entries below are this build's own
placeholder values, clearly not validated against anything, standing in
until the §10 360-image MVP test (10 images/cell, 6 templates x 6
categories) produces real numbers to replace them with.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowSpec:
    opacity: float
    blur: float
    offset_y: int


@dataclass(frozen=True)
class Template:
    id: str
    base_color: str  # "#RRGGBB"
    gradient_direction: str  # one of _GRADIENT_ANCHORS in background.py
    gradient_strength: float
    texture_strength: float
    shadow: ShadowSpec


TEMPLATES: dict[str, Template] = {
    "T01_warm_ivory": Template(
        id="T01_warm_ivory",
        base_color="#F1E8D8",
        gradient_direction="top_left",
        gradient_strength=0.08,
        texture_strength=0.015,
        shadow=ShadowSpec(opacity=0.12, blur=42, offset_y=18),
    ),
    # --- placeholders below (see module docstring) ---
    "T02_cool_white": Template(
        id="T02_cool_white",
        base_color="#F5F6F8",
        gradient_direction="top",
        gradient_strength=0.06,
        texture_strength=0.010,
        shadow=ShadowSpec(opacity=0.10, blur=38, offset_y=16),
    ),
    "T03_beige": Template(
        id="T03_beige",
        base_color="#E8DCC8",
        gradient_direction="bottom_right",
        gradient_strength=0.10,
        texture_strength=0.020,
        shadow=ShadowSpec(opacity=0.14, blur=40, offset_y=18),
    ),
    "T04_dark_premium": Template(
        id="T04_dark_premium",
        base_color="#2B2A28",
        gradient_direction="top",
        gradient_strength=0.14,
        texture_strength=0.020,
        shadow=ShadowSpec(opacity=0.35, blur=48, offset_y=20),
    ),
    "T05_japanese_editorial": Template(
        id="T05_japanese_editorial",
        base_color="#DCD3C3",
        gradient_direction="top_right",
        gradient_strength=0.09,
        texture_strength=0.025,
        shadow=ShadowSpec(opacity=0.16, blur=44, offset_y=20),
    ),
    "T06_fresh_natural": Template(
        id="T06_fresh_natural",
        base_color="#EFF2E6",
        gradient_direction="bottom",
        gradient_strength=0.07,
        texture_strength=0.015,
        shadow=ShadowSpec(opacity=0.11, blur=36, offset_y=14),
    ),
}
