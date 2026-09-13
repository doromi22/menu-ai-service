"""
Deterministic, GPU-free background rendering from Template parameters
(spec §9). No ML and no random noise - "texture" is a fixed procedural
pattern (a function of pixel position only), so the same template always
renders byte-identical output. That determinism matters beyond tidiness:
the Integrity Validator's transform-aware comparisons elsewhere in this
pipeline assume a given input always produces the same output.
"""
from __future__ import annotations

import numpy as np

from standard.templates.definitions import Template

_GRADIENT_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _gradient_field(canvas_size: tuple[int, int], direction: str) -> np.ndarray:
    """1.0 nearest the named anchor corner/edge, fading toward 0.0 at the opposite corner."""
    width, height = canvas_size
    anchor = _GRADIENT_ANCHORS.get(direction, _GRADIENT_ANCHORS["top_left"])
    ys, xs = np.mgrid[0:height, 0:width]
    xs_n = xs / max(width - 1, 1)
    ys_n = ys / max(height - 1, 1)
    dist = np.sqrt((xs_n - anchor[0]) ** 2 + (ys_n - anchor[1]) ** 2)
    return 1.0 - np.clip(dist / np.sqrt(2), 0.0, 1.0)


def _fixed_texture(canvas_size: tuple[int, int]) -> np.ndarray:
    """A fixed procedural pattern in [-1, 1] - purely a function of pixel position, no RNG involved."""
    width, height = canvas_size
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = np.sin(xs * 0.9) * np.cos(ys * 0.7) + np.sin((xs + ys) * 0.35)
    peak = np.abs(pattern).max()
    return pattern / peak if peak else pattern


def render_background(template: Template, canvas_size: tuple[int, int]) -> np.ndarray:
    """canvas_size = (width, height). Returns an (height, width, 3) uint8 array."""
    base = _hex_to_rgb(template.base_color)
    lighter = np.clip(base * 1.15, 0, 255)

    gradient = _gradient_field(canvas_size, template.gradient_direction)
    gradient_rgb = base + (lighter - base) * (gradient[..., None] * template.gradient_strength)

    texture = _fixed_texture(canvas_size) * template.texture_strength * 255.0
    combined = gradient_rgb + texture[..., None]

    return np.clip(combined, 0, 255).astype(np.uint8)
