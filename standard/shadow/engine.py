"""
Shadow Engine (spec §4/§1): deterministic rendering only, no policy
judgement - this stage never decides PASS/REVIEW/REJECT.

Produces a grayscale "shadow density" map (float32, 0..1 per pixel) rather
than a full RGBA composite: actual background pixels and color don't
exist yet at this point in the pipeline (the Graphic template renderer,
§9, is still pending - §11 build order), so this is exactly the
information a future renderer needs to darken its background layer
underneath the food, and no more.

Two contributions are combined:
- **Ambient**: one shadow for the whole group, driven by the *template's*
  shadow block (§9: `{"opacity", "blur", "offset_y"}`) - a single soft
  shadow under the whole composition.
- **Contact**: one per object, driven by that object's own
  `shadow_profile` (standard/shadow/profiles.py) - independent of the
  template, so different dishes in the same frame (e.g. Teishoku) can cast
  visibly different contact shadows, per spec §4's explicit requirement
  ("단일 giant shadow 금지").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from standard.geometry import translate_mask_by_center_delta
from standard.objects.food_object import FoodObject
from standard.shadow.profiles import resolve_contact_profile


@dataclass(frozen=True)
class AmbientShadowParams:
    """Mirrors a template's §9 `shadow` block."""

    opacity: float
    blur: float
    offset_y: int


# Note: `canvas_shape` below is a numpy array shape (height, width), matching
# FoodObject.mask.shape - unlike standard.layout's `canvas_size`, which is
# (width, height) to match the bbox/(x, y) convention used there.


def _placed_mask(obj: FoodObject) -> np.ndarray:
    return translate_mask_by_center_delta(obj.mask, obj.original_center, obj.final_center)


def render_contact_shadow(obj: FoodObject, canvas_shape: tuple[int, int]) -> np.ndarray:
    params = resolve_contact_profile(obj.shadow_profile)
    mask = _placed_mask(obj)
    offset = ndimage.shift(mask.astype(np.float32), shift=(params.offset_y, 0), order=0, cval=0.0)
    blurred = ndimage.gaussian_filter(offset, sigma=max(params.blur, 0.1))
    density = np.clip(blurred * params.opacity, 0.0, 1.0)
    assert density.shape == canvas_shape
    return density


def render_ambient_shadow(
    objects: list[FoodObject], canvas_shape: tuple[int, int], params: AmbientShadowParams
) -> np.ndarray:
    if not objects:
        return np.zeros(canvas_shape, dtype=np.float32)

    union = np.logical_or.reduce([_placed_mask(obj) for obj in objects])
    offset = ndimage.shift(union.astype(np.float32), shift=(params.offset_y, 0), order=0, cval=0.0)
    blurred = ndimage.gaussian_filter(offset, sigma=max(params.blur, 0.1))
    return np.clip(blurred * params.opacity, 0.0, 1.0)


@dataclass
class ShadowLayer:
    ambient: np.ndarray
    contacts: dict[str, np.ndarray]  # FoodObject.id -> density map
    combined: np.ndarray  # elementwise max(ambient, all contacts)


class ShadowEngine:
    def render(
        self, objects: list[FoodObject], canvas_shape: tuple[int, int], ambient_params: AmbientShadowParams
    ) -> ShadowLayer:
        ambient = render_ambient_shadow(objects, canvas_shape, ambient_params)
        contacts = {obj.id: render_contact_shadow(obj, canvas_shape) for obj in objects}

        combined = ambient
        for density in contacts.values():
            combined = np.maximum(combined, density)

        return ShadowLayer(ambient=ambient, contacts=contacts, combined=combined)
