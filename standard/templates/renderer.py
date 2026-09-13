"""
Scene compositor (spec §9, §11 step 6): places Layout Engine's positioned
FoodObjects and Shadow Engine's rendered shadow layer onto a Template's
background. This is the piece earlier modules' docstrings referred to as
not built yet (Shadow Engine's grayscale-only output, the full-pipeline
integration test's synthetic "background_graded" stand-in) - it exists
now, still GPU-free (PIL/numpy only).

Expects `graded_food_image_rgb` - the ALREADY color-graded source image
(spec §5's food_params applied, upstream of this module) - because per §9
the real background is this template's synthetic one, not a graded
version of the original photo's background region; grading the original
image's background pixels (as `color.grade.grade_food_and_background`
does) was only ever a stand-in for testing Color Grade in isolation
before this renderer existed.

Known simplification: Shadow Engine renders shadows from each object's
*unscaled* mask (§4/§11 step 2 predates scale-aware placement). When
`scale != 1.0` (grid-fallback only, §2), the rendered shadow's shape won't
match the resized food exactly. Not fixed here - flagged, like the other
provisional formulas in this build.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from standard.objects.food_object import FoodObject
from standard.shadow.engine import AmbientShadowParams, ShadowEngine
from standard.templates.background import render_background
from standard.templates.definitions import Template


def _place_object(
    obj: FoodObject, source_image_rgb: np.ndarray, canvas_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mask, image), both canvas-sized, with this object's own
    cropped mask/pixels resized by `scale` and recentered on
    `final_center` - the one place in the pipeline scale actually changes
    what gets rendered, not just what gets checked (spec §4).
    """
    canvas_h, canvas_w = canvas_shape
    x0, y0, x1, y1 = obj.bbox
    crop_mask = obj.mask[y0 : y1 + 1, x0 : x1 + 1]
    crop_image = source_image_rgb[y0 : y1 + 1, x0 : x1 + 1]

    if obj.scale != 1.0:
        new_w = max(1, round((x1 - x0 + 1) * obj.scale))
        new_h = max(1, round((y1 - y0 + 1) * obj.scale))
        crop_mask = np.asarray(Image.fromarray(crop_mask).resize((new_w, new_h), Image.NEAREST))
        crop_image = np.asarray(Image.fromarray(crop_image).resize((new_w, new_h), Image.BILINEAR))

    ch, cw = crop_mask.shape[:2]
    dst_x0 = round(obj.final_center[0] - cw / 2)
    dst_y0 = round(obj.final_center[1] - ch / 2)
    dst_x1, dst_y1 = dst_x0 + cw, dst_y0 + ch

    src_x0, src_y0 = 0, 0
    if dst_x0 < 0:
        src_x0, dst_x0 = -dst_x0, 0
    if dst_y0 < 0:
        src_y0, dst_y0 = -dst_y0, 0
    dst_x1 = min(dst_x1, canvas_w)
    dst_y1 = min(dst_y1, canvas_h)
    src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
    src_y1 = src_y0 + max(0, dst_y1 - dst_y0)

    full_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    full_image = np.zeros((canvas_h, canvas_w, 3), dtype=source_image_rgb.dtype)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        full_mask[dst_y0:dst_y1, dst_x0:dst_x1] = crop_mask[src_y0:src_y1, src_x0:src_x1]
        full_image[dst_y0:dst_y1, dst_x0:dst_x1] = crop_image[src_y0:src_y1, src_x0:src_x1]
    return full_mask, full_image


def place_food_layer(
    objects: list[FoodObject], canvas_shape: tuple[int, int], source_image_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composites just the food layer (mask + pixels) at each object's final
    position/scale, painted back-to-front by z_index - the exact placement
    `TemplateRenderer.render` uses for the real scene, factored out so a
    caller can place a DIFFERENT source array (e.g. the true original,
    ungraded photo) at those same final positions for a fair, aligned
    comparison against the real render.

    This is what makes the Integrity Validator's Module B/C wiring
    correct without assuming layout applied one uniform shift to every
    object - grid-fallback's per-object fit-scaling and the Retry state
    machine's `recompute_translation` (which nudges objects by different
    amounts depending on distance from the group centroid) both produce
    non-uniform per-object shifts, so re-deriving "one delta" and shifting
    a whole image by it would be wrong in those cases. Redoing the actual
    per-object placement is correct in every case.
    """
    canvas_h, canvas_w = canvas_shape
    combined_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    combined_image = np.zeros((canvas_h, canvas_w, 3), dtype=source_image_rgb.dtype)
    for obj in sorted(objects, key=lambda o: o.z_index):  # back to front
        mask, image = _place_object(obj, source_image_rgb, canvas_shape)
        combined_mask = combined_mask | mask
        combined_image = np.where(mask[..., None], image, combined_image)
    return combined_mask, combined_image


class TemplateRenderer:
    def __init__(self, shadow_engine: ShadowEngine | None = None) -> None:
        self._shadow_engine = shadow_engine or ShadowEngine()

    def render(
        self,
        template: Template,
        objects: list[FoodObject],
        canvas_size: tuple[int, int],
        graded_food_image_rgb: np.ndarray,
    ) -> np.ndarray:
        width, height = canvas_size
        canvas_shape = (height, width)

        background = render_background(template, canvas_size).astype(np.float32)

        ambient_params = AmbientShadowParams(
            opacity=template.shadow.opacity, blur=template.shadow.blur, offset_y=template.shadow.offset_y
        )
        shadow_layer = self._shadow_engine.render(objects, canvas_shape, ambient_params)
        scene = background * (1.0 - shadow_layer.combined[..., None])

        food_mask, food_image = place_food_layer(objects, canvas_shape, graded_food_image_rgb)
        scene = np.where(food_mask[..., None], food_image.astype(np.float32), scene)

        return np.clip(scene, 0, 255).astype(np.uint8)
