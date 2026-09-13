"""
Shared translation helpers for anything shaped (H, W, ...) on the pipeline's
canvas - FoodObject.mask (H, W) and RGB image arrays (H, W, 3) alike.
FoodObject.mask never changes shape after the Segmentation Gate -
repositioning is represented purely by original_center -> final_center, so
anything downstream that needs "what does this look like at its final
position" (Layout Engine's occlusion check, Shadow Engine's contact/ambient
rendering, the Template Renderer's food placement) shifts the same array
rather than re-segmenting or resampling.
"""
from __future__ import annotations

import numpy as np


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Shift an array by an integer (dy, dx) along its first two axes, filling
    the revealed area with zeros/False. Works for a 2D boolean mask (H, W)
    or an (H, W, C) image array equally - only the leading two axes matter.
    """
    if dy == 0 and dx == 0:
        return mask

    h, w = mask.shape[:2]
    result = np.zeros_like(mask)

    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return result


def center_delta(original_center: tuple[float, float], final_center: tuple[float, float]) -> tuple[int, int]:
    """(dy, dx) pixel shift implied by a center move, rounded to the nearest integer pixel."""
    dx = final_center[0] - original_center[0]
    dy = final_center[1] - original_center[1]
    return round(dy), round(dx)


def translate_mask_by_center_delta(
    mask: np.ndarray, original_center: tuple[float, float], final_center: tuple[float, float]
) -> np.ndarray:
    dy, dx = center_delta(original_center, final_center)
    return translate_mask(mask, dy, dx)
