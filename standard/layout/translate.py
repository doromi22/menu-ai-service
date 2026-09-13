"""Translation-first arrangement (spec §4): the default, non-fallback path."""
from __future__ import annotations

from dataclasses import replace

from standard.objects.food_object import FoodObject


def recenter_group(objects: list[FoodObject], canvas_size: tuple[int, int]) -> list[FoodObject]:
    """
    Translate every object by the same (dx, dy) so the group's combined
    bounding box is centered on the canvas - a rigid shift, so it never
    changes objects' scale, rotation, or relative positions (occlusion
    between objects is unaffected by a shared translation).
    """
    if not objects:
        return []

    x0 = min(obj.bbox[0] for obj in objects)
    y0 = min(obj.bbox[1] for obj in objects)
    x1 = max(obj.bbox[2] for obj in objects)
    y1 = max(obj.bbox[3] for obj in objects)
    group_cx, group_cy = (x0 + x1) / 2, (y0 + y1) / 2

    width, height = canvas_size
    dx, dy = width / 2 - group_cx, height / 2 - group_cy

    return [
        replace(obj, final_center=(obj.original_center[0] + dx, obj.original_center[1] + dy))
        for obj in objects
    ]
