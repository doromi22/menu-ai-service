"""
Grid fallback arrangement (spec §2: "unknown 비율 > 50%인 프레임은 layout
engine을 단순 grid/원본 배치로 낮추고 자동 REVIEW").

Simple, deterministic row-major grid. When an object's bbox is bigger than
its assigned cell, its scale is shrunk to fit - clamped into the 0.97-1.03
band via `clamp_scale`, so even the fallback path never produces an
out-of-band scale (spec §4).
"""
from __future__ import annotations

import math
from dataclasses import replace

from standard.layout.scale import clamp_scale
from standard.objects.food_object import FoodObject


def arrange_grid(objects: list[FoodObject], canvas_size: tuple[int, int]) -> list[FoodObject]:
    if not objects:
        return []

    width, height = canvas_size
    n = len(objects)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = width / cols
    cell_h = height / rows

    placed = []
    for i, obj in enumerate(objects):
        row, col = divmod(i, cols)
        cx = cell_w * (col + 0.5)
        cy = cell_h * (row + 0.5)

        obj_w = obj.bbox[2] - obj.bbox[0] + 1
        obj_h = obj.bbox[3] - obj.bbox[1] + 1
        fit_scale = min(cell_w / obj_w, cell_h / obj_h, 1.0) if obj_w and obj_h else 1.0
        scale = clamp_scale(fit_scale)

        placed_obj = replace(obj, final_center=(cx, cy), scale=scale)
        assert placed_obj.is_scale_within_bounds  # clamp_scale must guarantee this
        placed.append(placed_obj)
    return placed
