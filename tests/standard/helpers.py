"""Shared test builders for standard/ - not a test module itself (no test_ prefix)."""
from __future__ import annotations

import copy

import numpy as np

from standard.objects.food_object import FoodObject
from standard.policy.schema import Policy

BASE_POLICY_RAW = {
    "policy_version": "test_1.0.0",
    "mask": {"min_iou": 0.995, "max_area_delta": 0.01},
    "high_frequency": {"base_threshold": 0.04, "contrast_weight": 0.35, "sharpness_weight": 0.50},
    "food": {"max_saturation_gain": 0.05, "max_luminance_gain": 0.10},
    "layout": {
        "max_scale_delta_per_object": 0.03,
        "max_relative_scale_ratio_change": 0.02,
        "preserve_object_area_rank": True,
        "max_occlusion_change": 0.15,
        "preserve_front_back_order": True,
    },
    "severity": {
        "mask": {"min_iou": {"reject_multiplier": None}, "max_area_delta": {"reject_multiplier": None}},
        "high_frequency": {"base_threshold": {"reject_multiplier": None}},
        "food": {
            "max_saturation_gain": {"reject_multiplier": None},
            "max_luminance_gain": {"reject_multiplier": None},
        },
        "layout": {
            "max_scale_delta_per_object": {"reject_multiplier": None},
            "max_relative_scale_ratio_change": {"reject_multiplier": None},
            "max_occlusion_change": {"reject_multiplier": None},
        },
    },
}


def make_policy(overrides: dict | None = None) -> Policy:
    """A Policy built directly from a dict, bypassing PolicyLoader/lock-file
    machinery entirely - validator tests care about threshold/severity
    behavior, not the immutability workflow (that's test_policy_loader.py's job)."""
    raw = copy.deepcopy(BASE_POLICY_RAW)
    for section, keys in (overrides or {}).items():
        raw.setdefault(section, {}).update(keys)
    return Policy(version=raw["policy_version"], hash="sha256:" + "0" * 64, raw=raw)


def rect_mask(canvas_size: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """bbox = (x0, y0, x1, y1), inclusive, matching FoodObject.bbox convention."""
    width, height = canvas_size
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def make_food_object(
    id: str,
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int] = (100, 100),
    semantic_role: str = "unknown",
    role_confidence: float = 1.0,
    z_index: int = 0,
    scale: float = 1.0,
    final_center: tuple[float, float] | None = None,
    shadow_profile: str = "default",
) -> FoodObject:
    mask = rect_mask(canvas_size, bbox)
    x0, y0, x1, y1 = bbox
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    return FoodObject(
        id=id,
        mask=mask,
        bbox=bbox,
        original_center=center,
        final_center=final_center if final_center is not None else center,
        scale=scale,
        z_index=z_index,
        semantic_role=semantic_role,
        role_confidence=role_confidence,
        shadow_profile=shadow_profile,
    )
