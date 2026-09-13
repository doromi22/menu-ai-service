"""
Real BiRefNet integration test - exercises the actual `rembg` ONNX model,
not `FakeBackend`. Excluded from the default test run
(pyproject.toml's `addopts = -m "not integration"`) because it downloads
real model weights on first run and does real CPU inference - both slow
and, the first time, needing network access.

Run explicitly:
    ./venv/Scripts/python.exe -m pytest -m integration tests/integration/ -v -s
(-s so the printed timing/verdict lines are visible)
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from standard.reason_codes import GateResult
from standard.segmentation.backend import BiRefNetBackend
from standard.segmentation.gate import SegmentationGate

pytestmark = pytest.mark.integration


def _synthetic_food_like_image(size: int = 400) -> np.ndarray:
    """
    Not a real photo - see scripts/demo_birefnet_segmentation.py to run
    this against one. A soft-edged colored blob on a plain background:
    close enough to "an object on a surface" for BiRefNet's saliency-style
    segmentation to have something reasonable to find, without depending
    on any external image file or its licensing.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    radius = size * 0.3
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    background = np.full((size, size, 3), 235, dtype=np.float32)  # light table-like surface
    blob_color = np.array([180, 90, 60], dtype=np.float32)  # warm, food-like tone

    alpha = np.clip(1.0 - (dist - radius) / (size * 0.03), 0.0, 1.0)[..., None]
    image = background * (1 - alpha) + blob_color * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def test_real_birefnet_produces_a_usable_mask():
    backend = BiRefNetBackend()
    image = _synthetic_food_like_image()

    start = time.perf_counter()
    alpha = backend.infer_mask(image)
    elapsed = time.perf_counter() - start
    print(f"\nBiRefNet CPU inference took {elapsed:.2f}s for a {image.shape[0]}x{image.shape[1]} image")

    assert alpha.shape == image.shape[:2]
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert alpha.max() > 0.5  # found *something* salient


def test_real_birefnet_through_the_full_segmentation_gate():
    backend = BiRefNetBackend()
    image = _synthetic_food_like_image()

    gate = SegmentationGate(backend)
    result = gate.run(image, image_id="integration_test")

    assert result.result in (GateResult.PASS, GateResult.REVIEW, GateResult.REJECT)
    if result.result != GateResult.REJECT:
        assert len(result.food_objects) > 0
        assert result.food_objects[0].area > 0

    print(f"\nGate verdict: {result.result.value}, reasons: {[r.value for r in result.reasons]}")
    print(f"Metrics: {result.metrics}")
