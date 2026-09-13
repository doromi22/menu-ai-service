"""
Visual demo: runs real BiRefNet inference (not FakeBackend) through the
full Segmentation Gate on one image, and saves the original, the raw
mask, and a red-overlay visualization to outputs/ so you can eyeball
whether the segmentation is actually usable.

No CC0/licensed photo is bundled here (avoiding any license-verification
risk) - by default this generates the same synthetic "blob on a table"
stand-in used by tests/integration/test_birefnet_real_inference.py. Pass
your own photo with --image to see real results on real food:

    ./venv/Scripts/python.exe scripts/demo_birefnet_segmentation.py --image path/to/photo.jpg
    ./venv/Scripts/python.exe scripts/demo_birefnet_segmentation.py   # synthetic stand-in

First run downloads the birefnet-hrsod ONNX model (~973MB, cached to
~/.rembg/models/birefnet-hrsod/ - see standard/segmentation/README.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a plain script, not via pytest's pythonpath

import numpy as np
from PIL import Image

from standard.segmentation.backend import BiRefNetBackend
from standard.segmentation.gate import SegmentationGate

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _synthetic_food_like_image(size: int = 400) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    radius = size * 0.3
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    background = np.full((size, size, 3), 235, dtype=np.float32)
    blob_color = np.array([180, 90, 60], dtype=np.float32)

    alpha = np.clip(1.0 - (dist - radius) / (size * 0.03), 0.0, 1.0)[..., None]
    image = background * (1 - alpha) + blob_color * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def _save_overlay(image_rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = image_rgb.astype(np.float32).copy()
    red_tint = np.array([255, 40, 40], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.5 + red_tint * 0.5
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visually verify real BiRefNet segmentation on one image.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a photo. Omit to use a synthetic stand-in.")
    parser.add_argument("--name", type=str, default=None, help="Output filename prefix (default: derived from input).")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
        image_rgb = np.asarray(image)
        name = args.name or args.image.stem
    else:
        print("No --image given - using a synthetic 'blob on a table' stand-in (not a real photo).")
        image_rgb = _synthetic_food_like_image()
        name = args.name or "synthetic_demo"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BiRefNet (birefnet-hrsod via rembg) and running inference...")
    backend = BiRefNetBackend()
    start = time.perf_counter()
    gate = SegmentationGate(backend)
    result = gate.run(image_rgb, image_id=name)
    elapsed = time.perf_counter() - start

    Image.fromarray(image_rgb).save(OUTPUT_DIR / f"{name}_original.jpg")

    if result.food_objects:
        union_mask = np.logical_or.reduce([obj.mask for obj in result.food_objects])
        Image.fromarray((union_mask * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{name}_mask.png")
        _save_overlay(image_rgb, union_mask, OUTPUT_DIR / f"{name}_overlay.jpg")
    else:
        print("No food objects detected - no mask/overlay saved.")

    print(f"\nSegmentation Gate verdict: {result.result.value}")
    print(f"Reasons: {[r.value for r in result.reasons]}")
    print(f"Objects detected: {len(result.food_objects)}")
    print(f"Metrics: {result.metrics}")
    print(f"Inference + gate time: {elapsed:.2f}s")
    print(f"\nSaved to {OUTPUT_DIR}/{name}_original.jpg, _mask.png, _overlay.jpg")


if __name__ == "__main__":
    main()
