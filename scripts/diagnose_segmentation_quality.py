"""
Diagnostic (not a permanent module): for every image in a folder, runs the
real BiRefNet backend directly and saves BOTH the raw (soft, un-thresholded)
alpha mask and the binarized mask SegmentationGate actually uses, side by
side, so mask-completeness and edge-quality problems can be inspected
directly rather than inferred from the final composited render.

Run:
    ./venv/Scripts/python.exe scripts/diagnose_segmentation_quality.py photos/ramen photos/diagnosis
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from standard.segmentation.backend import BiRefNetBackend
from standard.config.loader import load_yaml_section

CONFIG_PATH = Path(__file__).resolve().parents[1] / "standard" / "config" / "segmentation.yaml"


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("photos/ramen")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("photos/diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_section(CONFIG_PATH, "segmentation")
    threshold = cfg["mask_binarize_threshold"] if "mask_binarize_threshold" in cfg else 0.5

    backend = BiRefNetBackend()

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image)

        alpha = backend.infer_mask(image_rgb)  # raw, soft, un-thresholded [0,1]
        binary = (alpha >= threshold).astype(np.uint8) * 255

        name = image_path.stem
        Image.fromarray((alpha * 255).astype(np.uint8)).save(output_dir / f"{name}_alpha_raw.png")
        Image.fromarray(binary).save(output_dir / f"{name}_alpha_binarized.png")

        # highlight pixels that are "soft" (neither near-0 nor near-1) - these
        # are exactly the edge/uncertain regions a hard threshold destroys.
        soft_band = ((alpha > 0.05) & (alpha < 0.95)).astype(np.uint8) * 255
        Image.fromarray(soft_band).save(output_dir / f"{name}_soft_edge_band.png")

        soft_pixel_ratio = float(((alpha > 0.05) & (alpha < 0.95)).mean())
        print(f"{name}: soft-edge-band pixel ratio = {soft_pixel_ratio:.4f}")


if __name__ == "__main__":
    main()
