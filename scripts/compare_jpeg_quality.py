"""
Compares Pillow JPEG quality levels (85/90/92/95/100) on a food-like
texture: file size, PSNR, and - most importantly for this repo - the
ACTUAL standard.validator.module_b.compute_hf_error value each quality
level produces (using the same contrast_weight/sharpness_weight
standard/policy/policy.yaml ships with), so the choice of JPEG_QUALITY
and the base_threshold: 0.04 calibration can be checked against the same
metric that decides PASS/REVIEW in production.

A flat-color block (used elsewhere in this repo's synthetic test fixtures)
compresses essentially losslessly at any of these qualities and would
tell us nothing here - this generates a textured, hard-edged "food photo"
stand-in on purpose, since that's where JPEG artifacts actually show up.

Run:
    ./venv/Scripts/python.exe scripts/compare_jpeg_quality.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from standard.validator.module_b import compute_hf_error

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CONTRAST_WEIGHT = 0.35
SHARPNESS_WEIGHT = 0.50
BASE_THRESHOLD = 0.04


def make_realistic_food_texture(size: int = 600, seed: int = 7) -> np.ndarray:
    """
    A round "bowl of something textured" on a dark table: fine grain noise
    (high spatial frequency, like rice/grain texture), scattered bright
    "seed" highlights (sharp local contrast), and a hard edge against a
    dark background (classic JPEG ringing-artifact trigger) - the kind of
    content real food photos actually have, unlike a flat color block.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = size * 0.35

    base = 150 + 40 * np.cos(np.clip(dist / radius, 0, 1) * np.pi / 2)
    texture = rng.normal(0, 12, (size, size))
    channel = base + texture

    for _ in range(150):
        sx, sy = rng.integers(0, size, 2)
        if dist[sy, sx] < radius:
            channel[max(0, sy - 2) : sy + 2, max(0, sx - 2) : sx + 2] += 60

    r = np.clip(channel * 1.15, 0, 255)
    g = np.clip(channel * 0.75, 0, 255)
    b = np.clip(channel * 0.45, 0, 255)
    food = np.stack([r, g, b], axis=-1)

    inside = (dist < radius)[..., None]
    background = np.full((size, size, 3), (30, 28, 25), dtype=np.float64)
    return np.clip(np.where(inside, food, background), 0, 255).astype(np.uint8)


def _encode_decode(image: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    size_bytes = buffer.tell()
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer).convert("RGB"))
    return decoded, size_bytes


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main() -> None:
    image = make_realistic_food_texture()
    mask = np.ones(image.shape[:2], dtype=bool)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(OUTPUT_DIR / "jpeg_quality_reference.png")  # lossless reference

    qualities = [85, 90, 92, 95, 100]
    rows = []
    for q in qualities:
        decoded, size_bytes = _encode_decode(image, q)
        hf_error = compute_hf_error(image, decoded, mask, CONTRAST_WEIGHT, SHARPNESS_WEIGHT)
        p = _psnr(image, decoded)
        rows.append((q, size_bytes / 1024, hf_error, p))
        Image.fromarray(decoded).save(OUTPUT_DIR / f"jpeg_quality_q{q}.jpg", format="JPEG", quality=q)

    print(f"{'quality':>7} | {'size(KB)':>9} | {'HF error':>9} | {'PSNR(dB)':>9} | vs base_threshold={BASE_THRESHOLD}")
    print("-" * 70)
    for q, kb, hf, p in rows:
        flag = "OK" if hf < BASE_THRESHOLD else "EXCEEDS"
        print(f"{q:>7} | {kb:>9.1f} | {hf:>9.4f} | {p:>9.2f} | {flag}")

    print(f"\nSaved reference + per-quality JPEGs to {OUTPUT_DIR}/jpeg_quality_*")


if __name__ == "__main__":
    main()
