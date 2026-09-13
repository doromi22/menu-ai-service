"""
Rebuilds the images in docs/images/ used by the top-level README.

Only freely-licensed Wikimedia Commons photos (CC0 / public domain) are
used - the real merchant-style test photos have unverified provenance and
are never published (see docs/images/CREDITS.md). Needs the Commons set
from scripts/collect_test_photos.py, since photos/ is not tracked.

Renders go through the same stages as standard/pipeline.py (gate -> stand-in
roles -> layout -> color grade -> renderer); validator/retry are skipped
because every photo used here PASSes, so they would not change the pixels.

Run:
    ./venv/Scripts/python.exe scripts/build_readme_assets.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from standard.color.grade import apply_grade
from standard.color.presets import STANDARD, food_params_for
from standard.layout.engine import LayoutEngine
from standard.pipeline import _assign_stand_in_roles
from standard.segmentation.backend import BiRefNetBackend
from standard.segmentation.gate import SegmentationGate
from standard.templates.definitions import TEMPLATES
from standard.templates.renderer import TemplateRenderer

PHOTOS = ROOT / "photos"
OUT = ROOT / "docs" / "images"

HERO_PHOTO = "pasta/pasta_002.jpg"
HERO_TEMPLATES = ["T01_warm_ivory", "T04_dark_premium", "T05_japanese_editorial"]

# Fork handle, a thin diagonal edge (x0, y0, x1, y1 as fractions of the render).
SOFT_EDGE_PHOTO = "dessert/dessert_009.jpg"
SOFT_EDGE_TEMPLATE = "T01_warm_ivory"
SOFT_EDGE_CROP = (0.63, 0.70, 0.695, 0.765)
SOFT_EDGE_ZOOM = 4

# Lowest general-vs-hrsod mask IoU among the CC0 photos that shows both the
# fix (bowl recovered) and the tradeoff (neighbouring props pulled in).
MODEL_PHOTO = "donburi/donburi_008.jpg"
MODEL_TEMPLATE = "T04_dark_premium"


def _load(relative: str) -> np.ndarray:
    return np.asarray(Image.open(PHOTOS / relative).convert("RGB"))


def _objects(image_rgb: np.ndarray, backend: BiRefNetBackend):
    result = SegmentationGate(backend).run(image_rgb)
    if not result.food_objects:
        raise RuntimeError(f"segmentation produced no objects ({result.result.value}: {result.reasons})")
    return _assign_stand_in_roles(result.food_objects)


def _render(image_rgb: np.ndarray, objects, template_id: str) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    placed = LayoutEngine().run(objects, (width, height)).objects
    graded = np.asarray(apply_grade(Image.fromarray(image_rgb), food_params_for(STANDARD)))
    return TemplateRenderer().render(TEMPLATES[template_id], placed, (width, height), graded)


def _thumb(image_rgb: np.ndarray, height: int) -> Image.Image:
    image = Image.fromarray(image_rgb)
    return image.resize((round(image.width * height / image.height), height), Image.LANCZOS)


def _strip(images: list[Image.Image], gap: int = 12) -> Image.Image:
    width = sum(i.width for i in images) + gap * (len(images) - 1)
    strip = Image.new("RGB", (width, images[0].height), (255, 255, 255))
    x = 0
    for image in images:
        strip.paste(image, (x, 0))
        x += image.width + gap
    return strip


def _crop(image_rgb: np.ndarray, frac: tuple[float, float, float, float], zoom: int) -> Image.Image:
    h, w = image_rgb.shape[:2]
    box = (round(frac[0] * w), round(frac[1] * h), round(frac[2] * w), round(frac[3] * h))
    crop = Image.fromarray(image_rgb).crop(box)
    return crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)  # NEAREST: keep pixel steps visible


def build_hero(backend: BiRefNetBackend) -> None:
    image = _load(HERO_PHOTO)
    objects = _objects(image, backend)
    panels = [_thumb(image, 360)] + [_thumb(_render(image, objects, t), 360) for t in HERO_TEMPLATES]
    _strip(panels).save(OUT / "hero.jpg", quality=90)


def build_soft_edges(backend: BiRefNetBackend) -> None:
    image = _load(SOFT_EDGE_PHOTO)
    soft_objects = _objects(image, backend)
    hard_objects = [replace(obj, alpha=None) for obj in soft_objects]
    _crop(_render(image, hard_objects, SOFT_EDGE_TEMPLATE), SOFT_EDGE_CROP, SOFT_EDGE_ZOOM).save(OUT / "edges_before.png")
    _crop(_render(image, soft_objects, SOFT_EDGE_TEMPLATE), SOFT_EDGE_CROP, SOFT_EDGE_ZOOM).save(OUT / "edges_after.png")


def build_model_comparison(hrsod: BiRefNetBackend) -> None:
    if not MODEL_PHOTO:
        return
    image = _load(MODEL_PHOTO)
    general = BiRefNetBackend(model_name="birefnet-general")
    _thumb(_render(image, _objects(image, general), MODEL_TEMPLATE), 420).save(OUT / "model_general.jpg", quality=90)
    _thumb(_render(image, _objects(image, hrsod), MODEL_TEMPLATE), 420).save(OUT / "model_hrsod.jpg", quality=90)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backend = BiRefNetBackend()  # production default (birefnet-hrsod)
    build_hero(backend)
    build_soft_edges(backend)
    build_model_comparison(backend)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
