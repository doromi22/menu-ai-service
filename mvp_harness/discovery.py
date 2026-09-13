"""Image folder scanning + category-from-filename parsing (spec §10's 6 categories)."""
from __future__ import annotations

from pathlib import Path

CATEGORIES = {"ramen", "donburi", "dessert", "pasta", "teishoku", "set_meal"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}  # .webp: real photos saved from phones/sites commonly use it; Pillow reads it fine


def parse_category(filename: str) -> str:
    """
    Expects a category prefix before the first run number, e.g.
    "ramen_001.jpg" -> "ramen", "set_meal_003.jpg" -> "set_meal". Checked
    two-token-first since "set_meal" itself contains an underscore -
    checking only the first token would misparse it as "set".
    """
    stem = Path(filename).stem
    tokens = stem.split("_")

    two_token = "_".join(tokens[:2])
    if two_token in CATEGORIES:
        return two_token

    if tokens[0] in CATEGORIES:
        return tokens[0]

    return "unknown"


def discover_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
