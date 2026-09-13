"""
Diagnostic (not a permanent module): run several rembg backends/configs on
the same known-problem real photos and save masks side by side, so a model
swap can be decided on evidence instead of assumption.

Candidates compared (all bundled in rembg already - zero new dependency):
  - birefnet-general            (current production backend, baseline)
  - birefnet-general + alpha_matting=True   (same model, matting refinement)
  - isnet-general-use           (different architecture, DIS-based)
  - bria-rmbg                   (BRIA RMBG, commercial-grade bg removal)

Run:
    ./venv/Scripts/python.exe scripts/compare_segmentation_backends.py photos/ramen photos/backend_compare ramen_003 ramen_008 ramen_011
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

CONFIGS: list[tuple[str, str, dict]] = [
    ("birefnet-general", "baseline", {}),
    ("birefnet-general", "alpha_matting", {"alpha_matting": True}),
    ("isnet-general-use", "baseline", {}),
    ("bria-rmbg", "baseline", {}),
]


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("photos/ramen")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("photos/backend_compare")
    stems = sys.argv[3:] if len(sys.argv) > 3 else None
    output_dir.mkdir(parents=True, exist_ok=True)

    from rembg import new_session, remove

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if stems:
        images = [p for p in images if p.stem in stems]

    sessions: dict[str, object] = {}

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        for model_name, label, kwargs in CONFIGS:
            if model_name not in sessions:
                print(f"loading session: {model_name} ...")
                sessions[model_name] = new_session(model_name, providers=["CPUExecutionProvider"])
            rgba = remove(image, session=sessions[model_name], **kwargs)
            alpha = np.asarray(rgba.split()[3], dtype=np.uint8)
            out_name = f"{image_path.stem}__{model_name}__{label}.png"
            Image.fromarray(alpha).save(output_dir / out_name)
            print(f"  {image_path.stem}: {model_name}/{label} -> {out_name}")


if __name__ == "__main__":
    main()
