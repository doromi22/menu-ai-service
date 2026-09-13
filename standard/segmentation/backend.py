"""
Segmentation backends for the Standard pipeline (spec §1: "BiRefNet
(segmentation)").

`SegmentationBackend` is a narrow protocol so `SegmentationGate` never
imports rembg/onnxruntime directly: unit tests inject a fake backend, and
swapping the underlying checkpoint later doesn't touch gate logic.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class SegmentationBackend(Protocol):
    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a float32 alpha mask in [0, 1], shape (H, W), for one RGB image."""
        ...


class BiRefNetBackend:
    """
    rembg's birefnet-hrsod model, CPU-only by default - the Standard
    pipeline must not require a GPU (spec §1). Model/session loading is
    lazy so importing this module never triggers a download.

    Chosen over rembg's default `birefnet-general` after comparing masks on
    real problem photos (see standard/segmentation/README.md's "Backend
    model comparison" section): `birefnet-hrsod` recovered thin-prop edges
    (chopsticks resting on a bowl rim) and full bowl silhouettes that
    `birefnet-general` lost, without the whole-object misdetections
    `birefnet-dis` introduced on the same photos. Same author (ZhengPeng7)
    and MIT license as `birefnet-general`.
    """

    def __init__(
        self,
        session=None,
        providers: list[str] | None = None,
        model_name: str = "birefnet-hrsod",
    ) -> None:
        self._session = session
        self._providers = providers or ["CPUExecutionProvider"]
        self._model_name = model_name

    def _get_session(self):
        if self._session is None:
            from rembg import new_session

            self._session = new_session(self._model_name, providers=self._providers)
        return self._session

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        from rembg import remove

        pil_image = Image.fromarray(image_rgb, mode="RGB")
        rgba = remove(pil_image, session=self._get_session())
        return np.asarray(rgba.split()[3], dtype=np.float32) / 255.0
