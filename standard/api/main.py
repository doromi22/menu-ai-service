"""
The Standard pipeline's own HTTP entrypoint (spec §11 step 4) -
implements the `POST /v1/standard/process` contract already sketched in
../../docs/standard-api-contract.md, now for real. Deliberately a separate
FastAPI app from ../../main.py / ../../modal_app.py (Premium) - see
README.md in this directory for why.

Run:
    cd ai-service
    ./venv/Scripts/python.exe -m uvicorn standard.api.main:app --port 8002
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from standard.pipeline import JPEG_QUALITY, PipelineResult, run_pipeline
from standard.policy.loader import PolicyLoader
from standard.policy.schema import Policy
from standard.retry.config import RetryPolicyConfig, load_retry_policy
from standard.segmentation.backend import BiRefNetBackend, SegmentationBackend
from standard.templates.definitions import TEMPLATES

_STANDARD_DIR = Path(__file__).resolve().parents[1]
_POLICY_PATH = _STANDARD_DIR / "policy" / "policy.yaml"
_POLICY_LOCK_PATH = _STANDARD_DIR / "policy" / "policy.lock.json"
_RETRY_POLICY_PATH = _STANDARD_DIR / "retry" / "retry_policy.yaml"
DEFAULT_TEMPLATE_ID = "T01_warm_ivory"

app = FastAPI(title="MenuAI Standard Pipeline")

class _FakeThresholdBackend:
    """
    TEST-ONLY stand-in, activated by the STANDARD_API_FAKE_SEGMENTATION=1
    environment variable - lets an end-to-end test start a REAL server
    process (e.g. from Laravel's PHPUnit suite, over real HTTP) without
    needing real BiRefNet inference. Same threshold trick used by
    tests/mvp_harness's fixtures: anything far enough from a fixed
    background color counts as food. NEVER set this env var in
    production - see standard/api/README.md.
    """

    _BACKGROUND_COLOR = np.array([230, 230, 230])

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        diff = np.abs(image_rgb.astype(np.int16) - self._BACKGROUND_COLOR).sum(axis=-1)
        return (diff > 60).astype(np.float32)


# Loaded once at import time, not per-request: the BiRefNet ONNX session is
# expensive to create (standard/segmentation/README.md: ~9-35s cold on
# CPU) and is safe to reuse across requests; Policy/Retry config are
# static files with no reason to re-parse every call.
_segmentation_backend = (
    _FakeThresholdBackend() if os.environ.get("STANDARD_API_FAKE_SEGMENTATION") == "1" else BiRefNetBackend()
)
_policy: Policy = PolicyLoader(_POLICY_LOCK_PATH).load(_POLICY_PATH)
_retry_config: RetryPolicyConfig = load_retry_policy(_RETRY_POLICY_PATH)


# FastAPI dependency functions (not called directly) rather than reading
# the module singletons above inline in the route - lets tests swap in a
# FakeBackend/synthetic Policy via `app.dependency_overrides` (see
# tests/api/test_process_endpoint.py) without touching module state or
# waiting on a real BiRefNet model load.
def get_segmentation_backend() -> SegmentationBackend:
    return _segmentation_backend


def get_policy() -> Policy:
    return _policy


def get_retry_config() -> RetryPolicyConfig:
    return _retry_config


def get_max_backend_retries() -> int:
    return 2


def get_backend_retry_backoff_seconds() -> float:
    return 1.5


def get_max_backend_retry_seconds() -> float:
    # Leaves headroom under this app's own ~60s client-side timeout
    # convention (docs/standard-api-contract.md) for the rest of the
    # pipeline (Layout/Shadow/Color/Validator/Retry) after segmentation.
    return 20.0


def _build_metadata(result: PipelineResult, policy: Policy) -> dict:
    """Matches ../../docs/standard-metadata.schema.json exactly."""
    return {
        "pipeline_version": "standard_1.0.0",
        "policy_version": policy.version,
        "policy_hash": policy.hash,
        "template_version": "graphic_1.0.0",
        "mode": "standard",
        "segmentation_status": result.segmentation_status.value,
        "segmentation_reasons": [r.value for r in result.segmentation_reasons],
        "is_infra_error": result.is_infra_error,
        "validator_status": result.validator_status.value if result.validator_status is not None else None,
        "validator_reasons": [r.value for r in result.validator_reasons],
        "retry_attempts_used": result.retry_attempts_used,
        "merchant_review": None,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/standard/process")
async def process_image(
    image: UploadFile = File(...),
    template_id: str = Form(DEFAULT_TEMPLATE_ID),
    segmentation_backend: SegmentationBackend = Depends(get_segmentation_backend),
    policy: Policy = Depends(get_policy),
    retry_config: RetryPolicyConfig = Depends(get_retry_config),
    max_backend_retries: int = Depends(get_max_backend_retries),
    backend_retry_backoff_seconds: float = Depends(get_backend_retry_backoff_seconds),
    max_backend_retry_seconds: float = Depends(get_max_backend_retry_seconds),
) -> dict:
    template = TEMPLATES.get(template_id)
    if template is None:
        raise HTTPException(status_code=422, detail=f"unknown template_id: {template_id!r}")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"unreadable image: {exc}") from exc

    image_rgb = np.asarray(pil_image)

    result = run_pipeline(
        image_rgb,
        template,
        segmentation_backend=segmentation_backend,
        policy=policy,
        retry_config=retry_config,
        max_backend_retries=max_backend_retries,
        backend_retry_backoff_seconds=backend_retry_backoff_seconds,
        max_backend_retry_seconds=max_backend_retry_seconds,
    )

    response: dict = {"metadata": _build_metadata(result, policy)}

    # §7: REJECT (incl. infra failure) = no usable output - no image to return.
    if result.rendered_image is not None:
        buffer = io.BytesIO()
        Image.fromarray(result.rendered_image).save(buffer, format="JPEG", quality=JPEG_QUALITY)
        response["image"] = {
            "content_type": "image/jpeg",
            "encoding": "base64",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    return response
