"""
Wiring tests for standard/api/main.py's `POST /v1/standard/process` -
request/response contract, dependency overrides, and the infra-error vs
data-quality-REJECT distinction actually reaching the HTTP response. Uses
a fake segmentation backend throughout - real BiRefNet inference is
already covered by tests/integration/test_birefnet_real_inference.py, so
this file is purely about the wiring.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from standard.api.main import (
    app,
    get_backend_retry_backoff_seconds,
    get_max_backend_retries,
    get_max_backend_retry_seconds,
    get_policy,
    get_retry_config,
    get_segmentation_backend,
)
from standard.retry.config import RetryPolicyConfig
from tests.standard.helpers import make_policy

BACKGROUND_COLOR = (230, 230, 230)
FOOD_BBOX = (30, 30, 89, 89)
SIZE = 120


class _FakeBackend:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask.astype(np.float32)

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._mask


class _ExplodingBackend:
    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated backend crash")


def _clean_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    x0, y0, x1, y1 = FOOD_BBOX
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _image_bytes() -> bytes:
    arr = np.full((SIZE, SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)
    x0, y0, x1, y1 = FOOD_BBOX
    arr[y0 : y1 + 1, x0 : x1 + 1] = (200, 80, 60)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def client():
    app.dependency_overrides[get_segmentation_backend] = lambda: _FakeBackend(_clean_mask())
    app.dependency_overrides[get_policy] = lambda: make_policy()
    app.dependency_overrides[get_retry_config] = lambda: RetryPolicyConfig(
        max_attempts=3, strategy_by_reason={}, final_action="review"
    )
    app.dependency_overrides[get_max_backend_retries] = lambda: 0
    app.dependency_overrides[get_backend_retry_backoff_seconds] = lambda: 0.0
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_pass_case_returns_metadata_and_image(client):
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/v1/standard/process", files=files)

    assert response.status_code == 200
    body = response.json()
    metadata = body["metadata"]

    assert metadata["mode"] == "standard"
    assert metadata["pipeline_version"] == "standard_1.0.0"
    assert metadata["segmentation_status"] == "PASS"
    assert metadata["validator_status"] == "PASS"
    assert metadata["is_infra_error"] is False
    assert metadata["merchant_review"] is None
    assert metadata["policy_hash"].startswith("sha256:")
    assert metadata["retry_attempts_used"] == 0

    assert "image" in body
    assert body["image"]["encoding"] == "base64"
    assert body["image"]["content_type"] == "image/jpeg"
    decoded = base64.b64decode(body["image"]["data"])
    image = Image.open(io.BytesIO(decoded))
    assert image.size == (SIZE, SIZE)


def test_process_defaults_to_the_default_template_when_unspecified(client):
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/v1/standard/process", files=files)

    assert response.status_code == 200  # no template_id given, server default used - didn't 422


def test_process_unknown_template_id_is_a_client_error(client):
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/v1/standard/process", files=files, data={"template_id": "does_not_exist"})

    assert response.status_code == 422


def test_process_infra_error_has_no_image_and_is_flagged(client):
    app.dependency_overrides[get_segmentation_backend] = lambda: _ExplodingBackend()
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/v1/standard/process", files=files)

    assert response.status_code == 200  # the REQUEST succeeded - the PIPELINE outcome is REJECT, carried in the body
    body = response.json()
    metadata = body["metadata"]

    assert metadata["segmentation_status"] == "REJECT"
    assert metadata["is_infra_error"] is True
    assert "SEGMENTATION_BACKEND_ERROR" in metadata["segmentation_reasons"]
    assert "image" not in body


def test_infra_retry_time_budget_is_wired_through_from_the_api(client):
    # High count cap so it isn't the limiting factor; a tight time budget
    # with a real (if tiny) backoff must still cut retrying short -
    # confirming get_max_backend_retry_seconds actually reaches
    # SegmentationGate through run_pipeline, not just get_max_backend_retries.
    app.dependency_overrides[get_segmentation_backend] = lambda: _ExplodingBackend()
    app.dependency_overrides[get_max_backend_retries] = lambda: 10
    app.dependency_overrides[get_backend_retry_backoff_seconds] = lambda: 0.05
    app.dependency_overrides[get_max_backend_retry_seconds] = lambda: 0.05
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    import time

    start = time.perf_counter()
    response = client.post("/v1/standard/process", files=files)
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert response.json()["metadata"]["is_infra_error"] is True
    assert elapsed < 2.0  # bounded by the time budget, not the count cap of 10


def test_process_data_quality_reject_is_distinguishable_from_infra_error(client):
    empty_mask = np.zeros((SIZE, SIZE), dtype=bool)
    app.dependency_overrides[get_segmentation_backend] = lambda: _FakeBackend(empty_mask)
    files = {"image": ("food.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/v1/standard/process", files=files)

    assert response.status_code == 200
    metadata = response.json()["metadata"]

    assert metadata["segmentation_status"] == "REJECT"
    assert metadata["is_infra_error"] is False
    assert "SEGMENTATION_NO_FOOD_DETECTED" in metadata["segmentation_reasons"]
