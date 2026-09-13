"""
Runs ONE (image, template) pair through the full Standard pipeline
(`standard.pipeline.run_pipeline`), saves the rendered image to disk, and
returns a `HarnessRecord`. Raises on any pipeline failure the pipeline
itself doesn't already turn into a controlled result (e.g. a bug
elsewhere in Layout/Shadow/Color/Validator); the caller (`cli.py`) decides
how to turn that into an ERROR row. Segmentation backend infra failures
are already handled inside the pipeline itself (§11 round: REJECT +
`SEGMENTATION_BACKEND_ERROR`, `is_infra_error=True`, never raised here).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from standard.pipeline import JPEG_QUALITY, run_pipeline
from standard.policy.schema import Policy
from standard.retry.config import RetryPolicyConfig
from standard.segmentation.backend import SegmentationBackend
from standard.templates.definitions import Template


@dataclass
class HarnessRecord:
    image_id: str
    category: str
    template_id: str
    validator_status: str  # PASS | REVIEW | REJECT
    reasons: list[str]
    retry_attempts_used: int
    processing_time_ms: float
    output_path: str | None


def run_one(
    image_path: Path,
    category: str,
    template: Template,
    *,
    segmentation_backend: SegmentationBackend,
    policy: Policy,
    retry_config: RetryPolicyConfig,
    output_dir: Path,
    max_backend_retries: int = 2,
    backend_retry_backoff_seconds: float = 1.5,
    max_backend_retry_seconds: float = 20.0,
    jpeg_quality: int = JPEG_QUALITY,
) -> HarnessRecord:
    start = time.perf_counter()
    image_id = image_path.stem
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))

    pipeline_result = run_pipeline(
        image_rgb,
        template,
        segmentation_backend=segmentation_backend,
        policy=policy,
        retry_config=retry_config,
        image_id=image_id,
        max_backend_retries=max_backend_retries,
        backend_retry_backoff_seconds=backend_retry_backoff_seconds,
        max_backend_retry_seconds=max_backend_retry_seconds,
        jpeg_quality=jpeg_quality,
    )

    output_path = None
    if pipeline_result.rendered_image is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}__{template.id}.jpg"
        Image.fromarray(pipeline_result.rendered_image).save(output_path, format="JPEG", quality=jpeg_quality)

    return HarnessRecord(
        image_id=image_id,
        category=category,
        template_id=template.id,
        validator_status=pipeline_result.overall_status.value,
        reasons=[r.value for r in pipeline_result.overall_reasons],
        retry_attempts_used=pipeline_result.retry_attempts_used,
        processing_time_ms=(time.perf_counter() - start) * 1000,
        output_path=str(output_path) if output_path else None,
    )
