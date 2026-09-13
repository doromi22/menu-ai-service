"""
Orchestrates a full harness run: discover images, apply all 6 templates
to each, run every (image, template) pair through `runner.run_one`, and
never let one pair's exception stop the run (spec requirement: the
harness itself must not crash on a single bad case like a segmentation
failure).
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from standard.pipeline import JPEG_QUALITY
from standard.policy.loader import PolicyLoader
from standard.policy.schema import Policy
from standard.retry.config import RetryPolicyConfig, load_retry_policy
from standard.segmentation.backend import BiRefNetBackend, SegmentationBackend
from standard.templates.definitions import TEMPLATES, Template

from mvp_harness.discovery import discover_images, parse_category
from mvp_harness.report import write_matrix_report, write_run_csv
from mvp_harness.runner import HarnessRecord, run_one

_STANDARD_DIR = Path(__file__).resolve().parents[1] / "standard"
DEFAULT_POLICY_PATH = _STANDARD_DIR / "policy" / "policy.yaml"
DEFAULT_POLICY_LOCK_PATH = _STANDARD_DIR / "policy" / "policy.lock.json"
DEFAULT_RETRY_POLICY_PATH = _STANDARD_DIR / "retry" / "retry_policy.yaml"


def _default_policy() -> Policy:
    return PolicyLoader(DEFAULT_POLICY_LOCK_PATH).load(DEFAULT_POLICY_PATH)


class _ReuseFirstSegmentation:
    """
    Wraps the backend for ONE photo's template loop. Segmentation doesn't
    depend on the template, and a real BiRefNet call costs seconds, so the
    first successful mask is reused for the remaining templates. Failures
    are not cached - the next template retries the backend normally.
    """

    def __init__(self, backend: SegmentationBackend) -> None:
        self._backend = backend
        self._alpha: np.ndarray | None = None

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        if self._alpha is None:
            self._alpha = self._backend.infer_mask(image_rgb)
        return self._alpha


def run_harness(
    input_dir: Path,
    output_dir: Path,
    run_csv_path: Path,
    matrix_csv_path: Path,
    error_log_path: Path,
    *,
    segmentation_backend: SegmentationBackend | None = None,
    policy: Policy | None = None,
    retry_config: RetryPolicyConfig | None = None,
    templates: dict[str, Template] | None = None,
    max_backend_retries: int = 2,
    backend_retry_backoff_seconds: float = 1.5,
    max_backend_retry_seconds: float = 20.0,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[HarnessRecord]:
    segmentation_backend = segmentation_backend or BiRefNetBackend()
    policy = policy or _default_policy()
    retry_config = retry_config or load_retry_policy(DEFAULT_RETRY_POLICY_PATH)
    templates = templates or TEMPLATES

    images = discover_images(Path(input_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[HarnessRecord] = []
    with open(error_log_path, "w", encoding="utf-8") as error_log:
        for image_path in images:
            category = parse_category(image_path.name)
            photo_backend = _ReuseFirstSegmentation(segmentation_backend)
            for template in templates.values():
                try:
                    record = run_one(
                        image_path,
                        category,
                        template,
                        segmentation_backend=photo_backend,
                        policy=policy,
                        retry_config=retry_config,
                        output_dir=output_dir,
                        max_backend_retries=max_backend_retries,
                        backend_retry_backoff_seconds=backend_retry_backoff_seconds,
                        max_backend_retry_seconds=max_backend_retry_seconds,
                        jpeg_quality=jpeg_quality,
                    )
                except Exception as exc:  # deliberately broad: one pair's crash must not stop the run
                    timestamp = datetime.now(timezone.utc).isoformat()
                    error_log.write(f"=== {timestamp} {image_path.name} / {template.id} ===\n")
                    error_log.write(traceback.format_exc())
                    error_log.write("\n")
                    record = HarnessRecord(
                        image_id=image_path.stem,
                        category=category,
                        template_id=template.id,
                        validator_status="ERROR",
                        reasons=[f"{type(exc).__name__}: {exc}"],
                        retry_attempts_used=0,
                        processing_time_ms=0.0,
                        output_path=None,
                    )
                records.append(record)

    write_run_csv(records, run_csv_path)
    write_matrix_report(records, matrix_csv_path)
    return records
