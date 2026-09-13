"""
Plumbing test for the §11 step 7 harness: proves it runs start-to-finish
without crashing, using synthetic images (real photos are a separate,
ongoing data-collection track - see mvp_harness/README.md) and a simple
threshold-based segmentation stand-in instead of the real BiRefNet
backend (this venv has no rembg/onnxruntime installed, and a solid-color
rectangle on a solid-color background doesn't need ML to segment).
Interpreting the results is explicitly out of scope - this only checks
the harness produces the expected files and statuses without dying.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from mvp_harness.cli import run_harness
from standard.policy.schema import Policy
from standard.reason_codes import ReasonCode
from standard.retry.config import RetryPolicyConfig
from standard.retry.strategies import Strategy
from standard.templates.definitions import TEMPLATES
from tests.standard.helpers import BASE_POLICY_RAW

BACKGROUND_COLOR = (230, 230, 230)


class _ThresholdBackend:
    """
    Test-only stand-in: flags any pixel far enough from a fixed background
    color as food. NOT a production segmentation backend - real photos
    need real ML segmentation (standard.segmentation.backend.BiRefNetBackend,
    the harness's actual default).
    """

    def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        diff = np.abs(image_rgb.astype(np.int16) - np.array(BACKGROUND_COLOR)).sum(axis=-1)
        return (diff > 60).astype(np.float32)


def _test_policy() -> Policy:
    raw = copy.deepcopy(BASE_POLICY_RAW)
    return Policy(version=raw["policy_version"], hash="sha256:" + "0" * 64, raw=raw)


def _test_retry_config() -> RetryPolicyConfig:
    return RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={
            ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
            ReasonCode.FOOD_LUMINANCE_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
            ReasonCode.CONTENT_HF_ERROR: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
            ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION],
            ReasonCode.OBJECT_SCALE_CHANGED: [Strategy.FORCE_SCALE_1_0],
            ReasonCode.RELATIVE_SCALE_CHANGED: [Strategy.FORCE_SCALE_1_0],
            ReasonCode.OBJECT_AREA_RANK_REVERSED: [Strategy.RECOMPUTE_TRANSLATION],
        },
        final_action="review",
    )


def _save_test_image(path: Path, size=(120, 120), food_bbox=(30, 30, 89, 89), food_color=(200, 80, 60)) -> None:
    arr = np.full((size[1], size[0], 3), BACKGROUND_COLOR, dtype=np.uint8)
    x0, y0, x1, y1 = food_bbox
    arr[y0 : y1 + 1, x0 : x1 + 1] = food_color
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


def test_harness_runs_end_to_end_on_synthetic_images(tmp_path):
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # PASS-ish: clean, well-proportioned food blob, away from every edge.
    _save_test_image(input_dir / "ramen_001.jpg", food_bbox=(30, 30, 89, 89))
    # REVIEW-ish: touches the image boundary heavily (segmentation-level REVIEW).
    _save_test_image(input_dir / "teishoku_002.jpg", food_bbox=(0, 0, 100, 100))

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_ThresholdBackend(),
        policy=_test_policy(),
        retry_config=_test_retry_config(),
        templates={"T01_warm_ivory": TEMPLATES["T01_warm_ivory"]},  # keep the sample run small and fast
    )

    assert len(records) == 2  # 2 images x 1 template, nothing lost
    statuses = {r.validator_status for r in records}
    assert "ERROR" not in statuses  # must not silently swallow a real crash either
    assert "PASS" in statuses
    assert "REVIEW" in statuses

    assert (output_dir / "run_results.csv").exists()
    assert (output_dir / "matrix_report.csv").exists()
    assert (output_dir / "errors.log").exists()

    with open(output_dir / "run_results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["validator_status"] for row in rows} == statuses

    for record in records:
        if record.validator_status != "REJECT":
            assert record.output_path is not None
            assert Path(record.output_path).exists()


def test_module_b_actually_fires_through_the_real_orchestrator(tmp_path):
    """
    Regression test for a real wiring bug: `runner.render_and_validate`
    used to pass the exact same array as both `expected_food_rgb` and
    `final_food_rgb`, so Module B's HF-error check was comparing a value
    against itself - guaranteed zero, regardless of what the pipeline
    actually did. Fixed by deriving `final_food_rgb` from the actual
    saved-and-reloaded JPEG (spec §6: "Actual Final Food" = what a
    merchant really receives) and aligning `expected_food_rgb`/`food_mask`/
    `original_food_rgb` via `place_food_layer` instead of assuming a
    uniform shift. This test proves the check now has real detection
    power by using a noisy (high-frequency) food region and a low JPEG
    quality - a flat-color block (this file's other tests) barely shows
    any compression artifact and shouldn't trip it; noise pushed through
    heavy compression should.
    """
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rng = np.random.default_rng(11)
    size = 120
    arr = np.full((size, size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    arr[30:90, 30:90] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)  # noisy "food" - hard to compress
    Image.fromarray(arr).save(input_dir / "ramen_001.jpg", format="JPEG", quality=95)

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_ThresholdBackend(),
        policy=_test_policy(),
        retry_config=_test_retry_config(),
        templates={"T01_warm_ivory": TEMPLATES["T01_warm_ivory"]},
        jpeg_quality=5,  # force visible compression artifacts
    )

    assert len(records) == 1
    assert ReasonCode.CONTENT_HF_ERROR.value in records[0].reasons


def test_retry_recompute_translation_path_keeps_module_b_aligned(tmp_path, monkeypatch):
    """
    Regression test for the RETRY-triggered path specifically - the other
    harness tests only ever exercise `render_and_validate` once, through
    plain `recenter_group` (a uniform group shift). This test forces an
    initial REJECT so `run_one`'s retry branch actually runs
    `recompute_translation` (which nudges each object by a DIFFERENT
    absolute amount, proportional to its own distance from the group
    centroid - not a uniform shift), then checks the re-rendered,
    saved-and-reloaded JPEG is still correctly aligned by
    `place_food_layer` for that non-uniform case. Flat-color food blocks
    are used deliberately: if alignment were broken, comparing food
    pixels against the wrong canvas region would show up as a spurious
    `CONTENT_HF_ERROR`, which a correctly-aligned comparison must not.

    Role assignment is monkeypatched to name two roles (`main` + `side`)
    because the harness's real stand-in (`assign_stand_in_roles`) only
    ever names one - through the real harness today, `recompute_translation`
    is unreachable (OCCLUSION_CHANGED/OBJECT_AREA_RANK_REVERSED, the only
    two reasons mapped to it, both require 2+ named roles - see §11
    round's report), so this test necessarily goes around that specific
    limitation to exercise the retry code path at all.
    """
    import standard.retry.state_machine as state_machine_module
    import standard.pipeline as pipeline_module

    def _two_role_assign(objects):
        if len(objects) < 2:
            return objects
        ordered = sorted(range(len(objects)), key=lambda i: -objects[i].area)
        result = list(objects)
        # z_index=1 (front) for "main", 0 (back) for "side": occlusion pairs
        # are read as (front_role, back_role) by z_index (standard/layout/occlusion.py),
        # and OCCLUSION_BASELINE only has a ("main", "side") entry, not the reverse.
        result[ordered[0]] = replace(objects[ordered[0]], semantic_role="main", role_confidence=0.9, z_index=1)
        result[ordered[1]] = replace(objects[ordered[1]], semantic_role="side", role_confidence=0.9, z_index=0)
        return result

    monkeypatch.setattr(pipeline_module, "assign_stand_in_roles", _two_role_assign)

    recompute_calls = []
    original_recompute = state_machine_module.recompute_translation

    def _spy_recompute(objects, canvas_size, attempt_index):
        recompute_calls.append(attempt_index)
        return original_recompute(objects, canvas_size, attempt_index)

    monkeypatch.setattr(state_machine_module, "recompute_translation", _spy_recompute)

    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Two disjoint, flat-color blocks -> zero occlusion, which is itself a
    # large deviation from OCCLUSION_BASELINE[("main","side")] = 0.18.
    size = 150
    arr = np.full((size, size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    arr[10:70, 10:70] = (200, 80, 60)  # "main", 60x60
    arr[90:120, 90:120] = (60, 120, 200)  # "side", 30x30
    image_path = input_dir / "ramen_001.jpg"
    Image.fromarray(arr).save(image_path, format="JPEG", quality=95)

    # reject_multiplier=1.0 on occlusion -> ANY exceedance of the 0.15
    # threshold is an immediate REJECT (no REVIEW buffer), forcing retry.
    raw = copy.deepcopy(BASE_POLICY_RAW)
    raw["severity"]["layout"]["max_occlusion_change"]["reject_multiplier"] = 1.0
    policy = Policy(version=raw["policy_version"], hash="sha256:" + "0" * 64, raw=raw)

    retry_config = RetryPolicyConfig(
        max_attempts=3,
        strategy_by_reason={ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION]},
        final_action="review",
    )

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_ThresholdBackend(),
        policy=policy,
        retry_config=retry_config,
        templates={"T01_warm_ivory": TEMPLATES["T01_warm_ivory"]},
    )

    assert len(records) == 1
    record = records[0]

    # retry actually ran recompute_translation - not just recenter_group.
    assert len(recompute_calls) >= 1
    assert record.retry_attempts_used >= 1

    # OCCLUSION_CHANGED could not be resolved (spacing apart doesn't fix a
    # "too little occlusion vs baseline" deviation - a separate, known
    # design point, not what this test is checking) and correctly persists.
    assert ReasonCode.OCCLUSION_CHANGED.value in record.reasons
    # The key assertion: no spurious content-integrity violation from
    # misaligned expected/final food regions after the non-uniform shift.
    assert ReasonCode.CONTENT_HF_ERROR.value not in record.reasons

    assert record.output_path is not None
    assert Path(record.output_path).exists()


def test_segmentation_backend_failure_resolves_to_a_clean_reject(tmp_path):
    """
    A backend crash (model load failure, inference OOM) is now caught
    inside SegmentationGate itself and turned into a controlled REJECT
    with SEGMENTATION_BACKEND_ERROR (standard/segmentation/gate.py) -
    it never reaches the harness as an uncaught exception at all, so this
    is NOT the "hasn't crashed" safety net (see the next test for that);
    it's proof the gate's own error handling is wired through correctly.
    """
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _save_test_image(input_dir / "dessert_001.jpg")

    class _ExplodingBackend:
        def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
            raise RuntimeError("simulated segmentation crash")

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_ExplodingBackend(),
        policy=_test_policy(),
        retry_config=_test_retry_config(),
        templates={"T01_warm_ivory": TEMPLATES["T01_warm_ivory"]},
        backend_retry_backoff_seconds=0,  # skip the real 1.5s x 2 infra-retry sleep in tests
    )

    assert len(records) == 1
    assert records[0].validator_status == "REJECT"
    assert records[0].reasons == [ReasonCode.SEGMENTATION_BACKEND_ERROR.value]
    assert records[0].retry_attempts_used == 0  # never retried - see reason_codes.py


def test_harness_survives_a_pipeline_crash_outside_segmentation(tmp_path):
    """
    The outer safety net (cli.py's per-pair try/except) for anything
    SegmentationGate's own error handling doesn't cover - a fault
    elsewhere in the pipeline (Layout/Shadow/Color/Validator/Retry/IO).
    Simulated here as a broken Policy to keep the fault injection simple
    and independent of any specific module's internals.
    """
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _save_test_image(input_dir / "dessert_001.jpg")

    class _BrokenPolicy:
        def threshold(self, section, key):
            raise RuntimeError("simulated pipeline crash outside segmentation")

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_ThresholdBackend(),
        policy=_BrokenPolicy(),
        retry_config=_test_retry_config(),
        templates={"T01_warm_ivory": TEMPLATES["T01_warm_ivory"]},
    )

    assert len(records) == 1
    assert records[0].validator_status == "ERROR"
    assert "simulated pipeline crash outside segmentation" in records[0].reasons[0]

    error_log_text = (output_dir / "errors.log").read_text(encoding="utf-8")
    assert "RuntimeError" in error_log_text
    assert "simulated pipeline crash outside segmentation" in error_log_text
    assert "Traceback" in error_log_text


def test_segmentation_runs_once_per_photo_not_once_per_template(tmp_path):
    """Segmentation doesn't depend on the template, and a real BiRefNet call
    costs seconds - re-running it for every template multiplied a batch's
    runtime by the template count."""
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _save_test_image(input_dir / "ramen_001.jpg")
    _save_test_image(input_dir / "ramen_002.jpg", food_bbox=(20, 20, 79, 79))

    class _CountingBackend(_ThresholdBackend):
        calls = 0

        def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
            type(self).calls += 1
            return super().infer_mask(image_rgb)

    templates = {key: TEMPLATES[key] for key in ("T01_warm_ivory", "T02_cool_white", "T04_dark_premium")}
    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_CountingBackend(),
        policy=_test_policy(),
        retry_config=_test_retry_config(),
        templates=templates,
    )

    assert len(records) == 6
    assert _CountingBackend.calls == 2


def test_a_failed_segmentation_is_not_reused_for_the_next_template(tmp_path):
    """Only successful inferences are reused: a transient backend failure on
    one template must not turn every remaining template into a REJECT."""
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _save_test_image(input_dir / "ramen_001.jpg")

    class _FailsOnceBackend(_ThresholdBackend):
        calls = 0

        def infer_mask(self, image_rgb: np.ndarray) -> np.ndarray:
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("transient failure")
            return super().infer_mask(image_rgb)

    records = run_harness(
        input_dir,
        output_dir,
        output_dir / "run_results.csv",
        output_dir / "matrix_report.csv",
        output_dir / "errors.log",
        segmentation_backend=_FailsOnceBackend(),
        policy=_test_policy(),
        retry_config=_test_retry_config(),
        templates={key: TEMPLATES[key] for key in ("T01_warm_ivory", "T02_cool_white")},
        max_backend_retries=0,
        backend_retry_backoff_seconds=0,
    )

    assert [r.validator_status for r in records] == ["REJECT", "PASS"]
