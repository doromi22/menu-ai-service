# Segmentation backend: rembg (BiRefNet) vs a direct checkpoint

**Decision: `rembg`'s `birefnet-hrsod` ONNX export, via `BiRefNetBackend`
in `backend.py`.** (Switched from `birefnet-general` - see "Backend model
comparison" below for why.) Rationale for using `rembg` at all, independent
of which BiRefNet checkpoint:

- **Already the working pattern in this codebase.** The existing
  Premium/generative engine (`../../main.py`, `../../modal_app.py`) already
  uses exactly this call shape (`rembg.new_session(...)` + `rembg.remove(...)`,
  there with the `birefnet-general` checkpoint) successfully in production.
  Reusing the `rembg` integration path - just with a different checkpoint
  name, decided independently for Standard's own real-photo failure cases -
  means one proven integration path instead of two.
- **No model/architecture code to own.** A direct checkpoint load would
  need PyTorch, BiRefNet's model definition, and our own pre/post-processing
  (resize, normalize, matting refinement) - `rembg` already packages all
  of that behind `new_session`/`remove`.
- **CPU-first, matching spec §1's "GPU 불필요" requirement for Standard.**
  `rembg` runs on plain `onnxruntime` with `CPUExecutionProvider` - no CUDA,
  no GPU driver dependency. A raw PyTorch checkpoint would default toward
  GPU-oriented tooling even if CPU-runnable.
- **Handles model download/caching itself** (see below) - no separate
  weights-hosting or version-pinning system to build for this step.

The `SegmentationBackend` Protocol (`backend.py`) is unchanged - this was
already the intended swap point (§11 step 2's own docstring: "swapping the
underlying checkpoint later doesn't touch gate logic"), so no new
config/factory system was needed to "switch mock vs real": unit tests
already inject `FakeBackend` (`tests/standard/test_segmentation_gate.py`)
directly and never import `rembg`; `mvp_harness/cli.py` already defaults to
constructing a real `BiRefNetBackend()` when no backend is injected. The
Protocol *is* the switch.

## Model caching, download, and timing (measured on this machine)

- **Cache location** (from `rembg`, not something this repo controls):
  `~/.rembg/models/birefnet-hrsod/birefnet-hrsod.onnx` -
  `C:\Users\<you>\.rembg\models\birefnet-hrsod\birefnet-hrsod.onnx`
  on Windows.
- **Model size**: 973 MB (measured directly - `ls -lah` on the cached file;
  slightly larger than `birefnet-general`'s 928 MB). First run needs to
  download this; time depends entirely on your network - budget several
  minutes on a typical connection, well under a minute on a fast one.
- **CPU inference timing**: not independently re-measured after the switch;
  `birefnet-hrsod` is the same BiRefNet architecture family as
  `birefnet-general` (same author, same rembg integration path), so the
  `birefnet-general` numbers below are a reasonable estimate, not a
  guarantee. Re-measure if inference latency becomes a concern.
  (AMD Ryzen 7 9800X3D, `onnxruntime` CPUExecutionProvider - confirmed via
  `onnxruntime.get_available_providers()`, no GPU provider present in this
  environment): a 400x400 test image took **~9.7s for session creation +
  first inference in a fresh process**, and **~5.4-5.5s per call once the
  ONNX session is warm** (same process, session reused - which is exactly
  what `BiRefNetBackend` does across repeated `infer_mask` calls, since
  `_get_session()` caches it on `self`). One cold-process run during testing
  took 34.75s for the same size image - likely OS file-cache variance for
  the ~930MB weights file, not a stable number; treat "several seconds to
  ~10s cold, ~5s warm" as the realistic range on comparable CPU hardware.
  **No GPU (e.g. A10G) timing is reported here** - this environment has no
  CUDA GPU to measure against; the existing Premium engine's `modal_app.py`
  (A10G, but running SDXL, a much heavier model) is the only real
  GPU-timing reference in this repo, and isn't directly comparable.
- **Batch/throughput implications**: `mvp_harness`'s 6-templates-per-image
  loop does NOT re-run segmentation per template (segmentation happens once,
  before the per-template loop in `runner.py`) - so this cost is paid once
  per photo, not once per (photo, template) pair.

## Backend model comparison (why `birefnet-hrsod`, not `birefnet-general`)

Phase 1 real-photo testing (13 real ramen photos, see project notes)
surfaced three quality complaints: intermittent bowl-edge loss, nori
sometimes missing, and staircase/aliasing on thin props (chopsticks). The
third is a hard-binarization/compositing issue, tracked separately. The
first two traced back to `birefnet-general`'s raw (pre-threshold) alpha
mask itself excluding real object pixels - not something `SegmentationGate`'s
own thresholding or noise filtering could be blamed for or fix, since the
confidence gap exists before either of those steps runs.

Rather than accept that as a fixed limit of "the model," three same-author
(ZhengPeng7), MIT-licensed BiRefNet checkpoints already bundled in `rembg`
(zero new dependency) were compared head-to-head on the two worst known
failure photos plus one known-good one, using
`scripts/compare_segmentation_backends.py`:

| Photo (failure mode) | `birefnet-general` (previous) | `birefnet-dis` | `birefnet-hrsod` (adopted) |
|---|---|---|---|
| Chopstick resting on bowl rim (edge notch) | Clear notch cut into bowl silhouette | Notch mostly filled, minor speckling | Notch essentially gone |
| Nori + chopsticks + napkin, full bowl (baseline-clean case) | Clean full circle (no problem here) | **Regression**: most of the bowl dropped, only a small fragment kept | Clean full circle, props included |
| Bowl on a round plate, strong top-edge glare | Clear notch cut into bowl silhouette | **Regression**: latched onto the plate as the salient object instead of the bowl, large soft-gray misfit | Notch still present but softened to gray (partial confidence) rather than hard-cut - smaller regression, no plate confusion |

`alpha_matting=True` (rembg's built-in trimap/matting refinement, no
checkpoint change) was tried first since it needs no download - it produced
masks visually indistinguishable from the un-refined baseline on all three
photos. Matting refines *already-uncertain* edge bands; it has nothing to
work with where the base model committed hard to "background" with no
uncertainty band at all, which is exactly the bowl/nori failure mode. So it
was ruled out before downloading either full alternative checkpoint.

**Decision rule used**: does the mask have a hard (non-gradient) break in
an object's silhouette that doesn't correspond to real occlusion in the
photo? `birefnet-dis` traded that failure for a worse one (whole-object
misdetection) on two of three photos, so it was rejected despite the
chopstick-case win. `birefnet-hrsod` won on the same case with no new
misdetection anywhere, and degraded the hardest case (top-glare bowl) from
a hard cut to a soft one rather than fully solving it - accepted as a real
but bounded remaining limitation, not a regression.

`birefnet_dynamic`, mentioned as a fallback candidate before this
comparison ran, **does not exist as an installed rembg session name** in
this environment (`rembg.sessions.sessions_names`) - only `birefnet-hrsod`,
`birefnet-dis`, `birefnet-general`, `birefnet-general-lite`,
`birefnet-portrait`, `birefnet-cod`, and `birefnet-massive` do. Not pursued
further since `birefnet-hrsod` already met the bar.

`isnet-general-use` and `bria-rmbg` were considered but never run:
`bria-rmbg` (BRIA RMBG 2.0) is CC BY-NC 4.0-licensed, incompatible with a
commercial service, and `isnet-general-use` was dropped alongside it to
save comparison time once the same-author BiRefNet family (already known
MIT-licensed) looked promising.

## Error handling: infrastructure failure vs data-quality REJECT

A model load failure or an inference crash/OOM is caught inside
`SegmentationGate.run()` itself (not left to propagate) and converted to a
REJECT with reason code `SEGMENTATION_BACKEND_ERROR` - deliberately kept
**out of `standard/retry/retry_policy.yaml`'s `strategy_by_reason`**, so it
is never retried. Rationale: the Retry state machine's whole model is
"try a different color/layout parameter" - none of those can fix a backend
that failed to load or ran out of memory, so retrying would just repeat the
same failure for no reason (§7's "단순 재실행 금지" applies here even though
this isn't one of §7's originally-listed reason codes). This is a different
kind of REJECT from the segmentation-quality ones (`SEGMENTATION_AREA_TOO_SMALL`,
etc.) - the photo itself might be perfectly fine; the failure is ours to fix
(infra/ops), not something a merchant or a different pipeline parameter can
address. See `tests/standard/test_segmentation_gate.py::test_backend_failure_rejects_with_its_own_reason_code_after_exhausting_retries`
and `tests/mvp_harness/test_harness_end_to_end.py::test_segmentation_backend_failure_resolves_to_a_clean_reject`.

**Retrying is bounded by elapsed time, not just an attempt count.**
`SegmentationGate.run()` retries a failed backend call with the same
parameters (plain infra retry, unrelated to `standard.retry.state_machine`),
governed by two independent limits - whichever is hit first stops
retrying:

- `max_backend_retry_seconds` (default `20.0`, primary): total elapsed
  time since the first attempt. A single BiRefNet call can itself take up
  to ~35s on CPU (this file's own measured worst case above), so a fixed
  *count* of retries has no bound on total time - two retries at 35s each
  would blow well past this app's own ~60s client-side timeout
  (`docs/standard-api-contract.md`) on retries alone, leaving nothing for
  the rest of the pipeline. Checked *before* starting each additional
  attempt, not mid-call - an in-flight synchronous inference call can't be
  cancelled without added thread/process machinery, which is out of scope
  here, so the first attempt is never cut short but no further attempt
  starts once the budget is spent.
- `max_backend_retries` (default `2`, secondary safety cap): guards the
  opposite case - a backend that fails near-instantly (e.g. connection
  refused) would otherwise retry many times within the time budget for no
  benefit.

Both are threaded through as parameters on `SegmentationGate.__init__`,
`standard.pipeline.run_pipeline`, and `standard/api/main.py`'s FastAPI
dependencies (`get_max_backend_retry_seconds` etc.) - see
`tests/standard/test_segmentation_gate.py::test_infra_retry_stops_on_time_budget_before_exhausting_the_count_cap`
and the matching API-level test in `tests/api/test_process_endpoint.py`
for proof the time budget actually binds, not just the count.

## Real-model integration tests

`tests/integration/test_birefnet_real_inference.py` runs actual `rembg`
inference (not `FakeBackend`) on a synthetic "blob on a table" stand-in
image (no bundled photo, to sidestep any license-verification risk).
Marked `integration` and **excluded from the default `pytest` run**
(`pyproject.toml`'s `addopts = -m "not integration"`) because it downloads
model weights on first run and takes real wall-clock seconds, not
milliseconds. Run explicitly:

```bash
cd ai-service
./venv/Scripts/python.exe -m pytest -m integration tests/integration/ -v -s
```

For a **visual** check (saves an original/mask/red-overlay image set to
`outputs/` so you can actually look at the segmentation quality):

```bash
./venv/Scripts/python.exe scripts/demo_birefnet_segmentation.py                       # synthetic stand-in
./venv/Scripts/python.exe scripts/demo_birefnet_segmentation.py --image path/to/photo.jpg   # your own photo
```
