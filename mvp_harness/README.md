# MVP test harness (spec §10, §11 step 7)

Runs every photo in a folder through the full Standard pipeline -
Segmentation → Layout → Shadow → Color → Template Render → Integrity
Validator → Retry - once per template (6 templates, `standard/templates/definitions.py`),
and writes out rendered images plus CSV reports. Built as **scaffolding
only**: real photos are a separate, ongoing data-collection track, and
interpreting the results (which template "wins") is explicitly out of
scope for this step - see "What this does NOT do" below.

## Status

Verified so far with **synthetic images only** (`tests/mvp_harness/test_harness_end_to_end.py`):
a solid-color background with a solid-color rectangle "food" region,
segmented with a trivial threshold stand-in instead of real ML. That
proves the harness's own plumbing (folder scanning, all 6 templates,
CSV/matrix writing, crash isolation) runs start to finish without dying.
**It has not been run against real store photos or the real BiRefNet
segmentation backend.** Do that once a real photo folder exists - see
"Running it for real" below.

## Phase 1 scope: 4 categories x 6 templates (Teishoku/Set Meal deferred)

§10's plan is 6 categories x 6 templates x 10 images/cell (360 total).
**Phase 1 real-photo collection and harness runs are narrowed to 4
categories - ramen / donburi / dessert / pasta - x all 6 templates.**
Teishoku and Set Meal are **deferred, not dropped**: see "No role
classifier" under "What this does NOT do" below for why (their typical
3+ food components trip today's unknown-majority grid fallback into
forced `REVIEW` almost every time, which isn't a meaningful signal about
template quality) and what unblocks them (a real role classifier, built
from the 4-category dataset once it exists).

## Input format

A folder of images (`.jpg`, `.jpeg`, `.png`), each filename starting with
a category prefix followed by an underscore. `discovery.py` still
recognizes all 6 of §10's categories (unchanged) - the narrowing above is
a Phase 1 *test-plan* decision about which photos to collect and run
right now, not a code restriction:

```
ramen_001.jpg
donburi_014.jpg
dessert_002.png
pasta_007.jpg
teishoku_003.jpg      <- recognized, but not part of the Phase 1 collection push
set_meal_009.jpg      <- checked as the two-token prefix "set_meal", not "set"
```

A filename that doesn't match any of the 6 is recorded with category
`"unknown"` rather than rejected outright.

## Running it for real

```bash
cd ai-service
./venv/Scripts/python.exe -m pip install rembg onnxruntime   # real segmentation backend - not installed by default
./venv/Scripts/python.exe -m mvp_harness path/to/photos path/to/output
```

That uses the real `BiRefNetBackend` (CPU) and the already-registered
`standard/policy/policy.yaml` / `standard/retry/retry_policy.yaml`. Optional
flags let you redirect the three report files:

```bash
./venv/Scripts/python.exe -m mvp_harness photos/ output/ \
  --run-csv output/run_results.csv \
  --matrix-csv output/matrix_report.csv \
  --error-log output/errors.log
```

Expect it to be slow on CPU (BiRefNet + 6 templates per image) - there's
no batching or parallelism here, matching the scaffolding-only scope of
this step.

## Output

- **Rendered images**: `output/<image_stem>__<template_id>.jpg` for every
  (image, template) pair that got far enough to render (i.e. segmentation
  didn't REJECT).
- **`run_results.csv`**: one row per (image, template) -
  `image_id, category, template_id, validator_status, reasons, retry_attempts_used, processing_time_ms`.
  `validator_status` is `PASS` / `REVIEW` / `REJECT` / `ERROR`;
  `REJECT` only ever appears from a Segmentation Gate REJECT (§7: no
  retry strategy exists for any `SEGMENTATION_*` reason) - anything that
  reaches the Integrity Validator and still fails there is resolved by
  the Retry state machine down to `PASS` or `REVIEW` before this column
  is written, never left as `REJECT`. `reasons` is semicolon-joined and
  is the union of the Segmentation Gate's reasons and whatever the
  Validator/Retry ended with.
- **`matrix_report.csv`**: one row per `(category, template_id)` actually
  seen, with PASS/REVIEW/REJECT/ERROR counts and a `pass_rate` - built
  specifically so a template that's disproportionately bad for one
  category is visible at a glance (§10's "cut the losers" goal). No
  ranking or recommendation is computed - that's a human decision once
  there's enough real data.
- **`errors.log`**: full traceback for every `(image, template)` pair
  that raised an exception, each entry timestamped and labeled with which
  pair it came from. A crash never stops the run (`cli.py`'s `run_harness`
  catches per-pair) - only that pair's `run_results.csv` row becomes
  `validator_status=ERROR` with the exception's type+message in `reasons`.

## What this does NOT do

- **No role classifier - kept as-is, deliberately.** `runner.py`
  hand-assigns the single largest segmented object as a confident
  `"main"`; everything else stays `"unknown"` (same stand-in used
  throughout this build's own tests). Verified by direct simulation
  (`LayoutEngine` + `unknown_ratio`): with N segmented objects and only
  one named, `unknown_ratio = (N-1)/N` - exactly `0.5` at N=2 (not
  `> 0.5`, so it does NOT trip the fallback), but `> 0.5` for every
  N >= 3 (0.667, 0.75, 0.8, 0.833, ...). Teishoku/Set Meal photos
  typically segment into 3+ components by definition (rice + soup + main
  + 1-2 sides), so under this rule they will almost always trip the
  unknown-majority grid fallback and land in forced `REVIEW` via
  `LAYOUT_UNKNOWN_ROLE_MAJORITY` - not a rare edge case for that category,
  close to guaranteed. **Decision**: don't paper over this with a
  catch-all "side" tag just to slip under the 50% line - per spec §2
  ("정확히 맞히는 것보다 모르면 unknown으로 빠지는 것이 안전"), staying
  honestly `"unknown"` when there's no real basis for a role is correct
  behavior, not a bug to route around. Instead, **Phase 1 real-photo
  testing is scoped to ramen / donburi / dessert / pasta** - single-bowl
  categories where one confident "main" is normally the whole story, so
  this stand-in is adequate. **Teishoku / Set Meal are deferred, not
  dropped**: once the 4-category dataset exists, build a real role
  classifier *from that real data* (rather than guessing thresholds now)
  and take on these two categories as their own follow-up.
- **One color preset for every run.** Always `STANDARD`
  (`standard/color/presets.py`) - Premium/Hero and per-category presets
  aren't wired in here.
- **Module B (Content Integrity) - fixed a real wiring bug, now genuinely
  live.** An earlier version of `runner.py` passed the *same array* as
  both `expected_food_rgb` and `final_food_rgb` - Module B was comparing
  a value against itself, guaranteed zero error regardless of what the
  pipeline did. Fixed: `final_food_rgb` now comes from the actual
  saved-and-reloaded JPEG (spec §6: "Actual Final Food" = what a merchant
  really receives, compression included), and `expected_food_rgb` /
  `food_mask` / `original_food_rgb` are aligned via
  `standard.templates.renderer.place_food_layer` (redoes the real
  per-object placement rather than assuming one uniform shift, so it
  stays correct through grid-fallback and Retry's non-uniform
  `recompute_translation` too). `test_module_b_actually_fires_through_the_real_orchestrator`
  proves it can now genuinely flag `CONTENT_HF_ERROR` on a noisy/
  hard-to-compress food region; a flat-color block (this file's other
  fixtures) legitimately shows near-zero error and correctly doesn't
  trigger it - that's Module B working, not a repeat of the old bug.
  Module A still only ever trivially passes in this pipeline (masks are
  never resampled, by design). Separately verified:
  `test_retry_recompute_translation_path_keeps_module_b_aligned` forces
  an initial REJECT so the Retry state machine actually runs
  `recompute_translation` (a *non-uniform* per-object shift, unlike
  `recenter_group`'s single group-wide shift) and confirms the
  re-rendered/re-saved food region still aligns correctly - the first
  fix's own test only ever exercised the plain `recenter_group` path,
  not retry's. Note in passing: through the *real* harness today (using
  `_assign_stand_in_roles`, which only ever names one role), that retry
  path is actually unreachable - `OCCLUSION_CHANGED` and
  `OBJECT_AREA_RANK_REVERSED`, the only two reasons mapped to
  `recompute_translation`, both require 2+ *named* roles to fire at all
  (see "No role classifier" below), so this test injects a second named
  role via monkeypatch specifically to exercise the code path.
- **No dataset judgment.** It produces numbers; deciding which of the 6
  templates to keep per category (§9: eventually compress to the top
  2-3) is a separate step once real volume exists.

## JPEG quality: decided (92)

`runner.py`'s `JPEG_QUALITY = 92` (used both for the file actually saved
and for the save-and-reload round-trip Module B validates against) is a
deliberate choice now, not a placeholder - measured via
`scripts/compare_jpeg_quality.py` against a synthetic but genuinely
textured food-like image (fine grain noise + scattered bright highlights +
a hard edge against a dark background - a flat-color block, used
elsewhere in this repo's test fixtures, tells you nothing about JPEG
artifacts since it compresses losslessly regardless of quality). The
script reuses `standard.validator.module_b.compute_hf_error` directly with
`policy.yaml`'s actual `contrast_weight`/`sharpness_weight`, so the numbers
below are exactly what Module B would see in production, not a proxy:

| quality | size (KB) | HF error | PSNR (dB) | vs `base_threshold: 0.04` |
|---|---|---|---|---|
| 85  | 48.1  | 0.0182 | 35.18 | OK (46% of budget) |
| 90  | 60.2  | 0.0140 | 36.89 | OK (35% of budget) |
| **92**  | **65.9**  | **0.0115** | **37.70** | **OK (29% of budget)** |
| 95  | 81.3  | 0.0078 | 38.85 | OK (20% of budget) |
| 100 | 157.7 | 0.0026 | 40.32 | OK (7% of budget) |

Visually, 85 vs 95 side-by-side (`outputs/jpeg_quality_q85.jpg` vs
`_q95.jpg`) are hard to tell apart at normal viewing size for this test
image. **Decision: 92** - a comfortable ~3.5x safety margin under
`base_threshold` from compression alone (real photos will add their own
genuine content on top of this baseline, so headroom matters), without
paying 100's near-2.5x file-size cost for a difference PSNR says is
real but this test image doesn't visibly show. Not a tight tradeoff
either way - 85 already uses under half the HF-error budget - so this
is a reasonable default rather than a finely-tuned optimum.

**`base_threshold: 0.04` itself is confirmed reasonable, not adjusted**:
even the most aggressive quality tested (85) leaves more than half the
threshold unused from compression alone. This check used a synthetic
texture, not real photos - if real §10 data ever shows meaningfully
different baseline HF-error numbers, revisit both this quality choice and
the threshold together (they're coupled, per the note in `runner.py`).

## Known caveat carried over from earlier build steps

Objects whose `scale` ends up != 1.0 (only reachable via the grid
fallback, which is always `REVIEW` - see `standard/README.md`'s Layout
Engine section) render with a shadow shaped from the *unscaled* mask.
Never reaches a silent `PASS` row in this harness's output, for the same
reason it doesn't in the pipeline generally.
