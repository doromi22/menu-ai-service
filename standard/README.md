# Standard MVP pipeline

Implements `MENUAI_V1_SPEC.md` Phase 1 (§1): **no generative-AI dependency**.
This lives beside the existing Premium/generative engine (`../main.py`,
`../modal_app.py` - BiRefNet + SDXL-Lightning collage) without touching it;
the two are meant to stay architecturally separate so a problem can be
attributed to "compositing engine vs generative model vs integrity policy"
(spec §1).

## Status (spec §11 build order)

| # | Item | Status |
|---|------|--------|
| 1 | Repo structure | done (this tree) |
| 2 | `FoodObject`, Segmentation Gate | done |
| 2 | Layout / Shadow / Color / Integrity Validator modules | done |
| 3 | Policy YAML schema + immutable/hash loader (§6) | done |
| 4 | Reason-code enum + metadata schema (§8), live endpoint | done - `standard/api/main.py` serves the `../docs/standard-metadata.schema.json` contract for real; `standard/pipeline.py` is the shared orchestrator both it and `mvp_harness` call |
| 5 | Retry policy state machine (§7) | done (decision logic only - see its README section for what's deliberately deferred) |
| 6 | Graphic template renderer (§9) | done (only `T01_warm_ivory` uses spec-given numbers; `T02`-`T06` are placeholders, see its README section) |
| 7 | 360-image MVP test harness (§10) | scaffolding done (`../mvp_harness/`), verified with synthetic images only - Phase 1 real-photo collection narrowed to 4 of the 6 categories (Teishoku/Set Meal deferred - see its own README) |
| 8 | Premium branch (SDXL/IC-Light) | explicitly out of scope here |

## Layout

```
standard/
  objects/food_object.py     FoodObject dataclass, SemanticRole, OCCLUSION_BASELINE (§2)
  segmentation/
    backend.py                SegmentationBackend protocol + BiRefNetBackend (rembg, CPU) - real inference wired up
    gate.py                   SegmentationGate -> PASS / REVIEW / REJECT (§3); backend failures get up to
                                2 same-parameters infra retries, then REJECT + SEGMENTATION_BACKEND_ERROR
                                (result.is_infra_error=True) - never handed to the Retry state machine
    README.md                  rembg-vs-checkpoint decision, measured model size/timing, error-handling rationale
  pipeline.py                  run_pipeline() - the shared orchestrator (Segmentation->Layout->Shadow->
                                Color->Validator->Retry) both mvp_harness/runner.py and standard/api/main.py call
  api/
    main.py                    FastAPI app: POST /v1/standard/process (§11 step 4) - separate service from
                                ../../main.py / ../../modal_app.py (Premium)
    README.md                  how to run it, how tests inject a fake backend, the E2E test-only env var
  geometry.py                  translate_mask() family - shared by layout.occlusion + shadow.engine
  layout/
    rank.py                    compute_role_area_rank / is_area_rank_preserved (§4) - reused by validator D
    occlusion.py                occlusion_ratio, compute_occlusion_changes (§4) - reused by validator D
    scale.py                    clamp_scale (§4, 0.97-1.03)
    translate.py                 recenter_group - default translation-first path
    grid.py                      arrange_grid - unknown>50% fallback, uses clamp_scale to fit cells
    engine.py                    LayoutEngine - wires the above into one LayoutResult
  shadow/
    profiles.py                 CONTACT_SHADOW_PROFILES (per-object) - default/heavy/glass
    engine.py                   ShadowEngine - per-object contact + one ambient density map (§4)
  color/
    presets.py                  STANDARD/PREMIUM/HERO (§5, exact) + food_params_for (conservative food pass)
    grade.py                    apply_grade, grade_food_and_background - CLAHE/adaptive ops never written
  validator/
    module_a.py                  Implementation Integrity: mask IoU/area delta
    module_b.py                  Content Integrity: transform-aware HF error (Expected vs Final)
    module_c.py                  Appearance Integrity: saturation/luminance gain vs true original
    module_d.py                  Layout Integrity: reuses layout.rank/occlusion, adds scale checks
    validator.py                 run_validator() - combines A-D into one PASS/REVIEW/REJECT + reasons
  policy/
    policy.yaml                Integrity Validator thresholds, matches §6 (+ severity tags)
    policy.lock.json            policy_version -> sha256, this policy's "git tag/digest" (§6)
    schema.py                   required structure + severity-tag validation
    loader.py                   PolicyLoader: register()/load(), immutability enforcement
  severity.py                   classify_violation() - shared by segmentation.gate, layout, AND validator
  config/
    segmentation.yaml          thresholds, matches §3 exactly (+ severity tags)
    loader.py                  plain yaml.safe_load wrapper (NOT the policy/ loader above)
  reason_codes.py              GateResult + ReasonCode enums, shared by every stage
  retry/
    retry_policy.yaml           max_attempts, strategy_by_reason, final_action - matches §7 exactly
    config.py                   load_retry_policy() -> typed RetryPolicyConfig
    strategies.py                conservative_color/neutral_color (reuse food_params_for),
                                  recompute_translation (reuses recenter_group), force_scale_1_0
    state_machine.py             RetryStateMachine - decision logic; evaluate() callback is injected
  templates/
    definitions.py                Template dataclass + 6 TEMPLATES - only T01_warm_ivory is spec-exact
    background.py                 render_background() - deterministic gradient + fixed procedural texture
    renderer.py                   TemplateRenderer - composites Shadow Engine's layer + positioned food;
                                    also exports place_food_layer() (the per-object placement alone,
                                    reused by ../mvp_harness/runner.py to align Module B/C's inputs)
../docs/
  standard-metadata.schema.json   §8 metadata JSON Schema - now what api/main.py actually returns
  standard-api-contract.md        OpenAPI contract (implemented) + Laravel storage decision + timeout notes
../mvp_harness/
  runner.py cli.py report.py discovery.py   360-image MVP test harness (§10, §11 step 7) - see its own README
```

## Open design decision: PASS/REVIEW/REJECT mapping in the Segmentation Gate

Spec §3 only fixes one case explicitly (non-food object overlapping food,
low confidence -> REVIEW, "자동 처리 금지"). It does not say which of the
five yaml thresholds should REJECT vs REVIEW - so this build tags each one
in `config/segmentation.yaml`'s `segmentation.severity` block:

| threshold | reject_multiplier | meaning |
|---|---|---|
| `min_food_area_ratio` | `0.5` | `food_area_ratio < 0.03 * 0.5 = 0.015` -> REJECT ("거의 음식이 없다"); `0.015`-`0.03` -> REVIEW |
| `max_food_area_ratio` | `1.0` | any exceedance of `0.95` -> REJECT immediately, no REVIEW buffer |
| `max_boundary_touch_ratio` | `null` | REVIEW-only, however far over |
| `max_hole_ratio` | `null` | REVIEW-only, however far over |
| `min_component_area_ratio` | n/a | not a metric threshold - it's the noise floor for dropping specks; if it filters away every component, that collapses to the "no food mask at all" REJECT case (`SEGMENTATION_NO_FOOD_DETECTED`), handled structurally rather than via a multiplier |

The rationale: §3 defines REVIEW as "사람이 crop·보정" (a human can generally
still salvage a bad-but-present mask) and REJECT as output that's unused
entirely (§7). Area is the one metric where the spec text itself implies
a hard floor/ceiling a human can't crop their way out of; boundary-touch
and hole-ratio are framing/segmentation-quality issues a crop or manual
touch-up can plausibly fix, so they stay REVIEW-only.

**This split is provisional, not derived from real data** - retune the
multipliers (or turn a `null` into a real one) once the §10 360-image MVP
test produces actual REVIEW-queue volume, and check it against the §7
operating target (PASS 900 / conservative retry 60 / human REVIEW 40 per
1000). If REVIEW volume runs far above that target, the boundary-touch or
hole-ratio checks are the first candidates to gain their own
`reject_multiplier` rather than staying `null` forever.

Two reason codes are defined but not yet produced by any gate:
`SEGMENTATION_LOW_CONFIDENCE` (needs a per-mask confidence score - rembg's
alpha is a soft mask, not a scalar confidence) and
`SEGMENTATION_NONFOOD_OVERLAP` (needs a non-food object classifier, e.g.
for chopsticks). Both are tracked as future work, not silently dropped.

One reason code was added beyond §6's literal enum listing:
`OBJECT_AREA_RANK_REVERSED`. §4 states object-area-rank preservation
(main > rice > soup > side) as its own REVIEW trigger in a sentence
separate from occlusion/front-back order, but §6's enum has no dedicated
code for it - without one, that REVIEW would be unreportable/uncountable.

## Policy YAML + immutable/hash loader (§6, §11 step 3)

`policy/policy.yaml` holds the Integrity Validator's thresholds (mask IoU,
HF error, food saturation/luminance gain, layout scale/occlusion) exactly
per §6, plus the same `severity.<section>.<key>.reject_multiplier` tagging
used in `config/segmentation.yaml` - both now share one classifier
(`standard/severity.py`), so "consistent design" here means literally the
same function, not just a similar yaml convention.

Every numeric threshold's `reject_multiplier` is `null` (REVIEW-only) for
the Phase-1 MVP - unlike segmentation's `max_food_area_ratio`, a
mask/content/appearance/layout drift here comes from the pipeline's own
transform, not the source photo, so nothing is prejudged as auto-reject
until §10 data says otherwise. The two boolean layout checks
(`preserve_object_area_rank`, `preserve_front_back_order`) carry no
severity tag at all - §4 already fixes their outcome to REVIEW.

"Immutable artifact, git tag/digest로 고정" (§6) is implemented without
touching git: `PolicyLoader.register(path)` computes sha256 over the
file's raw bytes and pins `{policy_version: hash}` into `policy.lock.json`
- a deliberate, human-triggered step. `PolicyLoader.load(path)` then
refuses to proceed if `policy_version` isn't yet registered, or if a
registered version's file content no longer matches its pinned hash (i.e.
someone edited a "locked" policy in place instead of bumping
`policy_version`). The hash itself can't live inside `policy.yaml` -
a file hashing its own bytes is circular - which is why the lock file is
a sibling, not a field.

`policy.lock.json` in this repo already has `food_integrity_1.0.1`
registered against the shipped `policy.yaml` (bumped once already, from
`1.0.0`, to add the `contrast_weight`/`sharpness_weight` provenance
comment below - a real demonstration of the intentional-version-up
workflow two bullets down, not just a description of it); re-running
`PolicyLoader.register(...)` on unchanged content is idempotent.

**Two workflows this enables, and how to tell them apart:**

- **Intentional version-up** (e.g. `v1.0.0` -> `v1.1.0`): edit `policy.yaml`
  *and* bump the `policy_version` field in the same change, then call
  `PolicyLoader.register(path)` once, deliberately, to pin the new
  version's hash. `load()` will accept it immediately - it's a version
  the lock file has never seen, registered on purpose.
- **Accidental edit of a shipped file** (`policy_version` left unchanged):
  `load()` raises `PolicyIntegrityError` on the very next call, because
  the file's hash no longer matches what's pinned for that version.
  `register()` also refuses to silently re-pin it (`"refusing to
  re-register ... bump policy_version instead"`). The fix is never to
  force-overwrite the lock entry - it's to either revert the edit or
  treat it as an intentional version-up per the bullet above.

## Layout / Shadow / Color / Integrity Validator (§4-6, §11 step 2 completion)

Built in dependency order - Layout's output feeds Shadow, Layout+Color
feed the Validator's B/C/D modules:

**Layout Engine** (`layout/`) is translation-first by default
(`translate.recenter_group` - a rigid group shift, so it can never change
area rank or relative occlusion on its own) and only switches to
`grid.arrange_grid` when `unknown_ratio(objects) > 0.5` (§2), which also
flags `LAYOUT_UNKNOWN_ROLE_MAJORITY` - a reason code added beyond §6's
enum (§6 predates this build step) so the fallback is countable, same
justification as `OBJECT_AREA_RANK_REVERSED` above. `grid.arrange_grid`
is the one real call site for `scale.clamp_scale`: an object bigger than
its assigned cell gets shrunk to fit, clamped into 0.97-1.03 rather than
using the raw fit ratio. `rank.py` and `occlusion.py` are pure
measurement functions independent of what produced "final" - deliberately
reused as-is by Validator module D rather than re-implemented, while the
Layout Engine's own `occlusion_change_violations` stays a simple
early-heads-up flag (not severity-classified) - see `layout/occlusion.py`'s
docstring for exactly where that split sits.

**Known limitation, not yet re-verified against real data**:
`occlusion.py`'s `_role_representative` picks one object per role when
computing occlusion pairs. For a Teishoku/Set Meal frame with more than
one object sharing a role (e.g. two side dishes), only that one
representative's occlusion is ever checked - a change in occlusion
involving any *other* same-role object goes undetected. Left as-is for
the Phase-1 MVP; revisit once the §10 360-image test includes real
multi-object-per-role frames and shows whether this actually matters.

**Spec-vs-implementation gap, found while tracing the shadow/scale
limitation below**: §4 says normal (non-fallback) layout can also make a
fine-grained 0.97-1.03 scale adjustment when needed. The actual normal
path (`translate.recenter_group`) only does a rigid translation -
scale-adjustment isn't implemented there at all (only the grid-fallback
path adjusts scale). Not implementing it now; revisit once the §10 real-
photo test shows whether translation-only actually leaves unresolved
cases (e.g. overlap a translation alone can't fix) that would need it.

**Shadow Engine** (`shadow/`) does deterministic rendering only, no
PASS/REVIEW/REJECT judgement. It outputs a grayscale shadow-density map
(not an RGBA composite) because the Graphic template renderer (§9) that
would composite real background pixels underneath doesn't exist yet.
Ambient shadow (one, for the whole group) uses the *template's* §9
`shadow` block; contact shadow (one per object) uses that object's own
`FoodObject.shadow_profile` (`shadow/profiles.py`) - giving that field an
actual job, and satisfying §4's explicit "단일 giant shadow 금지" for
Teishoku-style multi-bowl frames.

**Color Grade** (`color/`) implements exactly §5's allowed list - global
brightness/contrast/saturation, global white balance, one fixed
(non-adaptive) tone-curve LUT - and nothing else; CLAHE/local-contrast/
adaptive-tone-mapping are never written, guarded by
`test_forbidden_operations_are_not_implemented`. `food_params_for()`
scales a preset's delta-from-neutral by `FOOD_CONSERVATISM` (`0.5`,
provisional - §5 mandates food be "always more conservative" but gives no
exact ratio) and separately caps saturation at `FOOD_SATURATION_CAP`
(`1.05`, per §5). `grade_food_and_background()` returns the exact params
used for each region - required by Validator module B's transform-aware
comparison below.

**Integrity Validator** (`validator/`) - each module is an independent,
directly-testable pure function; `validator.run_validator()` combines all
four with the same aggregation rule as the Segmentation Gate (any REJECT
wins, else any violation means REVIEW, else PASS), reusing
`standard.severity.classify_violation` rather than a new one:

- **A - Implementation** (`module_a.py`): mask IoU / area delta between
  `original_mask` and `final_mask`. Always trivially passes today (Layout
  Engine never touches `mask`, only `final_center`/`scale`) - it exists to
  catch a future renderer (§9) that starts resampling pixels. Contour/
  alpha checks are named in §6's module description but have no policy
  threshold; not scored.
- **B - Content** (`module_b.py`): the transform-aware rule, verbatim from
  §6 - `Expected Food = apply_grade(Original, food_params)`, then compare
  `Expected <-> Final`, never `Original <-> Final`. The HF-error formula
  itself (a weighted blend of a local-contrast-difference map and a
  Laplacian-sharpness-difference map, using §6's `contrast_weight`/
  `sharpness_weight` literally as the blend weights) is this build's own
  design - §6 gives the two weights but not a combining formula.
- **C - Appearance** (`module_c.py`): saturation/luminance **gain**
  (increase only, matching the policy field names) of the food region,
  comparing the *true original* photo against the final output - unlike
  B, this one is allowed to compare against the real original, since it's
  measuring overall appearance drift, not verifying one specific
  transform. Red/orange emphasis and highlight are named in §6 but have
  no policy threshold; not scored, same as A's contour/alpha gap.
- **D - Layout** (`module_d.py`): reuses `layout.rank`/`layout.occlusion`
  directly, and takes an explicit `used_grid_fallback` flag from the
  Layout Engine's `LayoutResult` - if the frame fell back to grid layout
  (§2's unknown>50% rule), that alone forces REVIEW here too, not just as
  an informational flag on `LayoutResult` (an earlier draft of this build
  computed the flag but never actually wired it into the Validator's
  verdict - fixed once caught, with a regression test in
  `test_full_pipeline_integration.py`). Rank reversal is a fixed REVIEW
  (§4 fixes it, so no `classify_violation` call); occlusion change, per-object scale delta
  (`abs(scale - 1.0)`), and relative scale-ratio change (largest pairwise
  scale-ratio deviation from 1:1 - also this build's own definition, §6
  gives no formula) all go through `classify_violation` against
  `policy.yaml`'s thresholds. `preserve_front_back_order` folds into
  `OCCLUSION_CHANGED` (§4 states both in one sentence with one REVIEW
  consequence, and §6 has no separate code for it) - nothing built so far
  reorders `z_index`, so this branch has nothing live to check yet.

`tests/standard/test_full_pipeline_integration.py` runs one synthetic
photo through every stage above - Segmentation Gate -> Layout Engine ->
Shadow Engine -> Color Grade -> Integrity Validator - once to a genuine
PASS, once to a genuine REVIEW (simulating a compositing bug: food region
graded at background/HERO strength instead of its own conservative
`food_params`), and once to a forced REVIEW from `LAYOUT_UNKNOWN_ROLE_MAJORITY`
alone (§2's grid-fallback rule) with nothing else wrong. Role assignment
is hand-simulated after segmentation in these tests - no role classifier
exists yet, out of scope for this step.

## Retry Policy state machine (§7, §11 step 5)

`retry/state_machine.py`'s `RetryStateMachine` decides *what to try next
and when to give up* - it does not itself re-render anything. Actually
re-running the pipeline with new parameters needs the Graphic template
renderer (§9, still pending) and a pipeline orchestrator that doesn't
exist yet, so callers inject an `evaluate: Callable[[RetryParams],
ValidatorResult]` callback; in production that callback would re-run the
real pipeline, in tests it's a fake. The *parameters themselves* are not
faked - `strategies.py` calls the real `standard.color.presets.food_params_for`
and `standard.layout.translate.recenter_group` to produce them, per the
explicit instruction to reuse existing logic rather than rebuild it.

**Fallback strategies, all reused, not reimplemented:**

- `conservative_color` / `neutral_color` both call `food_params_for(STANDARD,
  conservatism=...)` - the exact function Color Grade itself uses to
  derive a food pass from a preset - with `conservatism=0.25` and `0.0`
  respectively. `0.0` collapses to the literal identity transform
  (`brightness=contrast=saturation=1.0, temperature_strength=0.0`): once
  even a conservative regrade hasn't worked, the final fallback is to
  stop touching color at all.
- `recompute_translation` reruns `recenter_group` as-is, then nudges every
  object away from the group centroid by a spacing factor that grows with
  the caller-supplied `attempt_index` (`1.05`, `1.10`, `1.15`, ...) - the
  minimal variation spec asked for, in the direction (more separation)
  that plausibly reduces occlusion drift.
- `force_scale_1_0` resets every object's `scale` to exactly `1.0`.

**Priority when multiple reasons are active in the same round**:
layout-category reasons (`OCCLUSION_CHANGED`, `OBJECT_SCALE_CHANGED`,
`RELATIVE_SCALE_CHANGED`, `OBJECT_AREA_RANK_REVERSED`) are ordered before
color-category ones (`CONTENT_HF_ERROR`, `FOOD_SATURATION_EXCEEDED`,
`FOOD_LUMINANCE_EXCEEDED`). Both still land in the *same* attempt's
`RetryParams` - only one `evaluate()` call happens per attempt regardless
of how many reasons it addresses - so in the current architecture this
ordering doesn't delay anything (color grading here doesn't depend on
object positions, and vice versa; §9's renderer would be where that
dependency could start to matter). It's kept as a deliberate, stable
convention anyway - fix structural/geometric problems before surface/color
polish - and it makes `RetryAttemptRecord.reasons_addressed` deterministic
for logging.

**Bundled per attempt, not consumed sequentially per category** - worth
stating explicitly since it directly affects whether `max_attempts: 3` is
actually enough: every currently-active reason's strategy is folded into
`needed` and applied together (one `RetryParams`, one `evaluate()` call)
within a single loop iteration - `retryable`/`ordered` are never filtered
down to "just this round's category" before the strategies are chosen.
A layout problem (1 strategy) and a color problem (2 strategies) present
together cost 2 attempts total, not 3+: attempt 0 applies both the layout
fix and the first color strategy at once; only the still-failing color
reason needs attempt 1. Proved by
`test_simultaneous_reasons_from_different_categories_use_attempts_efficiently`.
Relatedly, `current_reasons = list(result.reasons)` after every
`evaluate()` call is a full replace, not a diff against what was already
known - a brand-new reason that appears only *after* a fix (e.g. lowering
color correction fixes saturation but reveals an HF error) is picked up
and retried from its own first strategy, not missed because it "wasn't
in the original list". Proved by
`test_a_brand_new_reason_appearing_after_a_fix_is_detected_and_retried`.

**Anti-infinite-loop, by construction rather than by detection**: each
reason is retried at most `len(strategy_by_reason[reason])` times, tracked
by a plain per-reason counter. Once that count reaches the list's length,
the reason is dropped from the retryable set *before* the next `evaluate()`
call - it never receives a repeated strategy step, so there's no need to
fingerprint parameters to detect a loop after the fact. If dropping
exhausted reasons empties the retryable set, the loop stops immediately
instead of spending the rest of `max_attempts` on calls that cannot
change (§7: "단순 재실행 금지"). Reason codes with no entry in
`strategy_by_reason` at all (e.g. `MASK_CHANGED`, any `SEGMENTATION_*`
code) are treated the same way from the start - a pipeline bug or an
unusable source photo isn't fixable by a different color/layout
parameter, so they're never retried, only carried through to
`final_reasons` if they persist.

`RetryOutcome.result` is always `PASS` or `REVIEW`, never `REJECT` -
matching §7's `final_action: review`: exhausting every retryable reason
(whether by hitting `max_attempts` or by every reason's strategy list
running out first) always escalates to a human, it never silently
discards the result. `RetryOutcome.final_reasons` carries whatever reasons
were still present at the point retrying stopped, so a human reviewer
sees exactly why the fallbacks didn't work.

## Graphic Template Renderer (§9, §11 step 6)

`templates/renderer.py`'s `TemplateRenderer` is the piece several earlier
modules' docstrings pointed at as "doesn't exist yet" - Shadow Engine's
grayscale-only output, the full-pipeline integration test's synthetic
stand-in for a composited scene. It composites, per §9:

1. `background.render_background(template, canvas_size)` - a flat/graphic
   background built entirely from `Template` parameters (base color +
   directional gradient + a **fixed procedural texture**, not random
   noise, so the same template always renders byte-identical output -
   `_fixed_texture` is a function of pixel position only, no RNG involved).
2. `ShadowEngine.render(...)` (reused as-is, §11 step 2) using the
   template's own `shadow` block as `AmbientShadowParams` - darkens the
   background before any food is placed on top.
3. Each `FoodObject`, cropped to its bbox, resized by `scale`, and
   recentered on `final_center` - the one place in the pipeline `scale`
   actually changes rendered output rather than just being checked by the
   Integrity Validator. Painted back-to-front by `z_index`.

**Input is the already color-graded food image**, not the original photo.
Per §9 the real background is this template's synthetic one, not a
regraded version of the original photo's own background pixels - grading
the original image's background region (as
`color.grade.grade_food_and_background` does) was only ever a stand-in
for testing Color Grade in isolation before this renderer existed, and
that test file is unchanged - it's still validly testing Color Grade, on
its own terms.

**Only `T01_warm_ivory`'s parameters come from the spec** (§9's exact
example). §9/§10 name five more templates - Cool White, Beige, Dark
Premium, Japanese Editorial, Fresh Natural - without giving any numbers
for them; `definitions.py`'s entries for those five are this build's own
placeholders, explicitly flagged in its docstring, standing in until the
§10 360-image MVP test (10 images/cell x 6 templates x 6 categories)
produces real values. `Template` has no category-shaped field at all -
structurally enforcing §9's "Template ↔ Food category 완전 분리".

**Known simplification, not fixed here, and verified to be REVIEW-only**:
Shadow Engine renders each object's shadow from its *unscaled* mask
(`renderer.py` passes the same `objects` list to `ShadowEngine.render()`
that it later reads `obj.scale` from when placing food -
`shadow/engine.py`'s `_placed_mask` never looks at `scale` at all). When
`scale != 1.0`, the rendered shadow's shape won't exactly match the
resized food.

Spec §4 allows two paths to a non-1.0 scale: (1) the unknown-majority
grid fallback (§2), and (2) fine-grained 0.97-1.03 adjustment during
*normal* layout, which - if it existed - could reach a silent PASS with
no human ever looking at it. Traced through the actual code (not just
read the spec) to confirm which one this build's mismatch can reach:
a repo-wide grep for every place `FoodObject.scale` is ever set to
something other than its input value turns up exactly two call sites -
`layout/grid.py`'s `arrange_grid` (fit-to-cell shrinking) and
`retry/strategies.py`'s `force_scale_1_0` (which sets it to exactly
`1.0`, matching Shadow Engine's assumption, so no mismatch there either).
`layout/translate.py`'s `recenter_group` - the *only* function the
normal, non-fallback path calls - is a pure rigid translation that never
touches `scale`; path (2) simply isn't implemented in this codebase, so
it can't be the source of anything. `arrange_grid` only ever runs when
`unknown_ratio > 0.5` (`layout/engine.py`), and that flag now
unconditionally forces `REVIEW` in `validator/module_d.py` (the fix
earlier in this file's history). **Conclusion: this mismatch can only
ever reach a human-reviewed frame, never a silent PASS - no code change
needed now.** If a real path-(2) scale-adjustment feature is added later,
this gap needs to be revisited at that point, not before.

## Infra-error separation + live API endpoint (§11 step 4)

**Segmentation backend infra failures are not the same kind of REJECT as
a bad photo.** `SegmentationGate.run()` retries the backend call up to
`max_backend_retries` times (same parameters, `backend_retry_backoff_seconds`
apart - a plain infra retry, unrelated to `standard.retry.state_machine`,
which varies color/layout parameters to work around a real data-quality
REJECT and has no strategy that could ever fix a backend that's down).
Once exhausted, `SegmentationGateResult.is_infra_error` (a property
derived from `SEGMENTATION_BACKEND_ERROR in reasons`, not a separately-
settable field, so it can't drift out of sync) lets a caller tell this
apart from a real REJECT. **Deliberately not a new `GateResult` value**:
that enum is shared by the Validator and Retry state machine too, and
neither needs to represent "the backend was down" as one of their own
states - segmentation short-circuits before they ever run, so the
distinction only ever needs to exist at the `SegmentationGateResult`
level. This is what lets §11-4's API return a `metadata.is_infra_error`
boolean Laravel can use to show "일시적 오류, 잠시 후 다시 시도" instead of
"이 사진은 지원되지 않습니다" - see standard/segmentation/README.md and
tests/standard/test_segmentation_gate.py's infra-retry tests.

**`standard/pipeline.py`** is the orchestrator several earlier sections
of this README referred to as "doesn't exist yet" - it's the extraction
of what used to be `mvp_harness/runner.py`'s inline logic
(Segmentation -> Layout -> Shadow/Template -> Color -> Validator ->
Retry), so that both `mvp_harness/runner.py` (file-based, batch) and
`standard/api/main.py` (one HTTP request at a time) call the exact same
`run_pipeline()` rather than each re-implementing the wiring.

**`standard/api/main.py`** is a separate `FastAPI()` app from
`../main.py` / `../modal_app.py` (Premium) - see
`../docs/standard-api-contract.md` for the full rationale (different
infra requirements: CPU-only vs GPU) and the request/response contract.
On the Laravel side: `app/Services/StandardAiService.php` calls it and
persists the result via §11-2's pivot-table storage
(`Image::syncReasons()`); `tests/Feature/StandardPipelineEndToEndTest.php`
starts a real `uvicorn` process (with a fake segmentation backend via the
`STANDARD_API_FAKE_SEGMENTATION=1` test-only env var) and drives it over
real HTTP to prove the cross-language wiring end to end.

## Running tests

```bash
cd ai-service
./venv/Scripts/python.exe -m pytest -q
```

`pyproject.toml` sets `pythonpath = ["."]` so `import standard...` resolves
regardless of cwd. Tests inject a `FakeBackend` and never touch rembg/
onnxruntime, so they run without downloading any model weights.

Real-model tests (`tests/integration/`) are excluded by default
(`pyproject.toml`'s `addopts = -m "not integration"` - the bare command
above shows `N passed, 2 deselected`) since they download ~930MB of model
weights on first run and take real seconds, not milliseconds. Run them
explicitly:

```bash
./venv/Scripts/python.exe -m pytest -m integration tests/integration/ -v -s
```

See `segmentation/README.md` for what they check, measured timings, and
the visual demo script (`scripts/demo_birefnet_segmentation.py`).
