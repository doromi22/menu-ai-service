# Standard pipeline API contract (implemented)

This documents the HTTP contract between Laravel (`menu-ai`) and the
Standard pipeline endpoint. **Now implemented**: `standard/api/main.py`
(§11 step 4) serves exactly this contract; `tests/api/test_process_endpoint.py`
verifies it with a fake segmentation backend (real BiRefNet inference is
covered separately by `tests/integration/test_birefnet_real_inference.py`).

## Why this diverges from the current Premium endpoint

The current Premium engine (`ai-service/modal_app.py`'s `FoodRenderer.segment`,
called from `app/Jobs/ProcessImageJob.php`) returns **only image bytes**
(`media_type="image/jpeg"`) - Laravel writes the response body straight to
disk. That's sufficient for Premium because there's no §8 metadata to
carry: no policy hash, no validator status, no reason codes.

The Standard pipeline needs to return both the processed image *and* the
§8 metadata object in one response, so the contract below wraps both in a
JSON envelope instead.

## Standard's own FastAPI app - deliberately separate from Premium

`standard/api/main.py` is its own `FastAPI()` instance, run as its own
process (`uvicorn standard.api.main:app --port 8002`, distinct from
Premium's local dev server on 8001 in `main.py`) - **not** a new route
bolted onto `modal_app.py` or `main.py`. Rationale:

- **Different infrastructure requirements.** Standard is CPU-only by
  design (spec §1); Premium needs GPU (`modal_app.py` requests an A10G via
  Modal). Merging them would either force Standard onto GPU infra it
  doesn't need, or complicate Premium's deployment with a CPU-only route
  that doesn't belong in a GPU container image.
- **Matches the architectural separation spec §1 already asks for**
  ("compositing engine vs generative model vs integrity policy 문제를
  구분") - keeping them as separate deployable services (separate
  processes, separate ports, separate scaling/deployment targets - Standard
  could run on a cheap always-on CPU box; Premium stays on Modal, scaling
  to zero when idle) extends that same isolation to the infra layer, not
  just the codebase.
- Laravel gets a **new** env var (`STANDARD_AI_SERVICE_URL`) alongside the
  existing `AI_SERVICE_URL` (Premium) - see `app/Services/StandardAiService.php`.

## Request/response sketch (OpenAPI 3.0)

```yaml
openapi: 3.0.3
info:
  title: MenuAI Standard Pipeline
  version: "0.1.0-draft"
paths:
  /v1/standard/process:
    post:
      summary: Run one image through the Standard (no-generative-AI) pipeline
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [image]
              properties:
                image:
                  type: string
                  format: binary
                template_id:
                  type: string
                  description: Graphic template key (spec §9) - defaults server-side if omitted.
      responses:
        "200":
          description: >
            Processing ran to completion. Check `metadata.segmentation_status`
            and `metadata.validator_status` - 200 does NOT mean PASS; it means
            the request was handled (PASS, REVIEW, and REJECT are all valid
            outcomes carried inside the body, per §7).
          content:
            application/json:
              schema:
                type: object
                required: [metadata, image]
                properties:
                  metadata:
                    $ref: "./standard-metadata.schema.json"
                  image:
                    type: object
                    description: >
                      Present only when segmentation_status != REJECT (§7:
                      REJECT means "결과물 미사용" - a rejected request has
                      no usable output image, only metadata explaining why).
                    required: [content_type, encoding, data]
                    properties:
                      content_type:
                        type: string
                        example: image/jpeg
                      encoding:
                        type: string
                        enum: [base64]
                      data:
                        type: string
                        format: byte
        "422":
          description: Malformed request (missing/unreadable image, bad template_id).
```

See [`standard-metadata.schema.json`](./standard-metadata.schema.json) for
the full metadata shape (§8 fields, `is_infra_error`, `retry_attempts_used`,
and the `segmentation_status`/`segmentation_reasons` extension explained
there).

## Timeout: 60s starting point, not derived from an SLA

Laravel's HTTP client timeout calling this endpoint is set to **60
seconds** (`StandardAiService.php`). This is deliberately generous, not a
tight SLA figure - `standard/segmentation/README.md`'s measured numbers
(**~9.7s typical, one outlier at 34.75s**) are CPU-only, from a single
consumer desktop (AMD Ryzen 7 9800X3D) with no other load, covering only
the Segmentation Gate's BiRefNet call - not the full pipeline (Layout/
Shadow/Color/Validator/Retry add more, and Retry can mean multiple
re-renders). There is no GPU-environment measurement to base a tighter
number on. **Treat 60s as a starting point to be revisited once this runs
under real production load** (concurrent requests, real photo sizes, real
retry-rate) - both directions are possible: it could need to grow if
retry-heavy images are common, or shrink once there's data to justify a
tighter bound.

The segmentation backend's own infra-retry loop is deliberately capped at
`max_backend_retry_seconds` (default `20.0`, `standard/api/main.py`'s
`get_max_backend_retry_seconds`) rather than a fixed retry *count* -
counting attempts alone doesn't bound total time when a single attempt
can itself take ~35s, and two full-length retries could exhaust this
endpoint's entire 60s budget before Layout/Shadow/Color/Validator/Retry
even start. See `standard/segmentation/README.md`'s "Retrying is bounded
by elapsed time" section.

## Laravel-side storage: two sketches

Everything in `metadata` except `segmentation_reasons` and
`validator_reasons` is a single scalar value (`pipeline_version`,
`policy_hash`, `segmentation_status`, `validator_status`,
`merchant_review`, ...) - those are plain columns either way. The actual
decision is how to store the two **reason arrays**.

### (a) JSON column

Add `segmentation_reasons` and `validator_reasons` as JSON columns
directly on `images` (or wherever the scalar metadata columns live).

```php
// migration
$table->json('segmentation_reasons')->nullable();
$table->json('validator_reasons')->nullable();

// model
protected $casts = [
    'segmentation_reasons' => 'array',
    'validator_reasons' => 'array',
];
```

- **Pros**: one migration, one row write per image, zero schema churn
  when `standard/reason_codes.py` gains a new `ReasonCode` member, stores
  exactly what the API returned (good for audit/replay/debugging).
- **Cons**: "how many REVIEWs this month were `CONTENT_HF_ERROR`" needs
  MySQL JSON functions (`JSON_TABLE`/`JSON_CONTAINS`) instead of a plain
  `GROUP BY` - workable but slower and clunkier, especially once volume
  is high enough that this is the kind of dashboard query spec §6
  explicitly wants ("REVIEW의 62%가 segmentation 문제였다").

### (b) Normalized pivot table

```php
// migration
Schema::create('image_processing_reasons', function (Blueprint $table) {
    $table->id();
    $table->foreignId('image_id')->constrained()->cascadeOnDelete();
    $table->enum('stage', ['segmentation', 'validator']);
    $table->string('reason_code'); // plain string, not an FK-enforced enum -
                                    // see drift note below
    $table->timestamps();
});

// model
class Image extends Model {
    public function processingReasons(): HasMany {
        return $this->hasMany(ImageProcessingReason::class);
    }
}
```

- **Pros**: `GROUP BY reason_code` / `whereHas('processingReasons', ...)`
  is a normal indexed query - exactly what a Merchant Review UI filter
  ("show me all REVIEWs caused by `OCCLUSION_CHANGED`") or a bottleneck
  dashboard wants, with no JSON parsing.
- **Cons**: more upfront work (2 tables/models, a `HasMany` relation,
  multi-row inserts per image instead of one column update - needs a
  transaction), and the reason-code taxonomy now exists in two places
  (`standard/reason_codes.py` and whatever Laravel uses to validate
  `reason_code` values) with a real risk of drift if one side adds a code
  the other doesn't know about. Using a plain `string` column (as above)
  rather than a DB-level `ENUM` avoids a migration being *required* on
  drift, at the cost of losing DB-level integrity checking on the value.

### Decision: (b), normalized pivot table

Chosen for the reason-based filtering/stats the Merchant Review UI needs
(§6's own stated bottleneck-analysis use case). Implemented in the
`menu-ai` Laravel app:

- `database/migrations/2026_08_27_100000_add_standard_pipeline_metadata_to_images_table.php`
  adds the scalar columns (`pipeline_version`, `policy_version`,
  `policy_hash`, `template_version`, `mode`, `segmentation_status`,
  `validator_status`, `merchant_review`) directly to `images`.
- `database/migrations/2026_08_27_100100_create_image_processing_reasons_table.php`
  creates the pivot table (`image_id`, `stage`, `reason_code`, indexed on
  `(reason_code, stage)`), using a plain `string` column for `reason_code`
  rather than a DB-level `ENUM` - see the drift note above.
- `app/Models/Image.php` gained `processingReasons()` / `segmentationReasons()`
  / `validatorReasons()` relations and a `syncReasons(stage, codes)` helper.
- `app/Models/ImageProcessingReason.php` is the pivot model.

Both migrations have been applied to the local dev database and smoke-tested
(create -> syncReasons -> query -> cascade-delete, 0 orphaned rows).
