# Standard pipeline API (§11 step 4)

Implements the `POST /v1/standard/process` contract documented in
`../../docs/standard-api-contract.md` (see that file for the full
request/response schema and why this is a **separate FastAPI app** from
`../../main.py` / `../../modal_app.py`, Premium's engine).

## Running it

```bash
cd ai-service
./venv/Scripts/python.exe -m uvicorn standard.api.main:app --port 8002
```

`GET /health` returns `{"status": "ok"}` once the app (including the real
BiRefNet backend, Policy, and Retry config, all loaded once at import
time) is ready.

## Testing

- `tests/api/test_process_endpoint.py` - in-process wiring tests using
  FastAPI's `app.dependency_overrides` to swap in a fake segmentation
  backend, a synthetic `Policy`, and zero backend-retry backoff. No real
  network process involved.
- `tests/integration/test_birefnet_real_inference.py` - real BiRefNet,
  separately (see `standard/segmentation/README.md`).
- **Cross-language end-to-end**: `menu-ai`'s
  `tests/Feature/StandardPipelineEndToEndTest.php` starts a REAL
  `uvicorn` process (over real HTTP, not `TestClient`) with the
  `STANDARD_API_FAKE_SEGMENTATION=1` environment variable set, so a
  from-scratch server process can be driven by Laravel's PHPUnit suite
  without waiting on real model inference. **This env var is a test-only
  escape hatch - never set it outside a test process.** It swaps in
  `main.py`'s `_FakeThresholdBackend` (a simple background-color
  threshold, same trick `tests/mvp_harness`'s fixtures use) in place of
  `BiRefNetBackend` at import time.
