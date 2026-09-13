"""
360-image MVP test harness (spec §10, §11 step 7) - scaffolding only.

Lives outside `standard/` deliberately: everything under `standard/` is
pipeline *library* code (importable, no I/O side effects at import time);
this package is an offline batch-runner tool that drives that library
against a folder of photos and writes files. It has no ML dependency of
its own - segmentation is injected (see `runner.run_one`'s
`segmentation_backend` parameter), same pattern as `SegmentationGate`
itself, so this package imports cleanly even before rembg/onnxruntime are
installed in this venv.

Real 360-image dataset collection and any judgment about which templates
"win" are explicitly out of scope for this step - see ../standard/README.md's
harness section for what's proven so far (plumbing only, via synthetic
images) versus what's still pending (a real photo run).
"""
