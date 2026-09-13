"""
Standard MVP pipeline (MENUAI_V1_SPEC.md §1 Phase 1) - no generative-AI
dependency. Build order follows §11:

    1. repo structure (this package)                      <- done
    2. FoodObject + Segmentation Gate (§2, §3)             <- done
    2. Layout / Shadow / Color / Integrity Validator (§4-6) <- done
    3. Policy YAML schema + immutable/hash loader (§6)     <- done
    4. reason-code enum + metadata schema (§8)             <- done: enum in reason_codes.py, schema in
                                                                ../docs/, now served for real by api/main.py
    5. Retry policy state machine (§7)                     <- done (decision logic; its evaluate()
                                                                callback is now actually exercised by
                                                                ../mvp_harness/runner.py, see step 7)
    6. Graphic template renderer (§9)                      <- done (only T01_warm_ivory's numbers are
                                                                from the spec; T02-T06 are placeholders)
    7. 360-image MVP test harness (§10)                    <- scaffolding done (../mvp_harness/),
                                                                verified with synthetic images only -
                                                                no real photo run yet
    8. Premium (SDXL/IC-Light) branch                      <- explicitly out of scope here

This package must never import torch/diffusers/rembg's GPU providers at
module scope - anything Standard does has to run without a GPU.
"""
