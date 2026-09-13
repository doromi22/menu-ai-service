"""
Minimal YAML config loading for modules that just need a plain dict (e.g.
the Segmentation Gate's thresholds, §3).

This is deliberately not the Policy loader §6/§11 step 3 describes for the
Integrity Validator (immutable artifact, sha256-pinned, git tag/digest
locked) - that is a separate, later build step with much stricter
guarantees and should not be conflated with this one.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_section(path: str | Path, section: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get(section, {})
