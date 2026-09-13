"""
Policy schema for the Integrity Validator (spec §6): required structure of
the policy YAML, plus which numeric thresholds must carry a severity tag
(standard/severity.py) - consistent with how
standard/config/segmentation.yaml tags its own thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

REQUIRED_STRUCTURE: dict[str, tuple[str, ...]] = {
    "mask": ("min_iou", "max_area_delta"),
    "high_frequency": ("base_threshold", "contrast_weight", "sharpness_weight"),
    "food": ("max_saturation_gain", "max_luminance_gain"),
    "layout": (
        "max_scale_delta_per_object",
        "max_relative_scale_ratio_change",
        "preserve_object_area_rank",
        "max_occlusion_change",
        "preserve_front_back_order",
    ),
}

# Thresholds that must carry a severity.<section>.<key>.reject_multiplier tag.
# The two boolean layout checks (preserve_object_area_rank,
# preserve_front_back_order) are deliberately excluded: §4 already fixes
# their outcome to REVIEW ("순위 역전 시 REVIEW"; "앞/뒤 순서가 바뀌면 REVIEW"),
# so there's no "how far past the line" for a multiplier to act on.
SEVERITY_TAGGED: dict[str, tuple[str, ...]] = {
    "mask": ("min_iou", "max_area_delta"),
    "high_frequency": ("base_threshold",),
    "food": ("max_saturation_gain", "max_luminance_gain"),
    "layout": ("max_scale_delta_per_object", "max_relative_scale_ratio_change", "max_occlusion_change"),
}


class PolicySchemaError(Exception):
    pass


def validate_schema(data: dict) -> None:
    if not isinstance(data.get("policy_version"), str) or not data.get("policy_version"):
        raise PolicySchemaError("policy_version must be a non-empty string")

    for section, keys in REQUIRED_STRUCTURE.items():
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            raise PolicySchemaError(f"missing or invalid section: {section!r}")
        for key in keys:
            if key not in section_data:
                raise PolicySchemaError(f"missing key: {section}.{key}")

    severity = data.get("severity", {})
    for section, keys in SEVERITY_TAGGED.items():
        section_severity = severity.get(section, {})
        for key in keys:
            if "reject_multiplier" not in section_severity.get(key, {}):
                raise PolicySchemaError(f"missing severity tag: severity.{section}.{key}.reject_multiplier")


@dataclass(frozen=True)
class Policy:
    version: str
    hash: str  # "sha256:...", computed by PolicyLoader over the raw file bytes
    raw: dict

    def threshold(self, section: str, key: str):
        return self.raw[section][key]

    def reject_multiplier(self, section: str, key: str) -> float | None:
        return self.raw["severity"][section][key]["reject_multiplier"]
