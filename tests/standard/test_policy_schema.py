import copy

import pytest
import yaml

from standard.policy.schema import PolicySchemaError, validate_schema

VALID = {
    "policy_version": "food_integrity_1.0.0",
    "mask": {"min_iou": 0.995, "max_area_delta": 0.01},
    "high_frequency": {"base_threshold": 0.04, "contrast_weight": 0.35, "sharpness_weight": 0.50},
    "food": {"max_saturation_gain": 0.05, "max_luminance_gain": 0.10},
    "layout": {
        "max_scale_delta_per_object": 0.03,
        "max_relative_scale_ratio_change": 0.02,
        "preserve_object_area_rank": True,
        "max_occlusion_change": 0.15,
        "preserve_front_back_order": True,
    },
    "severity": {
        "mask": {"min_iou": {"reject_multiplier": None}, "max_area_delta": {"reject_multiplier": None}},
        "high_frequency": {"base_threshold": {"reject_multiplier": None}},
        "food": {"max_saturation_gain": {"reject_multiplier": None}, "max_luminance_gain": {"reject_multiplier": None}},
        "layout": {
            "max_scale_delta_per_object": {"reject_multiplier": None},
            "max_relative_scale_ratio_change": {"reject_multiplier": None},
            "max_occlusion_change": {"reject_multiplier": None},
        },
    },
}


def test_valid_policy_passes():
    validate_schema(copy.deepcopy(VALID))  # must not raise


def test_missing_policy_version_rejected():
    data = copy.deepcopy(VALID)
    del data["policy_version"]
    with pytest.raises(PolicySchemaError, match="policy_version"):
        validate_schema(data)


def test_missing_section_rejected():
    data = copy.deepcopy(VALID)
    del data["mask"]
    with pytest.raises(PolicySchemaError, match="mask"):
        validate_schema(data)


def test_missing_key_within_section_rejected():
    data = copy.deepcopy(VALID)
    del data["layout"]["preserve_front_back_order"]
    with pytest.raises(PolicySchemaError, match="layout.preserve_front_back_order"):
        validate_schema(data)


def test_missing_severity_tag_rejected():
    data = copy.deepcopy(VALID)
    del data["severity"]["food"]["max_saturation_gain"]
    with pytest.raises(PolicySchemaError, match="severity.food.max_saturation_gain"):
        validate_schema(data)


def test_boolean_layout_checks_do_not_require_a_severity_tag():
    # preserve_object_area_rank / preserve_front_back_order are fixed to
    # REVIEW by spec §4 and intentionally excluded from SEVERITY_TAGGED.
    data = copy.deepcopy(VALID)
    assert "preserve_object_area_rank" not in data["severity"]["layout"]
    validate_schema(data)  # still valid


def test_real_policy_yaml_matches_schema():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "standard" / "policy" / "policy.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_schema(data)  # must not raise
