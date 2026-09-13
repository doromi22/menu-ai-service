from pathlib import Path

from standard.config.loader import load_yaml_section

CONFIG_PATH = Path(__file__).resolve().parents[2] / "standard" / "config" / "segmentation.yaml"


def test_segmentation_yaml_matches_spec_section_3():
    cfg = load_yaml_section(CONFIG_PATH, "segmentation")
    assert cfg == {
        "min_food_area_ratio": 0.03,
        "max_food_area_ratio": 0.95,
        "max_boundary_touch_ratio": 0.20,
        "min_component_area_ratio": 0.002,
        "max_hole_ratio": 0.05,
        "multi_object_detection": True,
        "severity": {
            "min_food_area_ratio": {"reject_multiplier": 0.5},
            "max_food_area_ratio": {"reject_multiplier": 1.0},
            "max_boundary_touch_ratio": {"reject_multiplier": None},
            "max_hole_ratio": {"reject_multiplier": None},
        },
    }
