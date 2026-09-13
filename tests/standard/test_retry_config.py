from pathlib import Path

from standard.reason_codes import ReasonCode
from standard.retry.config import load_retry_policy
from standard.retry.strategies import Strategy

CONFIG_PATH = Path(__file__).resolve().parents[2] / "standard" / "retry" / "retry_policy.yaml"


def test_retry_policy_matches_spec_exactly():
    config = load_retry_policy(CONFIG_PATH)

    assert config.max_attempts == 3
    assert config.final_action == "review"
    assert config.strategy_by_reason == {
        ReasonCode.CONTENT_HF_ERROR: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        ReasonCode.FOOD_LUMINANCE_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        ReasonCode.OCCLUSION_CHANGED: [Strategy.RECOMPUTE_TRANSLATION],
        ReasonCode.OBJECT_SCALE_CHANGED: [Strategy.FORCE_SCALE_1_0],
        ReasonCode.RELATIVE_SCALE_CHANGED: [Strategy.FORCE_SCALE_1_0],
        ReasonCode.OBJECT_AREA_RANK_REVERSED: [Strategy.RECOMPUTE_TRANSLATION],
    }


def test_reason_codes_with_no_strategy_are_absent():
    config = load_retry_policy(CONFIG_PATH)

    assert ReasonCode.MASK_CHANGED not in config.strategy_by_reason
    assert ReasonCode.SEGMENTATION_NO_FOOD_DETECTED not in config.strategy_by_reason
