from pathlib import Path

from standard.reason_codes import ReasonCode
from standard.retry.config import load_retry_policy
from standard.retry.strategies import Strategy

CONFIG_PATH = Path(__file__).resolve().parents[2] / "standard" / "retry" / "retry_policy.yaml"


def test_production_retries_only_color_overshoot():
    """Narrower than spec §7 on purpose: only violations a computer can judge
    and fix by itself are retried; layout and HF-content changes go to a human."""
    config = load_retry_policy(CONFIG_PATH)

    assert config.max_attempts == 3
    assert config.final_action == "review"
    assert config.strategy_by_reason == {
        ReasonCode.FOOD_SATURATION_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
        ReasonCode.FOOD_LUMINANCE_EXCEEDED: [Strategy.CONSERVATIVE_COLOR, Strategy.NEUTRAL_COLOR],
    }


def test_reason_codes_with_no_strategy_are_absent():
    config = load_retry_policy(CONFIG_PATH)

    for reason in (
        ReasonCode.MASK_CHANGED,
        ReasonCode.SEGMENTATION_NO_FOOD_DETECTED,
        ReasonCode.CONTENT_HF_ERROR,
        ReasonCode.OCCLUSION_CHANGED,
        ReasonCode.OBJECT_SCALE_CHANGED,
        ReasonCode.RELATIVE_SCALE_CHANGED,
        ReasonCode.OBJECT_AREA_RANK_REVERSED,
    ):
        assert reason not in config.strategy_by_reason
