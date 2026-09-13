"""Loads standard/retry/retry_policy.yaml into typed structures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from standard.config.loader import load_yaml_section
from standard.reason_codes import ReasonCode
from standard.retry.strategies import Strategy


@dataclass(frozen=True)
class RetryPolicyConfig:
    max_attempts: int
    strategy_by_reason: dict[ReasonCode, list[Strategy]]
    final_action: str


def load_retry_policy(path: str | Path) -> RetryPolicyConfig:
    data = load_yaml_section(path, "retry")
    strategy_by_reason = {
        ReasonCode(reason): [Strategy(name) for name in strategies]
        for reason, strategies in data["strategy_by_reason"].items()
    }
    return RetryPolicyConfig(
        max_attempts=data["max_attempts"],
        strategy_by_reason=strategy_by_reason,
        final_action=data["final_action"],
    )
