"""Verifiable reward functions for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardSignal:
    value: float  # -1.0 to 1.0
    components: dict[str, float]
    explanation: str


def tool_accuracy_reward(
    expected_tools: list[str],
    called_tools: list[str],
) -> float:
    if not expected_tools:
        return 0.0
    matched = sum(1 for t in expected_tools if t in called_tools)
    return matched / len(expected_tools)


def response_quality_reward(
    response: str,
    expected_keywords: list[str] | None = None,
) -> float:
    if not response:
        return -0.5
    score = 0.3  # Base score for having a response
    if expected_keywords:
        response_lower = response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        score += 0.7 * (matched / len(expected_keywords))
    else:
        score += 0.2  # Partial credit for any response
    return min(score, 1.0)


def compute_reward(
    trace: Any,
    scenario: dict[str, Any],
) -> RewardSignal:
    components: dict[str, float] = {}

    # Tool accuracy
    expected_tools = scenario.get("expected_tool_calls", [])
    called_tools = [tc.name for tc in trace.tool_calls]
    components["tool_accuracy"] = tool_accuracy_reward(expected_tools, called_tools)

    # Response quality
    keywords = scenario.get("expected_keywords", [])
    components["response_quality"] = response_quality_reward(
        trace.final_response, keywords
    )

    # Penalize errors
    components["no_errors"] = 1.0 if not trace.error else -0.5

    # Weighted sum
    weights = {"tool_accuracy": 0.4, "response_quality": 0.4, "no_errors": 0.2}
    total = sum(components[k] * weights[k] for k in weights)

    explanation_parts = [f"{k}={v:.2f}" for k, v in components.items()]
    explanation = f"Reward={total:.2f} ({', '.join(explanation_parts)})"

    return RewardSignal(value=total, components=components, explanation=explanation)
