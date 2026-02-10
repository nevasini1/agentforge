"""Failure analysis using the local LLM — no external API keys needed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .local_agent import LocalAgent


@dataclass
class FailureAnalysis:
    scenario_id: str
    failure_type: str
    root_cause: str
    weakness_category: str
    suggested_difficulty_increase: str
    raw_analysis: str = ""


class FailureAnalyzer:
    """Analyzes agent failure traces using the local LLM."""

    def __init__(self, agent: LocalAgent | None = None):
        self.agent = agent or LocalAgent()

    def analyze(self, traces: list[Any]) -> list[FailureAnalysis]:
        failed_traces = [t for t in traces if not t.success and not t.error]
        if not failed_traces:
            failed_traces = [t for t in traces if t.error]
        if not failed_traces:
            return []

        analyses = []
        for trace in failed_traces:
            analysis = self._analyze_single(trace)
            analyses.append(analysis)
        return analyses

    def _analyze_single(self, trace: Any) -> FailureAnalysis:
        messages_str = "\n".join(
            f"[{m['role']}]: {m['content'][:200]}" for m in trace.messages
        )
        tool_calls_str = ", ".join(tc.name for tc in trace.tool_calls) or "none"

        prompt = (
            "Analyze this agent interaction trace and identify why it failed.\n\n"
            f"Scenario: {trace.scenario_id}\n"
            f"Tool calls made: {tool_calls_str}\n"
            f"Final response: {trace.final_response[:200] if trace.final_response else 'none'}\n"
            f"Error: {trace.error or 'none'}\n\n"
            f"Messages:\n{messages_str}\n\n"
            "Respond with JSON:\n"
            '{"failure_type": "...", "root_cause": "...", '
            '"weakness_category": "one of: tool_selection, argument_formatting, '
            'reasoning, instruction_following, error_recovery", '
            '"suggested_difficulty_increase": "..."}'
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.agent.generate(messages)

        # Parse the response
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = {
                "failure_type": "unknown",
                "root_cause": response[:200],
                "weakness_category": "reasoning",
                "suggested_difficulty_increase": "add more ambiguity",
            }

        return FailureAnalysis(
            scenario_id=trace.scenario_id,
            failure_type=parsed.get("failure_type", "unknown"),
            root_cause=parsed.get("root_cause", "unknown"),
            weakness_category=parsed.get("weakness_category", "reasoning"),
            suggested_difficulty_increase=parsed.get(
                "suggested_difficulty_increase", "increase complexity"
            ),
            raw_analysis=response,
        )

    def summarize(self, analyses: list[FailureAnalysis]) -> dict[str, Any]:
        if not analyses:
            return {"total_failures": 0, "categories": {}}

        categories: dict[str, int] = {}
        for a in analyses:
            categories[a.weakness_category] = categories.get(a.weakness_category, 0) + 1

        return {
            "total_failures": len(analyses),
            "categories": categories,
            "top_weakness": max(categories, key=categories.get) if categories else "none",
            "analyses": [
                {"scenario": a.scenario_id, "type": a.failure_type, "cause": a.root_cause}
                for a in analyses
            ],
        }
