"""Scenario generator using the local LLM — no external API keys needed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .local_agent import LocalAgent


@dataclass
class GeneratedScenario:
    id: str
    description: str
    user_message: str
    difficulty: str
    target_weakness: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)


class ScenarioGenerator:
    """Generates new test scenarios based on failure analyses, using the local LLM."""

    def __init__(self, agent: LocalAgent | None = None):
        self.agent = agent or LocalAgent()
        self._generated_count = 0

    def generate(
        self,
        failure_summary: dict[str, Any],
        existing_scenarios: list[dict[str, Any]],
        available_tools: list[str],
        num_scenarios: int = 3,
    ) -> list[GeneratedScenario]:
        scenarios = []
        top_weakness = failure_summary.get("top_weakness", "reasoning")

        for i in range(num_scenarios):
            scenario = self._generate_single(
                weakness=top_weakness,
                existing_scenarios=existing_scenarios,
                available_tools=available_tools,
                index=i,
            )
            scenarios.append(scenario)
        return scenarios

    def _generate_single(
        self,
        weakness: str,
        existing_scenarios: list[dict[str, Any]],
        available_tools: list[str],
        index: int,
    ) -> GeneratedScenario:
        self._generated_count += 1
        existing_desc = "\n".join(
            f"- {s.get('description', s.get('id', 'unknown'))}"
            for s in existing_scenarios[:5]
        )

        prompt = (
            "Generate a NEW test scenario for an AI agent. The scenario should target "
            f"this weakness: {weakness}\n\n"
            f"Available tools: {', '.join(available_tools)}\n\n"
            f"Existing scenarios (avoid duplicating these):\n{existing_desc}\n\n"
            "The new scenario should be HARDER than existing ones.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "description": "A brief description of the scenario",\n'
            '  "user_message": "What the user says to the agent",\n'
            '  "difficulty": "medium or hard",\n'
            '  "expected_tool_calls": ["tool1", "tool2"],\n'
            '  "success_criteria": ["criterion1", "criterion2"]\n'
            "}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.agent.generate(messages)

        try:
            # Try to find JSON in the response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = {
                "description": f"Generated scenario targeting {weakness}",
                "user_message": f"Please help me with a complex {weakness} task",
                "difficulty": "hard",
                "expected_tool_calls": available_tools[:2],
                "success_criteria": [f"Agent handles {weakness} correctly"],
            }

        return GeneratedScenario(
            id=f"gen_{self._generated_count}_{index}",
            description=parsed.get("description", f"Generated scenario {index}"),
            user_message=parsed.get("user_message", "Help me"),
            difficulty=parsed.get("difficulty", "hard"),
            target_weakness=weakness,
            initial_state=parsed.get("initial_state", {}),
            success_criteria=parsed.get("success_criteria", []),
            expected_tool_calls=parsed.get("expected_tool_calls", []),
        )
