"""Simulation environment for agent evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    function: Callable[..., Any]


@dataclass
class Scenario:
    id: str
    description: str
    user_message: str
    difficulty: str  # easy, medium, hard
    initial_state: dict[str, Any] = field(default_factory=dict)
    user_persona: str = ""
    success_criteria: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class EvalResult:
    scenario_id: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class SimulationEnvironment:
    """Configurable simulation environment loaded from YAML configs."""

    def __init__(self, config_path: str | None = None):
        self.tools: dict[str, Tool] = {}
        self.scenarios: list[Scenario] = []
        self._state: dict[str, Any] = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for tool_cfg in config.get("tools", []):
            name = tool_cfg["name"]
            self.tools[name] = Tool(
                name=name,
                description=tool_cfg.get("description", ""),
                parameters=tool_cfg.get("parameters", {}),
                function=self._make_mock_function(name, tool_cfg.get("mock_responses", {})),
            )

        for sc in config.get("scenarios", []):
            self.scenarios.append(
                Scenario(
                    id=sc["id"],
                    description=sc.get("description", ""),
                    user_message=sc.get("user_message", ""),
                    difficulty=sc.get("difficulty", "medium"),
                    initial_state=sc.get("initial_state", {}),
                    user_persona=sc.get("user_persona", ""),
                    success_criteria=sc.get("success_criteria", []),
                    expected_tool_calls=sc.get("expected_tool_calls", []),
                    expected_outcome=sc.get("expected_outcome", ""),
                )
            )

    def _make_mock_function(
        self, tool_name: str, mock_responses: dict[str, Any]
    ) -> Callable[..., Any]:
        def mock_fn(**kwargs) -> Any:
            key = json.dumps(kwargs, sort_keys=True)
            if key in mock_responses:
                return mock_responses[key]
            if "default" in mock_responses:
                return mock_responses["default"]
            return {"status": "ok", "tool": tool_name, "input": kwargs}
        return mock_fn

    def get_tools_for_agent(self) -> dict[str, Any]:
        result = {}
        for name, tool in self.tools.items():
            result[name] = {
                "description": tool.description,
                "parameters": tool.parameters,
                "function": tool.function,
            }
        return result

    def evaluate_trace(self, scenario: Scenario, trace: Any) -> EvalResult:
        score = 0.0
        details: dict[str, Any] = {}

        # Check tool calls
        if scenario.expected_tool_calls:
            called_tools = [tc.name for tc in trace.tool_calls]
            matched = sum(1 for t in scenario.expected_tool_calls if t in called_tools)
            tool_score = matched / len(scenario.expected_tool_calls)
            score += tool_score * 0.5
            details["tool_call_score"] = tool_score
            details["expected_tools"] = scenario.expected_tool_calls
            details["called_tools"] = called_tools

        # Check for final response
        if trace.final_response:
            score += 0.3
            details["has_response"] = True
        else:
            details["has_response"] = False

        # Check no errors
        if not trace.error:
            score += 0.2
            details["no_errors"] = True
        else:
            details["no_errors"] = False
            details["error"] = trace.error

        passed = score >= 0.5
        details["total_score"] = score

        return EvalResult(
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            details=details,
        )
