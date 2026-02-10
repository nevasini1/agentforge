"""Tests for LocalAgent — mocked so they don't need a real model."""

from unittest.mock import MagicMock, patch

import pytest

from agentforge.local_agent import AgentTrace, LocalAgent, ToolCall


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(name="lookup_order", arguments={"order_id": "123"})
        assert tc.name == "lookup_order"
        assert tc.arguments == {"order_id": "123"}


class TestAgentTrace:
    def test_defaults(self):
        trace = AgentTrace(scenario_id="test")
        assert trace.scenario_id == "test"
        assert trace.messages == []
        assert trace.tool_calls == []
        assert trace.final_response == ""
        assert trace.success is False
        assert trace.error is None


class TestLocalAgent:
    def test_init_defaults(self):
        agent = LocalAgent()
        assert agent.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert agent.max_new_tokens == 512

    def test_init_custom(self):
        agent = LocalAgent(model_name="custom/model", max_new_tokens=256)
        assert agent.model_name == "custom/model"
        assert agent.max_new_tokens == 256

    def test_parse_response_json(self):
        agent = LocalAgent()
        result = agent._parse_response('{"tool": "lookup_order", "args": {"order_id": "123"}}')
        assert result["tool"] == "lookup_order"

    def test_parse_response_final_answer(self):
        agent = LocalAgent()
        result = agent._parse_response('{"final_answer": "Order is shipped"}')
        assert result["final_answer"] == "Order is shipped"

    def test_parse_response_plain_text(self):
        agent = LocalAgent()
        result = agent._parse_response("Just a plain text response")
        assert "final_answer" in result

    def test_parse_response_code_block(self):
        agent = LocalAgent()
        result = agent._parse_response('```json\n{"tool": "check_inventory", "args": {}}\n```')
        assert result["tool"] == "check_inventory"

    def test_format_tools(self):
        agent = LocalAgent()
        tools = {
            "lookup_order": {
                "description": "Look up an order",
                "parameters": {"order_id": "string"},
            }
        }
        formatted = agent._format_tools(tools)
        assert "lookup_order" in formatted
        assert "Look up an order" in formatted

    @patch.object(LocalAgent, "generate")
    def test_run_scenario_with_final_answer(self, mock_generate):
        mock_generate.return_value = '{"final_answer": "Your order is shipped."}'

        agent = LocalAgent()
        scenario = {
            "id": "test_1",
            "user_message": "Check my order",
            "description": "Test scenario",
        }
        tools = {
            "lookup_order": {
                "description": "Look up an order",
                "parameters": {"order_id": "string"},
                "function": lambda **kwargs: {"status": "shipped"},
            }
        }

        trace = agent.run_scenario(scenario, tools, max_turns=3)
        assert trace.scenario_id == "test_1"
        assert trace.final_response == "Your order is shipped."
        assert trace.error is None

    @patch.object(LocalAgent, "generate")
    def test_run_scenario_with_tool_call(self, mock_generate):
        mock_generate.side_effect = [
            '{"tool": "lookup_order", "args": {"order_id": "123"}}',
            '{"final_answer": "Order 123 is shipped."}',
        ]

        agent = LocalAgent()
        scenario = {
            "id": "test_2",
            "user_message": "Check order 123",
            "description": "Test tool call",
        }
        tools = {
            "lookup_order": {
                "description": "Look up an order",
                "parameters": {"order_id": "string"},
                "function": lambda **kwargs: {"status": "shipped", "order_id": kwargs.get("order_id")},
            }
        }

        trace = agent.run_scenario(scenario, tools, max_turns=5)
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0].name == "lookup_order"
        assert trace.final_response == "Order 123 is shipped."

    @patch.object(LocalAgent, "generate")
    def test_run_scenario_unknown_tool(self, mock_generate):
        mock_generate.side_effect = [
            '{"tool": "nonexistent_tool", "args": {}}',
            '{"final_answer": "Sorry, I could not help."}',
        ]

        agent = LocalAgent()
        scenario = {"id": "test_3", "user_message": "Help me", "description": "Unknown tool"}
        tools = {}

        trace = agent.run_scenario(scenario, tools, max_turns=5)
        assert trace.tool_calls[0].name == "nonexistent_tool"


class TestEnvironment:
    def test_load_config(self):
        from agentforge.environment import SimulationEnvironment

        env = SimulationEnvironment(config_path="configs/customer_support.yaml")
        assert len(env.tools) == 3
        assert len(env.scenarios) == 3
        assert "lookup_order" in env.tools
        assert "issue_refund" in env.tools
        assert "check_inventory" in env.tools

    def test_get_tools_for_agent(self):
        from agentforge.environment import SimulationEnvironment

        env = SimulationEnvironment(config_path="configs/customer_support.yaml")
        tools = env.get_tools_for_agent()
        assert "lookup_order" in tools
        assert "function" in tools["lookup_order"]
        assert callable(tools["lookup_order"]["function"])


class TestRewards:
    def test_tool_accuracy_reward(self):
        from agentforge.rewards import tool_accuracy_reward

        assert tool_accuracy_reward(["a", "b"], ["a", "b"]) == 1.0
        assert tool_accuracy_reward(["a", "b"], ["a"]) == 0.5
        assert tool_accuracy_reward(["a", "b"], []) == 0.0
        assert tool_accuracy_reward([], []) == 0.0

    def test_response_quality_reward(self):
        from agentforge.rewards import response_quality_reward

        assert response_quality_reward("") == -0.5
        assert response_quality_reward("some response") > 0

    def test_compute_reward(self):
        from agentforge.rewards import compute_reward

        trace = AgentTrace(scenario_id="test")
        trace.final_response = "Here is your answer"
        scenario = {"expected_tool_calls": [], "expected_keywords": []}
        reward = compute_reward(trace, scenario)
        assert reward.value > -1.0


class TestCurriculum:
    def test_build_and_advance(self):
        from agentforge.curriculum import Curriculum

        c = Curriculum()
        scenarios = [
            {"id": "s1", "difficulty": "easy"},
            {"id": "s2", "difficulty": "medium"},
            {"id": "s3", "difficulty": "hard"},
        ]
        c.build_from_scenarios(scenarios)
        assert len(c.stages) == 3
        assert not c.is_complete()

        assert c.advance(0.8)  # pass easy
        assert c.advance(0.6)  # pass medium
        assert c.advance(0.9)  # pass hard
        assert c.is_complete()

    def test_fail_to_advance(self):
        from agentforge.curriculum import Curriculum

        c = Curriculum()
        c.build_from_scenarios([{"id": "s1", "difficulty": "easy"}])
        assert not c.advance(0.2)  # fail
        assert not c.is_complete()
