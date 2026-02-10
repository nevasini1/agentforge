"""Local LLM agent powered by Hugging Face Transformers (Qwen2.5-0.5B-Instruct)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentTrace:
    scenario_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str = ""
    success: bool = False
    error: str | None = None


class LocalAgent:
    """Agent that uses a local HuggingFace model for inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._load_model()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run_scenario(
        self,
        scenario: dict[str, Any],
        tools: dict[str, Any],
        max_turns: int = 5,
    ) -> AgentTrace:
        trace = AgentTrace(scenario_id=scenario.get("id", "unknown"))

        tools_description = self._format_tools(tools)
        system_msg = (
            f"You are a helpful agent. You have these tools:\n\n{tools_description}\n\n"
            "To call a tool, respond with JSON: {\"tool\": \"tool_name\", \"args\": {\"key\": \"value\"}}\n"
            "When you have a final answer, respond with: {\"final_answer\": \"your answer\"}\n"
            "Always respond with valid JSON only."
        )

        messages = [{"role": "system", "content": system_msg}]
        user_msg = scenario.get("user_message", scenario.get("description", ""))
        messages.append({"role": "user", "content": user_msg})
        trace.messages.append({"role": "user", "content": user_msg})

        for turn in range(max_turns):
            try:
                response = self.generate(messages)
                trace.messages.append({"role": "assistant", "content": response})

                parsed = self._parse_response(response)

                if parsed.get("final_answer"):
                    trace.final_response = parsed["final_answer"]
                    break

                if parsed.get("tool"):
                    tool_name = parsed["tool"]
                    tool_args = parsed.get("args", {})
                    trace.tool_calls.append(ToolCall(name=tool_name, arguments=tool_args))

                    if tool_name in tools:
                        tool_fn = tools[tool_name]["function"]
                        result = tool_fn(**tool_args)
                        tool_result = json.dumps(result) if not isinstance(result, str) else result
                    else:
                        tool_result = f"Error: Unknown tool '{tool_name}'"

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
                    trace.messages.append({"role": "tool", "content": tool_result})
                else:
                    trace.final_response = response
                    break

            except Exception as e:
                trace.error = str(e)
                break

        return trace

    def _format_tools(self, tools: dict[str, Any]) -> str:
        lines = []
        for name, spec in tools.items():
            desc = spec.get("description", "No description")
            params = spec.get("parameters", {})
            param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
            lines.append(f"- {name}({param_str}): {desc}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> dict[str, Any]:
        response = response.strip()
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find any JSON object
        match = re.search(r"\{[^{}]*\}", response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"final_answer": response}
