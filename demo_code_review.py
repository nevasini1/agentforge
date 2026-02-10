#!/usr/bin/env python3
"""AgentForge Code Review Demo — Run agent through code review scenarios."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from agentforge.environment import SimulationEnvironment
from agentforge.local_agent import LocalAgent
from agentforge.rewards import compute_reward

console = Console()


def main():
    parser = argparse.ArgumentParser(description="AgentForge Code Review Demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--config",
        default="configs/code_review.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]AgentForge — Code Review Agent Demo[/bold blue]\n"
        f"Model: {args.model}\n"
        f"Config: {args.config}",
        border_style="blue",
    ))

    env = SimulationEnvironment(config_path=args.config)
    console.print(f"\n  Loaded {len(env.scenarios)} code review scenarios with {len(env.tools)} tools\n")

    agent = LocalAgent(model_name=args.model)
    tools = env.get_tools_for_agent()
    total_passed = 0

    for i, scenario in enumerate(env.scenarios, 1):
        console.rule(f"Scenario {i}/{len(env.scenarios)}: {scenario.id}")
        console.print(f"  Difficulty: [yellow]{scenario.difficulty}[/yellow]")
        console.print(f"  Description: {scenario.description}")
        console.print(f"  User says: \"{scenario.user_message}\"")
        console.print()

        sc_dict = {
            "id": scenario.id,
            "user_message": scenario.user_message,
            "description": scenario.description,
            "expected_tool_calls": scenario.expected_tool_calls,
        }

        trace = agent.run_scenario(sc_dict, tools)
        eval_result = env.evaluate_trace(scenario, trace)
        reward = compute_reward(trace, sc_dict)

        status = "[green]PASSED[/green]" if eval_result.passed else "[red]FAILED[/red]"
        console.print(f"  Result: {status} (score: {eval_result.score:.2f})")
        console.print(f"  Reward: {reward.explanation}")

        if trace.tool_calls:
            for tc in trace.tool_calls:
                console.print(f"    → [cyan]{tc.name}[/cyan]({tc.arguments})")

        if trace.final_response:
            preview = trace.final_response[:200].replace("\n", " ")
            console.print(f"  Agent response: {preview}")

        if eval_result.passed:
            total_passed += 1
        console.print()

    console.rule("[bold]Code Review Results")
    total = len(env.scenarios)
    console.print(f"  Passed: {total_passed}/{total} ({total_passed / total * 100:.0f}%)")

    by_diff = {"easy": [], "medium": [], "hard": []}
    for sc, i in zip(env.scenarios, range(len(env.scenarios))):
        by_diff.setdefault(sc.difficulty, [])

    console.print("\n  This small model (0.5B) will struggle with hard scenarios.")
    console.print("  Try a larger model for better code review performance.\n")


if __name__ == "__main__":
    sys.exit(main())
