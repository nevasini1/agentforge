#!/usr/bin/env python3
"""AgentForge Demo — Run the agent through customer support scenarios."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from agentforge.environment import SimulationEnvironment
from agentforge.local_agent import LocalAgent
from agentforge.rewards import compute_reward

console = Console()


def main():
    parser = argparse.ArgumentParser(description="AgentForge Demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--config",
        default="configs/customer_support.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Max turns per scenario",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]AgentForge Demo[/bold green]\n"
        f"Model: {args.model}\n"
        f"Config: {args.config}",
        border_style="green",
    ))

    # Load environment
    console.print("\n[bold]Loading environment...[/bold]")
    env = SimulationEnvironment(config_path=args.config)
    console.print(f"  Loaded {len(env.scenarios)} scenarios with {len(env.tools)} tools")

    # Initialize agent
    console.print(f"\n[bold]Loading model: {args.model}...[/bold]")
    agent = LocalAgent(model_name=args.model)

    tools = env.get_tools_for_agent()
    total_passed = 0

    for i, scenario in enumerate(env.scenarios, 1):
        console.rule(f"Scenario {i}/{len(env.scenarios)}: {scenario.id}")
        console.print(f"  Difficulty: [yellow]{scenario.difficulty}[/yellow]")
        console.print(f"  User: {scenario.user_message[:80]}...")
        console.print()

        sc_dict = {
            "id": scenario.id,
            "user_message": scenario.user_message,
            "description": scenario.description,
            "expected_tool_calls": scenario.expected_tool_calls,
        }

        trace = agent.run_scenario(sc_dict, tools, max_turns=args.max_turns)
        eval_result = env.evaluate_trace(scenario, trace)
        reward = compute_reward(trace, sc_dict)

        # Print results
        status = "[green]PASSED[/green]" if eval_result.passed else "[red]FAILED[/red]"
        console.print(f"  Status: {status}")
        console.print(f"  Score: {eval_result.score:.2f}")
        console.print(f"  Reward: {reward.explanation}")

        if trace.tool_calls:
            tools_called = ", ".join(tc.name for tc in trace.tool_calls)
            console.print(f"  Tools called: [cyan]{tools_called}[/cyan]")
        else:
            console.print("  Tools called: [dim]none[/dim]")

        if trace.final_response:
            response_preview = trace.final_response[:150].replace("\n", " ")
            console.print(f"  Response: {response_preview}...")

        if trace.error:
            console.print(f"  Error: [red]{trace.error}[/red]")

        if eval_result.passed:
            total_passed += 1
        console.print()

    # Final summary
    console.rule("[bold]Final Results")
    total = len(env.scenarios)
    console.print(f"  Passed: {total_passed}/{total}")
    console.print(f"  Pass rate: {total_passed / total * 100:.0f}%")

    if total_passed == total:
        console.print("\n  [bold green]All scenarios passed![/bold green]")
    elif total_passed > 0:
        console.print("\n  [bold yellow]Some scenarios failed — room for improvement.[/bold yellow]")
    else:
        console.print("\n  [bold red]All scenarios failed — the agent needs work.[/bold red]")

    return 0 if total_passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
