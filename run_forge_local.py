#!/usr/bin/env python3
"""Run the full AgentForge co-evolutionary loop — 100% local, no API keys."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from agentforge.core import AgentForge
from agentforge.environment import SimulationEnvironment
from agentforge.local_agent import LocalAgent

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="AgentForge Co-Evolutionary Loop (Local)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--config",
        default="configs/customer_support.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of co-evolutionary rounds",
    )
    parser.add_argument(
        "--output",
        default="generated_scenarios",
        help="Output directory for generated scenarios",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]AgentForge — Co-Evolutionary Loop (100% Local)[/bold green]\n"
        f"Model: {args.model}\n"
        f"Config: {args.config}\n"
        f"Rounds: {args.rounds}\n"
        f"Output: {args.output}\n\n"
        "No API keys needed — everything runs on your machine.",
        border_style="green",
    ))

    console.print("\n[bold]Initializing...[/bold]")
    agent = LocalAgent(model_name=args.model)
    env = SimulationEnvironment(config_path=args.config)

    console.print(f"  Scenarios: {len(env.scenarios)}")
    console.print(f"  Tools: {', '.join(env.tools.keys())}")
    console.print()

    forge = AgentForge(env=env, agent=agent, output_dir=args.output)
    forge.run(num_rounds=args.rounds)

    console.print("\n[bold green]Done![/bold green]")
    console.print(f"Generated scenarios saved to: {args.output}/")


if __name__ == "__main__":
    sys.exit(main())
