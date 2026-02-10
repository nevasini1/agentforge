"""CLI interface for AgentForge."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="agentforge", help="AgentForge — Co-evolutionary agent training")
console = Console()


@app.command()
def demo(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="HuggingFace model name"
    ),
    config: str = typer.Option(
        "configs/customer_support.yaml", "--config", "-c", help="Config file path"
    ),
):
    """Run the AgentForge demo."""
    from .core import AgentForge
    from .environment import SimulationEnvironment
    from .local_agent import LocalAgent

    console.print(f"[bold green]AgentForge Demo[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Config: {config}\n")

    agent = LocalAgent(model_name=model)
    env = SimulationEnvironment(config_path=config)
    forge = AgentForge(env=env, agent=agent)
    forge.run(num_rounds=1)


@app.command()
def forge(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="HuggingFace model name"
    ),
    config: str = typer.Option(
        "configs/customer_support.yaml", "--config", "-c", help="Config file path"
    ),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Number of co-evolutionary rounds"),
    output_dir: str = typer.Option(
        "generated_scenarios", "--output", "-o", help="Output directory for generated scenarios"
    ),
):
    """Run the full co-evolutionary loop."""
    from .core import AgentForge
    from .environment import SimulationEnvironment
    from .local_agent import LocalAgent

    console.print(f"[bold green]AgentForge Co-Evolutionary Loop[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Config: {config}")
    console.print(f"Rounds: {rounds}\n")

    agent = LocalAgent(model_name=model)
    env = SimulationEnvironment(config_path=config)
    forge = AgentForge(env=env, agent=agent, output_dir=output_dir)
    forge.run(num_rounds=rounds)


if __name__ == "__main__":
    app()
