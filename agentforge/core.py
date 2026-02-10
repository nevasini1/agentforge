"""Main co-evolutionary loop — runs 100% locally with no API keys."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from .analyzer import FailureAnalyzer
from .curriculum import Curriculum
from .environment import SimulationEnvironment
from .generator import ScenarioGenerator
from .local_agent import LocalAgent
from .rewards import compute_reward

console = Console()


@dataclass
class ForgeRound:
    round_num: int
    base_results: list[dict[str, Any]] = field(default_factory=list)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    generated_scenarios: list[dict[str, Any]] = field(default_factory=list)
    generated_results: list[dict[str, Any]] = field(default_factory=list)


class AgentForge:
    """Co-evolutionary loop: evaluate → analyze failures → generate harder scenarios → repeat."""

    def __init__(
        self,
        env: SimulationEnvironment,
        agent: LocalAgent | None = None,
        output_dir: str = "generated_scenarios",
    ):
        self.env = env
        self.agent = agent or LocalAgent()
        self.analyzer = FailureAnalyzer(agent=self.agent)
        self.generator = ScenarioGenerator(agent=self.agent)
        self.output_dir = output_dir
        self.rounds: list[ForgeRound] = []
        os.makedirs(output_dir, exist_ok=True)

    def run_evaluation(
        self, scenarios: list[Any], label: str = "base"
    ) -> list[dict[str, Any]]:
        results = []
        tools = self.env.get_tools_for_agent()

        for sc in scenarios:
            sc_dict = {
                "id": sc.id if hasattr(sc, "id") else sc.get("id", "unknown"),
                "user_message": sc.user_message if hasattr(sc, "user_message") else sc.get("user_message", ""),
                "description": sc.description if hasattr(sc, "description") else sc.get("description", ""),
                "expected_tool_calls": (
                    sc.expected_tool_calls if hasattr(sc, "expected_tool_calls")
                    else sc.get("expected_tool_calls", [])
                ),
            }
            sc_id = sc_dict["id"]
            console.print(f"  [{label}] Running scenario: {sc_id}...", style="dim")

            trace = self.agent.run_scenario(sc_dict, tools)
            eval_result = self.env.evaluate_trace(sc, trace)
            reward = compute_reward(trace, sc_dict)
            trace.success = eval_result.passed

            results.append({
                "scenario_id": sc_id,
                "passed": eval_result.passed,
                "score": eval_result.score,
                "reward": reward.value,
                "reward_explanation": reward.explanation,
                "trace": trace,
            })

        return results

    def run_round(self, round_num: int) -> ForgeRound:
        forge_round = ForgeRound(round_num=round_num)
        console.rule(f"[bold blue]Round {round_num}")

        # 1. Evaluate on base scenarios
        console.print("\n[bold]Phase 1: Evaluating agent on base scenarios...[/bold]")
        forge_round.base_results = self.run_evaluation(self.env.scenarios, label="base")
        self._print_results(forge_round.base_results, "Base Scenario Results")

        # 2. Analyze failures
        console.print("\n[bold]Phase 2: Analyzing failures...[/bold]")
        traces = [r["trace"] for r in forge_round.base_results]
        analyses = self.analyzer.analyze(traces)
        forge_round.failure_summary = self.analyzer.summarize(analyses)
        if forge_round.failure_summary.get("top_weakness"):
            console.print(
                f"  Top weakness: [red]{forge_round.failure_summary['top_weakness']}[/red]"
            )

        # 3. Generate harder scenarios
        console.print("\n[bold]Phase 3: Generating harder scenarios...[/bold]")
        existing = [
            {"id": sc.id, "description": sc.description} for sc in self.env.scenarios
        ]
        tool_names = list(self.env.tools.keys())
        generated = self.generator.generate(
            failure_summary=forge_round.failure_summary,
            existing_scenarios=existing,
            available_tools=tool_names,
            num_scenarios=3,
        )
        forge_round.generated_scenarios = [
            {"id": g.id, "description": g.description, "difficulty": g.difficulty}
            for g in generated
        ]

        # Save generated scenarios
        save_path = os.path.join(self.output_dir, f"round_{round_num}.json")
        with open(save_path, "w") as f:
            json.dump(forge_round.generated_scenarios, f, indent=2)
        console.print(f"  Saved generated scenarios to {save_path}")

        # 4. Evaluate on generated scenarios
        console.print("\n[bold]Phase 4: Evaluating agent on generated scenarios...[/bold]")
        forge_round.generated_results = self.run_evaluation(generated, label="gen")
        self._print_results(forge_round.generated_results, "Generated Scenario Results")

        # 5. Comparison
        self._print_comparison(forge_round)

        self.rounds.append(forge_round)
        return forge_round

    def run(self, num_rounds: int = 3):
        console.print("[bold green]AgentForge Co-Evolutionary Loop[/bold green]")
        console.print(f"Running {num_rounds} rounds of diagnose → generate\n")

        for i in range(1, num_rounds + 1):
            self.run_round(i)
            console.print()

        self._print_final_summary()

    def _print_results(self, results: list[dict[str, Any]], title: str):
        table = Table(title=title)
        table.add_column("Scenario", style="cyan")
        table.add_column("Passed", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Reward", style="magenta")

        for r in results:
            passed = "[green]YES[/green]" if r["passed"] else "[red]NO[/red]"
            table.add_row(
                r["scenario_id"],
                passed,
                f"{r['score']:.2f}",
                f"{r['reward']:.2f}",
            )
        console.print(table)

    def _print_comparison(self, forge_round: ForgeRound):
        base_avg = (
            sum(r["score"] for r in forge_round.base_results) / len(forge_round.base_results)
            if forge_round.base_results else 0
        )
        gen_avg = (
            sum(r["score"] for r in forge_round.generated_results) / len(forge_round.generated_results)
            if forge_round.generated_results else 0
        )

        console.print(f"\n  Base avg score:      [green]{base_avg:.2f}[/green]")
        console.print(f"  Generated avg score: [red]{gen_avg:.2f}[/red]")
        if gen_avg < base_avg:
            console.print(
                "  [bold yellow]→ Generated scenarios are harder (expected!)[/bold yellow]"
            )
        else:
            console.print("  → Agent performed equally or better on generated scenarios")

    def _print_final_summary(self):
        console.rule("[bold green]Final Summary")
        table = Table(title="Performance Across Rounds")
        table.add_column("Round", style="cyan")
        table.add_column("Base Avg", style="green")
        table.add_column("Gen Avg", style="red")
        table.add_column("Top Weakness", style="yellow")

        for r in self.rounds:
            base_avg = (
                sum(res["score"] for res in r.base_results) / len(r.base_results)
                if r.base_results else 0
            )
            gen_avg = (
                sum(res["score"] for res in r.generated_results) / len(r.generated_results)
                if r.generated_results else 0
            )
            weakness = r.failure_summary.get("top_weakness", "n/a")
            table.add_row(
                str(r.round_num),
                f"{base_avg:.2f}",
                f"{gen_avg:.2f}",
                weakness,
            )
        console.print(table)
