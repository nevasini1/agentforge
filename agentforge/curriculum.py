"""Training curriculum strategies for progressive difficulty scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


@dataclass
class CurriculumStage:
    name: str
    difficulty: str
    scenario_ids: list[str] = field(default_factory=list)
    pass_threshold: float = 0.5


class Curriculum:
    """Manages a progressive training curriculum with difficulty scaling."""

    def __init__(self):
        self.stages: list[CurriculumStage] = []
        self.current_stage_idx: int = 0
        self.history: list[dict[str, Any]] = []

    def build_from_scenarios(self, scenarios: list[dict[str, Any]]):
        by_difficulty: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
        for sc in scenarios:
            diff = sc.get("difficulty", "medium")
            sid = sc.get("id", "unknown")
            by_difficulty.setdefault(diff, []).append(sid)

        for diff in ["easy", "medium", "hard"]:
            if by_difficulty.get(diff):
                self.stages.append(
                    CurriculumStage(
                        name=f"Stage: {diff}",
                        difficulty=diff,
                        scenario_ids=by_difficulty[diff],
                    )
                )

    @property
    def current_stage(self) -> CurriculumStage | None:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def advance(self, pass_rate: float) -> bool:
        stage = self.current_stage
        if stage is None:
            return False

        self.history.append({
            "stage": stage.name,
            "pass_rate": pass_rate,
            "advanced": pass_rate >= stage.pass_threshold,
        })

        if pass_rate >= stage.pass_threshold:
            self.current_stage_idx += 1
            return True
        return False

    def get_current_scenario_ids(self) -> list[str]:
        stage = self.current_stage
        return stage.scenario_ids if stage else []

    def is_complete(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def summary(self) -> dict[str, Any]:
        return {
            "total_stages": len(self.stages),
            "current_stage": self.current_stage_idx,
            "completed": self.is_complete(),
            "history": self.history,
        }
