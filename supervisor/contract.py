"""The task contract: the goal, setup commands, and milestones with checks.

Lives in <project>/.supervisor/contract.json, outside the agent's context, so
OpenHands' history condensation can never lose it. Saved after every change so
a run can be stopped and resumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = ".supervisor"
CONTRACT_FILE = "contract.json"

PENDING = "pending"
DONE = "done"
SKIPPED = "skipped"


@dataclass
class Milestone:
    title: str
    instructions: str
    check: str  # shell command, run from the project root; exit 0 == done
    status: str = PENDING
    restarts: int = 0
    notes: list[str] = field(default_factory=list)  # user guidance added on escalation


@dataclass
class Contract:
    goal: str
    milestones: list[Milestone]
    setup: list[str] = field(default_factory=list)  # run once per container start

    def pending(self) -> list[Milestone]:
        return [m for m in self.milestones if m.status == PENDING]

    def completed(self) -> list[Milestone]:
        return [m for m in self.milestones if m.status == DONE]

    def index_of(self, milestone: Milestone) -> int:
        return self.milestones.index(milestone) + 1

    # --- persistence -----------------------------------------------------

    @staticmethod
    def path_for(project_dir: Path) -> Path:
        return Path(project_dir) / STATE_DIR / CONTRACT_FILE

    def save(self, project_dir: Path) -> Path:
        path = self.path_for(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written contract
        return path

    @classmethod
    def load(cls, project_dir: Path) -> Contract:
        data = json.loads(cls.path_for(project_dir).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Contract:
        goal = str(data.get("goal", "")).strip()
        if not goal:
            raise ValueError("contract has no goal")
        raw = data.get("milestones") or []
        if not raw:
            raise ValueError("contract has no milestones")
        milestones = []
        for i, m in enumerate(raw, 1):
            title = str(m.get("title", "")).strip()
            check = str(m.get("check", "")).strip()
            if not title or not check:
                raise ValueError(f"milestone {i} needs both a title and a check command")
            milestones.append(
                Milestone(
                    title=title,
                    instructions=str(m.get("instructions", "")).strip(),
                    check=check,
                    status=m.get("status", PENDING),
                    restarts=int(m.get("restarts", 0)),
                    notes=[str(n) for n in m.get("notes", [])],
                )
            )
        setup = [str(s).strip() for s in data.get("setup", []) if str(s).strip()]
        return cls(goal=goal, milestones=milestones, setup=setup)
