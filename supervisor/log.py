"""Console output plus a JSONL record of every supervisor decision, so you can
see afterwards exactly what it caught and why."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .contract import STATE_DIR


class RunLog:
    def __init__(self, project_dir: Path):
        self.path = Path(project_dir) / STATE_DIR / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def say(self, text: str, **data) -> None:
        """Print a supervisor line (visually distinct from the agent's output) and log it."""
        print(f"\n\033[1;35m[supervisor]\033[0m {text}", file=sys.stderr, flush=True)
        self.event("say", text=text, **data)
