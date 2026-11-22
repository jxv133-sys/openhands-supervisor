"""Console output plus a JSONL record of every supervisor decision, so you can
see afterwards exactly what it caught and why. Listeners (the web UI) get every
record live, plus transient agent activity that isn't written to disk."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .contract import STATE_DIR

Listener = Callable[[dict], None]


class RunLog:
    def __init__(self, project_dir: Path, echo: bool = True):
        self.path = Path(project_dir) / STATE_DIR / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def publish(self, kind: str, persist: bool = False, **data) -> dict:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **data}
        if persist:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:
                pass  # a broken listener must never break the run
        return record

    def event(self, kind: str, **data) -> None:
        self.publish(kind, persist=True, **data)

    def say(self, text: str, **data) -> None:
        """Print a supervisor line (visually distinct from the agent's output) and log it."""
        if self.echo:
            print(f"\n\033[1;35m[supervisor]\033[0m {text}", file=sys.stderr, flush=True)
        self.event("say", text=text, **data)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
