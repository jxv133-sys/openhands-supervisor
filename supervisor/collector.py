"""Turns the OpenHands event stream for one turn into a TurnReport.

Matches events by class name and attributes instead of importing SDK classes,
so it is testable without the SDK and tolerant of SDK refactors.
"""

from __future__ import annotations

import threading
import time

from .watchdog import TurnEnd, TurnReport


def _text_of(message) -> str:
    content = getattr(message, "content", None) or []
    if isinstance(content, str):
        return content
    return "".join(getattr(c, "text", "") or "" for c in content).strip()


class TurnCollector:
    """Pass `on_event` as a Conversation callback; call `reset()` before each
    turn and `report()` after it."""

    def __init__(self, on_activity=None):
        self._lock = threading.Lock()
        self._on_activity = on_activity  # optional hook for live logging
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.actions = 0
            self.tool_counts: dict[str, int] = {}
            self.finished = False
            self.finish_message = ""
            self.last_agent_message = ""
            self.error_codes: list[str] = []
            self.errors: list[str] = []
            self.last_event_at = time.monotonic()

    def on_event(self, event) -> None:
        kind = type(event).__name__
        with self._lock:
            self.last_event_at = time.monotonic()
            if kind == "ActionEvent":
                self.actions += 1
                tool = getattr(event, "tool_name", "") or "?"
                self.tool_counts[tool] = self.tool_counts.get(tool, 0) + 1
                action = getattr(event, "action", None)
                if action is not None and type(action).__name__ == "FinishAction":
                    self.finished = True
                    self.finish_message = str(getattr(action, "message", "") or "")
            elif kind == "MessageEvent" and getattr(event, "source", "") == "agent":
                text = _text_of(getattr(event, "llm_message", None))
                if text:
                    self.last_agent_message = text
            elif kind == "ConversationErrorEvent":
                self.error_codes.append(str(getattr(event, "code", "")))
                self.errors.append(str(getattr(event, "detail", "")))
            elif kind == "AgentErrorEvent":
                # Tool-call validation problems etc. The agent sees these and
                # usually recovers by itself, so record but don't end the turn.
                self.errors.append(str(getattr(event, "error", ""))[:500])
        if self._on_activity is not None:
            self._on_activity(kind, event)

    def wait_until_quiet(self, quiet_seconds: float = 2.0, max_wait: float = 15.0) -> None:
        """Remote events can trail the end of run(); wait for them to settle."""
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            with self._lock:
                idle = time.monotonic() - self.last_event_at
            if idle >= quiet_seconds:
                return
            time.sleep(0.2)

    def report(self, status: str, exception: str = "", changed_files: int = 0) -> TurnReport:
        """`status` is the conversation's execution_status value after run()."""
        status = (status or "").lower()
        with self._lock:
            codes = " ".join(self.error_codes)
            errors = " | ".join(e for e in self.errors if e)
            iter_cap = "MaxIterationsReached" in codes or "maximum iterations" in (
                exception + " " + errors
            ).lower()
            if iter_cap:
                end = TurnEnd.ITER_CAP
            elif status == "stuck":
                end = TurnEnd.STUCK
            elif self.finished:
                end = TurnEnd.FINISHED
            elif status == "error" or exception:
                end = TurnEnd.ERROR
            else:
                end = TurnEnd.MESSAGE
            return TurnReport(
                end=end,
                agent_text=self.finish_message or self.last_agent_message,
                actions=self.actions,
                changed_files=changed_files,
                error=(exception or errors)[:1000],
            )
