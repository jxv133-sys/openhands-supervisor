"""Turns the OpenHands event stream for one turn into a TurnReport.

Matches events by class name and attributes instead of importing SDK classes,
so it is testable without the SDK and tolerant of SDK refactors.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time

from .watchdog import TurnEnd, TurnReport


def _text_of(message) -> str:
    content = getattr(message, "content", None) or []
    if isinstance(content, str):
        return content
    return "".join(getattr(c, "text", "") or "" for c in content).strip()


_NOISE = re.compile(r"\b\d+(\.\d+)?\s*(ms|s|sec|seconds)\b|0x[0-9a-f]+|\b\d{2}:\d{2}:\d{2}\b|/tmp/[\w.-]+", re.I)


def _args_of(action) -> str:
    if action is None:
        return ""
    dump = getattr(action, "model_dump", None)
    try:
        data = dump(exclude_none=True) if dump else vars(action)
    except Exception:
        return str(action)
    data = {k: v for k, v in data.items() if k not in ("kind", "thought", "summary", "security_risk")}
    return json.dumps(data, sort_keys=True, default=str)


def _obs_text(event) -> str:
    obs = getattr(event, "observation", None)
    for attr in ("text", "content", "output"):
        val = getattr(obs, attr, None)
        if val:
            if isinstance(val, list):
                return "".join(getattr(c, "text", "") or str(c) for c in val)
            return str(val)
    return str(obs or "")


def _digest(text: str) -> str:
    return hashlib.sha1(_NOISE.sub("#", text).encode("utf-8", "replace")).hexdigest()[:16]


def _label(tool: str, args: str) -> str:
    try:
        data = json.loads(args)
    except ValueError:
        data = {}
    main = data.get("command") or data.get("path") or args
    main = " ".join(str(main).split())
    return f"{tool}: {main[:140]}"


class TurnCollector:
    """Pass `on_event` as a Conversation callback; call `reset()` before each
    turn and `report()` after it."""

    def __init__(self, on_activity=None):
        self._lock = threading.Lock()
        self._on_activity = on_activity  # optional hook for live logging
        # (action signature, result digest) seen during this milestone. Kept
        # across turns and fresh conversations; the runner clears it per milestone.
        self.seen: set[tuple[str, str]] = set()
        self.reset()

    def new_milestone(self) -> None:
        with self._lock:
            self.seen = set()

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
            self.pending: dict[str, tuple[str, str]] = {}  # tool_call_id -> (signature, label)
            self.repeated = 0
            self.repeat_examples: list[str] = []

    def on_event(self, event) -> None:
        kind = type(event).__name__
        with self._lock:
            self.last_event_at = time.monotonic()
            if kind == "ActionEvent":
                self.actions += 1
                tool = getattr(event, "tool_name", "") or "?"
                self.tool_counts[tool] = self.tool_counts.get(tool, 0) + 1
                action = getattr(event, "action", None)
                args = _args_of(action)
                self.pending[str(getattr(event, "tool_call_id", "") or id(event))] = (f"{tool}|{args}", _label(tool, args))
                if action is not None and type(action).__name__ == "FinishAction":
                    self.finished = True
                    self.finish_message = str(getattr(action, "message", "") or "")
            elif kind == "MessageEvent" and getattr(event, "source", "") == "agent":
                text = _text_of(getattr(event, "llm_message", None))
                if text:
                    self.last_agent_message = text
            elif kind == "ObservationEvent":
                key = str(getattr(event, "tool_call_id", "") or "")
                if key in self.pending:
                    sig, label = self.pending.pop(key)
                    pair = (sig, _digest(_obs_text(event)))
                    if pair in self.seen:
                        self.repeated += 1
                        if label not in self.repeat_examples:
                            self.repeat_examples.append(label)
                    self.seen.add(pair)
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

    def report(self, status: str, exception: str = "", changed_files: int = 0, tokens: int = 0) -> TurnReport:
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
                repeated_actions=self.repeated,
                repeat_examples=list(self.repeat_examples[:10]),
                tokens=tokens,
            )
