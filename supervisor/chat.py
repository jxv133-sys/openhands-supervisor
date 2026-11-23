"""Turn OpenHands SDK events into chat records for the web UI's live chat.

Like collector.py, events are matched by class name and attributes (no SDK
imports), so this is testable with plain fakes and tolerant of SDK refactors.

Record shapes (all have "role"; long text is truncated):
  {"role": "supervisor", "text"}                 message sent to the agent
  {"role": "agent", "text"}                      agent's reply
  {"role": "thinking", "text"}                   agent's reasoning before an action
  {"role": "tool_call", "call_id", "tool", "summary", "args", "thinking"}
  {"role": "tool_result", "action_id", "tool", "text"}
  {"role": "system", "text", "level"}            errors, history condensation
"""

from __future__ import annotations

import json

TEXT_LIMIT = 8000
OUTPUT_LIMIT = 6000
ARG_LIMIT = 6000


def _cut(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + f"\n... ({len(text) - limit} characters cut) ...\n" + text[-(limit - head):]


def _texts(items) -> str:
    if items is None:
        return ""
    if isinstance(items, str):
        return items
    return "".join(getattr(c, "text", "") or "" for c in items).strip()


def _action_args(action) -> dict:
    if action is None:
        return {}
    dump = getattr(action, "model_dump", None)
    data = dump(exclude_none=True) if callable(dump) else dict(getattr(action, "__dict__", {}))
    data.pop("kind", None)
    out = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = _cut(value, ARG_LIMIT)
        else:
            try:
                out[key] = json.loads(json.dumps(value, default=str))
            except (TypeError, ValueError):
                out[key] = str(value)
    return out


def summarize_call(tool: str, args: dict) -> str:
    """One line describing a tool call: `$ cargo build`, `create src/main.rs`..."""
    if "command" in args and tool in ("terminal", "execute_bash", "bash"):
        return f"$ {args['command']}"
    if "path" in args:
        return f"{args.get('command', 'edit')} {args['path']}"
    if tool == "finish":
        return "finish"
    if "message" in args:
        return str(args["message"])[:200]
    if "command" in args:
        return str(args["command"])[:200]
    return tool


def chat_record(event) -> dict | None:
    kind = type(event).__name__
    if kind == "MessageEvent":
        message = getattr(event, "llm_message", None)
        text = _texts(getattr(message, "content", None))
        if not text:
            return None
        role = "supervisor" if getattr(event, "source", "") == "user" else "agent"
        record = {"role": role, "text": _cut(text, TEXT_LIMIT)}
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if role == "agent" and reasoning:
            record["thinking"] = _cut(reasoning, TEXT_LIMIT)
        return record

    if kind == "ActionEvent":
        tool = getattr(event, "tool_name", "") or "?"
        args = _action_args(getattr(event, "action", None))
        thought = _texts(getattr(event, "thought", None))
        reasoning = (getattr(event, "reasoning_content", None) or "").strip()
        return {
            "role": "tool_call",
            # not "id": the web event bus numbers every event with its own "id"
            "call_id": str(getattr(event, "id", "")),
            "tool": tool,
            "summary": summarize_call(tool, args),
            "args": args,
            "thinking": _cut("\n\n".join(t for t in (reasoning, thought) if t), TEXT_LIMIT),
        }

    if kind == "ObservationEvent":
        observation = getattr(event, "observation", None)
        text = _texts(getattr(observation, "to_llm_content", None))
        if not text and observation is not None:
            text = str(getattr(observation, "text", "") or "")
        return {
            "role": "tool_result",
            "action_id": str(getattr(event, "action_id", "")),
            "tool": getattr(event, "tool_name", ""),
            "text": _cut(text, OUTPUT_LIMIT),
        }

    if kind == "AgentErrorEvent":
        return {"role": "system", "level": "error", "text": _cut(str(getattr(event, "error", "")), 2000)}
    if kind == "ConversationErrorEvent":
        detail = getattr(event, "detail", "") or getattr(event, "code", "")
        return {"role": "system", "level": "error", "text": str(detail)[:2000]}
    if "Condensation" in kind and kind != "CondensationRequest":
        summary = getattr(event, "summary", None) or ""
        text = "Older conversation history was summarized to fit the context window."
        if summary:
            text += f"\n\n{_cut(str(summary), 3000)}"
        return {"role": "system", "level": "info", "text": text}
    return None


class LiveStream:
    """Batches the model's streamed tokens into a few updates per second.

    Deltas arrive one token at a time (dozens a second). They're grouped into
    one live message from the first delta until `end()` (called when the next
    real event, e.g. the finished message or tool call, replaces it). Each
    flush publishes only the newly arrived text, numbered with `seq`, so a
    page that joins mid-stream can take `snapshot()` and then apply later
    updates. Nothing here is saved to disk: the finished event is.
    """

    def __init__(self, publish, interval: float = 0.35):
        import threading

        self._publish = publish  # publish(**record): sends a transient "stream" record
        self._interval = interval
        self._lock = threading.Lock()
        self._timer = None
        # Keeps counting across messages, so a late update from an earlier
        # message can never be mistaken for one of the current message.
        self.seq = 0
        self._reset()

    def _reset(self) -> None:
        self.active = False
        self.text = ""
        self.thinking = ""
        self._pending_text = ""
        self._pending_thinking = ""
        self._last_flush = 0.0

    def delta(self, content: str | None, reasoning: str | None) -> None:
        import time

        if not content and not reasoning:
            return
        with self._lock:
            self.active = True
            self.text += content or ""
            self.thinking += reasoning or ""
            self._pending_text += content or ""
            self._pending_thinking += reasoning or ""
            if time.monotonic() - self._last_flush >= self._interval:
                self._flush_locked()
            elif self._timer is None:  # flush the tail even if no more tokens come soon
                import threading

                self._timer = threading.Timer(self._interval, self._flush_timer)
                self._timer.daemon = True
                self._timer.start()

    def _flush_timer(self) -> None:
        with self._lock:
            self._timer = None
            if self.active:
                self._flush_locked()

    def _flush_locked(self) -> None:
        import time

        if not self._pending_text and not self._pending_thinking:
            return
        self.seq += 1
        record = {"seq": self.seq, "text": self._pending_text, "thinking": self._pending_thinking}
        self._pending_text = self._pending_thinking = ""
        self._last_flush = time.monotonic()
        self._publish(**record)

    def end(self) -> None:
        """The streamed message is complete (its final event comes next)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if not self.active:
                return
            self._flush_locked()
            self.seq += 1
            self._publish(seq=self.seq, end=True)
            self._reset()

    def snapshot(self) -> dict | None:
        with self._lock:
            if not self.active:
                return None
            # Only what was already published up to `seq`; the unflushed tail
            # arrives in the next update (seq + 1), so nothing is shown twice.
            text = self.text[: len(self.text) - len(self._pending_text)]
            thinking = self.thinking[: len(self.thinking) - len(self._pending_thinking)]
            return {"seq": self.seq, "text": text, "thinking": thinking}
