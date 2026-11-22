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
