"""Auto-resolve: when the agent needs help and nobody answers in time, ask the
model for a diagnosis with a fresh pair of eyes, and retry with it as guidance.

The rescuer sees what a human would see on the help card (why the agent was
stopped, what it kept doing, the check output), plus the project notes, the
file list and the source code, and the guidance already tried. It can also
say the milestone can't be done as specified; then it is skipped so an
unattended run carries on with the rest.

Pure parts (prompt, parsing) are testable without a model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

RESCUE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "guidance": {"type": "string"},
        "give_up": {"type": "boolean"},
    },
    "required": ["diagnosis", "guidance", "give_up"],
}

RESCUE_PROMPT = """A coding agent is stuck on one milestone of a software project in a Linux (Debian) sandbox,
and the user is away. You are a senior engineer taking a fresh look. Find the real cause from the
evidence below and tell the agent exactly what to do next.

- Base the diagnosis on the evidence: the check output, the source code, what the agent tried.
  Name the file and the line or construct at fault when you can.
- The guidance is read by the agent. Make it concrete and short (at most 12 lines): the cause,
  the change to make, and how to verify it. Do not repeat guidance that was already tried
  unless you explain what was wrong with how it was applied.
- The check command is fixed: the agent must make it pass, never change, skip or weaken it.
- Set "give_up" to true only if the milestone cannot be done as specified (for example the check
  needs something the sandbox can never have). Then explain why in "diagnosis".

Goal of the project:
{goal}

Milestone: {title}
{instructions}
Check command (must exit 0):
{check}

Why the agent was stopped: {reason}
{details}

Current check output:
```
{check_output}
```

Guidance already tried for this milestone:
{tried}

Project notes (AGENTS.md):
{notes}

Project files:
{tree}

Source code:
{sources}

Reply with JSON only: {{"diagnosis": "...", "guidance": "...", "give_up": false}}"""


@dataclass
class Rescue:
    diagnosis: str
    guidance: str
    give_up: bool = False


def rescue_prompt(*, goal: str, title: str, instructions: str, check: str, reason: str, details: str,
                  check_output: str, tried: list[str], notes: str, tree: str, sources: str) -> str:
    return RESCUE_PROMPT.format(
        goal=goal, title=title, instructions=instructions or "", check=check, reason=reason,
        details=(f"Details:\n{details.strip()[-2500:]}" if details.strip() else ""),
        check_output=check_output.strip()[-3000:] or "(no output)",
        tried="\n".join(f"- {t}" for t in tried) or "- (none)",
        notes=notes.strip()[:4000] or "(none)", tree=tree, sources=sources,
    )


def parse_rescue(text: str) -> Rescue:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in the model's answer: {text[:300]!r}")
    data = json.loads(match.group(0))
    diagnosis = str(data.get("diagnosis", "")).strip()
    guidance = str(data.get("guidance", "")).strip()
    give_up = bool(data.get("give_up", False))
    if not guidance and not give_up:
        raise ValueError("the model gave no guidance")
    return Rescue(diagnosis, guidance, give_up)


def ask_rescuer(prompt: str, ollama_url: str, model: str, timeout: float) -> Rescue:
    from .planner import _post_json

    response = _post_json(
        ollama_url.rstrip("/") + "/api/chat",
        {
            "model": model,
            "stream": False,
            "format": RESCUE_SCHEMA,
            "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return parse_rescue(response["message"]["content"])
