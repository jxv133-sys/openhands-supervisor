"""Turns a plain-language goal into a draft contract with one LLM call.

The user reviews and edits the result before anything runs, so a mediocre plan
from a local model costs a minute of editing, not a failed run.
"""

from __future__ import annotations

import json
import re
import urllib.request

from .contract import Contract

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "setup": {"type": "array", "items": {"type": "string"}},
        "milestones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "instructions": {"type": "string"},
                    "check": {"type": "string"},
                },
                "required": ["title", "instructions", "check"],
            },
        },
    },
    "required": ["setup", "milestones"],
}

PLANNER_PROMPT = """You plan software projects for a coding agent that works in a Linux (Debian) sandbox.
Break the goal below into 3 to 8 small, sequential milestones. Each milestone must:
- be small enough to finish by editing a few files,
- build on the previous ones,
- have a "check": ONE shell command, run from the project root, that exits with code 0
  only when the milestone is truly done.
Rules for checks (they are run after EVERY step, many times, in any order):
- Prefer automated tests: tell the agent in the instructions to write tests for the
  feature (e.g. a Rust test named `test_add`), and make the check run just those tests
  (`cargo test test_add`, `python -m pytest tests/test_add.py -q`). Use `cargo build` only
  for setup milestones.
- A check must be self-contained and repeatable: it must not depend on files or data
  left behind by earlier checks or milestones, and must not change the project's real
  data files. If it needs data, it creates it in a temporary directory itself.
- When a check fails, its output is the ONLY error the agent sees. Never hide it:
  no `grep -q` on its own, no `> /dev/null`, no `--quiet`. If you must grep, keep the
  output visible with tee, exactly like this for Rust (a filter matching zero tests
  would otherwise "pass"):
  `cargo test test_add 2>&1 | tee /dev/stderr | grep -Eq 'test result: ok\\. [1-9]'`
  Do not use `--exact` (tests inside `mod tests` would not match).
  pytest needs no grep: `python -m pytest tests/test_add.py -q` already fails if nothing ran.
- No user input, no server left running, no GUI.
Also list "setup": shell commands that install the language toolchain the project
needs, written so they are safe to run again (e.g. for Rust:
`command -v cargo || (curl -sSf https://sh.rustup.rs | sh -s -- -y)`). Empty list if none needed.
Instructions should say WHAT to build, concretely, in 1-4 sentences. No code.

Goal:
{goal}

Reply with JSON only."""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def parse_plan(goal: str, text: str) -> Contract:
    """Parse the model's JSON (tolerating fences or stray prose around it)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"planner returned no JSON object: {text[:300]!r}")
    data = json.loads(match.group(0))
    data["goal"] = goal
    return Contract.from_dict(data)


def plan(goal: str, ollama_url: str, model: str, timeout: float = 900) -> Contract:
    response = _post_json(
        ollama_url.rstrip("/") + "/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": PLAN_SCHEMA,
            "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": PLANNER_PROMPT.format(goal=goal)}],
        },
        timeout,
    )
    return parse_plan(goal, response["message"]["content"])
