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

CHECK_RULES = """Rules for checks (they are run after EVERY step, many times, in any order):
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
- Rust: unit tests in `#[cfg(test)] mod tests`; for command-line behaviour, integration
  tests in `tests/*.rs` that run the real binary with
  `std::process::Command::new(env!("CARGO_BIN_EXE_<name>"))` in a fresh temporary
  directory (no extra crates needed). Check: `cargo test <test name>` with the tee/grep above.
- Python: pytest tests in `tests/test_*.py`; use the `tmp_path` fixture for files, and
  `subprocess.run([sys.executable, ...])` for command-line behaviour. Dependencies go in
  pyproject.toml or requirements.txt.
- Tests must check behaviour with real asserts (outputs, files written, error messages),
  not just that the code runs.
"""

PLANNER_PROMPT = """You plan software projects for a coding agent that works in a Linux (Debian) sandbox.
Break the goal below into 3 to 8 small, sequential milestones. Each milestone must:
- be small enough to finish by editing a few files,
- build on the previous ones,
- have a "check": ONE shell command, run from the project root, that exits with code 0
  only when the milestone is truly done.
{check_rules}Also list "setup": shell commands that install the language toolchain the project
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
            "messages": [{"role": "user", "content": PLANNER_PROMPT.format(goal=goal, check_rules=CHECK_RULES)}],
        },
        timeout,
    )
    return parse_plan(goal, response["message"]["content"])


MILESTONE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "instructions": {"type": "string"},
        "check": {"type": "string"},
    },
    "required": ["title", "instructions", "check"],
}

DRAFT_PROMPT = """You plan work for a coding agent that works in a Linux (Debian) sandbox.
The project already has a plan. Write ONE new milestone for what the user describes below.
The milestone must be small enough to finish by editing a few files, and must have a
"check": ONE shell command, run from the project root, that exits with code 0 only when
the milestone is truly done.
{check_rules}Use the language, file names and test style the project already uses (see its files and
the existing milestones' checks).
"title": short, starts with a verb. "instructions": WHAT to build, concretely, in 1-4
sentences, including which tests to write so the check can run them. No code.

Overall goal of the project:
{goal}

Existing milestones:
{milestones}

Project files:
{tree}

{request}

Reply with JSON only."""


def draft_milestone(contract: Contract, description: str, tree: str, ollama_url: str, model: str,
                    title: str = "", instructions: str = "", timeout: float = 900) -> dict:
    """One new milestone (title, instructions, check) from a plain-language
    description. If the user already wrote a title/instructions, keep those and
    only fill in what's missing (typically the check command)."""
    milestones = "\n".join(
        f"{i}. [{m.status}] {m.title} (check: {m.check})" for i, m in enumerate(contract.milestones, 1)
    ) or "(none)"
    if title or instructions:
        request = (
            "The user already started this milestone. Keep their intent; improve wording only if unclear.\n"
            f"Title: {title or '(none)'}\nInstructions: {instructions or '(none)'}"
            + (f"\nAlso: {description}" if description else "")
        )
    else:
        request = f"What the user wants next:\n{description}"
    prompt = DRAFT_PROMPT.format(check_rules=CHECK_RULES, goal=contract.goal, milestones=milestones,
                                 tree=tree, request=request)
    response = _post_json(
        ollama_url.rstrip("/") + "/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": MILESTONE_SCHEMA,
            "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return parse_milestone(response["message"]["content"])


def parse_milestone(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"model returned no JSON object: {text[:300]!r}")
    data = json.loads(match.group(0))
    out = {k: str(data.get(k, "") or "").strip() for k in ("title", "instructions", "check")}
    if not out["check"]:
        raise ValueError("model returned no check command")
    return out


# --- shaping the goal with the user ----------------------------------------------

GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "name": {"type": "string"},
        "note": {"type": "string"},
        "ready": {"type": "boolean"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "options"],
            },
        },
    },
    "required": ["goal", "questions", "ready"],
}

GOAL_PROMPT = """You help a user turn a rough idea into a precise goal for a coding agent. The agent
runs a local model in a Linux (Debian) sandbox. The goal will be split into small milestones,
each proven done by an automated command (usually tests), so the goal must be concrete and
testable: vague goals produce vague programs.

The user mostly builds Rust and Python projects. Suggest one of those unless they say otherwise.

Each time, return:
- "goal": the best goal you can write from everything known so far, in Markdown: a one-line
  summary, then short sections or bullets for: language and how it's run (CLI with exact
  commands, library API, service...), each feature with its exact inputs and outputs, where
  data is stored and in what format, error behaviour (bad input, missing files...), and what
  is out of scope. Use the user's answers exactly. Where something is still unknown, choose a
  sensible, simple default and say so in the goal (e.g. "Data: todos.json in the current
  directory (assumed)"). No code.
- "questions": at most 4 questions whose answers would really change what gets built, most
  important first. Never ask what the idea or earlier answers already settle. Each has 2-4
  short "options" (recommended first) and a short "why". Return [] when nothing important is open.
- "ready": true when the goal is concrete enough to plan and test.
- "note": one short sentence to the user about this round (what changed, or that it's ready).
- "name": a short project folder name (lowercase, hyphens), e.g. "todo-rs".

The user's idea:
{idea}

{history}
Reply with JSON only."""


def assist_goal(idea: str, answers: list[dict], current_goal: str, ollama_url: str, model: str,
                timeout: float = 900) -> dict:
    """One round of shaping the goal: an updated draft plus clarifying questions.
    `answers` is every question asked so far with the user's answer ("" = skipped)."""
    parts = []
    if answers:
        lines = []
        for qa in answers:
            answer = str(qa.get("answer", "")).strip() or "(skipped: choose a sensible default)"
            lines.append(f"Q: {qa.get('question', '')}\nA: {answer}")
        parts.append("Questions asked so far, with the user's answers:\n" + "\n\n".join(lines))
    if current_goal.strip():
        parts.append("The current goal draft (the user may have edited it; keep their edits):\n" + current_goal.strip())
    prompt = GOAL_PROMPT.format(idea=idea.strip(), history="\n\n".join(parts) + ("\n\n" if parts else ""))
    response = _post_json(
        ollama_url.rstrip("/") + "/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": GOAL_SCHEMA,
            "options": {"temperature": 0.4},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return parse_goal_round(response["message"]["content"], asked=[qa.get("question", "") for qa in answers])


def parse_goal_round(text: str, asked: list[str] = ()) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"model returned no JSON object: {text[:300]!r}")
    data = json.loads(match.group(0))
    goal = str(data.get("goal", "") or "").strip()
    if not goal:
        raise ValueError("model returned no goal")
    seen = {q.strip().lower() for q in asked if q}
    questions = []
    for q in data.get("questions") or []:
        if not isinstance(q, dict):
            continue
        text_q = str(q.get("question", "")).strip()
        if not text_q or text_q.lower() in seen:  # never ask the same thing twice
            continue
        seen.add(text_q.lower())
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()][:4]
        questions.append({"question": text_q, "why": str(q.get("why", "") or "").strip(), "options": options})
    name = re.sub(r"[^a-z0-9._-]+", "-", str(data.get("name", "") or "").lower()).strip("-.")[:40]
    return {
        "goal": goal,
        "questions": questions[:4],
        "ready": bool(data.get("ready")) or not questions,
        "note": str(data.get("note", "") or "").strip(),
        "name": name,
    }
