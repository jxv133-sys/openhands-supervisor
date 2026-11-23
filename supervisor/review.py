"""Final review: once every milestone is done, check the project as a whole.

Two parts, both decided by real commands, never by the model's opinion:

1. Regressions: every finished milestone's check is run again. A later
   milestone can break an earlier one; such milestones go back to pending.
2. Acceptance: one LLM call lists the goal's requirements, each with a shell
   command that tests it end to end. The supervisor runs those commands; every
   requirement that fails becomes a new milestone whose check is that command.

This module holds the pure parts (prompt, parsing, source gathering, report
I/O) so they are testable without the SDK or a model.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .contract import STATE_DIR, Contract, Milestone
from .progress import _iter_files

REVIEW_FILE = "review.json"
REVIEW_SOURCE = "review"  # Milestone.source of milestones added by the review

# Files that say little about whether requirements are met but eat the budget.
SKIP_NAMES = {"Cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "go.sum"}
MAX_SOURCE_FILE = 40_000

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "check": {"type": "string"},
                },
                "required": ["requirement", "check"],
            },
        },
    },
    "required": ["requirements"],
}

REVIEW_PROMPT = """You are the final reviewer of a software project built by a coding agent in a Linux (Debian)
sandbox. The agent says it is finished. Decide, with real commands, whether the project does
everything the goal asks for.

List every concrete requirement in the goal: each feature, command, option, input/output format,
file it must read or write, and error behaviour the goal mentions. Do not invent requirements the
goal does not state. Usually 3 to 12 requirements.

For each requirement write "check": ONE shell command, run from the project root with the project's
toolchain already installed, that exits 0 only if the requirement really works. Rules:
- Test the real program end to end the way a user would: build it, run it with arguments, and check
  its output or the files it writes. Existing unit tests alone are not enough.
- Work in a temporary directory for any data (`d=$(mktemp -d)`), never change the project's files.
- Keep failures visible: no `-q` on its own, no `> /dev/null`. To test output, use for example
  `out=$(cargo run -q -- list 2>&1); echo "$out"; echo "$out" | grep -F 'buy milk'`.
- Use the file names and commands that actually exist in the project files below.
- No user input, no server left running, no GUI. Must finish within a few minutes.
- Rust: build and run the binary (`cargo run -q -- <args>`) or the crate's tests. Python: run the
  program with `python3 ...` or `python -m <package> ...`.
- Include error behaviour the goal mentions (unknown ids, missing files, bad input), not only the
  main feature.
- If a requirement truly cannot be checked by a command, use "" as its check.

Goal:
{goal}

Milestones the agent completed:
{milestones}

Project files:
{tree}

Source code:
{sources}

Reply with JSON only."""


def gather_sources(root: Path, budget_chars: int) -> tuple[str, str]:
    """(file tree, concatenated file contents within budget). Small,
    top-level and source files first, so a big generated file can't crowd
    out main.rs."""
    root = Path(root)
    files = []
    for p in _iter_files(root):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        files.append((p.relative_to(root).as_posix(), p, size))
    files.sort(key=lambda f: f[0])
    tree = "\n".join(f[0] for f in files[:300]) or "(no files)"
    if len(files) > 300:
        tree += f"\n... and {len(files) - 300} more"

    def priority(f):
        rel, _, size = f
        return (rel.count("/"), size)

    chunks, used = [], 0
    for rel, path, size in sorted(files, key=priority):
        if path.name in SKIP_NAMES or size > MAX_SOURCE_FILE:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4000]:
            continue
        text = data.decode("utf-8", errors="replace")
        block = f"--- {rel} ---\n{text.rstrip()}\n"
        if used + len(block) > budget_chars:
            continue
        chunks.append((rel, block))
        used += len(block)
    chunks.sort(key=lambda c: c[0])
    return tree, "\n".join(b for _, b in chunks) or "(no readable source files)"


def review_prompt(contract: Contract, tree: str, sources: str) -> str:
    done = [m for m in contract.milestones if m.status == "done"]
    milestones = "\n".join(f"- {m.title} (check: {m.check})" for m in done) or "- (none)"
    return REVIEW_PROMPT.format(goal=contract.goal, milestones=milestones, tree=tree, sources=sources)


def parse_review(text: str) -> list[dict]:
    """[{requirement, check}] from the model's JSON (tolerating fences/prose)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"reviewer returned no JSON object: {text[:300]!r}")
    data = json.loads(match.group(0))
    out, seen = [], set()
    for item in data.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        req = str(item.get("requirement", "")).strip()
        check = str(item.get("check", "")).strip()
        if not req or req.lower() in seen:
            continue
        seen.add(req.lower())
        out.append({"requirement": req, "check": check})
    if not out:
        raise ValueError("reviewer listed no requirements")
    return out


def ask_reviewer(prompt: str, ollama_url: str, model: str, timeout: float) -> list[dict]:
    from .planner import _post_json

    response = _post_json(
        ollama_url.rstrip("/") + "/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": REVIEW_SCHEMA,
            "options": {"temperature": 0.2},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return parse_review(response["message"]["content"])


def fix_milestone(requirement: str, check: str, output: str) -> Milestone:
    title = requirement if len(requirement) <= 90 else requirement[:87] + "..."
    return Milestone(
        title=f"Meet requirement: {title}",
        instructions=(
            f"The final review found that this requirement from the goal is not met yet: {requirement}\n"
            "Make it work as the goal describes, without breaking what already works. "
            f"The check below currently fails with:\n{output.strip()[-1200:] or '(no output)'}"
        ),
        check=check,
        source=REVIEW_SOURCE,
    )


def regression_note(output: str) -> str:
    return (
        "The final review found this milestone broken again (its check passed earlier but fails now, "
        "probably because of a later change). Fix it without undoing later work. Check output:\n"
        + (output.strip()[-800:] or "(no output)")
    )


def review_path(project_dir: Path) -> Path:
    return Path(project_dir) / STATE_DIR / REVIEW_FILE


def save_report(project_dir: Path, report: dict) -> None:
    path = review_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_report(project_dir: Path) -> dict | None:
    try:
        return json.loads(review_path(project_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def new_report(round_n: int) -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "round": round_n,
        "regressions": [],    # [{milestone, title, output}]
        "requirements": [],   # [{requirement, check, passed (bool|None), output}]
        "added": [],          # titles of milestones added for failing requirements
        "error": "",          # reviewer could not run
        "verdict": "",        # pass | fail | regressions | error
    }
