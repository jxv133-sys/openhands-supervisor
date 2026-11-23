"""Messages sent to the coding agent. Kept short and explicit: local models do
best with one clear task, a concrete definition of done, and few rules."""

from __future__ import annotations

from .contract import Contract, Milestone

PROJECT_DIR = "/workspace/project"


def milestone_prompt(contract: Contract, milestone: Milestone, handoff: str = "") -> str:
    n = contract.index_of(milestone)
    total = len(contract.milestones)
    done = contract.completed()
    done_text = "\n".join(f"- {m.title}" for m in done) if done else "- (nothing yet, this is the first milestone)"
    notes = "\n".join(f"- {note}" for note in milestone.notes)
    parts = [
        f"You are working on a project in `{PROJECT_DIR}`.",
        f"Overall goal of the project: {contract.goal}",
        f"Milestones already completed (do not redo them):\n{done_text}",
        f"YOUR TASK NOW, milestone {n} of {total}: {milestone.title}",
    ]
    if milestone.instructions:
        parts.append(milestone.instructions)
    if notes:
        parts.append(f"Guidance from the user:\n{notes}")
    parts.append(
        "This milestone is done when this command exits successfully, run from "
        f"`{PROJECT_DIR}`:\n```\n{milestone.check}\n```"
    )
    parts.append(
        "Rules:\n"
        "- Only work on this milestone. Later milestones come later.\n"
        "- Work in small steps: create or edit ONE file per step, then run the check and fix errors.\n"
        "- Never write several files in one response.\n"
        "- Do not ask for permission or confirmation. Make reasonable decisions yourself.\n"
        "- Never skip (#[ignore], pytest.skip), weaken or delete tests to make the check pass: fix the code. "
        "The supervisor checks for this and will not accept it.\n"
        "- Declare every dependency (Cargo.toml, pyproject.toml or requirements.txt). The project must work "
        "on a fresh machine, not only in this sandbox.\n"
        "- Log important steps and every error at debug level (Rust: log/env_logger, RUST_LOG=debug; "
        "Python: logging). Error messages say what failed and with which input.\n"
        "- When the check command passes, call the finish tool with a one-line summary."
    )
    if handoff:
        parts.append(handoff)
    return "\n\n".join(parts)


def handoff_note(reason: str, tree: str, check_cmd: str, check_output: str) -> str:
    return (
        "NOTE: a previous attempt at this milestone was stopped "
        f"({reason}) and you are starting fresh. The files it left are still there; "
        "inspect them before changing anything, and keep what works.\n"
        f"Current project files:\n```\n{tree}\n```\n"
        f"Current output of `{check_cmd}`:\n```\n{check_output.strip()[-2000:] or '(no output)'}\n```"
    )
