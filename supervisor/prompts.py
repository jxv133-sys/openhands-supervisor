"""Messages sent to the coding agent. Kept short and explicit: local models do
best with one clear task, a concrete definition of done, and few rules."""

from __future__ import annotations

from .contract import Contract, Milestone

PROJECT_DIR = "/workspace/project"
NOTES_FILE = "AGENTS.md"  # the cross-tool name for "notes for coding agents"
NOTES_LIMIT = 4000

NOTES_RULE = (
    f"- Before calling finish, update `{NOTES_FILE}` (project notes for whoever works on this next, "
    "including you in a fresh conversation): what exists now (files and what each does), decisions and why, "
    "how to build, run and test, and known problems. Keep it short (under about 60 lines): rewrite parts "
    "that are out of date instead of appending."
)


def notes_section(notes: str) -> str:
    """The project notes as they go into the agent's instructions."""
    notes = notes.strip()
    if not notes:
        return (f"There are no project notes yet: create `{NOTES_FILE}` in the project root in this milestone "
                "(see the rules below).")
    if len(notes) > NOTES_LIMIT:
        notes = notes[:NOTES_LIMIT] + "\n...(cut: keep the notes shorter)"
    return (f"Project notes from earlier milestones (`{NOTES_FILE}`, kept by you; trust them, and fix them "
            f"if they're wrong):\n```markdown\n{notes}\n```")


def milestone_prompt(contract: Contract, milestone: Milestone, handoff: str = "", notes: str | None = None) -> str:
    """`notes`: AGENTS.md's text ("" if missing), or None when project notes are turned off."""
    n = contract.index_of(milestone)
    total = len(contract.milestones)
    done = contract.completed()
    done_text = "\n".join(f"- {m.title}" for m in done) if done else "- (nothing yet, this is the first milestone)"
    notes = "\n".join(f"- {note}" for note in milestone.notes)
    parts = [
        f"You are working on a project in `{PROJECT_DIR}`.",
        f"Overall goal of the project: {contract.goal}",
        f"Milestones already completed (do not redo them):\n{done_text}",
        *([notes_section(notes)] if notes is not None else []),
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
        "- The supervisor runs the check in a fresh login shell started in the project folder, not in your "
        "terminal session: what you `export`, `cd` into or activate in your terminal does not apply to it. If the "
        "project or its checks need anything outside the project folder (a link, a system-wide package, a tool), "
        "put the commands in `.openhands/setup.sh`: it runs in every fresh sandbox. Only doing it by hand makes "
        "the check pass here and fail later.\n"
        "- Log important steps and every error at debug level (Rust: log/env_logger, RUST_LOG=debug; "
        "Python: logging). Error messages say what failed and with which input.\n"
        + (NOTES_RULE + "\n" if notes is not None else "")
        + "- When the check command passes, call the finish tool with a one-line summary."
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
