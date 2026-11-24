"""Deterministic rules: given how a turn ended and what the check says, decide
what to do next. No LLM involved, and no SDK imports, so it is fully testable.

A "turn" is one conversation.run(): it ends when the agent calls finish, replies
with plain text, hits the per-turn action cap, gets flagged as looping, or errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class TurnEnd(str, Enum):
    FINISHED = "finished"  # agent called the finish tool
    MESSAGE = "message"  # agent replied with text and stopped (often a question)
    ITER_CAP = "iter_cap"  # hit max actions for one turn
    STUCK = "stuck"  # SDK stuck detector fired (repeating itself)
    LOOPING = "looping"  # supervisor cut off a reply that kept repeating itself
    ERROR = "error"  # LLM / server / sandbox error


@dataclass
class TurnReport:
    end: TurnEnd
    agent_text: str = ""  # finish message or last plain message from the agent
    actions: int = 0  # tool calls this turn
    changed_files: int = 0  # project files changed this turn (measured on disk)
    error: str = ""
    # Actions this turn that repeated an earlier action of this milestone AND
    # got the same result: they added no information. Examples for messages.
    repeated_actions: int = 0
    repeat_examples: list = field(default_factory=list)
    tokens: int = 0  # model tokens this turn (prompt + reply), 0 if unknown


@dataclass
class CheckResult:
    passed: bool
    exit_code: int
    output: str  # stdout+stderr, already trimmed to a tail
    seconds: float = 0.0  # how long the command ran
    # Set when the command couldn't run in a usable environment (project folder
    # not mounted, sandbox API error...). Such a result says nothing about the
    # project: callers report it as an infrastructure error, never a failure.
    infra: str = ""


class Action(str, Enum):
    DONE = "done"  # milestone complete
    NUDGE = "nudge"  # send `message` into the same conversation and run again
    RESTART = "restart"  # throw away the conversation, start fresh with a handoff
    ESCALATE = "escalate"  # ask the human


@dataclass
class Decision:
    action: Action
    reason: str
    message: str = ""


@dataclass
class Limits:
    max_turns_per_conversation: int = 8
    max_restarts_per_milestone: int = 3
    max_consecutive_errors: int = 3
    max_questions_per_milestone: int = 4
    stagnant_turns: int = 3  # same failing check output this many turns in a row
    check_output_chars: int = 2500  # how much check output to show the agent
    # Stagnation: the agent keeps working but learns nothing new.
    repeat_actions: int = 4  # actions in one turn that repeat earlier ones with the same result
    stall_turns: int = 4  # turns in a row without progress (no file change, same check result)
    stall_tokens: int = 400_000  # model tokens spent without progress (0 = no limit)


@dataclass
class MilestoneState:
    """Counters for one milestone. `turns`/`same_failure_turns`/`questions_in_conv`
    reset on restart; `restarts`/`questions`/`changed_files` persist."""

    turns: int = 0
    restarts: int = 0
    changed_files: int = 0
    consecutive_errors: int = 0
    questions: int = 0
    last_failure_sig: str = ""
    same_failure_turns: int = 0
    # Stagnation, kept across fresh conversations: a restart re-explores, it
    # doesn't make progress. Reset only when the human steps in.
    no_progress_turns: int = 0
    tokens_since_progress: int = 0
    progress_sig: str = ""  # check result last time something moved
    stagnation_warned: bool = False

    def human_stepped_in(self) -> None:
        self.no_progress_turns = 0
        self.tokens_since_progress = 0
        self.stagnation_warned = False

    def new_conversation(self) -> None:
        self.turns = 0
        self.consecutive_errors = 0
        self.last_failure_sig = ""
        self.same_failure_turns = 0


AUTONOMY_NUDGE = (
    "Yes, continue. Do not ask for permission or confirmation: make reasonable "
    "decisions yourself and keep working until the check command passes, then call finish."
)

_QUESTION_RE = re.compile(
    r"\?\s*$|\b(shall i|should i|would you like|do you want|want me to|let me know|"
    r"please confirm|can you confirm|proceed\?|which (option|approach)|before i (start|continue))\b",
    re.IGNORECASE | re.MULTILINE,
)


# The agent's way to ask the human (taught in the milestone prompt).
_HELP_RE = re.compile(r"^\s*STUCK\s*:|\b(i need (your |human )?help|i('| a)m (truly |completely )?stuck)\b", re.IGNORECASE)


def asks_for_help(text: str) -> bool:
    return bool(text and _HELP_RE.search(text))


def looks_like_question(text: str) -> bool:
    return bool(text and _QUESTION_RE.search(text.strip()))


_NOISE_RES = [
    re.compile(r"\b\d+(\.\d+)?\s*(ms|s|sec|secs|seconds)\b"),  # timings
    re.compile(r"0x[0-9a-f]+"),  # addresses
    re.compile(r"\s+"),
]


def failure_signature(output: str) -> str:
    """Normalize check output so "the same failure again" compares equal even
    when timings or addresses differ."""
    sig = output[-1500:].lower()
    sig = _NOISE_RES[0].sub("<t>", sig)
    sig = _NOISE_RES[1].sub("<addr>", sig)
    sig = _NOISE_RES[2].sub(" ", sig)
    return sig.strip()


def _tail(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else "...(truncated)...\n" + text[-n:]


def _check_feedback(check_cmd: str, check: CheckResult, limits: Limits) -> str:
    if not check.output.strip():
        return (
            f"The check command `{check_cmd}` currently FAILS (exit code {check.exit_code}) "
            "and prints nothing. Run its parts yourself (without any `grep -q`) to see why."
        )
    return (
        f"The check command `{check_cmd}` currently FAILS (exit code {check.exit_code}). "
        f"Output:\n```\n{_tail(check.output, limits.check_output_chars)}\n```"
    )


def decide(
    report: TurnReport,
    check: CheckResult,
    check_cmd: str,
    state: MilestoneState,
    limits: Limits,
) -> Decision:
    """Update `state` with this turn and return what to do next."""
    state.turns += 1
    state.changed_files += report.changed_files

    # 1. The check is the only authority on "done". The agent's claim alone
    #    never counts; but a passing check does count only once real work has
    #    happened (or the agent says it's finished), so a check that was
    #    already green before we started can't end a milestone instantly.
    if check.passed and (report.end == TurnEnd.FINISHED or state.changed_files > 0):
        return Decision(Action.DONE, f"check `{check_cmd}` passes")

    # 2. Infrastructure errors: retry a few times, then start over.
    if report.end == TurnEnd.ERROR:
        state.consecutive_errors += 1
        if state.consecutive_errors >= limits.max_consecutive_errors:
            return _restart(state, limits, f"{state.consecutive_errors} errors in a row: {report.error[:300]}")
        return Decision(
            Action.NUDGE,
            f"error, retrying ({state.consecutive_errors}/{limits.max_consecutive_errors}): {report.error[:200]}",
            "Continue working on the task from where you left off.",
        )
    state.consecutive_errors = 0

    # 2b. The agent says it's stuck: ask the human now, don't make it keep going.
    if report.end in (TurnEnd.MESSAGE, TurnEnd.FINISHED) and asks_for_help(report.agent_text):
        return Decision(Action.ESCALATE, "the agent asked for help", report.agent_text)

    # 2c. Stagnation: busy, but nothing new. Measured across restarts.
    stalled = _track_progress(report, check, state, limits)
    if report.repeated_actions >= limits.repeat_actions:
        if not state.stagnation_warned:
            state.stagnation_warned = True
            return Decision(Action.NUDGE, f"agent repeated {report.repeated_actions} earlier actions with the same results",
                            _repeat_message(report, check_cmd, check, limits))
        return Decision(Action.ESCALATE, "the agent keeps repeating actions that give no new information",
                        _stagnation_summary(report, state))
    if stalled:
        return Decision(Action.ESCALATE, stalled, _stagnation_summary(report, state))

    # 3. Looping: a fresh conversation is the cure.
    if report.end == TurnEnd.STUCK:
        return _restart(state, limits, "agent was repeating itself (stuck detector)")
    if report.end == TurnEnd.LOOPING:
        # The context that produced the loop tends to loop again: start fresh.
        return _restart(state, limits, "model kept repeating itself while thinking; cut it off")

    # 4. Same failure over and over: the current context is poisoned.
    if not check.passed:
        sig = failure_signature(check.output)
        if sig and sig == state.last_failure_sig:
            state.same_failure_turns += 1
        else:
            state.last_failure_sig = sig
            state.same_failure_turns = 1
        if state.same_failure_turns >= limits.stagnant_turns:
            return _restart(state, limits, f"identical check failure for {state.same_failure_turns} turns")

    # 5. Long conversations degrade small models; cap them.
    if state.turns >= limits.max_turns_per_conversation:
        return _restart(state, limits, f"conversation reached {state.turns} turns")

    # 6. Check passes but no work and no finish: let it confirm.
    if check.passed:
        return Decision(
            Action.NUDGE,
            "check passes but agent has not finished",
            f"The check command `{check_cmd}` passes. If this milestone is fully "
            "complete, call finish. Otherwise continue.",
        )

    feedback = _check_feedback(check_cmd, check, limits)

    if report.end == TurnEnd.FINISHED:
        return Decision(
            Action.NUDGE,
            "agent claimed done but check fails",
            "You said you were finished, but you are NOT done. "
            f"{feedback}\nFix this, run the check again, and only call finish once it passes.",
        )

    if report.end == TurnEnd.MESSAGE:
        if looks_like_question(report.agent_text):
            state.questions += 1
            if state.questions > limits.max_questions_per_milestone:
                return Decision(
                    Action.ESCALATE,
                    f"agent keeps asking questions ({state.questions})",
                    report.agent_text,
                )
            return Decision(Action.NUDGE, "agent asked a question", f"{AUTONOMY_NUDGE}\n\n{feedback}")
        return Decision(
            Action.NUDGE,
            "agent stopped without finishing",
            f"Keep going, you are not done. {feedback}\n"
            "Work in small steps: edit one file, then rerun the check.",
        )

    # ITER_CAP
    if report.changed_files == 0:
        return _restart(state, limits, "a full turn of actions changed no files")
    return Decision(
        Action.NUDGE,
        "turn action cap reached, progress made",
        f"Good progress. Keep going. {feedback}",
    )


def _track_progress(report: TurnReport, check: CheckResult, state: MilestoneState, limits: Limits) -> str:
    """Update the no-progress counters; return why to escalate, or ""."""
    sig = "pass" if check.passed else failure_signature(check.output)
    if report.changed_files > 0 or sig != state.progress_sig:
        state.progress_sig = sig
        state.no_progress_turns = 0
        state.tokens_since_progress = 0
        return ""
    state.no_progress_turns += 1
    state.tokens_since_progress += report.tokens
    if state.no_progress_turns >= limits.stall_turns:
        return f"no progress for {state.no_progress_turns} turns (no file changed, same check result)"
    if limits.stall_tokens and state.tokens_since_progress >= limits.stall_tokens:
        return f"{state.tokens_since_progress:,} tokens spent without progress"
    return ""


def _repeat_message(report: TurnReport, check_cmd: str, check: CheckResult, limits: Limits) -> str:
    listed = "\n".join(f"- {e}" for e in report.repeat_examples[:6])
    state_of_check = "passes" if check.passed else f"fails (exit {check.exit_code})"
    return (
        "Stop and take stock. You repeated actions you had already run, and they returned the same results, "
        f"so they told you nothing new:\n{listed}\n"
        f"The check `{check_cmd}` currently {state_of_check}. Before doing anything else, write down in one short "
        "list: what you know for certain, which explanations your evidence has ruled out, and the ONE next "
        "experiment that would tell the remaining explanations apart. Then run only that. Do not re-run commands "
        "whose results you already have unless you changed something.\n"
        "If you cannot find a new experiment, reply starting with `STUCK:` followed by what you tried, what you "
        "found, and what you need; the user will help."
    )


def _stagnation_summary(report: TurnReport, state: MilestoneState) -> str:
    parts = [f"No progress for {state.no_progress_turns} turn(s)"
             + (f", {state.tokens_since_progress:,} tokens" if state.tokens_since_progress else "") + "."]
    if report.repeat_examples:
        parts.append("It kept repeating:\n" + "\n".join(f"- {e}" for e in report.repeat_examples[:6]))
    if report.agent_text:
        parts.append("Its latest reasoning:\n" + report.agent_text[-1500:])
    return "\n\n".join(parts)


def _restart(state: MilestoneState, limits: Limits, reason: str) -> Decision:
    state.restarts += 1
    if state.restarts > limits.max_restarts_per_milestone:
        return Decision(
            Action.ESCALATE,
            f"{reason}; already restarted {state.restarts - 1} times",
            reason,
        )
    state.new_conversation()
    return Decision(Action.RESTART, reason)
