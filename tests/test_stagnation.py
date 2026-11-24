"""Stagnation: the agent keeps busy but learns nothing new. The supervisor
warns once, then hands over to the human instead of letting it burn tokens."""

from types import SimpleNamespace

from supervisor.collector import TurnCollector
from supervisor.watchdog import (
    Action, CheckResult, Limits, MilestoneState, TurnEnd, TurnReport, asks_for_help, decide,
)

CHECK = "pytest -q"
FAIL = CheckResult(False, 1, "E   ModuleNotFoundError: No module named 'main'")
FAIL2 = CheckResult(False, 1, "E   AssertionError: 'bye' != 'hi'")


class ActionEvent(SimpleNamespace):
    pass


class ObservationEvent(SimpleNamespace):
    pass


class TerminalAction(SimpleNamespace):
    pass


def act(c, call_id, command, output):
    c.on_event(ActionEvent(tool_name="terminal", tool_call_id=call_id, action=TerminalAction(command=command)))
    c.on_event(ObservationEvent(tool_call_id=call_id, observation=SimpleNamespace(text=output)))


# --- collector: repetition means same action AND same result ------------------------

def test_repeats_counted_across_turns_and_conversations():
    seen = set()
    c = TurnCollector(); c.seen = seen
    act(c, "1", "git status", "nothing to commit")
    act(c, "2", "python -c 'import main'", "ModuleNotFoundError")
    assert c.report("finished").repeated_actions == 0

    c2 = TurnCollector(); c2.seen = seen          # a fresh conversation, same milestone
    act(c2, "3", "git status", "nothing to commit")
    act(c2, "4", "python -c 'import main'", "ModuleNotFoundError")
    r = c2.report("finished")
    assert r.repeated_actions == 2
    assert r.repeat_examples == ["terminal: git status", "terminal: python -c 'import main'"]


def test_same_command_with_new_result_is_not_a_repeat():
    c = TurnCollector()
    act(c, "1", "cargo test", "test result: FAILED. 3 passed; 1 failed")
    act(c, "2", "cargo test", "test result: ok. 4 passed; 0 failed")
    assert c.report("finished").repeated_actions == 0


def test_timing_noise_does_not_hide_a_repeat():
    c = TurnCollector()
    act(c, "1", "cargo test", "FAILED in 1.32s")
    act(c, "2", "cargo test", "FAILED in 1.87s")
    assert c.report("finished").repeated_actions == 1


def test_new_milestone_forgets():
    c = TurnCollector()
    act(c, "1", "ls", "a b")
    c.new_milestone(); c.reset()
    act(c, "2", "ls", "a b")
    assert c.report("finished").repeated_actions == 0


# --- decisions -------------------------------------------------------------------

def turn(end=TurnEnd.MESSAGE, text="", changed=0, repeated=0, tokens=0):
    return TurnReport(end, text, changed_files=changed, repeated_actions=repeated,
                      repeat_examples=["terminal: git status"] * min(repeated, 1), tokens=tokens)


def test_stuck_reply_escalates_immediately():
    s = MilestoneState()
    d = decide(turn(text="STUCK: tried A and B, both fail the same way; need the expected format"), FAIL, CHECK, s, Limits())
    assert d.action == Action.ESCALATE and "asked for help" in d.reason and "need the expected format" in d.message
    assert asks_for_help("I'm stuck on this.") and asks_for_help("I need your help with the config")
    assert not asks_for_help("Stuck detection is enabled in the config.") and not asks_for_help("")


def test_repetition_warns_once_then_escalates():
    s, lim = MilestoneState(), Limits()
    d = decide(turn(repeated=5), FAIL, CHECK, s, lim)
    assert d.action == Action.NUDGE and "git status" in d.message and "STUCK:" in d.message
    assert "ruled out" in d.message
    d = decide(turn(repeated=5, text="Maybe it's the path again."), FAIL, CHECK, s, lim)
    assert d.action == Action.ESCALATE and "repeating" in d.reason
    assert "git status" in d.message and "Maybe it's the path again." in d.message


def test_no_progress_escalates_instead_of_restarting_forever():
    s, lim = MilestoneState(), Limits(stall_turns=4, stagnant_turns=99, max_turns_per_conversation=99)
    actions = [decide(turn(), FAIL, CHECK, s, lim).action for _ in range(5)]
    # first turn sets the baseline; four turns later nothing moved
    assert actions[:4] == [Action.NUDGE] * 4 and actions[4] == Action.ESCALATE
    assert s.no_progress_turns == 4


def test_restarts_do_not_reset_the_clock_but_the_human_does():
    s, lim = MilestoneState(), Limits(stall_turns=3, stagnant_turns=99)
    for _ in range(3):
        decide(turn(), FAIL, CHECK, s, lim)
    s.new_conversation()
    assert decide(turn(), FAIL, CHECK, s, lim).action == Action.ESCALATE
    s.human_stepped_in()
    assert decide(turn(), FAIL, CHECK, s, lim).action != Action.ESCALATE


def test_file_changes_or_a_new_check_result_are_progress():
    s, lim = MilestoneState(), Limits(stall_turns=2, stagnant_turns=99)
    decide(turn(), FAIL, CHECK, s, lim)
    decide(turn(changed=1), FAIL, CHECK, s, lim)
    decide(turn(), FAIL2, CHECK, s, lim)
    assert s.no_progress_turns == 0
    decide(turn(), FAIL2, CHECK, s, lim)
    assert decide(turn(), FAIL2, CHECK, s, lim).action == Action.ESCALATE


def test_token_budget_without_progress():
    s, lim = MilestoneState(), Limits(stall_turns=99, stall_tokens=100_000, stagnant_turns=99)
    decide(turn(tokens=90_000), FAIL, CHECK, s, lim)            # baseline turn
    assert decide(turn(tokens=60_000), FAIL, CHECK, s, lim).action != Action.ESCALATE
    d = decide(turn(tokens=60_000), FAIL, CHECK, s, lim)
    assert d.action == Action.ESCALATE and "tokens" in d.reason
    off = MilestoneState(); lim0 = Limits(stall_turns=99, stall_tokens=0, stagnant_turns=99)
    assert all(decide(turn(tokens=10**7), FAIL, CHECK, off, lim0).action != Action.ESCALATE for _ in range(5))


def test_passing_check_still_wins():
    s = MilestoneState()
    assert decide(turn(end=TurnEnd.FINISHED, repeated=9), CheckResult(True, 0, "ok"), CHECK, s, Limits()).action == Action.DONE
