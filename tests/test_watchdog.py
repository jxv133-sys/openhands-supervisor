from supervisor.watchdog import (
    Action,
    CheckResult,
    Limits,
    MilestoneState,
    TurnEnd,
    TurnReport,
    decide,
    failure_signature,
    looks_like_question,
)

CHECK = "cargo build"
FAIL = CheckResult(False, 101, "error[E0425]: cannot find value `x` in this scope\n --> src/main.rs:3:5")
FAIL2 = CheckResult(False, 101, "error[E0308]: mismatched types\n --> src/lib.rs:9:1")
PASS = CheckResult(True, 0, "Finished dev profile in 2.31s")


def run(report, check, state=None, limits=None):
    state = state or MilestoneState()
    return decide(report, check, CHECK, state, limits or Limits()), state


def test_finish_with_passing_check_is_done():
    d, _ = run(TurnReport(TurnEnd.FINISHED, "all done", changed_files=0), PASS)
    assert d.action == Action.DONE


def test_passing_check_with_work_is_done_even_without_finish():
    d, _ = run(TurnReport(TurnEnd.MESSAGE, "", changed_files=2), PASS)
    assert d.action == Action.DONE


def test_passing_check_without_work_or_finish_asks_to_confirm():
    # A check that was green before anything happened must not end the milestone.
    d, _ = run(TurnReport(TurnEnd.MESSAGE, "Looking at the code.", changed_files=0), PASS)
    assert d.action == Action.NUDGE
    assert "call finish" in d.message


def test_claimed_done_but_check_fails_gets_error_back():
    d, _ = run(TurnReport(TurnEnd.FINISHED, "Done!", changed_files=3), FAIL)
    assert d.action == Action.NUDGE
    assert "NOT done" in d.message
    assert "E0425" in d.message


def test_question_gets_autonomy_nudge():
    d, state = run(TurnReport(TurnEnd.MESSAGE, "Should I also add tests?"), FAIL)
    assert d.action == Action.NUDGE
    assert "Do not ask for permission" in d.message
    assert state.questions == 1


def test_too_many_questions_escalates():
    state = MilestoneState()
    limits = Limits(max_questions_per_milestone=2, stagnant_turns=99)
    checks = [FAIL, FAIL2, FAIL]
    for i in range(2):
        d = decide(TurnReport(TurnEnd.MESSAGE, "Want me to continue?"), checks[i], CHECK, state, limits)
        assert d.action == Action.NUDGE
    d = decide(TurnReport(TurnEnd.MESSAGE, "Want me to continue?"), checks[2], CHECK, state, limits)
    assert d.action == Action.ESCALATE


def test_stopping_without_question_gets_keep_going():
    d, _ = run(TurnReport(TurnEnd.MESSAGE, "Writing all remaining modules now in one go:"), FAIL)
    assert d.action == Action.NUDGE
    assert "Keep going" in d.message


def test_stuck_restarts():
    d, state = run(TurnReport(TurnEnd.STUCK), FAIL)
    assert d.action == Action.RESTART
    assert state.restarts == 1
    assert state.turns == 0  # fresh conversation counters


def test_restart_budget_then_escalate():
    state = MilestoneState()
    limits = Limits(max_restarts_per_milestone=2)
    for _ in range(2):
        d = decide(TurnReport(TurnEnd.STUCK), FAIL, CHECK, state, limits)
        assert d.action == Action.RESTART
    d = decide(TurnReport(TurnEnd.STUCK), FAIL, CHECK, state, limits)
    assert d.action == Action.ESCALATE


def test_errors_retry_then_restart():
    state = MilestoneState()
    limits = Limits(max_consecutive_errors=3)
    for _ in range(2):
        d = decide(TurnReport(TurnEnd.ERROR, error="ReadTimeout"), FAIL, CHECK, state, limits)
        assert d.action == Action.NUDGE
    d = decide(TurnReport(TurnEnd.ERROR, error="ReadTimeout"), FAIL, CHECK, state, limits)
    assert d.action == Action.RESTART


def test_error_counter_resets_after_good_turn():
    state = MilestoneState()
    limits = Limits(max_consecutive_errors=2, stagnant_turns=99)
    decide(TurnReport(TurnEnd.ERROR, error="x"), FAIL, CHECK, state, limits)
    decide(TurnReport(TurnEnd.ITER_CAP, changed_files=1), FAIL, CHECK, state, limits)
    d = decide(TurnReport(TurnEnd.ERROR, error="x"), FAIL, CHECK, state, limits)
    assert d.action == Action.NUDGE


def test_same_failure_repeated_restarts():
    state = MilestoneState()
    limits = Limits(stagnant_turns=3)
    for _ in range(2):
        d = decide(TurnReport(TurnEnd.ITER_CAP, changed_files=1), FAIL, CHECK, state, limits)
        assert d.action == Action.NUDGE
    d = decide(TurnReport(TurnEnd.ITER_CAP, changed_files=1), FAIL, CHECK, state, limits)
    assert d.action == Action.RESTART
    assert "identical check failure" in d.reason


def test_changing_failures_are_not_stagnation():
    state = MilestoneState()
    limits = Limits(stagnant_turns=3)
    for check in [FAIL, FAIL2, FAIL, FAIL2]:
        d = decide(TurnReport(TurnEnd.ITER_CAP, changed_files=1), check, CHECK, state, limits)
        assert d.action == Action.NUDGE


def test_iter_cap_without_changes_restarts():
    d, _ = run(TurnReport(TurnEnd.ITER_CAP, changed_files=0), FAIL)
    assert d.action == Action.RESTART


def test_iter_cap_with_changes_continues():
    d, _ = run(TurnReport(TurnEnd.ITER_CAP, changed_files=4), FAIL)
    assert d.action == Action.NUDGE


def test_conversation_turn_cap_restarts():
    state = MilestoneState()
    limits = Limits(max_turns_per_conversation=3, stagnant_turns=99)
    checks = [FAIL, FAIL2, FAIL]
    actions = [
        decide(TurnReport(TurnEnd.ITER_CAP, changed_files=1), c, CHECK, state, limits).action for c in checks
    ]
    assert actions == [Action.NUDGE, Action.NUDGE, Action.RESTART]


def test_check_output_is_truncated_for_agent():
    big = CheckResult(False, 1, "noise\n" * 5000 + "REAL ERROR AT END")
    d, _ = run(TurnReport(TurnEnd.FINISHED), big, limits=Limits(check_output_chars=200))
    assert "REAL ERROR AT END" in d.message
    assert len(d.message) < 1000


def test_silent_failing_check_tells_agent_how_to_see_why():
    d, _ = run(TurnReport(TurnEnd.FINISHED), CheckResult(False, 1, "  \n"))
    assert "prints nothing" in d.message
    assert "grep -q" in d.message


def test_bash_wrapper_uses_login_shell_and_pipefail():
    from supervisor.runner import _bash

    wrapped = _bash("cargo test x | tee /dev/stderr | grep -q 'ok'")
    assert wrapped.startswith("bash -lc ")
    assert "set -o pipefail; cargo test x" in wrapped


def test_looks_like_question():
    assert looks_like_question("Should I proceed with the parser?")
    assert looks_like_question("I can add logging. Let me know if you want that.")
    assert looks_like_question("Would you like me to continue")
    assert not looks_like_question("Writing all remaining modules now in one go:")
    assert not looks_like_question("")


def test_failure_signature_ignores_timings():
    a = "error: build failed in 2.31s at 0xdeadbeef"
    b = "error: build failed in 12.9s at 0xcafe"
    assert failure_signature(a) == failure_signature(b)
    assert failure_signature("error A") != failure_signature("error B")
