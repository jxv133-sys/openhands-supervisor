"""Nobody answers the help card: the supervisor diagnoses and retries by
itself, then skips, so an unattended run never sits idle for hours."""

import threading
import time

import pytest

from supervisor import rescue
from supervisor.config import Settings
from supervisor.contract import SKIPPED, Contract
from supervisor.log import RunLog
from supervisor.runner import Supervisor
from supervisor.watchdog import Action, CheckResult, Decision, MilestoneState
from supervisor.web import ApiError, App, Job

FAIL = CheckResult(False, 101, "error[E0425]: cannot find value `store` in this scope\n --> src/main.rs:12:5")
STUCK = Decision(Action.ESCALATE, "no progress for 4 turns", "It kept repeating:\n- terminal: cargo build")


def sup(tmp_path, attempts=2):
    s = Settings(); s.auto_resolve_attempts = attempts
    c = Contract.from_dict({"goal": "todo CLI", "milestones": [{"title": "Add command", "check": "cargo test add"}]})
    c.save(tmp_path)
    (tmp_path / "src").mkdir(); (tmp_path / "src" / "main.rs").write_text("fn main() { store.add(); }\n")
    (tmp_path / "AGENTS.md").write_text("Build: cargo build\n")
    return Supervisor(s, tmp_path, c, RunLog(tmp_path, echo=False), on_escalate="auto")


def test_parse_rescue():
    r = rescue.parse_rescue('<think>hm</think>```json\n{"diagnosis": "store is never created", "guidance": "add let store = Store::load()", "give_up": false}\n```')
    assert r.guidance.startswith("add let store") and not r.give_up
    assert rescue.parse_rescue('{"diagnosis": "needs network", "guidance": "", "give_up": true}').give_up
    with pytest.raises(ValueError):
        rescue.parse_rescue('{"diagnosis": "x", "guidance": "", "give_up": false}')
    with pytest.raises(ValueError):
        rescue.parse_rescue("I think you should...")


def test_prompt_has_the_evidence():
    p = rescue.rescue_prompt(goal="G", title="T", instructions="I", check="cargo test add", reason="R",
                             details="kept repeating cargo build", check_output=FAIL.output, tried=["use clap"],
                             notes="Build: cargo build", tree="src/main.rs", sources="--- src/main.rs ---\nfn main")
    for part in ("cargo test add", "kept repeating", "E0425", "- use clap", "Build: cargo build", "fn main", "never change"):
        assert part in p


def test_diagnosis_becomes_guidance_and_retries(tmp_path, monkeypatch):
    seen = {}

    def fake(prompt, url, model, timeout):
        seen["prompt"] = prompt
        return rescue.Rescue("`store` is used before it is created", "Create it with Store::load() at the top of main.")

    monkeypatch.setattr(rescue, "ask_rescuer", fake)
    s = sup(tmp_path); m = s.contract.milestones[0]; state = MilestoneState(restarts=3)
    assert s._escalate(m, STUCK, FAIL, state) is False                 # retried, not skipped
    assert "Store::load()" in m.notes[-1] and "user was away" in m.notes[-1]
    assert state.restarts == 0 and state.auto_resolves == 1           # fresh budget, attempt counted
    assert "fn main() { store.add(); }" in seen["prompt"] and "Build: cargo build" in seen["prompt"]
    kinds = [r.get("outcome") for r in RunLog(tmp_path, echo=False).read_all() if r["kind"] == "auto_resolve"]
    assert kinds == ["guidance"]


def test_attempts_run_out_then_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(rescue, "ask_rescuer", lambda *a: rescue.Rescue("d", "g"))
    s = sup(tmp_path, attempts=2); m = s.contract.milestones[0]; state = MilestoneState()
    assert s._escalate(m, STUCK, FAIL, state) is False
    assert s._escalate(m, STUCK, FAIL, state) is False
    assert s._escalate(m, STUCK, FAIL, state) is True
    assert m.status == SKIPPED and Contract.load(tmp_path).milestones[0].status == SKIPPED


def test_give_up_skips_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(rescue, "ask_rescuer", lambda *a: rescue.Rescue("the check needs internet access", "", True))
    s = sup(tmp_path); m = s.contract.milestones[0]
    assert s._escalate(m, STUCK, FAIL, MilestoneState()) is True and m.status == SKIPPED


def test_model_down_still_retries_with_generic_guidance(tmp_path, monkeypatch):
    def down(*a):
        raise ConnectionError("ollama unreachable")
    monkeypatch.setattr(rescue, "ask_rescuer", down)
    s = sup(tmp_path); m = s.contract.milestones[0]
    assert s._escalate(m, STUCK, FAIL, MilestoneState()) is False
    assert "different one" in m.notes[-1]


# --- web: the help card times out into an automatic answer --------------------------

def job(tmp_path, after):
    Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]}).save(tmp_path / "p")
    s = Settings(); s.auto_resolve_after = after
    return Job(App(s, tmp_path), "p")


def ask(j):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("answer", j._escalation(
        Contract.load(j.app.project_path("p")).milestones[0], STUCK, FAIL)))
    t.start()
    for _ in range(100):
        if j.escalation:
            break
        time.sleep(0.01)
    return t, out


def test_unanswered_help_resolves_automatically(tmp_path):
    j = job(tmp_path, after=1)
    t, out = ask(j)
    assert j.escalation["auto_at"] == pytest.approx(j.escalation["since"] + 1)
    t.join(5)
    assert out["answer"] == ("auto", "") and j.escalation is None


def test_wait_for_me_holds_and_an_answer_still_works(tmp_path):
    j = job(tmp_path, after=1)
    t, out = ask(j)
    j.answer("hold", "")
    assert j.escalation["auto_at"] is None
    time.sleep(1.3)
    assert t.is_alive()                               # still waiting for the human
    j.answer("guidance", "use serde")
    t.join(5)
    assert out["answer"] == ("guidance", "use serde")
    with pytest.raises(ApiError):
        j.answer("hold", "")                          # nothing to hold now


def test_zero_means_always_wait(tmp_path):
    j = job(tmp_path, after=0)
    t, out = ask(j)
    assert j.escalation["auto_at"] is None
    j.answer("auto", "")                              # "Resolve now"
    t.join(5)
    assert out["answer"] == ("auto", "")
