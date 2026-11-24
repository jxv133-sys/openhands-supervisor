"""Milestone checks and final-review checks run the same way, and the review
tells environment problems apart from project regressions.

Uses the real Supervisor code paths with real bash (LocalWorkspace runs
commands with subprocess, as the sandbox's agent-server does). A temp
"system root" stands in for the container filesystem outside the project.
"""

import threading

import pytest

from supervisor import review
from supervisor.config import Settings
from supervisor.contract import DONE, Contract
from supervisor.log import RunLog
from supervisor.runner import Supervisor, completion_reminder
from supervisor.sandbox import AGENT_SETUP, Sandbox, SandboxError
from supervisor.watchdog import Action, Decision
from supervisor.web import App, Job
from test_sandbox import LocalWorkspace


@pytest.fixture
def world(tmp_path, monkeypatch):
    proj, sysroot = tmp_path / "project", tmp_path / "sysroot"
    proj.mkdir(); sysroot.mkdir()
    (proj / "main.py").write_text('def hello():\n    return "hi"\n')
    monkeypatch.setattr(review, "ask_reviewer", lambda *a, **k: [])  # keep the review offline
    return proj, sysroot


def heredoc_check(sysroot):
    # The shape of check that "regressed": copy the project from a path
    # outside it into a temp dir and import it from a Python heredoc.
    return (f'd=$(mktemp -d) && cp -r {sysroot}/app/. "$d" && cd "$d" && python3 - <<\'EOF\'\n'
            'import main\nassert main.hello() == "hi"\nEOF')


def make(proj, sysroot, check, ws=None, test_guard=False):
    s = Settings(); s.test_guard = test_guard
    c = Contract.from_dict({"goal": "g", "milestones": [{"title": "Greeting", "check": check}]})
    c.save(proj)
    sup = Supervisor(s, proj, c, RunLog(proj, echo=False))
    sb = Sandbox(ws or LocalWorkspace(), proj, str(proj), watch=[str(sysroot)], skip=[])
    return sup, sb


def pass_milestone(sup, sb):
    """What _run_milestone does when a check passes: record the sandbox's
    changes outside the project on the milestone (see runner)."""
    m = sup.contract.milestones[0]
    res = sup._check(sb, m.check)
    assert res.passed, res.output
    m.status, m.env_changes = DONE, sb.drift()
    sup.contract.save(sup.project_dir)
    return m


def test_milestone_and_review_run_the_identical_command(world):
    proj, sysroot = world
    ws = LocalWorkspace()
    sup, sb = make(proj, sysroot, "python3 -c 'import main; assert main.hello() == \"hi\"'", ws)
    sb.prepare([])
    pass_milestone(sup, sb)
    milestone_cmd = [c for c in ws.commands if "import main" in c][-1]
    ws.commands.clear()
    assert sup._final_review(sb, 1, may_add=True) is False
    review_cmd = [c for c in ws.commands if "import main" in c][-1]
    assert milestone_cmd == review_cmd  # same cwd, shell, wrapper: same result
    assert review.load_report(proj)["verdict"] == "pass"


def test_fresh_sandbox_without_agent_made_link_is_not_a_regression(world):
    proj, sysroot = world
    sup, sb = make(proj, sysroot, heredoc_check(sysroot))
    sb.prepare([])
    (sysroot / "app").symlink_to(proj)        # the agent fixes things by hand in its sandbox...
    m = pass_milestone(sup, sb)
    assert m.env_changes == [f"added {sysroot}/app (symlink -> {proj})"]

    (sysroot / "app").unlink()                # ...and a later run starts a fresh sandbox
    fresh = Sandbox(LocalWorkspace(), proj, str(proj), watch=[str(sysroot)], skip=[]); fresh.prepare([])
    assert sup._final_review(fresh, 1, may_add=True) is False
    rep = review.load_report(proj)
    assert rep["verdict"] == "environment" and rep["regressions"] == []
    assert rep["environment"][0]["kind"] == "mismatch" and "app" in rep["environment"][0]["missing"][0]
    assert sup.contract.milestones[0].status == DONE  # not re-opened as if the code broke


def test_a_real_regression_is_still_a_regression(world):
    proj, sysroot = world
    sup, sb = make(proj, sysroot, "python3 -c 'import main; assert main.hello() == \"hi\"'")
    sb.prepare([])
    pass_milestone(sup, sb)
    (proj / "main.py").write_text('def hello():\n    return "bye"\n')  # a later change breaks it
    assert sup._final_review(sb, 1, may_add=True) is True
    rep = review.load_report(proj)
    assert rep["verdict"] == "regressions" and rep["environment"] == []
    assert sup.contract.milestones[0].status == "pending"


def test_declared_setup_makes_the_fresh_sandbox_equivalent(world):
    proj, sysroot = world
    sup, sb = make(proj, sysroot, heredoc_check(sysroot))
    (proj / ".openhands").mkdir()
    (proj / AGENT_SETUP).write_text(f"ln -sfn {proj} {sysroot}/app\n")
    sb.prepare([])
    pass_milestone(sup, sb)
    (sysroot / "app").unlink()                # fresh sandbox: setup.sh recreates it
    fresh = Sandbox(LocalWorkspace(), proj, str(proj), watch=[str(sysroot)], skip=[]); fresh.prepare([])
    assert sup._final_review(fresh, 1, may_add=True) is False
    assert review.load_report(proj)["verdict"] == "pass"


# --- initialization failures are infrastructure errors, never regressions ---------

def test_unmounted_project_is_reported_as_infrastructure(world, tmp_path):
    proj, sysroot = world
    sup, _ = make(proj, sysroot, "true")
    empty = tmp_path / "not-mounted"; empty.mkdir()
    sb = Sandbox(LocalWorkspace(), proj, str(empty), watch=[str(sysroot)], skip=[])
    with pytest.raises(SandboxError) as exc:
        sup._prepare(sb)
    sup._infra(exc.value)
    assert "not the project folder" in sup.infra_error
    events = [r for r in RunLog(proj, echo=False).read_all() if r["kind"] == "infra_error"]
    assert events and "not the project folder" in events[0]["error"]


def test_review_in_an_unusable_sandbox_reports_environment_not_regressions(world, tmp_path):
    proj, sysroot = world
    sup, sb = make(proj, sysroot, "python3 -c 'import main'")
    sb.prepare([])
    pass_milestone(sup, sb)
    (proj / sb.token).unlink()  # the mount went away before the review
    assert sup._final_review(sb, 1, may_add=True) is False
    rep = review.load_report(proj)
    assert rep["verdict"] == "environment" and rep["regressions"] == []
    assert rep["environment"][0]["kind"] == "infra"
    assert sup.contract.milestones[0].status == DONE


def test_failed_setup_never_reaches_checks(world):
    proj, sysroot = world
    ws = LocalWorkspace()
    sup, sb = make(proj, sysroot, "python3 -c 'import main'", ws)
    sup.contract.setup = ["exit 3"]
    with pytest.raises(SandboxError, match="setup command failed"):
        sup._prepare(sb)
    assert not any("import main" in c for c in ws.commands)


class InfraSupervisor:
    def __init__(self, settings, path, contract, log, on_escalate):
        self._stop = threading.Event(); self._milestone_n = 0; self.context = None
        self.infra_error = "the project folder isn't visible in the sandbox"

    def request_stop(self):
        pass

    def run(self):
        return False


def test_web_job_reports_sandbox_problem_not_project_failure(tmp_path, monkeypatch):
    Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]}).save(tmp_path / "p")
    monkeypatch.setattr("supervisor.runner.Supervisor", InfraSupervisor)
    job = Job(App(Settings(), tmp_path), "p")
    job.thread.start(); job.thread.join(5)
    assert job.status == "failed" and job.message.startswith("Sandbox problem; nothing was judged")


# --- the one combined "finish up" reminder --------------------------------------------

DONE_D = Decision(Action.DONE, "check passes")


def test_completion_reminder_combines_notes_and_environment():
    r = completion_reminder(DONE_D, notes=True, drift=["added /app (symlink -> /workspace/project)"])
    assert r.action == Action.NUDGE and "AGENTS.md" in r.message and ".openhands/setup.sh" in r.message
    assert "1." in r.message and "2." in r.message
    assert completion_reminder(DONE_D, notes=False, drift=[]) is None
    assert completion_reminder(Decision(Action.NUDGE, "x"), notes=True, drift=["x"]) is None
    only_env = completion_reminder(DONE_D, notes=False, drift=["added /app"])
    assert "AGENTS.md" not in only_env.message and "setup.sh" in only_env.message
