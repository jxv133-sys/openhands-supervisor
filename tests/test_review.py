import pytest

from supervisor import review
from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.log import RunLog
from supervisor.runner import Supervisor

def _sb(ws, project="/nonexistent"):
    """A prepared sandbox around a fake workspace (the fake ignores the wrapper)."""
    from supervisor.sandbox import Sandbox
    sb = Sandbox(ws, project, "/workspace/project")
    sb.ready = True
    return sb



class Result:
    def __init__(self, ok, out=""):
        self.exit_code = 0 if ok else 1
        self.stdout = out
        self.stderr = ""


class FakeWorkspace:
    """Commands listed in `failing` exit 1; everything else passes."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.ran = []

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.ran.append(cmd)
        return Result(not any(f in cmd for f in self.failing), "boom" if any(f in cmd for f in self.failing) else "ok")


def _supervisor(tmp_path, statuses=("done", "done")):
    contract = Contract.from_dict({"goal": "todo app with add and list", "milestones": [
        {"title": f"M{i}", "check": f"check{i}", "status": st} for i, st in enumerate(statuses, 1)]})
    contract.save(tmp_path)
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    return Supervisor(Settings(), tmp_path, contract, RunLog(tmp_path, echo=False))


def _reviewer(monkeypatch, reqs):
    monkeypatch.setattr(review, "ask_reviewer", lambda *a, **k: reqs)


REQS = [
    {"requirement": "add stores a task", "check": "req-add"},
    {"requirement": "list shows tasks", "check": "req-list"},
    {"requirement": "looks nice", "check": ""},
]


def test_all_requirements_pass(tmp_path, monkeypatch):
    sup = _supervisor(tmp_path)
    _reviewer(monkeypatch, REQS)
    assert sup._final_review(_sb(FakeWorkspace()), 1, may_add=True) is False
    report = review.load_report(tmp_path)
    assert report["verdict"] == "pass"
    assert [r["passed"] for r in report["requirements"]] == [True, True, None]
    assert len(sup.contract.milestones) == 2


def test_failing_requirement_becomes_milestone(tmp_path, monkeypatch):
    sup = _supervisor(tmp_path)
    _reviewer(monkeypatch, REQS)
    assert sup._final_review(_sb(FakeWorkspace(failing={"req-list"})), 1, may_add=True) is True
    added = sup.contract.milestones[-1]
    assert added.check == "req-list" and added.status == "pending" and added.source == "review"
    assert "list shows tasks" in added.title
    assert Contract.load(tmp_path).milestones[-1].check == "req-list"  # saved
    # Next round: same failure, but it was already tried as a milestone -> not added twice.
    added.status = "skipped"
    assert sup._final_review(_sb(FakeWorkspace(failing={"req-list"})), 2, may_add=True) is False
    assert len(sup.contract.milestones) == 3
    assert review.load_report(tmp_path)["verdict"] == "fail"


def test_last_round_only_reports(tmp_path, monkeypatch):
    sup = _supervisor(tmp_path)
    _reviewer(monkeypatch, REQS)
    assert sup._final_review(_sb(FakeWorkspace(failing={"req-add"})), 3, may_add=False) is False
    assert len(sup.contract.milestones) == 2
    assert review.load_report(tmp_path)["verdict"] == "fail"


def test_regression_reopens_milestone_and_skips_acceptance(tmp_path, monkeypatch):
    sup = _supervisor(tmp_path)

    def no_llm(*a, **k):
        raise AssertionError("fix regressions before reviewing requirements")

    monkeypatch.setattr(review, "ask_reviewer", no_llm)
    assert sup._final_review(_sb(FakeWorkspace(failing={"check1"})), 1, may_add=True) is True
    m1 = sup.contract.milestones[0]
    assert m1.status == "pending" and "broken again" in m1.notes[-1]
    assert review.load_report(tmp_path)["regressions"][0]["milestone"] == 1


def test_reviewer_failure_is_reported_not_raised(tmp_path, monkeypatch):
    sup = _supervisor(tmp_path)

    def down(*a, **k):
        raise ConnectionError("ollama unreachable")

    monkeypatch.setattr(review, "ask_reviewer", down)
    assert sup._final_review(_sb(FakeWorkspace()), 1, may_add=True) is False
    report = review.load_report(tmp_path)
    assert report["verdict"] == "error" and "unreachable" in report["error"]


def test_parse_review_tolerates_noise_and_dedupes():
    text = '<think>hmm</think>```json\n{"requirements": [{"requirement": "A", "check": "x"}, {"requirement": "a", "check": "y"}, {"requirement": "", "check": "z"}]}\n```'
    assert review.parse_review(text) == [{"requirement": "A", "check": "x"}]
    with pytest.raises(ValueError):
        review.parse_review('{"requirements": []}')


def test_gather_sources_skips_lockfiles_and_respects_budget(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "Cargo.lock").write_text("x" * 100)
    (tmp_path / "big.txt").write_text("y" * 5000)
    tree, sources = review.gather_sources(tmp_path, budget_chars=1000)
    assert "Cargo.lock" in tree and "src/main.rs" in tree
    assert "fn main" in sources and "xxxx" not in sources and "yyyy" not in sources
