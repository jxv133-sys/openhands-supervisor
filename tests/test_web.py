import threading
import time

import pytest

from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.web import ApiError, App, EventBus, Job


def _project(root, name="demo"):
    Contract.from_dict({"goal": "g", "milestones": [{"title": "M1", "check": "true"}]}).save(root / name)
    return name


def test_event_bus_answers_immediately_when_client_is_ahead():
    bus = EventBus()
    bus.publish({"kind": "x"})
    start = time.monotonic()
    events, last = bus.since(500, timeout=5)  # client remembers ids from before a restart
    assert time.monotonic() - start < 1
    assert events == [] and last == 1


def test_event_bus_returns_new_events():
    bus = EventBus()
    bus.publish({"kind": "a"})
    bus.publish({"kind": "b"})
    events, last = bus.since(1, timeout=0)
    assert [e["kind"] for e in events] == ["b"] and last == 2


class FakeSupervisor:
    instances = []

    def __init__(self, settings, path, contract, log, on_escalate):
        self.on_escalate = on_escalate
        self._stop = threading.Event()
        self._milestone_n = 1
        self.stop_requested = False
        FakeSupervisor.instances.append(self)

    def request_stop(self):
        self.stop_requested = True
        self._stop.set()

    def run(self):
        return not self._stop.is_set()


def test_stop_while_starting_is_not_lost(tmp_path, monkeypatch):
    app = App(Settings(), tmp_path)
    name = _project(tmp_path)
    monkeypatch.setattr("supervisor.runner.Supervisor", FakeSupervisor)
    job = Job(app, name)
    job.stop()  # before the thread even created the supervisor
    job.thread.start()
    job.thread.join(5)
    assert FakeSupervisor.instances[-1].stop_requested
    assert job.status == "stopped"
    assert job.ended_at is not None


def _escalating_job(tmp_path):
    app = App(Settings(), tmp_path)
    job = Job(app, _project(tmp_path))
    milestone = Contract.load(tmp_path / "demo").milestones[0]
    decision = type("D", (), {"reason": "stuck", "message": "stuck"})()
    check = type("C", (), {"output": "boom"})()
    results = []
    t = threading.Thread(target=lambda: results.append(job._escalation(milestone, decision, check)))
    t.start()
    for _ in range(100):
        if job.escalation is not None:
            break
        time.sleep(0.01)
    return job, t, results, (milestone, decision, check)


def test_double_answer_is_rejected_and_not_queued(tmp_path):
    job, t, results, args = _escalating_job(tmp_path)
    job.answer("guidance", "try harder")
    with pytest.raises(ApiError):
        job.answer("skip", "")
    t.join(5)
    assert results == [("guidance", "try harder")]
    assert job._answers.empty()


def test_stop_requested_skips_the_question(tmp_path):
    app = App(Settings(), tmp_path)
    job = Job(app, _project(tmp_path))
    job._stop_requested = True
    assert job._escalation(None, None, None) == ("stop", "")
    assert job.escalation is None


def test_save_rejects_unknown_status(tmp_path):
    app = App(Settings(), tmp_path)
    name = _project(tmp_path)
    with pytest.raises(ApiError):
        app.save(name, {"goal": "g", "milestones": [{"title": "M", "check": "true", "status": "wip"}]})


def test_file_view_blocks_path_traversal(tmp_path):
    app = App(Settings(), tmp_path / "projects")
    name = _project(app.root)
    (tmp_path / "secret.txt").write_text("x")
    with pytest.raises(ApiError):
        app.file(name, "../../secret.txt")


def test_run_end_message_counts_skipped(tmp_path, monkeypatch):
    class Skipper(FakeSupervisor):
        def __init__(self, settings, path, contract, log, on_escalate):
            super().__init__(settings, path, contract, log, on_escalate)
            self.contract = contract

        def run(self):
            self.contract.milestones[0].status = "skipped"
            return False

    app = App(Settings(), tmp_path)
    monkeypatch.setattr("supervisor.runner.Supervisor", Skipper)
    job = Job(app, _project(tmp_path))
    job.thread.start()
    job.thread.join(5)
    assert job.status == "finished"
    assert "1 skipped" in job.message


def test_delete_keep_code_removes_only_supervisor_state(tmp_path):
    app = App(Settings(), tmp_path)
    name = _project(tmp_path)
    (tmp_path / name / "main.rs").write_text("fn main() {}")
    app.delete(name, keep_code=True)
    assert (tmp_path / name / "main.rs").exists()
    assert not (tmp_path / name / ".supervisor").exists()
    assert [p["name"] for p in app.state()["projects"]] == []


def test_delete_everything(tmp_path):
    app = App(Settings(), tmp_path)
    name = _project(tmp_path)
    (tmp_path / name / "src").mkdir()
    (tmp_path / name / "src" / "main.rs").write_text("fn main() {}")
    app.delete(name, keep_code=False)
    assert not (tmp_path / name).exists()
    with pytest.raises(ApiError):
        app.delete(name, keep_code=False)  # already gone


def test_delete_refuses_running_project_and_symlinks(tmp_path, monkeypatch):
    app = App(Settings(), tmp_path / "projects")
    name = _project(app.root)
    monkeypatch.setattr(app, "_active_job_for", lambda n: True)
    with pytest.raises(ApiError, match="stop the run"):
        app.delete(name, keep_code=False)
    monkeypatch.undo()
    outside = tmp_path / "elsewhere"
    _project(tmp_path, "elsewhere")
    (app.root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ApiError, match="outside"):
        app.delete("link", keep_code=False)
    assert (outside / ".supervisor").exists()  # the link's target is untouched
