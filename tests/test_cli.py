from supervisor.cli import main
from supervisor.contract import Contract


def _write_contract(project):
    Contract.from_dict(
        {"goal": "g", "milestones": [{"title": "Scaffold", "instructions": "", "check": "true"}]}
    ).save(project)


def test_status_prints_contract(tmp_path, capsys):
    _write_contract(tmp_path)
    assert main(["status", str(tmp_path)]) == 0
    assert "Scaffold" in capsys.readouterr().out


class FakeSupervisor:
    calls = []

    def __init__(self, settings, project, contract, log, on_escalate="ask"):
        FakeSupervisor.calls.append((str(project), contract.goal, on_escalate))

    def run(self):
        return True


def _auto_env(monkeypatch, tmp_path, **env):
    monkeypatch.chdir(tmp_path)
    for key in ("PROJECT", "GOAL", "REVIEW_PLAN", "ON_ESCALATE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("supervisor.runner.Supervisor", FakeSupervisor)
    FakeSupervisor.calls = []


def _fake_plan(goal, *args, **kwargs):
    return Contract.from_dict({"goal": goal, "milestones": [{"title": "M1", "check": "true"}]})


def test_auto_without_goal_explains(tmp_path, monkeypatch, capsys):
    _auto_env(monkeypatch, tmp_path)
    assert main(["--config", "none.toml", "auto"]) == 1
    assert "GOAL" in capsys.readouterr().out
    assert FakeSupervisor.calls == []


def test_auto_plans_then_runs(tmp_path, monkeypatch):
    _auto_env(monkeypatch, tmp_path, PROJECT="demo", GOAL="build a thing")
    monkeypatch.setattr("supervisor.planner.plan", _fake_plan)
    assert main(["--config", "none.toml", "auto"]) == 0
    assert Contract.path_for(tmp_path / "demo").exists()
    assert FakeSupervisor.calls == [("demo", "build a thing", "auto")]  # unattended: resolve by itself


def test_auto_review_plan_stops_before_running(tmp_path, monkeypatch, capsys):
    _auto_env(monkeypatch, tmp_path, PROJECT="demo", GOAL="g", REVIEW_PLAN="1")
    monkeypatch.setattr("supervisor.planner.plan", _fake_plan)
    assert main(["--config", "none.toml", "auto"]) == 0
    assert FakeSupervisor.calls == []
    assert "REVIEW_PLAN" in capsys.readouterr().out


def test_auto_resumes_existing_plan_and_ignores_new_goal(tmp_path, monkeypatch, capsys):
    _auto_env(monkeypatch, tmp_path, PROJECT="demo", GOAL="a different goal", ON_ESCALATE="skip")
    _write_contract(tmp_path / "demo")

    def no_planning(*a, **k):
        raise AssertionError("must not re-plan an existing project")

    monkeypatch.setattr("supervisor.planner.plan", no_planning)
    assert main(["--config", "none.toml", "auto"]) == 0
    assert FakeSupervisor.calls == [("demo", "g", "skip")]
    assert "different goal" in capsys.readouterr().out


def test_auto_rejects_bad_escalate_mode(tmp_path, monkeypatch):
    _auto_env(monkeypatch, tmp_path, GOAL="g", ON_ESCALATE="panic")
    assert main(["--config", "none.toml", "auto"]) == 1


def test_config_path_from_env(tmp_path, monkeypatch, capsys):
    # A config with a typo must be picked up via SUPERVISOR_CONFIG and rejected.
    bad = tmp_path / "bad.toml"
    bad.write_text("[agent]\nnot_a_setting = 1\n")
    monkeypatch.setenv("SUPERVISOR_CONFIG", str(bad))
    _write_contract(tmp_path)
    try:
        main(["status", str(tmp_path)])
    except ValueError as exc:
        assert "not_a_setting" in str(exc)
    else:
        raise AssertionError("config from SUPERVISOR_CONFIG was not loaded")
