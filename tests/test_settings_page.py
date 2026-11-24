import pytest

from supervisor import settings_page as sp
from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.web import ApiError, App


@pytest.fixture
def paths(tmp_path):
    cfg = tmp_path / "supervisor.toml"
    cfg.write_text('[agent]\nmodel = "qwen3:14b"\n[limits]\nstall_turns = 5\n')
    return cfg, tmp_path / sp.OVERRIDES_FILE


def field(view, key):
    return next(f for f in view["fields"] if f["key"] == key)


def test_every_setting_is_in_the_schema():
    assert sp.undocumented() == []


def test_view_reports_sources(paths):
    cfg, ov = paths
    v = sp.view(cfg, ov)
    assert field(v, "model")["source"] == "file" and field(v, "model")["value"] == "qwen3:14b"
    assert field(v, "limits.stall_turns")["value"] == 5
    assert field(v, "limits.repeat_actions")["source"] == "default"


def test_save_keeps_only_real_changes_and_reset_removes_them(paths):
    cfg, ov = paths
    s = sp.save(cfg, ov, {"limits.stall_turns": "7", "model": "qwen3:14b"})
    assert s.limits.stall_turns == 7 and "model" not in ov.read_text() and "stall_turns = 7" in ov.read_text()
    assert field(sp.view(cfg, ov), "limits.stall_turns")["source"] == "ui"
    assert Settings.load(cfg, ov).limits.stall_turns == 7           # what a new run loads
    sp.save(cfg, ov, {"limits.stall_turns": 5})                     # back to the file's value
    assert not ov.exists()
    sp.save(cfg, ov, {"final_review": False}); sp.reset(cfg, ov)
    assert not ov.exists() and Settings.load(cfg, ov).final_review is True


@pytest.mark.parametrize("key, raw, msg", [
    ("limits.stall_turns", "x", "whole number"),
    ("limits.stall_turns", 2.5, "whole number"),
    ("limits.stall_turns", 0, "between"),
    ("ollama_url", "ollama:11434", "http"),
    ("model", "  ", "empty"),
    ("final_review", "maybe", "on or off"),
    ("nope", 1, "unknown"),
])
def test_bad_values_are_refused_and_nothing_is_written(paths, key, raw, msg):
    cfg, ov = paths
    with pytest.raises(ValueError, match=msg):
        sp.save(cfg, ov, {key: raw, "limits.stall_turns" if key != "limits.stall_turns" else "model": 9 if key != "limits.stall_turns" else "m"})
    assert not ov.exists()


def test_optional_values_can_be_cleared(paths):
    cfg, ov = paths
    sp.save(cfg, ov, {"chat_think": False})
    assert Settings.load(cfg, ov).chat_think is False
    sp.save(cfg, ov, {"chat_think": ""})
    assert Settings.load(cfg, ov).chat_think is None


def test_api_saves_applies_and_rejects(paths, monkeypatch):
    cfg, _ = paths
    root = cfg.parent / "projects"; root.mkdir()
    app = App(Settings.load(cfg), root, config_path=cfg)
    monkeypatch.setattr("supervisor.modelinfo._get", lambda *a, **k: {"models": [{"name": "qwen3:14b"}]})
    assert app.settings_view()["models"] == ["qwen3:14b"]
    app.save_settings({"limits.stall_tokens": 0})
    assert app.settings.limits.stall_tokens == 0
    with pytest.raises(ApiError):
        app.save_settings({"limits.stall_tokens": -1})
    with pytest.raises(ApiError):
        app.save_settings({})
    app.reset_settings()
    assert app.settings.limits.stall_tokens == 400_000


def test_project_api_returns_notes(tmp_path):
    Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]}).save(tmp_path / "p")
    app = App(Settings(), tmp_path)
    assert app.project("p")["notes"] == ""
    (tmp_path / "p" / "AGENTS.md").write_text("# Notes\nBuild with cargo.\n")
    assert "Build with cargo" in app.project("p")["notes"]
