import json

import pytest

from supervisor.contract import DONE, Contract
from supervisor.planner import parse_plan
from supervisor.prompts import handoff_note, milestone_prompt

RAW = {
    "goal": "a CLI todo app in Rust",
    "setup": ["command -v cargo || true", "  "],
    "milestones": [
        {"title": "Scaffold", "instructions": "cargo new", "check": "cargo build"},
        {"title": "Add command", "instructions": "", "check": "cargo test add"},
    ],
}


def test_roundtrip(tmp_path):
    c = Contract.from_dict(RAW)
    assert c.setup == ["command -v cargo || true"]  # blank entries dropped
    c.milestones[0].status = DONE
    c.milestones[1].notes.append("use clap")
    c.save(tmp_path)
    loaded = Contract.load(tmp_path)
    assert loaded == c
    assert [m.title for m in loaded.pending()] == ["Add command"]
    assert [m.title for m in loaded.completed()] == ["Scaffold"]


def test_validation():
    with pytest.raises(ValueError):
        Contract.from_dict({"goal": "", "milestones": RAW["milestones"]})
    with pytest.raises(ValueError):
        Contract.from_dict({"goal": "x", "milestones": []})
    with pytest.raises(ValueError):
        Contract.from_dict({"goal": "x", "milestones": [{"title": "a", "check": ""}]})


def test_parse_plan_tolerates_noise():
    body = {k: v for k, v in RAW.items() if k != "goal"}
    text = "<think>planning...</think>\nHere you go:\n```json\n" + json.dumps(body) + "\n```"
    c = parse_plan("my goal", text)
    assert c.goal == "my goal"
    assert len(c.milestones) == 2


def test_parse_plan_rejects_garbage():
    with pytest.raises(ValueError):
        parse_plan("g", "I cannot help with that")


def test_prompt_contents():
    c = Contract.from_dict(RAW)
    c.milestones[0].status = DONE
    c.milestones[1].notes.append("use clap")
    p = milestone_prompt(c, c.milestones[1], handoff="HANDOFF")
    assert "milestone 2 of 2: Add command" in p
    assert "- Scaffold" in p  # completed list
    assert "cargo test add" in p
    assert "use clap" in p
    assert "ONE file per step" in p
    assert p.endswith("HANDOFF")


def test_handoff_note():
    note = handoff_note("stuck", "src/main.rs", "cargo build", "error here")
    assert "stuck" in note and "src/main.rs" in note and "error here" in note


def test_parse_milestone_and_draft_prompt(monkeypatch):
    from supervisor import planner

    assert planner.parse_milestone('<think>x</think>{"title": "Add list", "instructions": "i", "check": "cargo test test_list"}') == {
        "title": "Add list", "instructions": "i", "check": "cargo test test_list"}
    with pytest.raises(ValueError):
        planner.parse_milestone('{"title": "t", "instructions": "", "check": ""}')
    sent = {}

    def fake_post(url, payload, timeout):
        sent["prompt"] = payload["messages"][0]["content"]
        return {"message": {"content": '{"title": "T", "instructions": "I", "check": "true"}'}}

    monkeypatch.setattr(planner, "_post_json", fake_post)
    c = Contract.from_dict(RAW)
    out = planner.draft_milestone(c, "", "src/main.rs", "http://x", "m", title="Add done", instructions="mark done")
    assert out["check"] == "true"
    p = sent["prompt"]
    assert "Title: Add done" in p and "cargo test add" in p and "src/main.rs" in p and "Rules for checks" in p
