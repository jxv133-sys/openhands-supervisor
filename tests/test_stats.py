from supervisor.contract import Contract
from supervisor.stats import compute, reason_group


def _contract():
    return Contract.from_dict({"goal": "g", "milestones": [
        {"title": "A", "check": "true", "status": "done"},
        {"title": "B", "check": "true"}]})


def turn(ts, n, **kw):
    rec = {"ts": ts, "kind": "turn", "milestone": n, "end": "message", "actions": 3, "changed": ["a.rs"],
           "check_passed": False, "decision": "nudge", "reason": "agent stopped without finishing",
           "tools": {"terminal": 2, "file_editor": 1}}
    rec.update(kw)
    return rec


def test_compute_totals_and_milestones():
    records = [
        {"ts": "2026-01-01T10:00:00", "kind": "milestone_start", "n": 1, "title": "A"},
        turn("2026-01-01T10:01:00", 1),  # old record: time estimated from timestamps
        turn("2026-01-01T10:02:00", 1, check_passed=True, decision="done", end="finished",
             duration=30, check_seconds=5, prompt_tokens=1000, completion_tokens=200, context_tokens=9000),
        {"ts": "2026-01-01T10:03:00", "kind": "milestone_start", "n": 2, "title": "B"},
        turn("2026-01-01T13:30:00", 2, decision="restart", reason="identical check failure for 3 turns"),
        {"ts": "2026-01-01T13:31:00", "kind": "escalate", "title": "B", "reason": "x"},
        {"ts": "2026-01-01T13:35:00", "kind": "guidance", "title": "B", "note": "hint"},
    ]
    s = compute(_contract(), records)
    t = s["totals"]
    assert t["turns"] == 3 and t["actions"] == 9
    assert t["seconds"] == 60 + 35  # 3.5h gap is a resume, not work
    assert t["checks_passed"] == 1 and t["checks_failed"] == 2
    assert t["restarts"] == 1 and t["escalations"] == 1 and t["guidance"] == 1
    assert t["prompt_tokens"] == 1000 and t["max_context"] == 9000
    assert t["files_changed"] == 1
    a, b = s["milestones"]
    assert (a["turns"], a["checks_passed"], b["escalations"], b["restarts"]) == (2, 1, 1, 1)
    assert dict(s["tools"]) == {"terminal": 6, "file_editor": 3}
    assert ("restart: identical check failure for turns", 1) in s["interventions"]


def test_compute_without_token_data_reports_unknown():
    s = compute(_contract(), [turn("2026-01-01T10:01:00", 1)])
    assert s["totals"]["prompt_tokens"] is None
    assert s["turns"][0]["tokens"] is None


def test_reason_group_drops_counts_and_details():
    assert reason_group("error, retrying (2/3): Timeout talking to model") == "error, retrying"
    assert reason_group("check `cargo build` passes") == "check passes"
