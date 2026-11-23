"""Numbers for the web UI's Stats tab, computed from the run log and contract.

Pure functions over the records in .supervisor/log.jsonl, so they work for
old logs too: turns logged before durations/tokens were recorded get their
time estimated from timestamps, and tokens simply count as unknown.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from .contract import Contract

# A gap longer than this between two log records means the run was stopped and
# resumed later (or waited on you); don't count it as working time.
MAX_TURN_GAP = 2 * 3600
TIMELINE_LIMIT = 400


def _ts(rec: dict) -> float | None:
    try:
        return datetime.strptime(rec.get("ts", ""), "%Y-%m-%dT%H:%M:%S").timestamp()
    except (TypeError, ValueError):
        return None


def reason_group(reason: str) -> str:
    """'error, retrying (2/3): Timeout...' -> 'error, retrying'. Groups decisions
    that differ only in counts and details."""
    text = (reason or "").split(":")[0]
    text = re.sub(r"`[^`]*`", "", text)  # check commands
    text = re.sub(r"\(?\d+(/\d+)?\)?", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text or "(no reason)"


def _milestone_row(i: int, m) -> dict:
    return {
        "n": i, "title": m.title, "status": m.status, "turns": 0, "actions": 0, "seconds": 0.0,
        "restarts": 0, "escalations": 0, "guidance": 0, "files": 0, "tokens": 0,
        "checks_passed": 0, "first_turn_ts": None, "last_turn_ts": None,
    }


def compute(contract: Contract, records: list[dict]) -> dict:
    milestones = [_milestone_row(i, m) for i, m in enumerate(contract.milestones, 1)]
    by_title = {m["title"]: m for m in milestones}

    def row(n) -> dict | None:
        return milestones[n - 1] if isinstance(n, int) and 1 <= n <= len(milestones) else None

    ends: Counter = Counter()
    decisions: Counter = Counter()
    interventions: Counter = Counter()
    tools: Counter = Counter()
    files: Counter = Counter()
    turns: list[dict] = []
    totals = Counter()
    has_tokens = False
    prev_ts: float | None = None
    max_context = 0
    last_context = 0
    current = None  # milestone row of the latest milestone_start (escalations carry only a title)

    for rec in records:
        kind = rec.get("kind")
        ts = _ts(rec)
        if kind in ("milestone_start", "guidance"):
            prev_ts = ts  # the clock for the next turn starts here
            if kind == "milestone_start":
                current = row(rec.get("n"))
                totals["milestone_starts"] += 1
            else:
                totals["guidance"] += 1
                target = by_title.get(rec.get("title")) or current
                if target:
                    target["guidance"] += 1
            continue
        if kind == "escalate":
            totals["escalations"] += 1
            target = by_title.get(rec.get("title")) or current
            if target:
                target["escalations"] += 1
            continue
        if kind == "skip":
            totals["skips"] += 1
            continue
        if kind != "turn":
            continue

        m = row(rec.get("milestone"))
        seconds = None
        if rec.get("duration") is not None:
            seconds = float(rec.get("duration") or 0) + float(rec.get("check_seconds") or 0)
        elif ts is not None and prev_ts is not None and 0 <= ts - prev_ts <= MAX_TURN_GAP:
            seconds = ts - prev_ts
        prev_ts = ts

        actions = int(rec.get("actions") or 0)
        changed = rec.get("changed") or []
        passed = bool(rec.get("check_passed"))
        decision = rec.get("decision") or ""
        tokens = int(rec.get("prompt_tokens") or 0) + int(rec.get("completion_tokens") or 0)
        if "prompt_tokens" in rec:
            has_tokens = True
        max_context = max(max_context, int(rec.get("context_tokens") or 0))
        last_context = int(rec.get("context_tokens") or 0) or last_context

        totals["turns"] += 1
        totals["actions"] += actions
        totals["checks_passed"] += passed
        totals["seconds"] += seconds or 0
        totals["agent_seconds"] += float(rec.get("duration") or 0)
        totals["check_seconds"] += float(rec.get("check_seconds") or 0)
        totals["prompt_tokens"] += int(rec.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(rec.get("completion_tokens") or 0)
        totals["restarts"] += decision == "restart"
        ends[rec.get("end") or "?"] += 1
        decisions[decision or "?"] += 1
        if decision in ("nudge", "restart", "escalate"):
            interventions[f"{decision}: {reason_group(rec.get('reason', ''))}"] += 1
        for tool, count in (rec.get("tools") or {}).items():
            tools[tool] += int(count or 0)
        for f in changed:
            files[f] += 1

        if m:
            m["turns"] += 1
            m["actions"] += actions
            m["seconds"] += seconds or 0
            m["restarts"] += decision == "restart"
            m["files"] += len(changed)
            m["tokens"] += tokens
            m["checks_passed"] += passed
            m["first_turn_ts"] = m["first_turn_ts"] or rec.get("ts")
            m["last_turn_ts"] = rec.get("ts")

        turns.append({
            "ts": rec.get("ts"), "milestone": rec.get("milestone"), "actions": actions,
            "changed": len(changed), "passed": passed, "decision": decision, "end": rec.get("end"),
            "reason": rec.get("reason", ""), "seconds": None if seconds is None else round(seconds, 1),
            "tokens": tokens if "prompt_tokens" in rec else None,
        })

    n_turns = totals["turns"]
    return {
        "totals": {
            "turns": n_turns,
            "actions": totals["actions"],
            "seconds": round(totals["seconds"], 1),
            "agent_seconds": round(totals["agent_seconds"], 1),
            "check_seconds": round(totals["check_seconds"], 1),
            "checks_passed": totals["checks_passed"],
            "checks_failed": n_turns - totals["checks_passed"],
            "restarts": totals["restarts"],
            "escalations": totals["escalations"],
            "guidance": totals["guidance"],
            "skips": totals["skips"],
            "files_changed": len(files),
            "prompt_tokens": totals["prompt_tokens"] if has_tokens else None,
            "completion_tokens": totals["completion_tokens"] if has_tokens else None,
            "max_context": max_context or None,
            "last_context": last_context or None,
            "milestones_done": sum(m.status == "done" for m in contract.milestones),
            "milestones_skipped": sum(m.status == "skipped" for m in contract.milestones),
            "milestones_total": len(contract.milestones),
            "actions_per_turn": round(totals["actions"] / n_turns, 1) if n_turns else None,
        },
        "milestones": milestones,
        "ends": ends.most_common(),
        "decisions": decisions.most_common(),
        "interventions": interventions.most_common(12),
        "tools": tools.most_common(12),
        "files": files.most_common(12),
        "turns": turns[-TIMELINE_LIMIT:],
        "turns_omitted": max(0, len(turns) - TIMELINE_LIMIT),
    }
