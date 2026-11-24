"""The web UI's Settings page: what each setting means, its current value and
where that value comes from, validation, and saving.

supervisor.toml is often mounted read-only (Docker), so settings changed in
the UI are saved to a separate overrides file of the same shape, applied on
top of supervisor.toml (see Settings.load). Only values that differ from
supervisor.toml (or the defaults) are written; "reset" deletes the file.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

from .config import Settings
from .watchdog import Limits

OVERRIDES_FILE = ".supervisor-settings.toml"

# key, group, label, help, type, extra (min/max/step/choices)
SCHEMA: list[dict] = [
    # --- Model & Ollama
    dict(key="ollama_url", group="Model & Ollama", label="Ollama address", type="str",
         help="Where Ollama runs, e.g. http://192.168.50.95:11434."),
    dict(key="model", group="Model & Ollama", label="Model", type="str",
         help="The Ollama model the agent, planner, review and Ideas chat use."),
    dict(key="max_input_tokens", group="Model & Ollama", label="Max prompt size (tokens)", type="int", min=2000, max=1_000_000,
         help="OpenHands summarizes older history to stay under this. Keep it well below the model's context window (Stats → Model shows it) or Ollama silently cuts prompts."),
    dict(key="max_output_tokens", group="Model & Ollama", label="Max reply size (tokens)", type="int", min=256, max=200_000,
         help="Longest single reply. Reasoning counts toward it."),
    dict(key="temperature", group="Model & Ollama", label="Temperature", type="float", min=0, max=2, step=0.05,
         help="Higher is more creative, lower more predictable. 0.6-0.7 suits coding."),
    dict(key="top_p", group="Model & Ollama", label="top_p", type="optfloat", min=0, max=1, step=0.01,
         help="Empty uses the model's default. Qwen's advice for thinking mode: 0.95."),
    dict(key="top_k", group="Model & Ollama", label="top_k", type="optint", min=1, max=1000,
         help="Empty uses the model's default. Qwen's advice: 20."),
    dict(key="presence_penalty", group="Model & Ollama", label="Presence penalty", type="float", min=0, max=2, step=0.1,
         help="0 is off. Up to 1.5 makes reasoning models repeat themselves less (can hurt code a little)."),
    dict(key="llm_timeout", group="Model & Ollama", label="Model call timeout (s)", type="int", min=30, max=7200,
         help="How long one model call may take before it counts as failed."),
    dict(key="stream", group="Model & Ollama", label="Stream replies", type="bool",
         help="Show the agent's reply as it's written. Turn off if your Ollama mishandles streaming with tool calls."),
    # --- Runs
    dict(key="server_image", group="Runs", label="Sandbox image", type="str",
         help="The OpenHands agent-server Docker image the agent works in."),
    dict(key="iterations_per_turn", group="Runs", label="Actions per turn", type="int", min=1, max=500,
         help="Tool calls the agent may make before the supervisor checks in."),
    dict(key="setup_timeout", group="Runs", label="Setup timeout (s)", type="int", min=10, max=36000,
         help="Per setup command (installing toolchains etc.)."),
    dict(key="check_timeout", group="Runs", label="Check timeout (s)", type="int", min=10, max=36000,
         help="Per check command."),
    dict(key="loop_guard", group="Runs", label="Cut off thinking loops", type="bool",
         help="Stop a reply that keeps repeating itself and restart the milestone fresh (needs streaming)."),
    dict(key="notes", group="Runs", label="Project notes (AGENTS.md)", type="bool",
         help="The agent keeps AGENTS.md up to date and sees it at the start of every milestone, so fresh conversations don't forget decisions."),
    # --- Checks & review
    dict(key="test_guard", group="Checks & review", label="Test guard (Rust/Python)", type="bool",
         help="A check that ran 0 tests, or passed because tests were skipped or deleted, counts as failed."),
    dict(key="final_review", group="Checks & review", label="Final review", type="bool",
         help="After the last milestone: re-run every check, run the whole test suite, and test each requirement of the goal."),
    dict(key="review_rounds", group="Checks & review", label="Review rounds", type="int", min=0, max=10,
         help="How often failing requirements may become new milestones. 0 = report only."),
    # --- Ideas chat
    dict(key="chat_web", group="Ideas chat", label="Read web pages", type="bool",
         help="Let the Ideas chat fetch public web pages (never addresses on your network)."),
    dict(key="chat_think", group="Ideas chat", label="Thinking", type="optbool",
         help="Model default, or force thinking on/off. Off makes thinking models answer faster."),
    # --- Limits
    dict(key="limits.max_turns_per_conversation", group="Limits", label="Turns per conversation", type="int", min=1, max=100,
         help="After this many turns the milestone restarts in a fresh conversation (long ones confuse small models)."),
    dict(key="limits.max_restarts_per_milestone", group="Limits", label="Restarts before asking you", type="int", min=0, max=50,
         help="Fresh starts the agent gets before the supervisor asks for your help."),
    dict(key="limits.max_consecutive_errors", group="Limits", label="Errors in a row", type="int", min=1, max=50,
         help="Model/sandbox errors in a row before the milestone restarts."),
    dict(key="limits.max_questions_per_milestone", group="Limits", label="Questions per milestone", type="int", min=0, max=50,
         help="How often the agent may ask questions (answered 'decide yourself') before it's escalated to you."),
    dict(key="limits.stagnant_turns", group="Limits", label="Same failure, turns", type="int", min=1, max=50,
         help="The identical check failure this many turns in a row restarts the milestone."),
    dict(key="limits.repeat_actions", group="Limits", label="Repeated actions before warning", type="int", min=1, max=100,
         help="Actions in one turn that repeat earlier ones with the same result. First time: the agent is told to "
              "take stock; again: you're asked for help."),
    dict(key="limits.stall_turns", group="Limits", label="Turns without progress", type="int", min=1, max=100,
         help="Turns in a row with no file changed and the same check result (across restarts) before you're asked."),
    dict(key="limits.stall_tokens", group="Limits", label="Tokens without progress", type="int", min=0, max=100_000_000,
         help="Model tokens spent without progress before you're asked. 0 = no limit."),
    dict(key="limits.check_output_chars", group="Limits", label="Check output shown to agent", type="int", min=200, max=50_000,
         help="Characters of failing check output sent back to the agent."),
]
_BY_KEY = {f["key"]: f for f in SCHEMA}


def _get(settings: Settings, key: str):
    if key.startswith("limits."):
        return getattr(settings.limits, key.split(".", 1)[1])
    return getattr(settings, key)


def coerce(field: dict, raw):
    """The value as its proper type, or ValueError with a message for the user."""
    t, label = field["type"], field["label"]
    if t in ("optfloat", "optint", "optbool") and (raw is None or raw == ""):
        return None
    if t in ("bool", "optbool"):
        if isinstance(raw, bool):
            return raw
        if str(raw).lower() in ("true", "1", "on", "yes"):
            return True
        if str(raw).lower() in ("false", "0", "off", "no"):
            return False
        raise ValueError(f"{label}: choose on or off")
    if t == "str":
        value = str(raw).strip()
        if not value:
            raise ValueError(f"{label} can't be empty")
        if field["key"] == "ollama_url" and not value.startswith(("http://", "https://")):
            raise ValueError(f"{label} must start with http:// or https://")
        return value
    try:
        value = int(raw) if t in ("int", "optint") else float(raw)
        if t in ("int", "optint") and float(raw) != int(raw):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a {'whole ' if t in ('int', 'optint') else ''}number") from None
    lo, hi = field.get("min"), field.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{label} must be between {lo:,} and {hi:,}")
    return value


def view(config_path: Path | None, overrides_path: Path) -> dict:
    """Every setting with its value, default, and where the value comes from."""
    defaults, base = Settings(), Settings.load(config_path)
    effective = Settings.load(config_path, overrides_path)
    changed = _read_overrides(overrides_path)
    out = []
    for f in SCHEMA:
        key = f["key"]
        source = "ui" if key in changed else ("file" if _get(base, key) != _get(defaults, key) else "default")
        out.append({**f, "value": _get(effective, key), "default": _get(defaults, key), "file_value": _get(base, key), "source": source})
    return {"fields": out, "overrides_path": str(overrides_path), "config_path": str(config_path) if config_path else None}


def _read_overrides(path: Path) -> dict:
    """Flat {key: value} of what the overrides file sets."""
    import tomllib

    if not Path(path).exists():
        return {}
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    flat = {**data.get("ollama", {}), **data.get("agent", {})}
    flat.update({f"limits.{k}": v for k, v in data.get("limits", {}).items()})
    return flat


def _toml_value(v) -> str:
    if v is None:
        return '""'  # "not set" (see Settings.apply)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))  # a TOML basic string is a JSON string for these values


def save(config_path: Path | None, overrides_path: Path, values: dict) -> Settings:
    """Validate `values` ({key: raw}) and store those that differ from
    supervisor.toml/defaults. Returns the new effective Settings."""
    base = Settings.load(config_path)
    current = _read_overrides(overrides_path)
    errors = []
    for key, raw in values.items():
        field = _BY_KEY.get(key)
        if field is None:
            errors.append(f"unknown setting {key!r}")
            continue
        try:
            value = coerce(field, raw)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if value == _get(base, key):
            current.pop(key, None)  # same as supervisor.toml: no override needed
        else:
            current[key] = value
    if errors:
        raise ValueError("; ".join(errors))
    lines = ["# Settings changed in the web UI. They apply on top of supervisor.toml.",
             "# Delete this file (or use \"Reset all\" in the UI) to go back to supervisor.toml.", ""]
    plain = {k: v for k, v in current.items() if not k.startswith("limits.")}
    limits = {k.split(".", 1)[1]: v for k, v in current.items() if k.startswith("limits.")}
    if plain:
        lines.append("[agent]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in sorted(plain.items())]
        lines.append("")
    if limits:
        lines.append("[limits]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in sorted(limits.items())]
        lines.append("")
    overrides_path = Path(overrides_path)
    if not current:
        overrides_path.unlink(missing_ok=True)
    else:
        tmp = overrides_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp, overrides_path)
    return Settings.load(config_path, overrides_path)  # also proves the file parses


def reset(config_path: Path | None, overrides_path: Path) -> Settings:
    Path(overrides_path).unlink(missing_ok=True)
    return Settings.load(config_path)


# Every Settings field (and limit) must be described above, so a new setting
# can't silently miss the page. Checked by the tests.
def undocumented() -> list[str]:
    keys = {f.name for f in fields(Settings)} - {"limits"}
    keys |= {f"limits.{f.name}" for f in fields(Limits)}
    return sorted(keys - set(_BY_KEY))
