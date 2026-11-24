"""Settings, loaded from supervisor.toml (every key optional)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .watchdog import Limits


@dataclass
class Settings:
    # Ollama
    ollama_url: str = "http://192.168.50.95:11434"
    model: str = "qwen-abl:latest"
    # Keep input well under the model's num_ctx (48K for qwen-abl) so there is
    # always room to answer. OpenHands' condenser summarizes history to stay under this.
    max_input_tokens: int = 34000
    max_output_tokens: int = 8000
    temperature: float = 0.7
    llm_timeout: int = 900
    # Stream the model's output token by token, so the web UI can show it live.
    stream: bool = True
    # Rust/Python: a check that ran 0 tests, or passed because the agent skipped
    # or deleted tests, doesn't count; the final review runs the whole suite.
    test_guard: bool = True
    # Ideas chat: may it fetch public web pages? And Ollama's `think` option for
    # it (None = the model's default; false = faster answers from thinking models).
    chat_web: bool = True
    chat_think: bool | None = None
    # Cut off a reply that keeps repeating itself mid-generation (needs stream).
    loop_guard: bool = True
    # Sampling knobs that make reasoning models loop less. None/0 = Ollama's default.
    # Qwen's advice for thinking mode: top_p 0.95, top_k 20, presence_penalty up to 1.5.
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float = 0.0
    # Sandbox
    server_image: str = "ghcr.io/openhands/agent-server:latest-python"
    iterations_per_turn: int = 40
    setup_timeout: int = 1800
    check_timeout: int = 900
    # After the last milestone: re-run every check, then test each requirement
    # of the goal end to end; failures become new milestones. Rounds of that.
    final_review: bool = True
    review_rounds: int = 2
    # The agent keeps AGENTS.md (project notes) up to date and sees it at the
    # start of every milestone, so fresh conversations don't forget decisions.
    notes: bool = True
    limits: Limits = field(default_factory=Limits)

    @property
    def openai_base_url(self) -> str:
        return self.ollama_url.rstrip("/") + "/v1"

    @classmethod
    def load(cls, path: Path | None, overrides: Path | None = None) -> Settings:
        """supervisor.toml, then the settings changed in the web UI (a second
        TOML file of the same shape, kept where it's writable)."""
        settings = cls()
        for source in (path, overrides):
            if source is not None and Path(source).exists():
                settings.apply(tomllib.loads(Path(source).read_text(encoding="utf-8")), source)
        return settings

    def apply(self, data: dict, source="settings") -> None:
        flat = {**data.get("ollama", {}), **data.get("agent", {})}
        known = {f.name for f in fields(type(self))} - {"limits"}
        for key, value in flat.items():
            if key not in known:
                raise ValueError(f"unknown setting {key!r} in {source}")
            # TOML has no null: "" means "not set" for optional settings (top_p...)
            setattr(self, key, None if value == "" else value)
        limit_names = {f.name for f in fields(Limits)}
        for key, value in data.get("limits", {}).items():
            if key not in limit_names:
                raise ValueError(f"unknown limit {key!r} in {source}")
            setattr(self.limits, key, value)
