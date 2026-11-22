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
    # Sandbox
    server_image: str = "ghcr.io/openhands/agent-server:latest-python"
    iterations_per_turn: int = 40
    setup_timeout: int = 1800
    check_timeout: int = 900
    limits: Limits = field(default_factory=Limits)

    @property
    def openai_base_url(self) -> str:
        return self.ollama_url.rstrip("/") + "/v1"

    @classmethod
    def load(cls, path: Path | None) -> Settings:
        settings = cls()
        if path is None or not Path(path).exists():
            return settings
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        flat = {**data.get("ollama", {}), **data.get("agent", {})}
        known = {f.name for f in fields(cls)} - {"limits"}
        for key, value in flat.items():
            if key not in known:
                raise ValueError(f"unknown setting {key!r} in {path}")
            setattr(settings, key, value)
        limit_names = {f.name for f in fields(Limits)}
        for key, value in data.get("limits", {}).items():
            if key not in limit_names:
                raise ValueError(f"unknown limit {key!r} in {path}")
            setattr(settings.limits, key, value)
        return settings
