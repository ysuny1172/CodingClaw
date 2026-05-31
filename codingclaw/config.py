from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    workspace: Path
    model: str
    base_url: str
    api_key: str | None
    max_steps: int = 20

    @classmethod
    def from_env(
        cls,
        workspace: str | Path | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 20,
    ) -> "Config":
        return cls(
            workspace=Path(workspace or os.getcwd()).resolve(),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=(base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
            max_steps=max_steps,
        )
