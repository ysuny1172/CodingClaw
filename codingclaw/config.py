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
    context_window: int = 128_000
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000
    auto_compact: bool = True

    @classmethod
    def from_env(
        cls,
        workspace: str | Path | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 20,
        context_window: int | None = None,
        reserve_tokens: int | None = None,
        keep_recent_tokens: int | None = None,
        auto_compact: bool | None = None,
    ) -> "Config":
        return cls(
            workspace=Path(workspace or os.getcwd()).resolve(),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=(base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
            max_steps=max_steps,
            context_window=context_window or int(os.getenv("CODINGCLAW_CONTEXT_WINDOW", "128000")),
            reserve_tokens=reserve_tokens or int(os.getenv("CODINGCLAW_RESERVE_TOKENS", "16384")),
            keep_recent_tokens=keep_recent_tokens or int(os.getenv("CODINGCLAW_KEEP_RECENT_TOKENS", "20000")),
            auto_compact=auto_compact if auto_compact is not None else os.getenv("CODINGCLAW_AUTO_COMPACT", "1") not in {"0", "false", "False"},
        )
