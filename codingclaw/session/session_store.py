from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionStore:
    def __init__(self, workspace_root: Path, session_id: str | None = None) -> None:
        self.workspace_root = workspace_root
        self.session_id = session_id or uuid4().hex
        self.dir = workspace_root / ".codingclaw" / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.dir / f"{timestamp}_{self.session_id}.jsonl"
        self._append(
            {
                "type": "session",
                "version": 1,
                "id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": str(workspace_root),
            }
        )

    def append_message(self, message: dict) -> None:
        self._append(
            {
                "type": "message",
                "id": uuid4().hex[:8],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": message,
            }
        )

    def append_model_change(self, model: str) -> None:
        self._append(
            {
                "type": "model_change",
                "id": uuid4().hex[:8],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
            }
        )

    def _append(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
