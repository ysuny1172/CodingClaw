from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from codingclaw.agent.types import Message


class SessionStore:
    def __init__(self, workspace_root: Path, session_id: str | None = None, path: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.dir = self.workspace_root / ".codingclaw" / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid4().hex
        self.path = path or self._new_path(self.session_id)
        self.is_new = path is None
        if path is not None:
            self.session_id = self._load_session_id() or self.session_id
            return
        self._append(
            {
                "type": "session",
                "version": 1,
                "id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": str(workspace_root),
            }
        )

    @classmethod
    def open(cls, workspace_root: Path, path: str | Path) -> "SessionStore":
        workspace_root = workspace_root.resolve()
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = workspace_root / resolved
        resolved = resolved.resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError as error:
            raise ValueError(f"Session path escapes workspace: {resolved}") from error
        if not resolved.exists():
            raise FileNotFoundError(f"Session file does not exist: {resolved}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Session path is not a file: {resolved}")
        return cls(workspace_root, path=resolved)

    @classmethod
    def open_latest(cls, workspace_root: Path) -> "SessionStore":
        session_dir = workspace_root.resolve() / ".codingclaw" / "sessions"
        candidates = sorted(session_dir.glob("*.jsonl"), key=lambda path: (path.stat().st_mtime, path.name))
        if not candidates:
            raise FileNotFoundError(f"No session files found in: {session_dir}")
        return cls.open(workspace_root, candidates[-1])

    def load_messages(self) -> list[Message]:
        messages: list[Message] = []
        if not self.path.exists():
            return messages
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            message = entry.get("message")
            if entry.get("type") == "message" and isinstance(message, dict):
                messages.append(message)
        return messages

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

    def _new_path(self, session_id: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.dir / f"{timestamp}_{session_id}.jsonl"

    def _load_session_id(self) -> str | None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "session":
                session_id = entry.get("id")
                return str(session_id) if session_id else None
        return None
