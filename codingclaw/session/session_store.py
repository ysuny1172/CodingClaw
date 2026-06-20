from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from codingclaw.agent.types import Message
from codingclaw.unicode import sanitize_json_value


class SessionStore:
    def __init__(self, workspace_root: Path, session_id: str | None = None, path: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.dir = self.workspace_root / ".codingclaw" / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid4().hex
        self.path = path or self._new_path(self.session_id)
        self.is_new = path is None
        self.unicode_repairs = 0
        if path is not None:
            self.unicode_repairs = self._repair_unicode_entries()
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
        entries = self.load_entries()
        latest_compaction_index = self._latest_compaction_index(entries)
        if latest_compaction_index is None:
            return [entry["message"] for entry in entries if entry.get("type") == "message" and isinstance(entry.get("message"), dict)]

        compaction = entries[latest_compaction_index]
        first_kept_message_id = self._first_kept_id(compaction)
        messages: list[Message] = [self._summary_message(str(compaction.get("summary") or ""), compaction.get("tokens_before"))]

        found_first_kept = False
        for entry in entries[:latest_compaction_index]:
            if entry.get("type") != "message" or not isinstance(entry.get("message"), dict):
                continue
            if entry.get("id") == first_kept_message_id:
                found_first_kept = True
            if found_first_kept:
                messages.append(entry["message"])

        for entry in entries[latest_compaction_index + 1 :]:
            if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
                messages.append(entry["message"])
        return messages

    def load_entries(self) -> list[dict]:
        entries: list[dict] = []
        if not self.path.exists():
            return entries
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(sanitize_json_value(json.loads(line)))
        return entries

    def active_message_entries(self) -> tuple[list[dict], dict | None]:
        entries = self.load_entries()
        latest_compaction_index = self._latest_compaction_index(entries)
        if latest_compaction_index is None:
            return [entry for entry in entries if entry.get("type") == "message" and isinstance(entry.get("message"), dict)], None

        compaction = entries[latest_compaction_index]
        first_kept_message_id = self._first_kept_id(compaction)
        active: list[dict] = []
        found_first_kept = False
        for entry in entries[:latest_compaction_index]:
            if entry.get("type") != "message" or not isinstance(entry.get("message"), dict):
                continue
            if entry.get("id") == first_kept_message_id:
                found_first_kept = True
            if found_first_kept:
                active.append(entry)
        if first_kept_message_id and not found_first_kept:
            active = [
                entry
                for entry in entries[latest_compaction_index + 1 :]
                if entry.get("type") == "message" and isinstance(entry.get("message"), dict)
            ]
            return active, compaction
        active.extend(
            entry
            for entry in entries[latest_compaction_index + 1 :]
            if entry.get("type") == "message" and isinstance(entry.get("message"), dict)
        )
        return active, compaction

    def load_raw_messages(self) -> list[Message]:
        messages: list[Message] = []
        if not self.path.exists():
            return messages
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = sanitize_json_value(json.loads(line))
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
                "message": sanitize_json_value(message),
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

    def append_compaction(
        self,
        summary: str,
        first_kept_message_id: str | None = None,
        tokens_before: int = 0,
        reason: str = "manual",
        *,
        first_kept_entry_id: str | None = None,
        details: dict | None = None,
        from_hook: bool = False,
    ) -> None:
        kept_id = first_kept_entry_id or first_kept_message_id
        if kept_id is None:
            raise ValueError("first_kept_entry_id is required")
        parent_id = self._latest_entry_id()
        self._append(
            {
                "type": "compaction",
                "id": uuid4().hex[:8],
                "parent_id": parent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "first_kept_entry_id": kept_id,
                "first_kept_message_id": kept_id,
                "tokens_before": tokens_before,
                "reason": reason,
                "details": details or {"read_files": [], "modified_files": []},
                "from_hook": from_hook,
            }
        )

    def _append(self, entry: dict) -> None:
        entry = sanitize_json_value(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _new_path(self, session_id: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.dir / f"{timestamp}_{session_id}.jsonl"

    def _load_session_id(self) -> str | None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = sanitize_json_value(json.loads(line))
            if entry.get("type") == "session":
                session_id = entry.get("id")
                return str(session_id) if session_id else None
        return None

    def _latest_compaction_index(self, entries: list[dict]) -> int | None:
        for index in range(len(entries) - 1, -1, -1):
            if entries[index].get("type") == "compaction":
                return index
        return None

    def _latest_entry_id(self) -> str | None:
        entries = self.load_entries()
        for entry in reversed(entries):
            entry_id = entry.get("id")
            if entry_id:
                return str(entry_id)
        return None

    def _first_kept_id(self, compaction: dict) -> object:
        return compaction.get("first_kept_entry_id") or compaction.get("first_kept_message_id")

    def _repair_unicode_entries(self) -> int:
        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: list[dict] = []
        repairs = 0
        for line in raw_lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            sanitized = sanitize_json_value(entry)
            if sanitized != entry:
                repairs += 1
            entries.append(sanitized)
        if repairs:
            temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            temporary_path.replace(self.path)
        return repairs

    def _summary_message(self, summary: str, tokens_before: object) -> Message:
        token_text = f" tokens_before={tokens_before}" if isinstance(tokens_before, int) else ""
        return {
            "role": "user",
            "content": f"<context_summary{token_text}>\n{summary}\n</context_summary>",
        }
