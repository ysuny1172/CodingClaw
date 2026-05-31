from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class TraceLogger:
    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid4().hex
        self.dir = root / ".codingclaw" / "traces"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.run_id}.jsonl"

    def log(self, event: dict) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
