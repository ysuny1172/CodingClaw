from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionContext:
    workspace_root: Path
    session_file: Path
    trace_file: Path
