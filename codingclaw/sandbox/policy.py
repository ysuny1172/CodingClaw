from __future__ import annotations

from pathlib import Path

from .commands import CommandSandbox
from .paths import PathSandbox


class SandboxPolicy:
    def __init__(self, workspace_root: Path) -> None:
        self.paths = PathSandbox(workspace_root)
        self.commands = CommandSandbox()
