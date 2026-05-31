from __future__ import annotations

from pathlib import Path

from codingclaw.errors import SandboxError


class PathSandbox:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def resolve_read_path(self, path: str) -> Path:
        resolved = self._resolve(path)
        self._assert_inside(resolved)
        return resolved

    def resolve_write_path(self, path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self.workspace_root / target
        parent = target.parent.resolve()
        self._assert_inside(parent)
        resolved = parent / target.name
        self._assert_inside(resolved)
        return resolved

    def _resolve(self, path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self.workspace_root / target
        return target.resolve()

    def _assert_inside(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.workspace_root)
        except ValueError as error:
            raise SandboxError(f"Path escapes workspace: {path}") from error
