from __future__ import annotations

from pathlib import Path
from typing import Any

from codingclaw.sandbox import SandboxPolicy
from .base import ToolContext, ToolResult

MAX_READ_BYTES = 200_000


class ListFilesTool:
    name = "list_files"
    description = "List files and directories under a workspace path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path to list. Defaults to '.'."}
        },
        "required": [],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy = SandboxPolicy(context.workspace_root)
        path = policy.paths.resolve_read_path(args.get("path") or ".")
        if not path.exists():
            return ToolResult.failure("FileNotFoundError", f"Path does not exist: {path}")
        if not path.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Path is not a directory: {path}")
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            relative = child.resolve().relative_to(context.workspace_root.resolve())
            entries.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return ToolResult.success({"entries": entries})

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path to read."}
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy = SandboxPolicy(context.workspace_root)
        path = policy.paths.resolve_read_path(args["path"])
        if not path.exists():
            return ToolResult.failure("FileNotFoundError", f"File does not exist: {path}")
        if not path.is_file():
            return ToolResult.failure("IsADirectoryError", f"Path is not a file: {path}")
        raw = path.read_bytes()
        truncated = len(raw) > MAX_READ_BYTES
        text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return ToolResult.success(
            {
                "path": str(path.relative_to(context.workspace_root.resolve())).replace("\\", "/"),
                "content": text,
                "truncated": truncated,
                "bytes_read": min(len(raw), MAX_READ_BYTES),
            }
        )

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class WriteFileTool:
    name = "write_file"
    description = "Write a UTF-8 text file inside the workspace. Creates parent directories."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path to write."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy = SandboxPolicy(context.workspace_root)
        path: Path = policy.paths.resolve_write_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return ToolResult.success(
            {
                "path": str(path.relative_to(context.workspace_root.resolve())).replace("\\", "/"),
                "bytes_written": len(args["content"].encode("utf-8")),
            }
        )

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class EditFileTool:
    name = "edit_file"
    description = "Edit a UTF-8 text file by replacing an exact text segment. Defaults to exactly one replacement."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path to edit."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "expected_replacements": {
                "type": "integer",
                "description": "Expected number of replacements. Defaults to 1.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy = SandboxPolicy(context.workspace_root)
        path: Path = policy.paths.resolve_write_path(args["path"])
        if not path.exists():
            return ToolResult.failure("FileNotFoundError", f"File does not exist: {path}")
        if not path.is_file():
            return ToolResult.failure("IsADirectoryError", f"Path is not a file: {path}")

        old_text = args["old_text"]
        if old_text == "":
            return ToolResult.failure("ValueError", "old_text must not be empty")

        expected_replacements = args.get("expected_replacements", 1)
        if expected_replacements < 1:
            return ToolResult.failure("ValueError", "expected_replacements must be at least 1")

        content = path.read_text(encoding="utf-8")
        actual_replacements = content.count(old_text)
        if actual_replacements != expected_replacements:
            return ToolResult.failure(
                "ReplacementCountMismatch",
                f"Expected {expected_replacements} replacement(s), found {actual_replacements}.",
            )

        updated = content.replace(old_text, args["new_text"], expected_replacements)
        path.write_text(updated, encoding="utf-8")
        return ToolResult.success(
            {
                "path": str(path.relative_to(context.workspace_root.resolve())).replace("\\", "/"),
                "replacements": expected_replacements,
                "bytes_written": len(updated.encode("utf-8")),
            }
        )

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
