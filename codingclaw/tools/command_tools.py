from __future__ import annotations

import subprocess
from typing import Any

from codingclaw.sandbox import SandboxPolicy
from codingclaw.unicode import decode_utf8_output
from .base import ToolContext, ToolResult

MAX_OUTPUT_CHARS = 40_000


class RunCommandTool:
    name = "run_command"
    description = "Run a shell command in the workspace with a timeout and basic sandbox checks."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout in seconds. Defaults to 30 and is capped at 120.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy = SandboxPolicy(context.workspace_root)
        command = args["command"]
        policy.commands.assert_allowed(command)
        timeout = min(max(int(args.get("timeout_seconds") or 30), 1), 120)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=context.workspace_root,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            return ToolResult.failure("TimeoutExpired", f"Command timed out after {timeout}s: {error}")

        stdout = decode_utf8_output(completed.stdout)
        stderr = decode_utf8_output(completed.stderr)
        stdout_truncated = len(stdout) > MAX_OUTPUT_CHARS
        stderr_truncated = len(stderr) > MAX_OUTPUT_CHARS
        return ToolResult.success(
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": stdout[:MAX_OUTPUT_CHARS],
                "stderr": stderr[:MAX_OUTPUT_CHARS],
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        )

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
