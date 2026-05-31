from __future__ import annotations

import re

from codingclaw.errors import SandboxError


class CommandSandbox:
    def __init__(self) -> None:
        self._blocked_patterns = [
            r"\brm\s+(-[^\s]*r[^\s]*f|-[^\s]*f[^\s]*r)\b",
            r"\bdel\s+(/s|/q)\b",
            r"\brmdir\s+(/s|/q)\b",
            r"\bformat\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bgit\s+reset\s+--hard\b",
            r"\bgit\s+clean\s+-[^\s]*f",
            r"\bcurl\b.*\|\s*(sh|bash|powershell|pwsh)\b",
            r"\bwget\b.*\|\s*(sh|bash|powershell|pwsh)\b",
            r"\binvoke-webrequest\b.*\|\s*(iex|invoke-expression)\b",
            r"\biwr\b.*\|\s*(iex|invoke-expression)\b",
            r"\bpowershell\b.*\s-enc(odedcommand)?\b",
            r"\bpwsh\b.*\s-enc(odedcommand)?\b",
        ]

    def assert_allowed(self, command: str) -> None:
        normalized = command.strip().lower()
        if not normalized:
            raise SandboxError("Empty command is not allowed")
        if "\x00" in normalized:
            raise SandboxError("Command contains NUL byte")
        for pattern in self._blocked_patterns:
            if re.search(pattern, normalized):
                raise SandboxError(f"Command rejected by sandbox policy: {command}")
