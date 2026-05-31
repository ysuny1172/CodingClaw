class CodingClawError(Exception):
    """Base exception for CodingClaw."""


class ConfigError(CodingClawError):
    """Configuration is missing or invalid."""


class SandboxError(CodingClawError):
    """An operation was rejected by the sandbox."""


class ToolError(CodingClawError):
    """A tool failed or was called incorrectly."""
