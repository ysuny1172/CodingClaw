from __future__ import annotations

from typing import Any


REPLACEMENT_CHARACTER = "\ufffd"


def sanitize_text(text: str) -> str:
    """Replace isolated UTF-16 surrogate code points with valid Unicode."""
    if not any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return text
    return "".join(REPLACEMENT_CHARACTER if 0xD800 <= ord(char) <= 0xDFFF else char for char in text)


def sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize strings in JSON-compatible values."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_text(key) if isinstance(key, str) else key: sanitize_json_value(item)
            for key, item in value.items()
        }
    return value


def decode_utf8_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return sanitize_text(value)
