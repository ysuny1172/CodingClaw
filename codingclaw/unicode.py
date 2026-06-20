from __future__ import annotations

from typing import Any


REPLACEMENT_CHARACTER = "\ufffd"


def sanitize_text(text: str) -> str:
    """Combine valid surrogate pairs and replace isolated surrogates."""
    if not any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return text

    sanitized: list[str] = []
    index = 0
    while index < len(text):
        code = ord(text[index])
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(text):
            next_code = ord(text[index + 1])
            if 0xDC00 <= next_code <= 0xDFFF:
                scalar = 0x10000 + ((code - 0xD800) << 10) + (next_code - 0xDC00)
                sanitized.append(chr(scalar))
                index += 2
                continue
        if 0xD800 <= code <= 0xDFFF:
            sanitized.append(REPLACEMENT_CHARACTER)
        else:
            sanitized.append(text[index])
        index += 1
    return "".join(sanitized)


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
