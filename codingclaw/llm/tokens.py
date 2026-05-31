from __future__ import annotations

import json
from typing import Any

from codingclaw.agent.types import Message


def estimate_prompt_tokens(*, system_prompt: str, messages: list[Message], tools: list[dict[str, Any]]) -> int:
    token_count = estimate_text_tokens(system_prompt) if system_prompt else 0
    token_count += 2

    for message in messages:
        token_count += 4
        token_count += estimate_text_tokens(str(message.get("role", "")))
        content = message.get("content")
        if content is not None:
            token_count += estimate_text_tokens(str(content))
        if "name" in message:
            token_count += estimate_text_tokens(str(message["name"]))
        if "tool_call_id" in message:
            token_count += estimate_text_tokens(str(message["tool_call_id"]))
        if "tool_calls" in message:
            token_count += estimate_text_tokens(_compact_json(message["tool_calls"]))

    if tools:
        token_count += estimate_text_tokens(_compact_json(tools))

    return max(token_count, 0)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return (ascii_count + 3) // 4 + non_ascii_count


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
