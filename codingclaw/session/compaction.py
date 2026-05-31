from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from codingclaw.agent.types import LLMClient, Message
from codingclaw.config import Config
from codingclaw.llm.tokens import estimate_text_tokens


CompactionReason = Literal["manual", "threshold", "overflow"]


@dataclass(frozen=True)
class CompactionPreparation:
    first_kept_message_id: str
    messages_to_summarize: list[Message]
    kept_messages: list[Message]
    tokens_before: int
    previous_summary: str | None = None


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_message_id: str
    tokens_before: int
    reason: CompactionReason


SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""


SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use the same structured format as the existing summary."""


def should_compact(context_tokens: int, config: Config) -> bool:
    if not config.auto_compact:
        return False
    return context_tokens > config.context_window - config.reserve_tokens


def prepare_compaction(
    active_message_entries: list[dict],
    *,
    previous_compaction: dict | None,
    tokens_before: int,
    keep_recent_tokens: int,
) -> CompactionPreparation | None:
    if not active_message_entries:
        return None

    first_kept_index = _find_first_kept_index(active_message_entries, keep_recent_tokens)
    if first_kept_index <= 0:
        return None

    messages_to_summarize = [entry["message"] for entry in active_message_entries[:first_kept_index]]
    kept_messages = [entry["message"] for entry in active_message_entries[first_kept_index:]]
    if not messages_to_summarize or not kept_messages:
        return None

    first_kept_message_id = str(active_message_entries[first_kept_index]["id"])
    previous_summary = None
    if previous_compaction:
        previous_summary = str(previous_compaction.get("summary") or "")

    return CompactionPreparation(
        first_kept_message_id=first_kept_message_id,
        messages_to_summarize=messages_to_summarize,
        kept_messages=kept_messages,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
    )


def generate_summary(
    *,
    llm: LLMClient,
    model: str,
    preparation: CompactionPreparation,
    reserve_tokens: int,
) -> str:
    conversation = serialize_messages(preparation.messages_to_summarize)
    prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if preparation.previous_summary:
        prompt += f"<previous-summary>\n{preparation.previous_summary}\n</previous-summary>\n\n"
        prompt += UPDATE_SUMMARIZATION_PROMPT
    else:
        prompt += SUMMARIZATION_PROMPT

    response = llm.chat(
        model=model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    summary = response.content.strip()
    if not summary:
        raise RuntimeError("Compaction summarization returned an empty summary.")
    return summary[: max(reserve_tokens * 16, 8_000)]


def serialize_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            content = message.get("content")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            content = message.get("content")
            if content:
                parts.append(f"[Assistant]: {content}")
            tool_calls = message.get("tool_calls")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {_compact_json(tool_calls)}")
        elif role == "tool":
            name = message.get("name") or "tool"
            content = str(message.get("content") or "")
            parts.append(f"[Tool result: {name}]: {_truncate(content, 2_000)}")
        else:
            parts.append(f"[{role}]: {message}")
    return "\n\n".join(parts)


def is_context_overflow_error(error: Exception) -> bool:
    text = str(error).lower()
    needles = [
        "context_length_exceeded",
        "maximum context length",
        "context window",
        "too many tokens",
        "context limit",
    ]
    return any(needle in text for needle in needles)


def _find_first_kept_index(entries: list[dict], keep_recent_tokens: int) -> int:
    accumulated = 0
    first_kept_index = 0
    for index in range(len(entries) - 1, -1, -1):
        message = entries[index]["message"]
        accumulated += estimate_message_tokens(message)
        first_kept_index = index
        if accumulated >= keep_recent_tokens:
            break

    while first_kept_index > 0 and entries[first_kept_index]["message"].get("role") == "tool":
        first_kept_index -= 1
    return first_kept_index


def estimate_message_tokens(message: Message) -> int:
    role = str(message.get("role", ""))
    content = str(message.get("content", ""))
    total = estimate_text_tokens(role) + estimate_text_tokens(content) + 4
    if "tool_calls" in message:
        total += estimate_text_tokens(_compact_json(message["tool_calls"]))
    if "name" in message:
        total += estimate_text_tokens(str(message["name"]))
    return total


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"
