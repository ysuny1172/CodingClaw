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
    first_kept_entry_id: str
    messages_to_summarize: list[Message]
    kept_messages: list[Message]
    tokens_before: int
    previous_summary: str | None = None
    details: dict | None = None
    is_split_turn: bool = False
    instructions: str | None = None

    @property
    def first_kept_message_id(self) -> str:
        return self.first_kept_entry_id


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    reason: CompactionReason
    details: dict | None = None
    from_hook: bool = False

    @property
    def first_kept_message_id(self) -> str:
        return self.first_kept_entry_id


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


CUSTOM_INSTRUCTIONS_PROMPT = """Additional user instructions for this compaction:
{instructions}

Follow these instructions while preserving the required summary structure."""


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
    instructions: str | None = None,
) -> CompactionPreparation | None:
    if not active_message_entries:
        return None

    first_kept_index, is_split_turn = _find_first_kept_index(active_message_entries, keep_recent_tokens)
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
    details = collect_compaction_details(messages_to_summarize, previous_compaction)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_message_id,
        messages_to_summarize=messages_to_summarize,
        kept_messages=kept_messages,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        details=details,
        is_split_turn=is_split_turn,
        instructions=instructions,
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
    if preparation.is_split_turn:
        prompt += "\n\nThe kept context starts in the middle of a large turn. Summarize the earlier part as turn-prefix context."
    if preparation.instructions:
        prompt += "\n\n" + CUSTOM_INSTRUCTIONS_PROMPT.format(instructions=preparation.instructions)

    response = llm.chat(
        model=model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    summary = response.content.strip()
    if not summary:
        raise RuntimeError("Compaction summarization returned an empty summary.")
    summary = append_details_to_summary(summary, preparation.details or {})
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


def collect_compaction_details(messages: list[Message], previous_compaction: dict | None = None) -> dict[str, list[str]]:
    read_files = _detail_list(previous_compaction, "read_files")
    modified_files = _detail_list(previous_compaction, "modified_files")

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name, args = _tool_call_name_args(call)
            path = _normalized_path(args.get("path")) if isinstance(args, dict) else None
            if not path:
                continue
            if name == "read_file":
                _append_unique(read_files, path)
            elif name in {"write_file", "edit_file"}:
                _append_unique(modified_files, path)

    return {"read_files": read_files, "modified_files": modified_files}


def append_details_to_summary(summary: str, details: dict) -> str:
    read_files = [str(item) for item in details.get("read_files") or []]
    modified_files = [str(item) for item in details.get("modified_files") or []]
    blocks: list[str] = []
    if read_files:
        blocks.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        blocks.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not blocks:
        return summary
    return summary.rstrip() + "\n\n" + "\n\n".join(blocks)


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


def _find_first_kept_index(entries: list[dict], keep_recent_tokens: int) -> tuple[int, bool]:
    accumulated = 0
    first_kept_index = 0
    for index in range(len(entries) - 1, -1, -1):
        message = entries[index]["message"]
        accumulated += estimate_message_tokens(message)
        first_kept_index = index
        if accumulated >= keep_recent_tokens:
            break

    first_kept_index = _avoid_orphan_tool(entries, first_kept_index)
    turn_start = _turn_start_index(entries, first_kept_index)
    turn_tokens = _entries_tokens(entries[turn_start:])
    if turn_start > 0 and turn_tokens <= keep_recent_tokens:
        return turn_start, False
    return first_kept_index, turn_start < first_kept_index


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


def _avoid_orphan_tool(entries: list[dict], index: int) -> int:
    while index > 0 and entries[index]["message"].get("role") == "tool":
        index -= 1
    return index


def _turn_start_index(entries: list[dict], index: int) -> int:
    current = index
    while current > 0:
        role = entries[current]["message"].get("role")
        if role == "user":
            return current
        previous_role = entries[current - 1]["message"].get("role")
        if previous_role == "user":
            return current - 1
        current -= 1
    return 0


def _entries_tokens(entries: list[dict]) -> int:
    return sum(estimate_message_tokens(entry["message"]) for entry in entries)


def _detail_list(compaction: dict | None, key: str) -> list[str]:
    details = compaction.get("details") if isinstance(compaction, dict) else None
    if not isinstance(details, dict):
        return []
    value = details.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _normalized_path(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.replace("\\", "/")


def _tool_call_name_args(call: object) -> tuple[str, dict]:
    if not isinstance(call, dict):
        return "", {}
    function = call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or call.get("name") or "")
        args = function.get("arguments") or {}
    else:
        name = str(call.get("name") or "")
        args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = {}
        args = parsed
    return name, args if isinstance(args, dict) else {}
