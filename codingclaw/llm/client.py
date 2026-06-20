from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from codingclaw.agent.types import AssistantResponse, Message, TokenUsage, ToolCall
from codingclaw.errors import ConfigError
from codingclaw.unicode import sanitize_json_value


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible Chat Completions client using only stdlib."""

    def __init__(self, *, base_url: str, api_key: str | None, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AssistantResponse:
        if not self.api_key:
            raise ConfigError("OPENAI_API_KEY is required for the real LLM client.")

        request_messages: list[Message] = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload = sanitize_json_value(payload)

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = sanitize_json_value(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error

        choice = raw.get("choices", [{}])[0]
        message = choice.get("message") or {}
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [ToolCall.from_openai(call, index=i) for i, call in enumerate(raw_tool_calls)]
        return AssistantResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            raw=raw,
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage.from_openai(raw.get("usage")),
        )
