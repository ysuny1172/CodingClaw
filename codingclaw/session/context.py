from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from codingclaw.agent.events import make_event
from codingclaw.agent.types import AgentEvent, LLMClient, Message, TokenUsage
from codingclaw.config import Config
from codingclaw.hooks import BeforeCompactionContext, HookRegistry
from codingclaw.llm.tokens import estimate_prompt_tokens
from codingclaw.session.compaction import (
    CompactionReason,
    CompactionResult,
    append_details_to_summary,
    generate_summary,
    is_context_overflow_error,
    prepare_compaction,
    should_compact,
)
from codingclaw.session.session_store import SessionStore


@dataclass(frozen=True)
class SessionContext:
    workspace_root: Path
    session_file: Path
    trace_file: Path


ContextUsageSource = Literal["provider", "estimate", "unknown"]


@dataclass(frozen=True)
class ContextUsage:
    tokens: int | None
    context_window: int
    percent: float | None
    source: ContextUsageSource
    is_stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "context_window": self.context_window,
            "percent": self.percent,
            "source": self.source,
            "is_stale": self.is_stale,
        }


class ContextManager:
    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        store: SessionStore,
        hooks: HookRegistry,
        workspace_root: Path,
        get_system_prompt: Callable[[], str],
        get_messages: Callable[[], list[Message]],
        set_messages: Callable[[list[Message]], None],
        get_tool_schemas: Callable[[], list[dict[str, Any]]],
        emit: Callable[[AgentEvent], None],
    ) -> None:
        self.config = config
        self.llm = llm
        self.store = store
        self.hooks = hooks
        self.workspace_root = workspace_root
        self._get_system_prompt = get_system_prompt
        self._get_messages = get_messages
        self._set_messages = set_messages
        self._get_tool_schemas = get_tool_schemas
        self._emit = emit
        self.latest_usage: TokenUsage | None = None
        self._provider_usage_stale = False

    def record_usage(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.latest_usage = usage
        self._provider_usage_stale = False

    def context_usage(self, *, extra_messages: list[Message] | None = None) -> ContextUsage:
        if (
            extra_messages is None
            and self.latest_usage
            and self.latest_usage.prompt_tokens is not None
            and not self.latest_usage.is_estimate
            and not self._provider_usage_stale
        ):
            return self._usage(self.latest_usage.prompt_tokens, "provider", is_stale=False)
        try:
            return self._usage(self.context_token_estimate(extra_messages=extra_messages), "estimate", is_stale=False)
        except Exception:
            return ContextUsage(
                tokens=None,
                context_window=self.config.context_window,
                percent=None,
                source="unknown",
                is_stale=self._provider_usage_stale,
            )

    def context_token_estimate(self, *, extra_messages: list[Message] | None = None) -> int:
        messages = list(self._get_messages())
        if extra_messages:
            messages.extend(extra_messages)
        return estimate_prompt_tokens(
            system_prompt=self._get_system_prompt(),
            messages=messages,
            tools=self._get_tool_schemas(),
        )

    def context_tokens_label(self) -> str:
        usage = self.context_usage()
        if usage.tokens is None:
            return f"?/{usage.context_window:,} tokens (unknown)"
        prefix = "~" if usage.source == "estimate" else ""
        suffix = " estimate" if usage.source == "estimate" else ""
        return f"{prefix}{usage.tokens:,}/{usage.context_window:,} tokens{suffix}"

    def latest_usage_label(self) -> str | None:
        if not self.latest_usage or self.latest_usage.prompt_tokens is None or self._provider_usage_stale:
            return None
        prefix = "~" if self.latest_usage.is_estimate else ""
        parts = [f"{prefix}{self.latest_usage.prompt_tokens:,} prompt"]
        if self.latest_usage.completion_tokens is not None:
            parts.append(f"{self.latest_usage.completion_tokens:,} completion")
        if self.latest_usage.total_tokens is not None:
            parts.append(f"{self.latest_usage.total_tokens:,} total")
        return " / ".join(parts) + " tokens"

    def maybe_compact_before_prompt(self, text: str) -> None:
        usage = self.context_usage(extra_messages=[{"role": "user", "content": text}])
        if self._should_compact_usage(usage):
            self.compact(reason="threshold")

    def maybe_auto_compact(self) -> None:
        usage = self.context_usage()
        if self._should_compact_usage(usage):
            self.compact(reason="threshold")

    def compact(self, reason: CompactionReason = "manual", instructions: str | None = None) -> CompactionResult | None:
        usage = self.context_usage()
        tokens_before = usage.tokens if usage.tokens is not None else self.context_token_estimate()
        self._emit(
            make_event(
                "compaction_start",
                reason=reason,
                tokens_before=tokens_before,
                usage=usage.to_dict(),
                instructions=instructions,
            )
        )
        active_entries, previous_compaction = self.store.active_message_entries()
        preparation = prepare_compaction(
            active_entries,
            previous_compaction=previous_compaction,
            tokens_before=tokens_before,
            keep_recent_tokens=self.config.keep_recent_tokens,
            instructions=instructions,
        )
        if not preparation:
            self._emit(make_event("compaction_end", reason=reason, result=None, aborted=False, will_retry=False))
            return None

        try:
            decision = self.hooks.run_before_compaction(
                BeforeCompactionContext(
                    reason=reason,
                    usage=usage,
                    messages_to_summarize=preparation.messages_to_summarize,
                    kept_messages=preparation.kept_messages,
                    previous_summary=preparation.previous_summary,
                    details=preparation.details or {},
                    workspace_root=self.workspace_root,
                    instructions=instructions,
                )
            )
        except Exception as error:
            self._emit(
                make_event(
                    "compaction_end",
                    reason=reason,
                    result=None,
                    aborted=False,
                    will_retry=False,
                    error_message=str(error),
                )
            )
            return None

        details = decision.details or preparation.details or {}
        if not decision.allow:
            self._emit(
                make_event(
                    "compaction_end",
                    reason=reason,
                    result=None,
                    aborted=True,
                    will_retry=False,
                    error_message=decision.reason,
                )
            )
            return None

        from_hook = decision.summary is not None
        try:
            if decision.summary is not None:
                summary = append_details_to_summary(decision.summary.strip(), details)
                if not summary:
                    raise RuntimeError("Compaction hook returned an empty summary.")
            else:
                summary = generate_summary(
                    llm=self.llm,
                    model=self.config.model,
                    preparation=replace(preparation, details=details),
                    reserve_tokens=self.config.reserve_tokens,
                )
        except Exception as error:
            self._emit(
                make_event(
                    "compaction_end",
                    reason=reason,
                    result=None,
                    aborted=False,
                    will_retry=False,
                    error_message=str(error),
                )
            )
            raise

        result = CompactionResult(
            summary=summary,
            first_kept_entry_id=preparation.first_kept_entry_id,
            tokens_before=tokens_before,
            reason=reason,
            details=details,
            from_hook=from_hook,
        )
        self.store.append_compaction(
            summary=result.summary,
            first_kept_entry_id=result.first_kept_entry_id,
            tokens_before=result.tokens_before,
            reason=reason,
            details=result.details,
            from_hook=result.from_hook,
        )
        self._set_messages(self.store.load_messages())
        self._provider_usage_stale = True
        self._emit(
            make_event(
                "compaction_end",
                reason=reason,
                result=result.__dict__,
                aborted=False,
                will_retry=reason == "overflow",
            )
        )
        return result

    def is_context_overflow_error(self, error: Exception) -> bool:
        return is_context_overflow_error(error)

    def _should_compact_usage(self, usage: ContextUsage) -> bool:
        if usage.tokens is None:
            return False
        return should_compact(usage.tokens, self.config)

    def _usage(self, tokens: int, source: ContextUsageSource, *, is_stale: bool) -> ContextUsage:
        percent = tokens / self.config.context_window if self.config.context_window else None
        return ContextUsage(
            tokens=tokens,
            context_window=self.config.context_window,
            percent=percent,
            source=source,
            is_stale=is_stale,
        )
