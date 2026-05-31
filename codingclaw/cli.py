from __future__ import annotations

import argparse
import sys
from typing import Callable, TextIO

from codingclaw.config import Config
from codingclaw.llm import OpenAICompatibleClient
from codingclaw.session import Session
from codingclaw.session.session_store import SessionStore


HELP_TEXT = """Commands:
  /help     Show this help message.
  /session  Show the current session and trace files.
  /compact  Compact the current session context.
  /exit     Exit interactive mode.
  /quit     Exit interactive mode.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodingClaw minimal coding-agent harness.")
    parser.add_argument("task", nargs="*", help="Task for the coding agent. Omit to start interactive mode.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run an initial task, then stay in interactive mode.",
    )
    mode.add_argument(
        "--print",
        dest="print_mode",
        action="store_true",
        help="Run a single task and exit. This is the default when a task is provided.",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the latest session in this workspace.",
    )
    resume.add_argument(
        "--session",
        dest="session_path",
        default=None,
        help="Resume a specific session JSONL file.",
    )
    parser.add_argument("--workspace", default=None, help="Workspace directory. Defaults to current directory.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to OPENAI_MODEL.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum agent loop steps.")
    parser.add_argument("--context-window", type=int, default=None, help="Model context window in tokens. Defaults to CODINGCLAW_CONTEXT_WINDOW or 128000.")
    parser.add_argument("--reserve-tokens", type=int, default=None, help="Tokens reserved for the model response. Defaults to CODINGCLAW_RESERVE_TOKENS or 16384.")
    parser.add_argument("--keep-recent-tokens", type=int, default=None, help="Approximate recent context tokens to keep during compaction. Defaults to CODINGCLAW_KEEP_RECENT_TOKENS or 20000.")
    parser.add_argument("--no-auto-compact", action="store_true", help="Disable automatic threshold compaction.")
    return parser


def should_run_interactive(args: argparse.Namespace) -> bool:
    if args.print_mode:
        return False
    return args.interactive or not args.task


def run_prompt(session: Session, text: str, *, output: TextIO, error_output: TextIO) -> bool:
    try:
        final_text = session.prompt(text)
    except Exception as error:
        print(f"codingclaw: {error}", file=error_output)
        return False
    if final_text:
        print(final_text, file=output)
    return True


def run_compact(session: Session, *, output: TextIO, error_output: TextIO) -> bool:
    try:
        result = session.compact(reason="manual")
    except Exception as error:
        print(f"codingclaw: {error}", file=error_output)
        return False
    if not result:
        print("No compaction was needed or possible.", file=output)
        return True
    print(
        f"Compacted {result.tokens_before:,} tokens. Kept from message {result.first_kept_message_id}.",
        file=output,
    )
    print(f"Session: {session.store.path}", file=output)
    return True


def interactive_prompt(session: Session) -> str:
    return f"claw [{session.context_tokens_label()}]> "


def resolve_session_store(args: argparse.Namespace, config: Config) -> SessionStore | None:
    if args.continue_session:
        return SessionStore.open_latest(config.workspace)
    if args.session_path:
        return SessionStore.open(config.workspace, args.session_path)
    return None


def run_interactive(
    session: Session,
    *,
    initial_task: str | None = None,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    error_output: TextIO | None = None,
) -> int:
    output = output or sys.stdout
    error_output = error_output or sys.stderr

    print("CodingClaw interactive mode. Type /help for commands.", file=output)
    print(f"Session: {session.store.path}", file=output)
    print(f"Trace:   {session.trace.path}", file=output)
    print(f"Context: {session.context_tokens_label()}", file=output)

    if initial_task:
        run_prompt(session, initial_task, output=output, error_output=error_output)

    while True:
        try:
            text = input_fn(interactive_prompt(session))
        except EOFError:
            print("", file=output)
            return 0
        except KeyboardInterrupt:
            print("", file=output)
            return 0

        text = text.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/help":
            print(HELP_TEXT, file=output)
            continue
        if text == "/session":
            print(f"Session: {session.store.path}", file=output)
            print(f"Trace:   {session.trace.path}", file=output)
            print(f"Context: {session.context_tokens_label()}", file=output)
            latest_usage = session.latest_usage_label()
            if latest_usage:
                print(f"Last request: {latest_usage}", file=output)
            continue
        if text == "/compact":
            run_compact(session, output=output, error_output=error_output)
            continue
        if text.startswith("/"):
            print(f"Unknown command: {text}. Type /help for commands.", file=error_output)
            continue

        run_prompt(session, text, output=output, error_output=error_output)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_mode and not args.task:
        parser.error("--print requires a task.")

    config = Config.from_env(
        workspace=args.workspace,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_steps=args.max_steps,
        context_window=args.context_window,
        reserve_tokens=args.reserve_tokens,
        keep_recent_tokens=args.keep_recent_tokens,
        auto_compact=not args.no_auto_compact,
    )
    llm = OpenAICompatibleClient(base_url=config.base_url, api_key=config.api_key)
    try:
        store = resolve_session_store(args, config)
        session = Session(config=config, llm=llm, store=store)
    except Exception as error:
        print(f"codingclaw: {error}", file=sys.stderr)
        return 1

    task = " ".join(args.task).strip()
    if should_run_interactive(args):
        return run_interactive(session, initial_task=task or None)

    if not run_prompt(session, task, output=sys.stdout, error_output=sys.stderr):
        return 1
    print(f"\nSession: {session.store.path}")
    print(f"Trace:   {session.trace.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
