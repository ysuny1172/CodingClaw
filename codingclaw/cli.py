from __future__ import annotations

import argparse
import sys

from codingclaw.config import Config
from codingclaw.llm import OpenAICompatibleClient
from codingclaw.session import Session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodingClaw minimal coding-agent harness.")
    parser.add_argument("task", nargs="+", help="Task for the coding agent.")
    parser.add_argument("--workspace", default=None, help="Workspace directory. Defaults to current directory.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to OPENAI_MODEL.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum agent loop steps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.from_env(
        workspace=args.workspace,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_steps=args.max_steps,
    )
    llm = OpenAICompatibleClient(base_url=config.base_url, api_key=config.api_key)
    session = Session(config=config, llm=llm)
    try:
        final_text = session.prompt(" ".join(args.task))
    except Exception as error:
        print(f"codingclaw: {error}", file=sys.stderr)
        return 1
    print(final_text)
    print(f"\nSession: {session.store.path}")
    print(f"Trace:   {session.trace.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
