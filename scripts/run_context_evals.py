from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codingclaw.config import Config
from codingclaw.llm import OpenAICompatibleClient
from codingclaw.session import Session
from codingclaw.session.session_store import SessionStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "context_cases.json"
DEFAULT_OUT_DIR = ROOT / "evals" / "context_runs"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("context eval cases file must contain a JSON list")
    return data


def validate_environment() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "OPENAI_API_KEY is required to run live context evals.\n"
        "PowerShell example:\n"
        '  $env:OPENAI_API_KEY="sk-..."\n'
        "You can also set OPENAI_BASE_URL and OPENAI_MODEL if you use an OpenAI-compatible provider."
    )


def write_setup_files(workspace: Path, setup_files: dict[str, str]) -> None:
    for relative, content in setup_files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def calculate_pass_rate(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return passed / total * 100


def calculate_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_compaction_metric(*, reason: str, tokens_before: int, tokens_after: int) -> dict[str, Any]:
    tokens_saved = tokens_before - tokens_after
    token_savings_rate = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0.0
    return {
        "reason": reason,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "token_savings_rate": token_savings_rate,
    }


def expand_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for turn in turns:
        repeat_noise = turn.get("repeat_noise")
        if not isinstance(repeat_noise, dict):
            expanded.append(turn)
            continue

        count = int(repeat_noise.get("count", 1))
        template = str(repeat_noise.get("template") or turn.get("user") or "{noise}")
        noise = str(repeat_noise.get("noise") or "")
        noise_blocks = repeat_noise.get("noise_blocks")
        for index in range(1, count + 1):
            copied = {key: value for key, value in turn.items() if key != "repeat_noise"}
            block = ""
            if isinstance(noise_blocks, list) and noise_blocks:
                block = str(noise_blocks[(index - 1) % len(noise_blocks)])
            copied["user"] = template.format(index=index, noise=noise, block=block)
            expanded.append(copied)
    return expanded


def new_session(workspace: Path, case_config: dict[str, Any], *, store: SessionStore | None = None) -> Session:
    config = Config.from_env(
        workspace=workspace,
        model=case_config.get("model"),
        base_url=case_config.get("base_url"),
        api_key=case_config.get("api_key"),
        max_steps=int(case_config.get("max_steps", 12)),
        context_window=case_config.get("context_window"),
        reserve_tokens=case_config.get("reserve_tokens"),
        keep_recent_tokens=case_config.get("keep_recent_tokens"),
        auto_compact=case_config.get("auto_compact"),
    )
    llm = OpenAICompatibleClient(base_url=config.base_url, api_key=config.api_key)
    return Session(config=config, llm=llm, store=store)


def preload_history(session: Session, messages: list[dict[str, Any]]) -> None:
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or content is None:
            continue
        normalized = {"role": role, "content": str(content)}
        session.store.append_message(normalized)
        session.agent.state.messages.append(normalized)


def compaction_entries(session: Session) -> list[dict[str, Any]]:
    return [entry for entry in session.store.load_entries() if entry.get("type") == "compaction"]


def check_phrase_groups(label: str, text: str, groups: list[list[str]]) -> list[str]:
    failures: list[str] = []
    lowered = text.lower()
    for group in groups:
        if not any(str(item).lower() in lowered for item in group):
            failures.append(f"{label} missing required phrase group: {group}")
    return failures


def check_forbidden_phrases(label: str, text: str, phrases: list[str]) -> list[str]:
    failures: list[str] = []
    lowered = text.lower()
    for phrase in phrases:
        if str(phrase).lower() in lowered:
            failures.append(f"{label} contains forbidden phrase: {phrase}")
    return failures


def turn_rule(rules: Any, index: int) -> list[str]:
    if isinstance(rules, list):
        if index < len(rules) and isinstance(rules[index], list):
            return [str(item) for item in rules[index]]
        return []
    if isinstance(rules, dict):
        value = rules.get(str(index + 1), rules.get(index + 1, []))
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def check_assertions(
    *,
    case: dict[str, Any],
    answer: str,
    summaries: list[str],
    compactions: list[dict[str, Any]],
    tool_calls_by_turn: list[list[str]],
) -> list[str]:
    assertions = case.get("assertions", {})
    failures: list[str] = []

    if assertions.get("require_compaction") and not compactions:
        failures.append("expected at least one compaction")

    min_compactions = assertions.get("min_compactions")
    if isinstance(min_compactions, int) and len(compactions) < min_compactions:
        failures.append(f"expected at least {min_compactions} compaction(s), found {len(compactions)}")

    expected_reason = assertions.get("expected_compaction_reason")
    if isinstance(expected_reason, str):
        for index, compaction in enumerate(compactions):
            if compaction.get("reason") != expected_reason:
                failures.append(
                    f"compaction {index + 1} expected reason {expected_reason}, got {compaction.get('reason')}"
                )

    min_tokens_before = assertions.get("min_tokens_before")
    if isinstance(min_tokens_before, int):
        largest_tokens_before = max((int(item.get("tokens_before") or 0) for item in compactions), default=0)
        if largest_tokens_before < min_tokens_before:
            failures.append(
                f"expected a compaction with tokens_before >= {min_tokens_before}, got {largest_tokens_before}"
            )

    failures.extend(check_phrase_groups("answer", answer, assertions.get("must_include_any", [])))
    failures.extend(check_forbidden_phrases("answer", answer, assertions.get("must_not_include", [])))

    summary_text = "\n\n".join(summaries)
    failures.extend(check_phrase_groups("summary", summary_text, assertions.get("summary_must_include_any", [])))
    failures.extend(check_forbidden_phrases("summary", summary_text, assertions.get("summary_must_not_include", [])))

    expected_tools_by_turn = assertions.get("expected_tools_by_turn", [])
    forbidden_tools_by_turn = assertions.get("forbidden_tools_by_turn", [])
    for index, calls in enumerate(tool_calls_by_turn):
        for tool in turn_rule(expected_tools_by_turn, index):
            if tool not in calls:
                failures.append(f"turn {index + 1} missing expected tool call: {tool}")
        for tool in turn_rule(forbidden_tools_by_turn, index):
            if tool in calls:
                failures.append(f"turn {index + 1} called forbidden tool: {tool}")

    return failures


def run_case(case: dict[str, Any], *, keep_workspaces: bool) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"codingclaw-context-eval-{case['id']}-"))
    write_setup_files(temp_dir, case.get("setup_files", {}))

    session = new_session(temp_dir, case.get("config", {}))
    preload_history(session, case.get("history_messages", []))
    compactions: list[dict[str, Any]] = []
    summaries: list[str] = []
    tool_calls_by_turn: list[list[str]] = []
    seen_compaction_ids: set[str] = set()
    current_turn_tool_calls: list[str] = []

    def listener(event: dict[str, Any]) -> None:
        if event.get("type") == "tool_execution_start":
            name = event.get("tool_name")
            if isinstance(name, str):
                current_turn_tool_calls.append(name)

    session.subscribe(listener)

    answer = ""
    failures: list[str] = []
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        for turn_index, turn in enumerate(expand_turns(case.get("turns", []))):
            current_turn_tool_calls = []
            answer = session.prompt(str(turn.get("user", "")))
            tool_calls_by_turn.append(list(current_turn_tool_calls))

            for entry in compaction_entries(session):
                entry_id = str(entry.get("id") or "")
                if entry_id in seen_compaction_ids:
                    continue
                seen_compaction_ids.add(entry_id)
                tokens_before = int(entry.get("tokens_before") or 0)
                tokens_after = session.context_token_estimate()
                compactions.append(
                    build_compaction_metric(
                        reason=str(entry.get("reason") or "unknown"),
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                    )
                )
                summaries.append(str(entry.get("summary") or ""))

            if turn.get("force_compact_after"):
                tokens_before = session.context_token_estimate()
                result = session.compact(reason="manual")
                if result:
                    tokens_after = session.context_token_estimate()
                    compactions.append(
                        build_compaction_metric(
                            reason=result.reason,
                            tokens_before=tokens_before,
                            tokens_after=tokens_after,
                        )
                    )
                    summaries.append(result.summary)
                    seen_compaction_ids.update(str(entry.get("id") or "") for entry in compaction_entries(session))

            if turn.get("resume_after"):
                store = SessionStore.open(session.workspace_root, session.store.path)
                session = new_session(temp_dir, case.get("config", {}), store=store)
                session.subscribe(listener)

    except Exception as error:
        failures.append(f"case execution failed: {error}")

    if not failures:
        failures.extend(
            check_assertions(
                case=case,
                answer=answer,
                summaries=summaries,
                compactions=compactions,
                tool_calls_by_turn=tool_calls_by_turn,
            )
        )

    total_tokens_saved = sum(int(item["tokens_saved"]) for item in compactions)
    average_savings_rate = calculate_average([float(item["token_savings_rate"]) for item in compactions])
    result = {
        "id": case["id"],
        "category": case.get("category"),
        "source": case.get("source"),
        "primary_skill": case.get("primary_skill"),
        "stressors": case.get("stressors", []),
        "passed": not failures,
        "failures": failures,
        "answer": answer,
        "compaction_occurred": bool(compactions),
        "compactions": compactions,
        "total_tokens_saved": total_tokens_saved,
        "average_savings_rate": average_savings_rate,
        "tool_calls_by_turn": tool_calls_by_turn,
        "workspace": str(temp_dir),
        "started_at": started_at,
        "manual_judge": case.get("manual_judge", ""),
    }

    if not keep_workspaces:
        shutil.rmtree(temp_dir, ignore_errors=True)
        result["workspace"] = None

    return result


def print_case_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"{status} {result['id']}", flush=True)
    for failure in result["failures"]:
        print(f"  - {failure}", flush=True)

    compactions = result.get("compactions") or []
    print(f"  Compactions: {len(compactions)}", flush=True)
    if compactions:
        first_before = compactions[0]["tokens_before"]
        last_after = compactions[-1]["tokens_after"]
        print(f"  Tokens before: {first_before:,}", flush=True)
        print(f"  Tokens after: {last_after:,}", flush=True)
        print(f"  Tokens saved: {result['total_tokens_saved']:,}", flush=True)
        print(f"  Savings rate: {result['average_savings_rate']:.2f}%", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CodingClaw context and compaction eval cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case-id", action="append", default=[], help="Run only a specific case id. Can be repeated.")
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep temp workspaces for debugging.")
    args = parser.parse_args()

    validate_environment()

    cases = load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get("id") in wanted]
        missing = wanted - {case.get("id") for case in cases}
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(sorted(missing))}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.out_dir / f"{run_id}.jsonl"

    results = []
    with report_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            print(f"running {case['id']} ...", flush=True)
            result = run_case(case, keep_workspaces=args.keep_workspaces)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print_case_result(result)

    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    pass_rate = calculate_pass_rate(passed, len(results))
    total_tokens_saved = sum(int(result["total_tokens_saved"]) for result in results)
    savings_rates = [
        float(result["average_savings_rate"])
        for result in results
        if result.get("compactions")
    ]
    average_savings_rate = calculate_average(savings_rates)

    print(f"\nReport: {report_path}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"Total tokens saved: {total_tokens_saved:,}")
    print(f"Average savings rate: {average_savings_rate:.2f}%")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
