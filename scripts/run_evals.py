from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "eval_cases.json"
DEFAULT_OUT_DIR = ROOT / "evals" / "runs"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("eval cases file must contain a JSON list")
    return data


def validate_environment() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "OPENAI_API_KEY is required to run live evals.\n"
        "PowerShell example:\n"
        '  $env:OPENAI_API_KEY="sk-..."\n'
        "You can also set OPENAI_BASE_URL and OPENAI_MODEL if you use an OpenAI-compatible provider."
    )


def write_setup_files(workspace: Path, setup_files: dict[str, str]) -> None:
    for relative, content in setup_files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def read_tool_calls(workspace: Path) -> list[str]:
    trace_path = latest_trace_path(workspace)
    if not trace_path:
        return []

    calls: list[str] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "tool_execution_start":
                name = event.get("tool_name")
                if isinstance(name, str):
                    calls.append(name)
    return calls


def latest_trace_path(workspace: Path) -> Path | None:
    trace_dir = workspace / ".codingclaw" / "traces"
    if not trace_dir.exists():
        return None
    trace_files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not trace_files:
        return None
    return trace_files[-1]


def read_trace_answer(workspace: Path) -> str:
    trace_path = latest_trace_path(workspace)
    if not trace_path:
        return ""

    answer = ""
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message_end":
                continue
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                answer = content
    return answer


def extract_answer(stdout: str) -> str:
    marker = "\nSession:"
    if marker not in stdout:
        return stdout
    return stdout.split(marker, 1)[0].rstrip()


def check_case(case: dict[str, Any], output: str, tool_calls: list[str]) -> list[str]:
    failures: list[str] = []
    lowered_output = output.lower()

    for tool in case.get("expected_tools", []):
        if tool not in tool_calls:
            failures.append(f"missing expected tool call: {tool}")

    for tool in case.get("forbidden_tools", []):
        if tool in tool_calls:
            failures.append(f"forbidden tool was called: {tool}")

    for group in case.get("must_include_any", []):
        if not any(str(item).lower() in lowered_output for item in group):
            failures.append(f"missing required phrase group: {group}")

    for phrase in case.get("must_not_include", []):
        if str(phrase).lower() in lowered_output:
            failures.append(f"forbidden phrase appeared: {phrase}")

    if case.get("json_only"):
        stripped = output.strip()
        try:
            json.loads(stripped)
        except json.JSONDecodeError as error:
            failures.append(f"output is not valid JSON: {error}")

    return failures


def calculate_pass_rate(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return passed / total * 100


def run_case(case: dict[str, Any], *, keep_workspaces: bool, timeout_seconds: int) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"codingclaw-eval-{case['id']}-"))
    write_setup_files(temp_dir, case.get("setup_files", {}))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    command = [
        sys.executable,
        "-m",
        "codingclaw.cli",
        "--print",
        "--workspace",
        str(temp_dir),
        "--max-steps",
        "12",
        case["user_input"],
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        completed = error
        timed_out = True

    stdout = completed.stdout or ""
    stdout_answer = extract_answer(stdout)
    trace_answer = read_trace_answer(temp_dir)
    answer = stdout_answer or trace_answer
    answer_source = "stdout" if stdout_answer else "trace" if trace_answer else "none"
    stderr = completed.stderr or ""
    tool_calls = read_tool_calls(temp_dir)
    failures = [f"process timed out after {timeout_seconds}s"] if timed_out else []
    process_failed = not timed_out and getattr(completed, "returncode", 1) != 0
    if process_failed:
        failures.append(f"process exited with code {completed.returncode}")
    if not timed_out and not process_failed:
        failures.extend(check_case(case, answer, tool_calls))

    result = {
        "id": case["id"],
        "category": case.get("category"),
        "passed": not failures,
        "failures": failures,
        "tool_calls": tool_calls,
        "answer": answer,
        "answer_source": answer_source,
        "stdout": stdout,
        "stderr": stderr,
        "workspace": str(temp_dir),
        "started_at": started_at,
        "manual_judge": case.get("manual_judge", ""),
    }

    if not keep_workspaces:
        shutil.rmtree(temp_dir, ignore_errors=True)
        result["workspace"] = None

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CodingClaw behavioral eval cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case-id", action="append", default=[], help="Run only a specific case id. Can be repeated.")
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep temp workspaces for debugging.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
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
            result = run_case(case, keep_workspaces=args.keep_workspaces, timeout_seconds=args.timeout_seconds)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {case['id']}", flush=True)
            for failure in result["failures"]:
                print(f"  - {failure}", flush=True)

    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    pass_rate = calculate_pass_rate(passed, len(results))
    print(f"\nReport: {report_path}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.2f}%")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
