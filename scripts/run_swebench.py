from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "evals" / "swebench_runs"
DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
EXCLUDED_PATCH_PARTS = {
    ".codingclaw",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_instance_ids(*, instance_id: str | None, instances_file: Path | None) -> list[str]:
    if bool(instance_id) == bool(instances_file):
        raise ValueError("Provide exactly one of --instance-id or --instances-file.")
    raw_ids: list[Any]
    if instance_id:
        raw_ids = [instance_id]
    else:
        data = read_json(instances_file.expanduser())  # type: ignore[union-attr]
        if not isinstance(data, list):
            raise ValueError("Instances file must contain a JSON list.")
        raw_ids = data

    result: list[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every instance ID must be a non-empty string.")
        value = value.strip()
        if value not in seen:
            result.append(value)
            seen.add(value)
    if not result:
        raise ValueError("At least one instance ID is required.")
    return result


def load_dataset_instances(dataset_name: str, split: str, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Run this script from the SWE-bench virtual environment."
        ) from error

    dataset = load_dataset(dataset_name, split=split)
    by_id = {row["instance_id"]: dict(row) for row in dataset if row["instance_id"] in instance_ids}
    missing = [item for item in instance_ids if item not in by_id]
    if missing:
        raise ValueError(f"Instance IDs not found in {dataset_name}/{split}: {', '.join(missing)}")
    return [by_id[item] for item in instance_ids]


def build_agent_prompt(instance: dict[str, Any]) -> str:
    return (
        "Fix the following issue in the current repository.\n\n"
        f"Instance ID: {instance['instance_id']}\n"
        f"Repository: {instance['repo']}\n\n"
        "Issue:\n"
        f"{str(instance['problem_statement']).strip()}\n\n"
        "Instructions:\n"
        "- Inspect the repository and relevant tests before changing code.\n"
        "- Implement the smallest complete fix for the issue.\n"
        "- Run relevant existing tests when the local environment supports them.\n"
        "- Do not only explain the solution; modify files in the workspace.\n"
        "- Do not read SWE-bench evaluation logs, hidden tests, gold patches, or other benchmark answers.\n"
        "- Do not commit changes. Finish with a concise summary of the files changed and tests run."
    )


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def require_success(completed: subprocess.CompletedProcess[str], description: str) -> None:
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout or "").strip()
    raise RuntimeError(f"{description} failed with exit code {completed.returncode}: {detail}")


def prepare_repository(
    instance: dict[str, Any],
    repo_dir: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone = command_runner(
        ["git", "clone", f"https://github.com/{instance['repo']}.git", str(repo_dir)],
        cwd=repo_dir.parent,
    )
    require_success(clone, f"Cloning {instance['repo']}")
    checkout = command_runner(["git", "checkout", str(instance["base_commit"])], cwd=repo_dir)
    require_success(checkout, f"Checking out {instance['base_commit']}")


def is_excluded_patch_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if any(part in EXCLUDED_PATCH_PARTS for part in path.parts):
        return True
    return path.suffix in {".pyc", ".pyo"}


def collect_model_patch(
    repo_dir: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
) -> str:
    untracked = command_runner(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_dir,
    )
    require_success(untracked, "Listing untracked files")
    untracked_paths = [
        item for item in untracked.stdout.split("\0") if item and not is_excluded_patch_path(item)
    ]
    if untracked_paths:
        intent = command_runner(["git", "add", "--intent-to-add", "--", *untracked_paths], cwd=repo_dir)
        require_success(intent, "Marking new files for diff")

    excluded_pathspecs = [
        ":(exclude).codingclaw/**",
        ":(exclude)**/.codingclaw/**",
        ":(exclude)**/__pycache__/**",
        ":(exclude)**/.pytest_cache/**",
        ":(exclude)**/.mypy_cache/**",
        ":(exclude)**/.ruff_cache/**",
        ":(exclude)**/*.pyc",
        ":(exclude)**/*.pyo",
    ]
    diff = command_runner(
        ["git", "-c", "core.fileMode=false", "diff", "--binary", "HEAD", "--", ".", *excluded_pathspecs],
        cwd=repo_dir,
    )
    require_success(diff, "Collecting model patch")
    return diff.stdout


def find_latest_artifact(repo_dir: Path, directory: str) -> str | None:
    artifact_dir = repo_dir / ".codingclaw" / directory
    if not artifact_dir.exists():
        return None
    files = [path for path in artifact_dir.glob("*.jsonl") if path.is_file()]
    if not files:
        return None
    return str(max(files, key=lambda path: path.stat().st_mtime))


def make_prediction(instance_id: str, model_name: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def write_predictions(path: Path, predictions: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def instance_is_generated(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        result = read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("generation_status") == "completed"


def load_completed_prediction(instance_dir: Path, instance_id: str, model_name: str) -> dict[str, str]:
    patch_path = instance_dir / "model.patch"
    patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    return make_prediction(instance_id, model_name, patch)


def run_agent_for_instance(
    instance: dict[str, Any],
    *,
    run_dir: Path,
    codingclaw_executable: Path,
    model_name: str,
    max_steps: int,
    agent_timeout: int,
    resume: bool,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, str]:
    instance_id = str(instance["instance_id"])
    instance_dir = run_dir / "instances" / instance_id
    result_path = instance_dir / "result.json"
    if resume and instance_is_generated(result_path):
        print(f"SKIP {instance_id}: prediction already generated", flush=True)
        return load_completed_prediction(instance_dir, instance_id, model_name)

    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True)
    repo_dir = instance_dir / "repo"
    prompt = build_agent_prompt(instance)
    (instance_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    started_at = utc_now()
    started = time.monotonic()
    exit_code: int | None = None
    error: str | None = None
    stdout = ""
    stderr = ""
    agent_attempted = False

    try:
        prepare_repository(instance, repo_dir, command_runner=command_runner)
        command = [
            str(codingclaw_executable),
            "--print",
            "--workspace",
            str(repo_dir),
            "--max-steps",
            str(max_steps),
            prompt,
        ]
        agent_attempted = True
        completed = command_runner(
            command,
            cwd=repo_dir,
            env=os.environ.copy(),
            timeout=agent_timeout,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if exit_code != 0:
            error = f"CodingClaw exited with code {exit_code}"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        error = f"CodingClaw timed out after {agent_timeout} seconds"
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"

    (instance_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (instance_dir / "stderr.log").write_text(stderr, encoding="utf-8")

    patch = ""
    if repo_dir.exists():
        try:
            patch = collect_model_patch(repo_dir, command_runner=command_runner)
        except Exception as exc:
            error = error or f"Patch collection failed: {exc}"
    (instance_dir / "model.patch").write_text(patch, encoding="utf-8")

    result = {
        "instance_id": instance_id,
        "generation_status": "completed" if agent_attempted else "failed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "agent_exit_code": exit_code,
        "agent_attempted": agent_attempted,
        "timed_out": error is not None and "timed out" in error,
        "error": error,
        "patch_bytes": len(patch.encode("utf-8")),
        "empty_patch": not bool(patch.strip()),
        "session_path": find_latest_artifact(repo_dir, "sessions") if repo_dir.exists() else None,
        "trace_path": find_latest_artifact(repo_dir, "traces") if repo_dir.exists() else None,
    }
    write_json(result_path, result)
    status = "EMPTY" if result["empty_patch"] else "READY"
    if error:
        status = "ERROR"
    print(f"{status} {instance_id}: {result['patch_bytes']} patch bytes", flush=True)
    return make_prediction(instance_id, model_name, patch)


def create_run_metadata(
    *,
    run_id: str,
    dataset_name: str,
    split: str,
    instance_ids: Sequence[str],
    model_name: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "split": split,
        "instance_ids": list(instance_ids),
        "model_name_or_path": model_name,
        "mode": "pass@1",
        "generation_status": "pending",
        "evaluation_status": "pending",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def generate(args: argparse.Namespace) -> Path:
    if not args.codingclaw_executable.is_file():
        raise RuntimeError(f"CodingClaw executable does not exist: {args.codingclaw_executable}")

    run_dir = args.runs_dir / args.run_id
    metadata_path = run_dir / "run.json"
    instance_ids = load_instance_ids(instance_id=args.instance_id, instances_file=args.instances_file)

    if metadata_path.exists():
        if not args.resume:
            raise RuntimeError(f"Run already exists: {run_dir}. Use --resume or choose another --run-id.")
        metadata = read_json(metadata_path)
        expected = {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "instance_ids": instance_ids,
            "model_name_or_path": args.model_name,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Cannot resume: run metadata field {key!r} does not match.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = create_run_metadata(
            run_id=args.run_id,
            dataset_name=args.dataset_name,
            split=args.split,
            instance_ids=instance_ids,
            model_name=args.model_name,
        )
        write_json(metadata_path, metadata)

    instances = load_dataset_instances(args.dataset_name, args.split, instance_ids)
    metadata["generation_status"] = "running"
    metadata["updated_at"] = utc_now()
    write_json(metadata_path, metadata)

    predictions = []
    for instance in instances:
        print(f"GENERATE {instance['instance_id']}", flush=True)
        predictions.append(
            run_agent_for_instance(
                instance,
                run_dir=run_dir,
                codingclaw_executable=args.codingclaw_executable,
                model_name=args.model_name,
                max_steps=args.max_steps,
                agent_timeout=args.agent_timeout,
                resume=args.resume,
            )
        )
        write_predictions(run_dir / "predictions.jsonl", predictions)

    metadata["generation_status"] = "completed"
    metadata["generation_finished_at"] = utc_now()
    metadata["updated_at"] = utc_now()
    metadata["prediction_count"] = len(predictions)
    metadata["empty_patch_count"] = sum(not item["model_patch"].strip() for item in predictions)
    metadata["generation_failure_count"] = sum(
        read_json(run_dir / "instances" / item["instance_id"] / "result.json").get("generation_status")
        != "completed"
        for item in predictions
    )
    write_json(metadata_path, metadata)
    print(f"Predictions: {run_dir / 'predictions.jsonl'}")
    return run_dir


def build_harness_command(
    *,
    swebench_python: Path,
    dataset_name: str,
    split: str,
    instance_ids: Sequence[str],
    predictions_path: Path,
    run_id: str,
    timeout: int,
) -> list[str]:
    return [
        str(swebench_python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--instance_ids",
        *instance_ids,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        "1",
        "--timeout",
        str(timeout),
        "--run_id",
        run_id,
    ]


def find_harness_summary(swebench_root: Path, model_name: str, run_id: str) -> Path | None:
    expected = swebench_root / f"{model_name.replace('/', '__')}.{run_id}.json"
    if expected.exists():
        return expected
    matches = sorted(swebench_root.glob(f"*.{run_id}.json"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def summarize_harness_report(report: dict[str, Any]) -> dict[str, Any]:
    total = int(report.get("total_instances", 0))
    resolved = int(report.get("resolved_instances", 0))
    return {
        "total_instances": total,
        "completed_instances": int(report.get("completed_instances", 0)),
        "resolved_instances": resolved,
        "unresolved_instances": int(report.get("unresolved_instances", 0)),
        "error_instances": int(report.get("error_instances", 0)),
        "pass_rate": (resolved / total * 100) if total else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.swebench_root.is_dir():
        raise RuntimeError(f"SWE-bench root does not exist: {args.swebench_root}")
    if not args.swebench_python.is_file():
        raise RuntimeError(f"SWE-bench Python does not exist: {args.swebench_python}")

    run_dir = args.runs_dir / args.run_id
    metadata_path = run_dir / "run.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not metadata_path.exists() or not predictions_path.exists():
        raise RuntimeError(f"Run has no generated predictions: {run_dir}")

    metadata = read_json(metadata_path)
    command = build_harness_command(
        swebench_python=args.swebench_python,
        dataset_name=metadata["dataset_name"],
        split=metadata["split"],
        instance_ids=metadata["instance_ids"],
        predictions_path=predictions_path,
        run_id=args.run_id,
        timeout=args.harness_timeout,
    )
    metadata["evaluation_status"] = "running"
    metadata["evaluation_started_at"] = utc_now()
    metadata["updated_at"] = utc_now()
    write_json(metadata_path, metadata)

    print(f"EVALUATE {len(metadata['instance_ids'])} instance(s) with run id {args.run_id}", flush=True)
    try:
        completed = run_command(command, cwd=args.swebench_root, timeout=args.harness_process_timeout)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (run_dir / "harness.stdout.log").write_text(stdout, encoding="utf-8")
        (run_dir / "harness.stderr.log").write_text(stderr, encoding="utf-8")
        metadata["evaluation_status"] = "failed"
        metadata["evaluation_error"] = (
            f"Harness process timed out after {args.harness_process_timeout} seconds"
        )
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
        raise RuntimeError(metadata["evaluation_error"]) from error
    (run_dir / "harness.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "harness.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        metadata["evaluation_status"] = "failed"
        metadata["evaluation_error"] = f"Harness exited with code {completed.returncode}"
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-20:])
            raise RuntimeError(f"{metadata['evaluation_error']}:\n{detail}")
        raise RuntimeError(metadata["evaluation_error"])

    summary_path = find_harness_summary(args.swebench_root, metadata["model_name_or_path"], args.run_id)
    if summary_path is None:
        metadata["evaluation_status"] = "failed"
        metadata["evaluation_error"] = "SWE-bench completed but its summary JSON could not be found."
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
        raise RuntimeError(metadata["evaluation_error"])
    report = read_json(summary_path)
    summary = summarize_harness_report(report)
    write_json(run_dir / "evaluation_summary.json", summary)
    metadata["evaluation_status"] = "completed"
    metadata["evaluation_finished_at"] = utc_now()
    metadata["updated_at"] = utc_now()
    metadata["harness_summary_path"] = str(summary_path)
    metadata["summary"] = summary
    write_json(metadata_path, metadata)
    print(
        f"Resolved: {summary['resolved_instances']}/{summary['total_instances']} "
        f"({summary['pass_rate']:.2f}%)"
    )
    return summary


def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True, help="Stable identifier for this pass@1 run.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    instances = parser.add_mutually_exclusive_group(required=True)
    instances.add_argument("--instance-id")
    instances.add_argument("--instances-file", type=Path)


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_run_arguments(parser)
    parser.add_argument("--codingclaw-executable", type=Path, required=True)
    parser.add_argument("--model-name", default=os.getenv("OPENAI_MODEL", "codingclaw"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--agent-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CodingClaw against SWE-bench in strict pass@1 mode.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate one patch per instance.")
    add_generate_arguments(generate_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate existing predictions.")
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    evaluate_parser.add_argument("--swebench-root", type=Path, required=True)
    evaluate_parser.add_argument("--swebench-python", type=Path, default=Path(sys.executable))
    evaluate_parser.add_argument("--harness-timeout", type=int, default=1800)
    evaluate_parser.add_argument("--harness-process-timeout", type=int, default=None)

    all_parser = subparsers.add_parser("all", help="Generate predictions, then evaluate them.")
    add_generate_arguments(all_parser)
    all_parser.add_argument("--swebench-root", type=Path, required=True)
    all_parser.add_argument("--swebench-python", type=Path, default=Path(sys.executable))
    all_parser.add_argument("--harness-timeout", type=int, default=1800)
    all_parser.add_argument("--harness-process-timeout", type=int, default=None)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("runs_dir", "instances_file", "swebench_root"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    for name in ("codingclaw_executable", "swebench_python"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            expanded = value.expanduser()
            if not expanded.is_absolute():
                expanded = Path.cwd() / expanded
            setattr(args, name, expanded.absolute())
    return args


def main(argv: list[str] | None = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    try:
        if args.command == "generate":
            generate(args)
        elif args.command == "evaluate":
            evaluate(args)
        else:
            generate(args)
            evaluate(args)
    except Exception as error:
        print(f"run_swebench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
