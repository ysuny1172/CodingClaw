import importlib.util
import json
import subprocess
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvalRunnersTest(unittest.TestCase):
    def test_run_evals_calculates_pass_rate(self):
        run_evals = load_script("run_evals")

        self.assertEqual(run_evals.calculate_pass_rate(0, 0), 0.0)
        self.assertEqual(run_evals.calculate_pass_rate(1, 1), 100.0)
        self.assertEqual(run_evals.calculate_pass_rate(3, 4), 75.0)

    def test_context_eval_builds_compaction_metric(self):
        run_context_evals = load_script("run_context_evals")

        metric = run_context_evals.build_compaction_metric(
            reason="manual",
            tokens_before=4820,
            tokens_after=1340,
        )

        self.assertEqual(metric["reason"], "manual")
        self.assertEqual(metric["tokens_before"], 4820)
        self.assertEqual(metric["tokens_after"], 1340)
        self.assertEqual(metric["tokens_saved"], 3480)
        self.assertAlmostEqual(metric["token_savings_rate"], 72.199, places=2)

    def test_context_eval_empty_compaction_average_is_zero(self):
        run_context_evals = load_script("run_context_evals")

        self.assertEqual(run_context_evals.calculate_average([]), 0.0)

    def test_context_eval_expands_repeat_noise_turns(self):
        run_context_evals = load_script("run_context_evals")

        turns = run_context_evals.expand_turns(
            [
                {"user": "first"},
                {
                    "repeat_noise": {
                        "count": 2,
                        "template": "noise {index}: {noise}",
                        "noise": "payload",
                    }
                },
            ]
        )

        self.assertEqual([turn["user"] for turn in turns], ["first", "noise 1: payload", "noise 2: payload"])

    def test_context_eval_checks_compaction_reason_and_token_floor(self):
        run_context_evals = load_script("run_context_evals")

        failures = run_context_evals.check_assertions(
            case={
                "assertions": {
                    "require_compaction": True,
                    "expected_compaction_reason": "threshold",
                    "min_compactions": 1,
                    "min_tokens_before": 2000,
                }
            },
            answer="Northstar CN-8842 2026-09-30",
            summaries=["Northstar CN-8842 2026-09-30"],
            compactions=[
                {
                    "reason": "manual",
                    "tokens_before": 1500,
                    "tokens_after": 900,
                    "tokens_saved": 600,
                    "token_savings_rate": 40.0,
                }
            ],
            tool_calls_by_turn=[],
        )

        self.assertIn("compaction 1 expected reason threshold, got manual", failures)
        self.assertIn("expected a compaction with tokens_before >= 2000, got 1500", failures)

    def test_swebench_load_instance_ids_deduplicates_in_order(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.json"
            path.write_text(json.dumps(["a__a-1", "b__b-2", "a__a-1"]), encoding="utf-8")

            result = run_swebench.load_instance_ids(instance_id=None, instances_file=path)

        self.assertEqual(result, ["a__a-1", "b__b-2"])

    def test_swebench_load_dataset_instances_rejects_missing_ids(self):
        run_swebench = load_script("run_swebench")
        fake_datasets = type(
            "FakeDatasets",
            (),
            {"load_dataset": staticmethod(lambda _name, split: [{"instance_id": "a", "repo": "o/r"}])},
        )

        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            with self.assertRaisesRegex(ValueError, "missing"):
                run_swebench.load_dataset_instances("dataset", "test", ["a", "missing"])

    def test_swebench_prompt_marks_run_as_pass_at_one(self):
        run_swebench = load_script("run_swebench")

        prompt = run_swebench.build_agent_prompt(
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "problem_statement": "Fix the thing.",
            }
        )

        self.assertIn("Fix the thing.", prompt)
        self.assertIn("Do not read SWE-bench evaluation logs", prompt)
        self.assertIn("Do not commit changes", prompt)

    def test_swebench_collect_patch_includes_new_files_and_excludes_runtime_files(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
            (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
            (repo / "new.txt").write_text("new\n", encoding="utf-8")
            (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
            subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
            runtime = repo / ".codingclaw"
            runtime.mkdir()
            (runtime / "trace.jsonl").write_text("{}\n", encoding="utf-8")

            model_patch = run_swebench.collect_model_patch(repo)

        self.assertIn("tracked.txt", model_patch)
        self.assertIn("new.txt", model_patch)
        self.assertIn("staged.txt", model_patch)
        self.assertNotIn(".codingclaw", model_patch)

    def test_swebench_prediction_jsonl_format(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            prediction = run_swebench.make_prediction("a__a-1", "codingclaw/model", "diff")
            run_swebench.write_predictions(path, [prediction])
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["instance_id"], "a__a-1")
        self.assertEqual(loaded["model_patch"], "diff")

    def test_swebench_resume_recognizes_completed_generation(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            run_swebench.write_json(result_path, {"generation_status": "completed"})

            self.assertTrue(run_swebench.instance_is_generated(result_path))

    def test_swebench_resume_skips_agent_process(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            instance_dir = run_dir / "instances" / "a__a-1"
            instance_dir.mkdir(parents=True)
            (instance_dir / "model.patch").write_text("saved patch", encoding="utf-8")
            run_swebench.write_json(instance_dir / "result.json", {"generation_status": "completed"})

            prediction = run_swebench.run_agent_for_instance(
                {
                    "instance_id": "a__a-1",
                    "repo": "a/a",
                    "base_commit": "abc",
                    "problem_statement": "Fix.",
                },
                run_dir=run_dir,
                codingclaw_executable=Path("/venv/bin/codingclaw"),
                model_name="model",
                max_steps=2,
                agent_timeout=1,
                resume=True,
                command_runner=lambda *_args, **_kwargs: self.fail("agent process should be skipped"),
            )

        self.assertEqual(prediction["model_patch"], "saved patch")

    def test_swebench_agent_timeout_still_writes_empty_prediction(self):
        run_swebench = load_script("run_swebench")

        def fake_runner(command, **kwargs):
            if command[:2] == ["git", "clone"]:
                repo = Path(command[-1])
                repo.mkdir(parents=True)
                subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:2] == ["git", "checkout"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[0].endswith("codingclaw"):
                raise subprocess.TimeoutExpired(command, 1)
            return run_swebench.run_command(command, **kwargs)

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prediction = run_swebench.run_agent_for_instance(
                {
                    "instance_id": "a__a-1",
                    "repo": "a/a",
                    "base_commit": "abc",
                    "problem_statement": "Fix.",
                },
                run_dir=run_dir,
                codingclaw_executable=Path("/venv/bin/codingclaw"),
                model_name="model",
                max_steps=2,
                agent_timeout=1,
                resume=False,
                command_runner=fake_runner,
            )
            result = run_swebench.read_json(run_dir / "instances" / "a__a-1" / "result.json")

        self.assertEqual(prediction["model_patch"], "")
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["empty_patch"])
        self.assertEqual(result["generation_status"], "completed")

    def test_swebench_setup_failure_is_retryable(self):
        run_swebench = load_script("run_swebench")

        def fake_runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 1, "", "clone failed")

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_swebench.run_agent_for_instance(
                {
                    "instance_id": "a__a-1",
                    "repo": "a/a",
                    "base_commit": "abc",
                    "problem_statement": "Fix.",
                },
                run_dir=run_dir,
                codingclaw_executable=Path("/venv/bin/codingclaw"),
                model_name="model",
                max_steps=2,
                agent_timeout=1,
                resume=False,
                command_runner=fake_runner,
            )
            result_path = run_dir / "instances" / "a__a-1" / "result.json"

            self.assertFalse(run_swebench.instance_is_generated(result_path))
            self.assertEqual(run_swebench.read_json(result_path)["generation_status"], "failed")

    def test_swebench_nonzero_agent_exit_is_recorded_as_completed_attempt(self):
        run_swebench = load_script("run_swebench")

        def fake_runner(command, **kwargs):
            if command[:2] == ["git", "clone"]:
                repo = Path(command[-1])
                repo.mkdir(parents=True)
                subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
                subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
                subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
                (repo / "base.txt").write_text("base\n", encoding="utf-8")
                subprocess.run(["git", "add", "base.txt"], cwd=repo, check=True)
                subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:2] == ["git", "checkout"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[0].endswith("codingclaw"):
                return subprocess.CompletedProcess(command, 2, "", "model failed")
            return run_swebench.run_command(command, **kwargs)

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_swebench.run_agent_for_instance(
                {
                    "instance_id": "a__a-1",
                    "repo": "a/a",
                    "base_commit": "abc",
                    "problem_statement": "Fix.",
                },
                run_dir=run_dir,
                codingclaw_executable=Path("/venv/bin/codingclaw"),
                model_name="model",
                max_steps=2,
                agent_timeout=1,
                resume=False,
                command_runner=fake_runner,
            )
            result_path = run_dir / "instances" / "a__a-1" / "result.json"
            result = run_swebench.read_json(result_path)
            is_generated = run_swebench.instance_is_generated(result_path)

        self.assertTrue(is_generated)
        self.assertEqual(result["agent_exit_code"], 2)
        self.assertEqual(result["generation_status"], "completed")

    def test_swebench_harness_command_is_sequential(self):
        run_swebench = load_script("run_swebench")

        command = run_swebench.build_harness_command(
            swebench_python=Path("/swebench/bin/python"),
            dataset_name="SWE-bench/SWE-bench_Lite",
            split="test",
            instance_ids=["a", "b"],
            predictions_path=Path("/runs/predictions.jsonl"),
            run_id="run-1",
            timeout=900,
        )

        self.assertEqual(command[command.index("--max_workers") + 1], "1")
        self.assertEqual(
            command[command.index("--instance_ids") + 1 : command.index("--predictions_path")],
            ["a", "b"],
        )

    def test_swebench_report_summary_calculates_pass_rate(self):
        run_swebench = load_script("run_swebench")

        summary = run_swebench.summarize_harness_report(
            {
                "total_instances": 5,
                "completed_instances": 5,
                "resolved_instances": 2,
                "unresolved_instances": 3,
                "error_instances": 0,
            }
        )

        self.assertEqual(summary["pass_rate"], 40.0)

    def test_swebench_normalize_args_preserves_executable_symlinks(self):
        run_swebench = load_script("run_swebench")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "python-real"
            target.write_text("", encoding="utf-8")
            link = root / "python"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Creating symlinks is not available on this system")
            args = Namespace(
                runs_dir=root / ".",
                instances_file=None,
                swebench_root=root / ".",
                codingclaw_executable=link,
                swebench_python=link,
            )

            normalized = run_swebench.normalize_args(args)

        self.assertEqual(normalized.swebench_python, link.absolute())
        self.assertNotEqual(normalized.swebench_python, target.resolve())


if __name__ == "__main__":
    unittest.main()
