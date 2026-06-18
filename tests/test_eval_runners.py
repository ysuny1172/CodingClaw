import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
