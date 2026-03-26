# ruff: noqa: E402
"""Regression coverage for WebArena Verified subset selection and summary."""

from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from test.browser_benchmarks.webarena_verified_subset import (
    WebArenaVerifiedBenchmarkCase,
    choose_evenly_spaced_positions,
    select_stratified_cases,
    summarize_benchmark_runs,
)


class WebArenaVerifiedSubsetTests(unittest.TestCase):
    def test_choose_evenly_spaced_positions_spreads_indices(self) -> None:
        self.assertEqual(choose_evenly_spaced_positions(length=6, limit=4), [0, 2, 3, 5])

    def test_select_stratified_cases_picks_per_site_spread(self) -> None:
        cases = [
            WebArenaVerifiedBenchmarkCase(task_id=21, site_name="shopping", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="a"),
            WebArenaVerifiedBenchmarkCase(task_id=124, site_name="shopping", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="b"),
            WebArenaVerifiedBenchmarkCase(task_id=191, site_name="shopping", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="c"),
            WebArenaVerifiedBenchmarkCase(task_id=388, site_name="shopping", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="d"),
            WebArenaVerifiedBenchmarkCase(task_id=11, site_name="shopping_admin", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="e"),
            WebArenaVerifiedBenchmarkCase(task_id=110, site_name="shopping_admin", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="f"),
            WebArenaVerifiedBenchmarkCase(task_id=196, site_name="shopping_admin", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="g"),
            WebArenaVerifiedBenchmarkCase(task_id=491, site_name="shopping_admin", expected_action="RETRIEVE", eval_types=("AgentResponseEvaluatorCfg",), intent="h"),
        ]
        selected = select_stratified_cases(cases=cases, per_site=2)
        self.assertEqual(
            [case.task_id for case in selected],
            [21, 388, 11, 491],
        )

    def test_summarize_benchmark_runs_counts_site_breakdown(self) -> None:
        summary = summarize_benchmark_runs(
            runs=[
                {"site_name": "shopping", "runner_status": "completed", "browser_ok": True, "eval_pass": True},
                {"site_name": "shopping", "runner_status": "completed", "browser_ok": False, "eval_pass": False},
                {"site_name": "reddit", "runner_status": "runner_error", "browser_ok": False, "eval_pass": False},
            ]
        )
        self.assertEqual(summary["total_tasks"], 3)
        self.assertEqual(summary["passed_tasks"], 1)
        self.assertEqual(summary["runner_errors"], 1)
        self.assertEqual(summary["by_site"]["shopping"]["passed"], 1)
        self.assertEqual(summary["by_site"]["reddit"]["browser_ok"], 0)


if __name__ == "__main__":
    unittest.main()
