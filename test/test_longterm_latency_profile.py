"""Cover long-term latency profile summarization and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.latency_profile import (
    LongTermLatencyProfileQueryRun,
    LongTermLatencyProfileResult,
    _summarize_decisions,
    _summarize_span_stats,
    default_longterm_latency_profile_dir,
    run_longterm_latency_profile,
    write_longterm_latency_profile_artifacts,
)


class LongTermLatencyProfileTests(unittest.TestCase):
    """Validate the latency-profile report helpers without live providers."""

    def test_run_profile_bootstraps_required_remote_before_timing_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")
            output_dir = Path(tmp_dir) / "report"
            calls: list[str] = []

            class _FakeService:
                def probe_remote_ready(self, *, bootstrap: bool = True, include_archive: bool = True):
                    calls.append(f"probe:{bootstrap}:{include_archive}")
                    return object()

                def build_provider_context(self, query_text: str):
                    calls.append(f"build:{query_text}")
                    return object()

                def shutdown(self, timeout_s: float = 0.0) -> None:
                    calls.append("shutdown")

            with patch(
                "twinr.memory.longterm.evaluation.latency_profile.LongTermMemoryService.from_config",
                return_value=_FakeService(),
            ):
                result = run_longterm_latency_profile(
                    env_path=env_path,
                    queries=["Was weisst du ueber xyz?"],
                    runs_per_query=1,
                    output_dir=output_dir,
                    target="provider_context",
                )

        self.assertEqual(result.query_runs[0].error_type, None)
        self.assertEqual(calls[:2], ["probe:True:False", "build:Was weisst du ueber xyz?"])
        self.assertIn("shutdown", calls)

    def test_summarize_span_stats_ranks_total_duration(self) -> None:
        records = [
            {"kind": "span_end", "msg": "slow_span", "kpi": {"duration_ms": 210.0}},
            {"kind": "span_end", "msg": "slow_span", "kpi": {"duration_ms": 190.0}},
            {"kind": "span_end", "msg": "fast_span", "kpi": {"duration_ms": 25.0}},
            {
                "kind": "exception",
                "msg": "ignored_exception_name",
                "details": {"span": "slow_span"},
                "kpi": {"duration_ms": 5.0},
            },
        ]

        stats = _summarize_span_stats(records)

        self.assertEqual(stats[0].name, "slow_span")
        self.assertEqual(stats[0].count, 3)
        self.assertEqual(stats[0].error_count, 1)
        self.assertAlmostEqual(stats[0].total_duration_ms, 405.0)
        self.assertEqual(stats[1].name, "fast_span")

    def test_summarize_decisions_counts_selected_ids(self) -> None:
        records = [
            {"kind": "decision", "msg": "route_choice", "reason": {"selected": {"id": "topk_records"}}},
            {"kind": "decision", "msg": "route_choice", "reason": {"selected": {"id": "topk_records"}}},
            {"kind": "decision", "msg": "route_choice", "reason": {"selected": {"id": "retrieve"}}},
            {"kind": "decision", "msg": "other_choice", "reason": {"selected": {"id": "local_selector"}}},
        ]

        summaries = _summarize_decisions(records)

        self.assertEqual(summaries[0].msg, "route_choice")
        self.assertEqual(summaries[0].count, 3)
        self.assertEqual(summaries[0].selected_counts["topk_records"], 2)
        self.assertEqual(summaries[0].selected_counts["retrieve"], 1)
        self.assertEqual(summaries[1].msg, "other_choice")

    def test_write_profile_artifacts_persists_profile_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir) / "report"
            result = LongTermLatencyProfileResult(
                profile_id="profile123",
                captured_at_utc="2026-03-22T12:00:00Z",
                env_file="/home/thh/twinr/.env",
                report_dir=str(report_dir),
                workflow_run_id="run123",
                workflow_run_dir=str(report_dir / "workflow" / "run123"),
                trace_mode="forensic",
                query_rewrite_enabled=False,
                subtext_compiler_enabled=False,
                background_store_turns_enabled=False,
                read_cache_ttl_s=0.0,
                recall_limit=3,
                midterm_limit=4,
                query_runs=(
                    LongTermLatencyProfileQueryRun(
                        query_index=1,
                        iteration=1,
                        trace_id="trace123",
                        duration_ms=3210.5,
                        query_sha256="abc123",
                        query_chars=27,
                        durable_context=True,
                        episodic_context=True,
                        graph_context=True,
                        conflict_context=False,
                        midterm_context=False,
                        subtext_context=False,
                    ),
                ),
                top_spans=(),
                decision_summaries=(),
                exception_counts={},
                workflow_summary={"slow_spans": []},
                workflow_metrics={"duration_ms": 3210.5},
            )

            output_path = write_longterm_latency_profile_artifacts(result)

            self.assertEqual(output_path, report_dir / "profile.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "twinr_longterm_latency_profile_v1")
            self.assertEqual(payload["profile_id"], "profile123")
            self.assertEqual(payload["query_runs"][0]["duration_ms"], 3210.5)

    def test_default_profile_dir_uses_report_namespace(self) -> None:
        target = default_longterm_latency_profile_dir(Path("/home/thh/twinr"), profile_id="profile123")
        self.assertEqual(
            target,
            Path("/home/thh/twinr/artifacts/reports/longterm_latency_profile/profile123"),
        )


if __name__ == "__main__":
    unittest.main()
