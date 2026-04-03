from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.subtext_eval import (
    SubtextEvalCaseResult,
    SubtextEvalDiagnostics,
    SubtextJudgeResult,
    _build_case_config,
    _build_case_diagnostics,
    _apply_seed_actions,
    _flush_seed_memory,
    _prime_live_provider_front,
    _summarize,
    _extract_json_object,
    contains_explicit_memory_announcement,
    default_subtext_eval_cases,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.core.models import LongTermMemoryContext


class SubtextResponseEvalTests(unittest.TestCase):
    def test_default_cases_cover_expected_shape(self) -> None:
        cases = default_subtext_eval_cases()
        self.assertEqual(len(cases), 8)
        self.assertGreaterEqual(sum(1 for case in cases if case.should_use_personal_context), 4)
        self.assertGreaterEqual(sum(1 for case in cases if not case.should_use_personal_context), 3)

    def test_explicit_memory_detection_catches_common_phrases(self) -> None:
        self.assertTrue(contains_explicit_memory_announcement("Wenn ich mich richtig erinnere, magst du Melitta."))
        self.assertTrue(contains_explicit_memory_announcement("I remember that you said this yesterday."))
        self.assertFalse(contains_explicit_memory_announcement("Melitta könnte heute gut passen."))

    def test_extract_json_object_accepts_fenced_or_wrapped_content(self) -> None:
        payload = _extract_json_object("```json\n{\"passed\": true, \"reason\": \"ok\"}\n```")
        self.assertEqual(payload["passed"], True)
        self.assertEqual(payload["reason"], "ok")

    def test_build_case_config_copies_personality_into_temp_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            source_personality = project_root / "personality"
            source_personality.mkdir()
            (source_personality / "SYSTEM.md").write_text("System context", encoding="utf-8")

            case_root = project_root / "case-run"
            config = _build_case_config(
                base_config=TwinrConfig(
                    project_root=str(project_root),
                    personality_dir="personality",
                ),
                temp_root=case_root,
            )

            self.assertEqual(config.project_root, str(case_root))
            self.assertEqual(config.personality_dir, "personality")
            self.assertTrue((case_root / "state").is_dir())
            self.assertEqual(config.runtime_state_path, str(case_root / "state" / "runtime-state.json"))
            self.assertEqual(config.reminder_store_path, str(case_root / "state" / "reminders.json"))
            self.assertEqual(config.automation_store_path, str(case_root / "state" / "automations.json"))
            self.assertEqual(config.voice_profile_store_path, str(case_root / "state" / "voice_profile.json"))
            self.assertEqual(config.adaptive_timing_store_path, str(case_root / "state" / "adaptive_timing.json"))
            self.assertEqual(
                (case_root / "personality" / "SYSTEM.md").read_text(encoding="utf-8"),
                "System context",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            self.assertEqual(graph_store._lock_path, case_root / "state" / "locks" / "twinr_graph_v1.json.lock")

    def test_episode_seed_actions_persist_when_background_writers_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            source_personality = project_root / "personality"
            source_personality.mkdir()
            (source_personality / "SYSTEM.md").write_text("System context", encoding="utf-8")

            case_root = project_root / "case-run"
            config = _build_case_config(
                base_config=TwinrConfig(
                    project_root=str(project_root),
                    personality_dir="personality",
                    long_term_memory_enabled=True,
                ),
                temp_root=case_root,
            )
            runtime = TwinrRuntime(config)
            case = next(item for item in default_subtext_eval_cases() if item.case_id == "episodic_knee")
            try:
                _apply_seed_actions(runtime, case.seed_actions)
                _flush_seed_memory(runtime)
                stored_objects = runtime.long_term_memory.object_store.load_objects()
            finally:
                runtime.shutdown(timeout_s=1.0)

        self.assertTrue(any(item.kind == "episode" for item in stored_objects))

    def test_prime_live_provider_front_reports_materialized_source(self) -> None:
        calls: list[str] = []

        class _LongTermMemory:
            def materialize_live_provider_context(self, query_text: str):
                calls.append(query_text)
                return type(
                    "_Resolution",
                    (),
                    {
                        "context": LongTermMemoryContext(durable_context="live"),
                        "source": "materialized_built_sync",
                    },
                )()

            def latest_context_snapshot(self, *, profile: str):
                assert profile == "provider"
                return type(
                    "_Snapshot",
                    (),
                    {
                        "source": "materialized_local_cache_hit",
                        "query_profile": type(
                            "_QueryProfile",
                            (),
                            {
                                "original_text": "Wer ist Janina?",
                                "canonical_english_text": "Who is Janina?",
                                "retrieval_text": "Who is Janina?",
                            },
                        )(),
                        "context": LongTermMemoryContext(
                            subtext_context="twinr_silent_personalization_context_v1 Corinna physiotherapist",
                            durable_context="twinr_long_term_durable_context_v1 Corinna contact",
                            graph_context="twinr_graph_memory_context_v1 Corinna physiotherapist",
                        ),
                    },
                )()

        runtime = type("_Runtime", (), {"long_term_memory": _LongTermMemory()})()

        result = _prime_live_provider_front(runtime, "Wer ist Janina?")
        diagnostics = _build_case_diagnostics(
            case=next(item for item in default_subtext_eval_cases() if item.case_id == "contact_role"),
            conversation=(("system", "twinr_long_term_durable_context_v1"),),
            stage_timings_s={"live_front_prime": 0.1},
            runtime=runtime,
            live_front_prime=result,
            failure_stage=None,
        )

        self.assertEqual(calls, ["Wer ist Janina?"])
        self.assertTrue(result.primed)
        self.assertEqual(result.source, "materialized_built_sync")
        self.assertTrue(diagnostics.live_front_primed)
        self.assertEqual(diagnostics.live_front_prime_source, "materialized_built_sync")
        self.assertEqual(diagnostics.provider_context_source, "materialized_local_cache_hit")
        self.assertEqual(diagnostics.query_profile_canonical_english_text, "Who is Janina?")
        self.assertTrue(diagnostics.has_durable_context)
        self.assertTrue(diagnostics.has_subtext_context)
        self.assertEqual(diagnostics.subtext_seed_hits, ("corinna", "physiotherapist"))
        self.assertEqual(diagnostics.graph_seed_hits, ("corinna", "physiotherapist"))
        self.assertGreater(diagnostics.subtext_context_chars, 0)
        self.assertIn("twinr_silent_personalization_context_v1", diagnostics.subtext_context_preview or "")

    def test_summarize_distinguishes_execution_failures_from_personalization_gaps(self) -> None:
        failed_execution = SubtextEvalCaseResult(
            case_id="contact_role",
            category="contact",
            query_text="Wer ist Corinna?",
            should_use_personal_context=True,
            response_text="",
            explicit_memory_announcement=False,
            response_seed_hits=(),
            judge=SubtextJudgeResult(
                helpful_context_used=False,
                subtle_not_explicit=True,
                unforced=False,
                addresses_request=False,
                naturalness_score=1,
                passed=False,
                reason="Case execution failed.",
            ),
            model=None,
            request_id=None,
            response_id=None,
            prompt_tokens=None,
            output_tokens=None,
            diagnostics=SubtextEvalDiagnostics(
                stage_timings_s={"runtime_init": 0.8},
                system_message_count=0,
                system_message_chars=0,
                query_profile_original_text="Wer ist Corinna?",
                query_profile_canonical_english_text="Who is Corinna?",
                query_profile_retrieval_text="Who is Corinna?",
                has_subtext_context=False,
                has_topic_context=False,
                has_midterm_context=False,
                has_durable_context=False,
                has_episodic_context=False,
                has_graph_context=False,
                has_conflict_context=False,
                subtext_context_chars=0,
                durable_context_chars=0,
                episodic_context_chars=0,
                graph_context_chars=0,
                subtext_context_preview=None,
                durable_context_preview=None,
                episodic_context_preview=None,
                graph_context_preview=None,
                subtext_seed_hits=(),
                durable_seed_hits=(),
                episodic_seed_hits=(),
                graph_seed_hits=(),
                live_front_primed=False,
                live_front_prime_source=None,
                provider_context_source=None,
                execution_error_text="LongTermRemoteUnavailableError: queue_saturated",
                execution_root_cause_text="ChonkyDBError: status=429",
                execution_remote_write_context={"snapshot_kind": "graph_nodes"},
                failure_stage="seed_actions",
            ),
        )
        judged_but_not_personalized = SubtextEvalCaseResult(
            case_id="coffee_brand",
            category="preference",
            query_text="Wo kann ich heute Kaffee kaufen?",
            should_use_personal_context=True,
            response_text="Du koenntest heute im Supermarkt nach Kaffee schauen.",
            explicit_memory_announcement=False,
            response_seed_hits=(),
            judge=SubtextJudgeResult(
                helpful_context_used=False,
                subtle_not_explicit=True,
                unforced=True,
                addresses_request=True,
                naturalness_score=4,
                passed=False,
                reason="The reply stayed generic despite relevant personal context.",
            ),
            model="gpt-test",
            request_id="req-1",
            response_id="resp-1",
            prompt_tokens=100,
            output_tokens=30,
            diagnostics=SubtextEvalDiagnostics(
                stage_timings_s={"provider_context": 0.1, "model_response": 0.2, "judge": 0.1},
                system_message_count=2,
                system_message_chars=240,
                query_profile_original_text="Wo kann ich heute Kaffee kaufen?",
                query_profile_canonical_english_text="Where can I buy coffee today?",
                query_profile_retrieval_text="Where can I buy coffee today?",
                has_subtext_context=True,
                has_topic_context=False,
                has_midterm_context=False,
                has_durable_context=True,
                has_episodic_context=False,
                has_graph_context=False,
                has_conflict_context=False,
                subtext_context_chars=180,
                durable_context_chars=120,
                episodic_context_chars=0,
                graph_context_chars=0,
                subtext_context_preview="Melitta and Markt Z are relevant.",
                durable_context_preview="coffee preference",
                episodic_context_preview=None,
                graph_context_preview=None,
                subtext_seed_hits=("melitta", "markt z"),
                durable_seed_hits=("melitta",),
                episodic_seed_hits=(),
                graph_seed_hits=(),
                live_front_primed=True,
                live_front_prime_source="materialized_built_sync",
                provider_context_source="materialized_local_cache_hit",
                execution_error_text=None,
                execution_root_cause_text=None,
                execution_remote_write_context=None,
                failure_stage=None,
            ),
        )
        clean_control = SubtextEvalCaseResult(
            case_id="math_control",
            category="control",
            query_text="Was ist 17 mal 8?",
            should_use_personal_context=False,
            response_text="17 mal 8 ist 136.",
            explicit_memory_announcement=False,
            response_seed_hits=(),
            judge=SubtextJudgeResult(
                helpful_context_used=True,
                subtle_not_explicit=True,
                unforced=True,
                addresses_request=True,
                naturalness_score=5,
                passed=True,
                reason="The answer is direct and does not leak hidden memory.",
            ),
            model="gpt-test",
            request_id="req-2",
            response_id="resp-2",
            prompt_tokens=60,
            output_tokens=12,
            diagnostics=SubtextEvalDiagnostics(
                stage_timings_s={"provider_context": 0.05, "model_response": 0.06, "judge": 0.03},
                system_message_count=1,
                system_message_chars=64,
                query_profile_original_text="Was ist 17 mal 8?",
                query_profile_canonical_english_text="What is 17 times 8?",
                query_profile_retrieval_text="What is 17 times 8?",
                has_subtext_context=False,
                has_topic_context=False,
                has_midterm_context=False,
                has_durable_context=False,
                has_episodic_context=False,
                has_graph_context=False,
                has_conflict_context=False,
                subtext_context_chars=0,
                durable_context_chars=0,
                episodic_context_chars=0,
                graph_context_chars=0,
                subtext_context_preview=None,
                durable_context_preview=None,
                episodic_context_preview=None,
                graph_context_preview=None,
                subtext_seed_hits=(),
                durable_seed_hits=(),
                episodic_seed_hits=(),
                graph_seed_hits=(),
                live_front_primed=True,
                live_front_prime_source="materialized_built_sync",
                provider_context_source="materialized_local_cache_hit",
                execution_error_text=None,
                execution_root_cause_text=None,
                execution_remote_write_context=None,
                failure_stage=None,
            ),
        )

        summary = _summarize((failed_execution, judged_but_not_personalized, clean_control))

        self.assertEqual(summary.total_cases, 3)
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.execution_failed_cases, 1)
        self.assertEqual(summary.judge_failed_cases, 0)
        self.assertEqual(summary.judged_cases, 2)
        self.assertEqual(summary.failure_stage_counts, {"seed_actions": 1})
        self.assertEqual(summary.personalization_expected_cases, 2)
        self.assertEqual(summary.personalization_expected_with_context_cases, 1)
        self.assertEqual(summary.personalization_expected_with_seed_grounding_cases, 1)
        self.assertEqual(summary.personalization_expected_with_response_seed_hits, 0)
        self.assertEqual(summary.personalization_expected_helpful_cases, 0)
        self.assertEqual(summary.control_cases, 1)
        self.assertEqual(summary.control_cases_without_leak, 1)


if __name__ == "__main__":
    unittest.main()
