from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrGraphContactOption, TwinrGraphWriteResult
from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalGoldsetCase,
    UnifiedRetrievalGoldsetCaseResult,
    run_unified_retrieval_cases,
    seed_unified_retrieval_fixture,
    unified_retrieval_case_profile_memory_type_coverage,
    unified_retrieval_goldset_cases,
    unified_retrieval_case_summary,
    wait_for_unified_retrieval_cases,
)
from twinr.memory.longterm.evaluation.eval import _seed_contacts
from twinr.memory.longterm.runtime.service import LongTermMemoryService


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        user_display_name="Erika",
        openai_web_search_timezone="Europe/Berlin",
    )


class UnifiedRetrievalGoldsetSharedTests(unittest.TestCase):
    def test_expanded_profile_has_50_cases_and_memory_type_coverage(self) -> None:
        cases = unified_retrieval_goldset_cases(profile="expanded")
        coverage = dict(unified_retrieval_case_profile_memory_type_coverage(profile="expanded"))

        self.assertEqual(len(cases), 50)
        for memory_type in ("adaptive", "conflict", "durable", "episodic", "graph", "midterm"):
            self.assertGreaterEqual(coverage.get(memory_type, 0), 30)

    def test_local_fixture_covers_full_stack_and_graph_only_cases(self) -> None:
        local_cases = (
            UnifiedRetrievalGoldsetCase(
                case_id="local_corinna_phone_full_stack",
                query_text="What is Corinna Maier's phone number?",
                canonical_query_text="What is Corinna Maier's phone number?",
                required_candidate_sources=("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
                required_selected_ids={
                    "episodic_entry_ids": ("episode:corinna_called",),
                    "durable_memory_ids": ("fact:corinna_phone_current", "fact:corinna_phone_old"),
                    "conflict_slot_keys": ("contact:person:corinna_maier:phone",),
                    "midterm_packet_ids": ("midterm:corinna_today",),
                    "adaptive_packet_ids": ("adaptive:confirmed:fact_corinna_phone_current",),
                },
                required_join_anchors={
                    "person_ref:person:corinna_maier": ("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
                },
                required_access_path=("structured_query_first", "graph_document_load"),
                required_sections=(
                    "subtext_context",
                    "midterm_context",
                    "durable_context",
                    "episodic_context",
                    "graph_context",
                    "conflict_context",
                ),
                required_context_terms={
                    "durable_context": ("+15555558877",),
                    "episodic_context": ("Corinna called earlier today",),
                    "graph_context": ("Corinna Maier",),
                    "conflict_context": ("contact:person:corinna_maier:phone",),
                },
            ),
            UnifiedRetrievalGoldsetCase(
                case_id="local_anna_email_graph_only",
                query_text="What is Anna Becker's email address?",
                canonical_query_text="What is Anna Becker's email address?",
                required_candidate_sources=("graph",),
                required_access_path=("graph_document_load",),
                required_sections=("graph_context",),
                forbidden_sections=(
                    "midterm_context",
                    "durable_context",
                    "episodic_context",
                    "conflict_context",
                ),
                required_context_terms={"graph_context": ("Anna Becker", "anna.becker@example.com")},
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "personality").mkdir(parents=True, exist_ok=True)
            service = LongTermMemoryService.from_config(_config(temp_dir))
            try:
                seed_stats = seed_unified_retrieval_fixture(service)
                case_results = run_unified_retrieval_cases(
                    service=service,
                    cases=local_cases,
                    phase="local",
                )
            finally:
                service.shutdown(timeout_s=30.0)

        total_cases, passed_cases, failed_case_ids = unified_retrieval_case_summary(case_results)
        self.assertEqual(seed_stats.episodic_objects, 2)
        self.assertEqual(seed_stats.durable_objects, 3)
        self.assertEqual(seed_stats.conflict_count, 1)
        self.assertEqual(seed_stats.midterm_packets, 2)
        self.assertGreaterEqual(seed_stats.graph_nodes, 4)
        self.assertGreaterEqual(seed_stats.graph_edges, 2)
        self.assertEqual(total_cases, 2)
        self.assertEqual(passed_cases, 2, failed_case_ids)
        self.assertFalse(failed_case_ids)

    def test_mixed_graph_seed_creates_anna_becker_after_existing_becker_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "personality").mkdir(parents=True, exist_ok=True)
            service = LongTermMemoryService.from_config(_config(temp_dir))
            try:
                _seed_contacts(service.graph_store)
                seed_unified_retrieval_fixture(service)
                lookup = service.graph_store.lookup_contact(
                    name="Anna",
                    family_name="Becker",
                    role="Daughter",
                )
            finally:
                service.shutdown(timeout_s=30.0)

        self.assertEqual(lookup.status, "found")
        assert lookup.match is not None
        self.assertEqual(lookup.match.label, "Anna Becker")
        self.assertEqual(lookup.match.emails, ("anna.becker@example.com",))

    def test_seed_fixture_fails_closed_when_graph_contact_seed_needs_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "personality").mkdir(parents=True, exist_ok=True)
            service = LongTermMemoryService.from_config(_config(temp_dir))
            original_remember_contact = service.graph_store.remember_contact

            def _side_effect(*args, **kwargs):
                if kwargs.get("given_name") == "Anna" and kwargs.get("family_name") == "Becker":
                    return TwinrGraphWriteResult(
                        status="needs_clarification",
                        label="Anna",
                        node_id="",
                        question="Did you mean Chris Becker?",
                        options=(
                            TwinrGraphContactOption(
                                person_node_id="person:chris_becker",
                                label="Chris Becker",
                                role="Physiotherapist",
                                emails=("chris.becker@example.com",),
                            ),
                        ),
                    )
                return original_remember_contact(*args, **kwargs)

            try:
                with patch.object(service.graph_store, "remember_contact", side_effect=_side_effect):
                    with self.assertRaisesRegex(RuntimeError, "Anna Becker"):
                        seed_unified_retrieval_fixture(service)
            finally:
                service.shutdown(timeout_s=30.0)

    def test_wait_for_unified_retrieval_cases_reruns_only_pending_cases(self) -> None:
        cases = (
            UnifiedRetrievalGoldsetCase(
                case_id="case_a",
                query_text="A",
                canonical_query_text="A",
            ),
            UnifiedRetrievalGoldsetCase(
                case_id="case_b",
                query_text="B",
                canonical_query_text="B",
            ),
        )
        initial_results = (
            UnifiedRetrievalGoldsetCaseResult(
                case_id="case_a",
                phase="writer",
                query_text="A",
            ),
            UnifiedRetrievalGoldsetCaseResult(
                case_id="case_b",
                phase="writer",
                query_text="B",
                missing_sections=("graph_context",),
            ),
        )
        rerun_results = (
            UnifiedRetrievalGoldsetCaseResult(
                case_id="case_b",
                phase="writer",
                query_text="B",
            ),
        )
        observed_case_ids: list[tuple[str, ...]] = []

        def _fake_run(*, service, cases, phase):
            del service, phase
            observed_case_ids.append(tuple(case.case_id for case in cases))
            if len(observed_case_ids) == 1:
                return initial_results
            return rerun_results

        with (
            patch(
                "twinr.memory.longterm.evaluation._unified_retrieval_shared.run_unified_retrieval_cases",
                side_effect=_fake_run,
            ),
            patch("twinr.memory.longterm.evaluation._unified_retrieval_shared.time.sleep"),
        ):
            results = wait_for_unified_retrieval_cases(
                service=object(),
                cases=cases,
                phase="writer",
                timeout_s=5.0,
                poll_interval_s=0.1,
            )

        self.assertEqual(observed_case_ids, [("case_a", "case_b"), ("case_b",)])
        self.assertEqual(tuple(item.case_id for item in results), ("case_a", "case_b"))
        self.assertTrue(all(item.passed for item in results))


if __name__ == "__main__":
    unittest.main()
