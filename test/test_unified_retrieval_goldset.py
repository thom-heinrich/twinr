from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalGoldsetCase,
    run_unified_retrieval_cases,
    seed_unified_retrieval_fixture,
    unified_retrieval_case_summary,
)
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


if __name__ == "__main__":
    unittest.main()
