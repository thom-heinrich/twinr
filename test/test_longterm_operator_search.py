from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.context_store import PersistentMemoryEntry
from twinr.memory.longterm.core.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.retrieval.operator_search import LongTermOperatorSearch
from twinr.memory.query_normalization import LongTermQueryProfile


class _FakeQueryRewriter:
    def profile(
        self,
        query_text: str | None,
        *,
        wait_for_rewrite_s: float = 0.0,
    ) -> LongTermQueryProfile:
        del wait_for_rewrite_s
        return LongTermQueryProfile.from_text(query_text, canonical_english_text="lea lentil soup")


class _FakeObjectStore:
    def select_relevant_objects(self, *, query_text: str | None, limit: int = 0):
        return (
            LongTermMemoryObjectV1(
                memory_id="fact:lea_visit",
                kind="fact",
                summary="Lea bringt heute Abend Linsensuppe vorbei.",
                details="Heute um 19 Uhr mit Thermoskanne.",
                status="active",
                confidence=0.93,
                source=LongTermSourceRefV1(source_type="conversation"),
            ),
        )[:limit]

    def select_relevant_episodic_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 0,
        fallback_limit: int = 0,
        require_query_match: bool = False,
    ):
        return (
            LongTermMemoryObjectV1(
                memory_id="episode:lea_dropoff",
                kind="episode",
                summary="Lea kuendigte den Suppenbesuch an.",
                details="Sie bringt eine Thermoskanne mit.",
                status="active",
                confidence=0.85,
                source=LongTermSourceRefV1(source_type="conversation"),
            ),
        )[:limit]


class _FakeMidtermStore:
    def select_relevant_packets(self, query_text: str | None, *, limit: int = 0):
        return (
            LongTermMidtermPacketV1(
                packet_id="midterm:lea_soup",
                kind="visit",
                summary="Lea bringt heute Abend Suppe vorbei.",
                details="Thermoskanne, 19 Uhr.",
                query_hints=("lea", "linsensuppe"),
            ),
        )[:limit]


class _FakeRetriever:
    def _episodic_entry_from_object(self, item: LongTermMemoryObjectV1):
        return PersistentMemoryEntry(
            entry_id=item.memory_id,
            kind=item.kind,
            summary=item.summary,
            details=item.details,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def select_conflict_queue(self, *, query: LongTermQueryProfile, limit: int | None = None):
        return (
            LongTermConflictQueueItemV1(
                slot_key="favorite_drink",
                question="Welches Getraenk stimmt jetzt?",
                reason="Zwei widerspruechliche Erinnerungen sind noch offen.",
                candidate_memory_id="fact:oolong",
                options=(
                    LongTermConflictOptionV1(
                        memory_id="fact:oolong",
                        summary="Oolong-Tee",
                        status="candidate",
                    ),
                    LongTermConflictOptionV1(
                        memory_id="fact:coffee",
                        summary="Kaffee",
                        status="active",
                    ),
                ),
            ),
        )[: limit or 1]


class _FakeGraphStore:
    def build_prompt_context(self, query_text: str | None):
        return "Graph says Lea is the daughter and the visit is tonight."


class _FailingQueryRewriter:
    def profile(
        self,
        query_text: str | None,
        *,
        wait_for_rewrite_s: float = 0.0,
    ) -> LongTermQueryProfile:
        del query_text
        del wait_for_rewrite_s
        raise AssertionError("conversation recap operator search must not call the query rewriter")


class _RecapOnlyObjectStore(_FakeObjectStore):
    def select_relevant_objects(self, *, query_text: str | None, limit: int = 0):
        del query_text
        del limit
        raise AssertionError("conversation recap operator search must skip durable-object selection")


class _FailingMidtermStore(_FakeMidtermStore):
    def select_relevant_packets(self, query_text: str | None, *, limit: int = 0):
        del query_text
        del limit
        raise AssertionError("conversation recap operator search must skip midterm selection")


class _FailingRetriever(_FakeRetriever):
    def select_conflict_queue(self, *, query: LongTermQueryProfile, limit: int | None = None):
        del query
        del limit
        raise AssertionError("conversation recap operator search must skip conflict selection")


class _FailingGraphStore(_FakeGraphStore):
    def build_prompt_context(self, query_text: str | None):
        del query_text
        raise AssertionError("conversation recap operator search must skip graph-context rendering")


class LongTermOperatorSearchTests(unittest.TestCase):
    def test_search_groups_real_memory_sections(self) -> None:
        search = LongTermOperatorSearch(
            query_rewriter=_FakeQueryRewriter(),
            retriever=_FakeRetriever(),
            graph_store=_FakeGraphStore(),
            object_store=_FakeObjectStore(),
            midterm_store=_FakeMidtermStore(),
            result_limit=4,
        )

        result = search.search("Was bringt Lea heute Abend vorbei?")

        self.assertEqual(result.query_profile.canonical_english_text, "lea lentil soup")
        self.assertEqual(len(result.durable_objects), 1)
        self.assertEqual(len(result.episodic_entries), 1)
        self.assertEqual(len(result.midterm_packets), 1)
        self.assertEqual(len(result.conflict_queue), 1)
        self.assertEqual(result.total_hits, 4)
        self.assertIn("daughter", result.graph_context or "")

    def test_conversation_recap_search_uses_episodic_only_fast_plan(self) -> None:
        search = LongTermOperatorSearch(
            query_rewriter=_FailingQueryRewriter(),
            retriever=_FailingRetriever(),
            graph_store=_FailingGraphStore(),
            object_store=_RecapOnlyObjectStore(),
            midterm_store=_FailingMidtermStore(),
            result_limit=4,
        )

        result = search.search("Worüber haben wir heute gesprochen?")

        self.assertIsNone(result.query_profile.canonical_english_text)
        self.assertEqual(len(result.episodic_entries), 1)
        self.assertEqual(len(result.durable_objects), 0)
        self.assertEqual(len(result.midterm_packets), 0)
        self.assertEqual(len(result.conflict_queue), 0)
        self.assertIsNone(result.graph_context)
        self.assertEqual(result.total_hits, 1)


if __name__ == "__main__":
    unittest.main()
