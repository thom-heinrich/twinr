from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm import (
    LongTermConflictResolver,
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermRetriever,
    LongTermSourceRefV1,
    LongTermStructuredStore,
)
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.query_normalization import LongTermQueryProfile


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_recall_limit=3,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        user_display_name="Erika",
    )


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


class LongTermRetrieverTests(unittest.TestCase):
    def _make_retriever(self, temp_dir: str) -> tuple[LongTermRetriever, LongTermStructuredStore, PromptContextStore, TwinrPersonalGraphStore]:
        config = _config(temp_dir)
        prompt_context_store = PromptContextStore.from_config(config)
        graph_store = TwinrPersonalGraphStore.from_config(config)
        object_store = LongTermStructuredStore.from_config(config)
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=prompt_context_store,
            graph_store=graph_store,
            object_store=object_store,
            conflict_resolver=LongTermConflictResolver(),
            subtext_builder=LongTermSubtextBuilder(config=config, graph_store=graph_store),
        )
        return retriever, object_store, prompt_context_store, graph_store

    def test_build_context_assembles_structured_memory_packet(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            details="Use the mobile number ending in 1234.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555558877.",
            details="Use the office number ending in 8877.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555558877",
            attributes={"person_ref": "person:corinna_maier"},
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, prompt_context_store, graph_store = self._make_retriever(temp_dir)
            graph_store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone="5551234",
                role="Physiotherapist",
            )
            prompt_context_store.memory_store.remember(
                kind="episodic_turn",
                summary='Conversation about "Corinna called earlier today."',
                details='User said: "Corinna called earlier today." Twinr answered: "I can keep that in mind."',
            )
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(existing,),
                    deferred_objects=(candidate,),
                    conflicts=(conflict,),
                    graph_edges=(),
                )
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Was ist Corinnas Nummer?",
                    canonical_english_text="What is Corinna's phone number?",
                ),
                original_query_text="Was ist Corinnas Nummer?",
            )

        self.assertIsNotNone(context.episodic_context)
        self.assertIsNotNone(context.durable_context)
        self.assertIsNotNone(context.graph_context)
        self.assertIsNotNone(context.conflict_context)
        self.assertIn("Corinna called earlier today", context.episodic_context or "")
        self.assertIn("+15555551234", context.durable_context or "")
        self.assertIn("Corinna Maier", context.graph_context or "")
        self.assertIn("contact:person:corinna_maier:phone", context.conflict_context or "")

    def test_select_conflict_queue_respects_canonical_query_profile(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555558877.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555558877",
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store = self._make_retriever(temp_dir)
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(existing,),
                    deferred_objects=(candidate,),
                    conflicts=(conflict,),
                    graph_edges=(),
                )
            )

            matching = retriever.select_conflict_queue(
                query=LongTermQueryProfile.from_text(
                    "Wie lautet Corinnas Telefonnummer?",
                    canonical_english_text="What is Corinna's phone number?",
                )
            )
            unrelated = retriever.select_conflict_queue(
                query=LongTermQueryProfile.from_text(
                    "Was ist 27 mal 14?",
                    canonical_english_text="What is 27 times 14?",
                )
            )

        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertEqual(unrelated, ())


if __name__ == "__main__":
    unittest.main()
