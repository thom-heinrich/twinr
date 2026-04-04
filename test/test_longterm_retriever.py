from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphNodeV1
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm import (
    LongTermConflictResolver,
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMidtermPacketV1,
    LongTermMemoryObjectV1,
    LongTermReflectionResultV1,
    LongTermMidtermStore,
    LongTermProactiveCandidateV1,
    LongTermProactiveStateStore,
    LongTermRetriever,
    LongTermSourceRefV1,
    LongTermStructuredStore,
)
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.retrieval.unified_plan import (
    UnifiedGraphSelectionInput,
    attach_adaptive_packets_to_query_plan,
    build_episodic_selection_input,
    build_unified_retrieval_selection,
)
from twinr.memory.query_normalization import LongTermQueryProfile


_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"
_TEST_CORINNA_GRAPH_PHONE = "5551234"


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
    def _make_retriever(
        self,
        temp_dir: str,
        *,
        adaptive_policy_builder: LongTermAdaptivePolicyBuilder | None = None,
    ) -> tuple[
        LongTermRetriever,
        LongTermStructuredStore,
        PromptContextStore,
        TwinrPersonalGraphStore,
        LongTermMidtermStore,
    ]:
        config = _config(temp_dir)
        prompt_context_store = PromptContextStore.from_config(config)
        graph_store = TwinrPersonalGraphStore.from_config(config)
        object_store = LongTermStructuredStore.from_config(config)
        midterm_store = LongTermMidtermStore.from_config(config)
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=prompt_context_store,
            graph_store=graph_store,
            object_store=object_store,
            midterm_store=midterm_store,
            conflict_resolver=LongTermConflictResolver(),
            subtext_builder=LongTermSubtextBuilder(config=config, graph_store=graph_store),
            adaptive_policy_builder=adaptive_policy_builder,
        )
        return retriever, object_store, prompt_context_store, graph_store, midterm_store

    def test_build_context_assembles_structured_memory_packet(self) -> None:
        episode = LongTermMemoryObjectV1(
            memory_id="episode:corinna_called",
            kind="episode",
            summary='Conversation about "Corinna called earlier today."',
            details='User said: "Corinna called earlier today." Twinr answered: "I can keep that in mind."',
            source=_source("turn:0"),
            status="active",
            confidence=1.0,
            attributes={
                "raw_transcript": "Corinna called earlier today.",
                "raw_response": "I can keep that in mind.",
            },
        )
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
            details="Use the mobile number ending in 1234.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_OLD,
            attributes={"person_ref": "person:corinna_maier"},
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
            details="Use the office number ending in 8877.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_NEW,
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
            retriever, object_store, prompt_context_store, graph_store, midterm_store = self._make_retriever(temp_dir)
            graph_store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_GRAPH_PHONE,
                role="Physiotherapist",
            )
            prompt_context_store.memory_store.remember(
                kind="episodic_turn",
                summary='Conversation about "This local-only memory should not be used."',
                details='User said: "This local-only memory should not be used." Twinr answered: "Ignored."',
            )
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(episode,),
                    durable_objects=(existing,),
                    deferred_objects=(candidate,),
                    conflicts=(conflict,),
                    graph_edges=(),
                )
            )
            midterm_store.save_packets(
                packets=(
                    midterm_store.packet_type(
                        packet_id="midterm:corinna_today",
                        kind="recent_contact_bundle",
                        summary="Corinna Maier is a recent practical contact and current phone questions may require disambiguation.",
                        details="Useful when the user asks for Corinna's number or whether to call her.",
                        source_memory_ids=("fact:corinna_phone_old", "fact:corinna_phone_new"),
                        query_hints=("corinna", "phone", "number"),
                        sensitivity="normal",
                    ),
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
        self.assertIsNotNone(context.midterm_context)
        self.assertIsNotNone(context.durable_context)
        self.assertIsNotNone(context.graph_context)
        self.assertIsNotNone(context.conflict_context)
        self.assertIn("Corinna called earlier today", context.episodic_context or "")
        self.assertIn("recent_contact_bundle", context.midterm_context or "")
        self.assertIn(_TEST_CORINNA_PHONE_OLD, context.durable_context or "")
        self.assertIn("Corinna Maier", context.graph_context or "")
        self.assertIn("contact:person:corinna_maier:phone", context.conflict_context or "")
        self.assertNotIn("This local-only memory should not be used", context.episodic_context or "")

    def test_unified_retrieval_selection_joins_graph_and_structured_candidates_on_explicit_person_anchor(self) -> None:
        related_object = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            source=_source("turn:corinna"),
            status="active",
            confidence=0.97,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        unrelated_object = LongTermMemoryObjectV1(
            memory_id="fact:janina_phone",
            kind="contact_method_fact",
            summary="Janina can be reached at +15555550001.",
            source=_source("turn:janina"),
            status="active",
            confidence=0.9,
            slot_key="contact:person:janina:phone",
            value_key="+15555550001",
            attributes={"person_ref": "person:janina"},
        )
        related_conflict = LongTermConflictQueueItemV1(
            slot_key="contact:person:corinna_maier:phone",
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
            candidate_memory_id="fact:corinna_phone",
            options=(
                LongTermConflictOptionV1(
                    memory_id="fact:corinna_phone",
                    summary=related_object.summary,
                    status="active",
                    value_key="+15555551234",
                ),
            ),
        )
        unrelated_conflict = LongTermConflictQueueItemV1(
            slot_key="contact:person:janina:phone",
            question="Which phone number should I use for Janina?",
            reason="Conflicting phone numbers exist.",
            candidate_memory_id="fact:janina_phone",
            options=(
                LongTermConflictOptionV1(
                    memory_id="fact:janina_phone",
                    summary=unrelated_object.summary,
                    status="active",
                    value_key="+15555550001",
                ),
            ),
        )
        graph_selection = UnifiedGraphSelectionInput(
            document=TwinrGraphDocumentV1(
                subject_node_id="user:main",
                graph_id="graph:user_main",
                created_at="2026-03-30T09:00:00Z",
                updated_at="2026-03-30T09:00:00Z",
                nodes=(
                    TwinrGraphNodeV1(
                        node_id="user:main",
                        node_type="user",
                        label="Erika",
                    ),
                    TwinrGraphNodeV1(
                        node_id="person:corinna_maier",
                        node_type="person",
                        label="Corinna Maier",
                    ),
                ),
                edges=(),
                metadata={"kind": "personal_graph"},
            ),
            query_plan={
                "mode": "remote_query_first_subgraph",
                "matched_node_ids": ["person:corinna_maier"],
                "selected_node_ids": ["person:corinna_maier"],
                "selected_edge_ids": [],
                "access_path": ["graph_neighbors_query"],
            },
        )

        selection = build_unified_retrieval_selection(
            query_texts=("What is Corinna's phone number?",),
            episodic_entries=(),
            durable_objects=(unrelated_object, related_object),
            conflict_queue=(unrelated_conflict, related_conflict),
            conflict_supporting_objects=(unrelated_object, related_object),
            midterm_packets=(),
            graph_selection=graph_selection,
        )

        self.assertEqual(selection.durable_objects[0].memory_id, "fact:corinna_phone")
        self.assertEqual(selection.conflict_queue[0].slot_key, "contact:person:corinna_maier:phone")
        join_anchors = {
            item["anchor"]: item["sources"]
            for item in selection.query_plan["join_anchors"]
        }
        self.assertEqual(
            join_anchors["person_ref:person:corinna_maier"],
            ["conflict", "durable", "graph"],
        )
        self.assertIn("structured_query_first", selection.query_plan["access_path"])
        self.assertIn("graph_neighbors_query", selection.query_plan["access_path"])

    def test_unified_retrieval_selection_reorders_midterm_packets_on_shared_explicit_anchors(self) -> None:
        related_object = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            source=_source("turn:corinna"),
            status="active",
            confidence=0.97,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        unrelated_object = LongTermMemoryObjectV1(
            memory_id="fact:janina_phone",
            kind="contact_method_fact",
            summary="Janina can be reached at +15555550001.",
            source=_source("turn:janina"),
            status="active",
            confidence=0.9,
            slot_key="contact:person:janina:phone",
            value_key="+15555550001",
            attributes={"person_ref": "person:janina"},
        )
        related_midterm = LongTermMidtermPacketV1(
            packet_id="midterm:corinna_today",
            kind="recent_contact_bundle",
            summary="Corinna may matter for today's practical phone decision.",
            source_memory_ids=("fact:corinna_phone",),
            attributes={"person_ref": "person:corinna_maier"},
        )
        unrelated_midterm = LongTermMidtermPacketV1(
            packet_id="midterm:janina_today",
            kind="recent_contact_bundle",
            summary="Janina may matter for another phone decision.",
            source_memory_ids=("fact:janina_phone",),
            attributes={"person_ref": "person:janina"},
        )
        graph_selection = UnifiedGraphSelectionInput(
            document=TwinrGraphDocumentV1(
                subject_node_id="user:main",
                graph_id="graph:user_main",
                created_at="2026-03-30T09:00:00Z",
                updated_at="2026-03-30T09:00:00Z",
                nodes=(
                    TwinrGraphNodeV1(
                        node_id="user:main",
                        node_type="user",
                        label="Erika",
                    ),
                    TwinrGraphNodeV1(
                        node_id="person:corinna_maier",
                        node_type="person",
                        label="Corinna Maier",
                    ),
                ),
                edges=(),
                metadata={"kind": "personal_graph"},
            ),
            query_plan={
                "mode": "remote_query_first_subgraph",
                "matched_node_ids": ["person:corinna_maier"],
                "selected_node_ids": ["person:corinna_maier"],
                "selected_edge_ids": [],
                "access_path": ["graph_neighbors_query"],
            },
        )

        selection = build_unified_retrieval_selection(
            query_texts=("Should I call Corinna today?",),
            episodic_entries=(),
            durable_objects=(unrelated_object, related_object),
            conflict_queue=(),
            conflict_supporting_objects=(unrelated_object, related_object),
            midterm_packets=(unrelated_midterm, related_midterm),
            graph_selection=graph_selection,
        )

        self.assertEqual(selection.midterm_packets[0].packet_id, "midterm:corinna_today")
        self.assertEqual(
            selection.query_plan["selected"]["midterm_packet_ids"],
            ["midterm:corinna_today"],
        )
        join_anchors = {
            item["anchor"]: item["sources"]
            for item in selection.query_plan["join_anchors"]
        }
        self.assertEqual(
            join_anchors["person_ref:person:corinna_maier"],
            ["graph", "midterm"],
        )

    def test_unified_retrieval_selection_keeps_disconnected_practical_meta_memory_when_only_graph_direct_match_is_user_main(self) -> None:
        jam_generic = LongTermMemoryObjectV1(
            memory_id="fact:jam_generic",
            kind="fact",
            summary="User usually likes some jam on bread at breakfast.",
            source=_source("turn:jam_generic"),
            status="active",
            confidence=0.85,
            slot_key="fact:user:breakfast:jam",
            value_key="jam_on_bread_at_breakfast",
        )
        jam_new = LongTermMemoryObjectV1(
            memory_id="fact:jam_preference_new",
            kind="fact",
            summary="Inzwischen magst du lieber Aprikosenmarmelade.",
            details="Neuere Vorliebe fuer das Fruehstueck.",
            source=_source("turn:jam_new"),
            status="active",
            confirmed_by_user=True,
            confidence=0.95,
            slot_key="preference:breakfast:jam",
            value_key="apricot",
        )
        graph_selection = UnifiedGraphSelectionInput(
            document=TwinrGraphDocumentV1(
                subject_node_id="user:main",
                graph_id="graph:user_main",
                created_at="2026-03-30T09:00:00Z",
                updated_at="2026-03-30T09:00:00Z",
                nodes=(
                    TwinrGraphNodeV1(
                        node_id="user:main",
                        node_type="user",
                        label="Main user",
                    ),
                ),
                edges=(),
                metadata={"kind": "personal_graph"},
            ),
            query_plan={
                "mode": "remote_query_first_subgraph",
                "matched_node_ids": ["user:main"],
                "selected_node_ids": ["user:main"],
                "selected_edge_ids": [],
                "access_path": ["graph_path_query"],
            },
        )

        selection = build_unified_retrieval_selection(
            query_texts=("Welche Marmelade ist jetzt als bestaetigt gespeichert?",),
            episodic_entries=(),
            durable_objects=(jam_new, jam_generic),
            conflict_queue=(),
            conflict_supporting_objects=(),
            midterm_packets=(),
            graph_selection=graph_selection,
        )

        self.assertEqual(selection.query_plan["pruning"]["mode"], "practical")
        self.assertEqual(selection.query_plan["selected"]["durable_memory_ids"][0], "fact:jam_preference_new")
        self.assertEqual(
            tuple(item.memory_id for item in selection.durable_objects[:2]),
            ("fact:jam_preference_new", "fact:jam_generic"),
        )

    def test_unified_retrieval_selection_prefers_confirmed_meta_memory_fact_across_query_variants(self) -> None:
        thermos = LongTermMemoryObjectV1(
            memory_id="fact:thermos_location_old",
            kind="fact",
            summary="Früher stand die rote Thermoskanne im Flurschrank.",
            details="Historische Ortsangabe zur roten Thermoskanne.",
            source=_source("turn:thermos"),
            status="active",
            confidence=0.99,
            confirmed_by_user=True,
            slot_key="object:red_thermos:location",
            value_key="hallway_cupboard",
        )
        jam_generic = LongTermMemoryObjectV1(
            memory_id="fact:jam_generic",
            kind="fact",
            summary="User usually likes some jam on bread at breakfast.",
            details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
            source=_source("turn:jam_generic"),
            status="active",
            confidence=0.84,
            slot_key="fact:user:breakfast:jam",
            value_key="jam_on_bread_at_breakfast",
            attributes={
                "fact_type": "general",
                "memory_domain": "general",
                "value_text": "jam on bread at breakfast",
            },
        )
        jam_new = LongTermMemoryObjectV1(
            memory_id="fact:jam_preference_new",
            kind="fact",
            summary="Inzwischen magst du lieber Aprikosenmarmelade.",
            details="Neuere Vorliebe fuer das Fruehstueck.",
            source=_source("turn:jam_new"),
            status="active",
            confirmed_by_user=True,
            confidence=0.95,
            slot_key="preference:breakfast:jam",
            value_key="apricot",
        )
        jam_restart = LongTermMidtermPacketV1(
            packet_id="adaptive:restart:fact_jam_preference_new",
            kind="restart_recall",
            summary="Der bestaetigte aktuelle Marmeladeneintrag lautet Aprikosenmarmelade.",
            source_memory_ids=("fact:jam_preference_new",),
            attributes={"memory_domain": "general"},
        )
        thermos_restart = LongTermMidtermPacketV1(
            packet_id="adaptive:restart:fact_thermos_location_old",
            kind="restart_recall",
            summary="Die rote Thermoskanne stand frueher im Flurschrank.",
            source_memory_ids=("fact:thermos_location_old",),
            attributes={"memory_domain": "general"},
        )

        selection = build_unified_retrieval_selection(
            query_texts=(
                "Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                "Which marmalade is currently saved as confirmed?",
            ),
            episodic_entries=(),
            durable_objects=(jam_new, thermos, jam_generic),
            conflict_queue=(),
            conflict_supporting_objects=(),
            midterm_packets=(jam_restart, thermos_restart),
            graph_selection=None,
        )

        self.assertEqual(selection.durable_objects[0].memory_id, "fact:jam_preference_new")
        self.assertEqual(
            selection.query_plan["selected"]["durable_memory_ids"],
            ["fact:jam_preference_new", "fact:thermos_location_old", "fact:jam_generic"],
        )
        kept_payload_by_id = {
            item["id"]: item
            for item in selection.query_plan["candidates"]
            if item["source"] == "durable"
        }
        self.assertIn("confirmed", kept_payload_by_id["fact:jam_preference_new"]["matched_query_terms"])

    def test_unified_retrieval_selection_reorders_episodic_entries_on_shared_explicit_anchors(self) -> None:
        related_object = LongTermMemoryObjectV1(
            memory_id="episode:corinna_called",
            kind="episode",
            summary='Conversation about "Corinna called earlier today."',
            details='User said: "Corinna called earlier today."',
            source=_source("turn:corinna"),
            status="active",
            confidence=1.0,
            attributes={"person_ref": "person:corinna_maier"},
        )
        unrelated_object = LongTermMemoryObjectV1(
            memory_id="episode:janina_called",
            kind="episode",
            summary='Conversation about "Janina called earlier today."',
            details='User said: "Janina called earlier today."',
            source=_source("turn:janina"),
            status="active",
            confidence=1.0,
            attributes={"person_ref": "person:janina"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            related_entry = retriever._episodic_entry_from_object(related_object)
            unrelated_entry = retriever._episodic_entry_from_object(unrelated_object)
            self.assertIsNotNone(related_entry)
            self.assertIsNotNone(unrelated_entry)
            graph_selection = UnifiedGraphSelectionInput(
                document=TwinrGraphDocumentV1(
                    subject_node_id="user:main",
                    graph_id="graph:user_main",
                    created_at="2026-03-30T09:00:00Z",
                    updated_at="2026-03-30T09:00:00Z",
                    nodes=(
                        TwinrGraphNodeV1(
                            node_id="user:main",
                            node_type="user",
                            label="Erika",
                        ),
                        TwinrGraphNodeV1(
                            node_id="person:corinna_maier",
                            node_type="person",
                            label="Corinna Maier",
                        ),
                    ),
                    edges=(),
                    metadata={"kind": "personal_graph"},
                ),
                query_plan={
                    "mode": "remote_query_first_subgraph",
                    "matched_node_ids": ["person:corinna_maier"],
                    "selected_node_ids": ["person:corinna_maier"],
                    "selected_edge_ids": [],
                    "access_path": ["graph_neighbors_query"],
                },
            )

            selection = build_unified_retrieval_selection(
                query_texts=("What happened with Corinna earlier today?",),
                episodic_entries=(
                    build_episodic_selection_input(entry=unrelated_entry, source_object=unrelated_object),
                    build_episodic_selection_input(entry=related_entry, source_object=related_object),
                ),
                durable_objects=(),
                conflict_queue=(),
                conflict_supporting_objects=(),
                midterm_packets=(),
                graph_selection=graph_selection,
            )

        self.assertEqual(selection.episodic_entries[0].entry_id, "episode:corinna_called")
        self.assertEqual(
            selection.query_plan["selected"]["episodic_entry_ids"],
            ["episode:corinna_called"],
        )
        join_anchors = {
            item["anchor"]: item["sources"]
            for item in selection.query_plan["join_anchors"]
        }
        self.assertEqual(
            join_anchors["person_ref:person:corinna_maier"],
            ["episodic", "graph"],
        )

    def test_unified_retrieval_selection_prunes_practical_candidates_for_continuity_queries(self) -> None:
        episodic_object = LongTermMemoryObjectV1(
            memory_id="episode:corinna_called",
            kind="episode",
            summary='Conversation about "Corinna called earlier today."',
            details='User said: "Corinna called earlier today."',
            source=_source("turn:episode"),
            status="active",
            confidence=1.0,
            attributes={"person_ref": "person:corinna_maier"},
        )
        durable_object = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            source=_source("turn:durable"),
            status="active",
            confidence=0.97,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        related_midterm = LongTermMidtermPacketV1(
            packet_id="midterm:corinna_today",
            kind="recent_contact_bundle",
            summary="Corinna called earlier today and may matter for current continuity.",
            source_memory_ids=("fact:corinna_phone",),
            attributes={"person_ref": "person:corinna_maier"},
        )
        related_conflict = LongTermConflictQueueItemV1(
            slot_key="contact:person:corinna_maier:phone",
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
            candidate_memory_id="fact:corinna_phone",
            options=(
                LongTermConflictOptionV1(
                    memory_id="fact:corinna_phone",
                    summary=durable_object.summary,
                    status="active",
                    value_key="+15555551234",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            episodic_entry = retriever._episodic_entry_from_object(episodic_object)
            self.assertIsNotNone(episodic_entry)
            graph_selection = UnifiedGraphSelectionInput(
                document=TwinrGraphDocumentV1(
                    subject_node_id="user:main",
                    graph_id="graph:user_main",
                    created_at="2026-03-30T09:00:00Z",
                    updated_at="2026-03-30T09:00:00Z",
                    nodes=(
                        TwinrGraphNodeV1(
                            node_id="user:main",
                            node_type="user",
                            label="Erika",
                        ),
                        TwinrGraphNodeV1(
                            node_id="person:corinna_maier",
                            node_type="person",
                            label="Corinna Maier",
                        ),
                        TwinrGraphNodeV1(
                            node_id="phone:5551234",
                            node_type="phone",
                            label="5551234",
                        ),
                    ),
                    edges=(),
                    metadata={"kind": "personal_graph"},
                ),
                query_plan={
                    "mode": "remote_query_first_subgraph",
                    "matched_node_ids": ["person:corinna_maier"],
                    "selected_node_ids": ["user:main", "person:corinna_maier", "phone:5551234"],
                    "selected_edge_ids": [],
                    "access_path": ["graph_neighbors_query"],
                },
            )

            selection = build_unified_retrieval_selection(
                query_texts=("Did Corinna call earlier today?",),
                episodic_entries=(build_episodic_selection_input(entry=episodic_entry, source_object=episodic_object),),
                durable_objects=(durable_object,),
                conflict_queue=(related_conflict,),
                conflict_supporting_objects=(durable_object,),
                midterm_packets=(related_midterm,),
                graph_selection=graph_selection,
            )

        self.assertEqual([item.memory_id for item in selection.durable_objects], [])
        self.assertEqual([item.slot_key for item in selection.conflict_queue], [])
        self.assertEqual(
            selection.query_plan["selected"]["graph_node_ids"],
            ["person:corinna_maier"],
        )
        self.assertEqual(
            {candidate["source"] for candidate in selection.query_plan["candidates"]},
            {"episodic", "graph", "midterm"},
        )
        self.assertEqual(selection.query_plan["pruning"]["mode"], "continuity")

    def test_unified_retrieval_selection_prefers_turn_continuity_for_generic_recap_queries(self) -> None:
        continuity_packet = LongTermMidtermPacketV1(
            packet_id="midterm:turn:recap",
            kind="recent_turn_continuity",
            summary="Recent conversation continuity from the latest user-assistant turn.",
            details="This packet preserves immediate continuity until slower durable-memory enrichment finishes.",
            query_hints=("medikamente",),
            attributes={"persistence_scope": "turn_continuity"},
        )
        graph_selection = UnifiedGraphSelectionInput(
            document=TwinrGraphDocumentV1(
                subject_node_id="user:main",
                graph_id="graph:user_main",
                created_at="2026-03-30T09:00:00Z",
                updated_at="2026-03-30T09:00:00Z",
                nodes=(
                    TwinrGraphNodeV1(
                        node_id="user:main",
                        node_type="user",
                        label="Erika",
                    ),
                    TwinrGraphNodeV1(
                        node_id="preference:coffee",
                        node_type="preference",
                        label="Melitta coffee",
                    ),
                ),
                edges=(),
                metadata={"kind": "personal_graph"},
            ),
            query_plan={
                "mode": "remote_query_first_subgraph",
                "matched_node_ids": ["user:main"],
                "selected_node_ids": ["user:main", "preference:coffee"],
                "selected_edge_ids": [],
                "access_path": ["graph_neighbors_query"],
            },
        )

        selection = build_unified_retrieval_selection(
            query_texts=("Worüber haben wir heute gesprochen?",),
            episodic_entries=(),
            durable_objects=(),
            conflict_queue=(),
            conflict_supporting_objects=(),
            midterm_packets=(continuity_packet,),
            graph_selection=graph_selection,
        )

        self.assertEqual(
            selection.query_plan["selected"]["midterm_packet_ids"],
            ["midterm:turn:recap"],
        )
        self.assertEqual(selection.query_plan["selected"]["graph_node_ids"], [])
        self.assertEqual(selection.query_plan["pruning"]["mode"], "continuity")

    def test_attach_adaptive_packets_to_query_plan_adds_selected_ids_and_candidates(self) -> None:
        related_object = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            source=_source("turn:corinna"),
            status="active",
            confidence=0.97,
            confirmed_by_user=True,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier", "support_count": 3},
        )
        selection = build_unified_retrieval_selection(
            query_texts=("What is Corinna's phone number?",),
            episodic_entries=(),
            durable_objects=(related_object,),
            conflict_queue=(),
            conflict_supporting_objects=(related_object,),
            midterm_packets=(),
            graph_selection=None,
        )
        adaptive_packets = LongTermAdaptivePolicyBuilder().build_packets(
            query_text="What is Corinna's phone number?",
            durable_objects=(related_object,),
        )

        attached_packets = attach_adaptive_packets_to_query_plan(
            query_plan=selection.query_plan,
            durable_objects=selection.durable_objects,
            adaptive_packets=adaptive_packets,
        )

        self.assertTrue(attached_packets)
        self.assertEqual(
            selection.query_plan["selected"]["adaptive_packet_ids"],
            [packet.packet_id for packet in attached_packets],
        )
        self.assertEqual(selection.query_plan["sources"]["adaptive_count"], len(attached_packets))
        adaptive_candidates = [
            candidate
            for candidate in selection.query_plan["candidates"]
            if candidate["source"] == "adaptive"
        ]
        self.assertEqual(len(adaptive_candidates), len(attached_packets))
        join_anchors = {
            item["anchor"]: item["sources"]
            for item in selection.query_plan["join_anchors"]
        }
        self.assertEqual(
            join_anchors["person_ref:person:corinna_maier"],
            ["adaptive", "durable"],
        )

    def test_build_context_surfaces_environment_reflection_in_durable_and_midterm_context(self) -> None:
        environment_summary = LongTermMemoryObjectV1(
            memory_id="environment_reflection:home_main:2026-03-16",
            kind="summary",
            summary="Recent home activity on 2026-03-16 differs from the usual pattern: less activity than usual.",
            details=(
                "Room-agnostic smart-home environment reflection compiled from motion and device-health history. "
                "Recent deviations: Observed less activity than usual compared with the rolling baseline. "
                "Quality flags: device_offline_present. Caution: sensor_quality_limited."
            ),
            source=_source("turn:env"),
            status="active",
            confidence=0.8,
            slot_key="environment_reflection:home:main",
            value_key="2026-03-16",
            valid_from="2026-03-16",
            valid_to="2026-03-16",
            attributes={
                "memory_domain": "smart_home_environment",
                "summary_type": "environment_reflection",
                "environment_id": "home:main",
                "profile_day": "2026-03-16",
                "deviation_types": ("daily_activity_drop",),
                "deviation_labels": ("less activity than usual",),
                "quality_flags": ("device_offline_present",),
                "blocked_by": ("sensor_quality_limited",),
                "active_node_count": 1,
                "active_epoch_count": 2,
                "baseline_weekday_class": "all_days",
                "query_hints": ("home activity", "movement pattern", "daily routine", "less activity than usual"),
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, midterm_store = self._make_retriever(temp_dir)
            object_store.apply_reflection(
                LongTermReflectionResultV1(
                    reflected_objects=(),
                    created_summaries=(environment_summary,),
                    midterm_packets=(),
                )
            )
            midterm_store.save_packets(
                packets=(
                    midterm_store.packet_type(
                        packet_id="midterm:environment:home_main:2026-03-16",
                        kind="recent_environment_pattern",
                        summary=environment_summary.summary,
                        details=environment_summary.details,
                        source_memory_ids=(environment_summary.memory_id,),
                        query_hints=("home activity", "movement pattern", "daily routine", "less activity than usual"),
                        sensitivity="low",
                        valid_from="2026-03-16",
                        valid_to="2026-03-16",
                        attributes={
                            "memory_domain": "smart_home_environment",
                            "packet_scope": "recent_environment_reflection",
                            "environment_id": "home:main",
                            "profile_day": "2026-03-16",
                        },
                    ),
                )
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Wie war die Aktivität im Haus heute?",
                    canonical_english_text="How has home activity been today?",
                ),
                original_query_text="Wie war die Aktivität im Haus heute?",
            )

        self.assertIsNotNone(context.durable_context)
        self.assertIsNotNone(context.midterm_context)
        self.assertIn("smart_home_environment", context.durable_context or "")
        self.assertIn("environment_reflection", context.durable_context or "")
        self.assertIn("less activity than usual", context.durable_context or "")
        self.assertIn("recent_environment_pattern", context.midterm_context or "")
        self.assertIn("sensor_quality_limited", context.midterm_context or "")

    def test_durable_context_serializes_environment_change_and_quality_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            change_point = LongTermMemoryObjectV1(
                memory_id="environment_change_point:home_main:2026-03-22",
                kind="summary",
                summary="Smart-home environment transition detected.",
                details="Recent room-agnostic smart-home markers suggest an ongoing transition away from the older behavior regime.",
                source=_source("turn:cp"),
                status="candidate",
                confidence=0.81,
                slot_key="environment_change_point:home:main",
                value_key="2026-03-22",
                valid_from="2026-03-18",
                valid_to="2026-03-22",
                attributes={
                    "memory_domain": "smart_home_environment",
                    "summary_type": "environment_change_point",
                    "environment_id": "home:main",
                    "change_started_on": "2026-03-18",
                    "severity": "moderate",
                    "markers": (
                        {"name": "active_epoch_count_day", "observed": 3.2, "baseline_median": 6.0, "delta_ratio": -0.46},
                    ),
                    "quality_flags": (),
                    "blocked_by": (),
                },
            )
            quality_state = LongTermMemoryObjectV1(
                memory_id="environment_quality_state:home_main:2026-03-22",
                kind="summary",
                summary="Smart-home environment quality state for 2026-03-22: caution.",
                details="Environment quality is usable with caution.",
                source=_source("turn:quality"),
                status="active",
                confidence=0.84,
                slot_key="environment_quality_state:home:main",
                value_key="2026-03-22",
                valid_from="2026-03-22",
                valid_to="2026-03-22",
                attributes={
                    "memory_domain": "smart_home_environment",
                    "summary_type": "environment_quality_state",
                    "environment_id": "home:main",
                    "classification": "caution",
                    "quality_flags": ("possible_visitor_or_multi_person_activity",),
                    "blocked_by": (),
                    "evidence_markers": ("active_epoch_count_day", "node_entropy_day"),
                },
            )

            change_record = retriever._durable_context_record(change_point)
            quality_record = retriever._durable_context_record(quality_state)

        self.assertEqual(change_record["smart_home_environment"]["summary_type"], "environment_change_point")
        self.assertEqual(change_record["smart_home_environment"]["change_started_on"], "2026-03-18")
        self.assertEqual(quality_record["smart_home_environment"]["summary_type"], "environment_quality_state")
        self.assertEqual(quality_record["smart_home_environment"]["classification"], "caution")

    def test_build_context_ignores_local_markdown_when_no_structured_episode_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            prompt_context_store.memory_store.remember(
                kind="episodic_turn",
                summary='Conversation about "Local markdown only."',
                details='User said: "Local markdown only." Twinr answered: "This should stay local."',
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "What did we talk about?",
                    canonical_english_text="What did we talk about?",
                ),
                original_query_text="What did we talk about?",
            )

        self.assertIsNone(context.episodic_context)

    def test_build_context_reuses_nonempty_episodic_matches_for_subtext(self) -> None:
        section_calls: list[tuple[str, ...]] = []

        def _select_context_object_sections(self, query_texts):
            del self
            section_calls.append(tuple(query_texts))
            entry = type(
                "Entry",
                (),
                {
                    "entry_id": "episode:one",
                    "kind": "episodic_turn",
                    "summary": "Stored conversation excerpt",
                    "details": 'User said: "Hallo"',
                    "created_at": datetime(2026, 3, 20, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    "updated_at": datetime(2026, 3, 20, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                },
            )()
            source_object = type(
                "Object",
                (),
                {
                    "memory_id": "episode:one",
                    "slot_key": None,
                    "value_key": None,
                    "attributes": {"person_ref": "person:corinna_maier"},
                },
            )()
            return [build_episodic_selection_input(entry=entry, source_object=source_object)], ()

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            with (
                patch.object(LongTermRetriever, "_normalize_query_text", lambda self, query, fallback_text=None: "Hallo"),
                patch.object(LongTermRetriever, "_query_text_variants", lambda self, query, fallback_text=None: ("Hallo",)),
                patch.object(LongTermRetriever, "_select_context_object_sections", _select_context_object_sections),
                patch.object(LongTermRetriever, "_select_midterm_packets", lambda self, query_texts: ()),
                patch.object(LongTermRetriever, "_build_adaptive_packets", lambda self, retrieval_text, durable_objects: ()),
                patch.object(
                    LongTermRetriever,
                    "_select_conflict_queue_selection_for_texts",
                    lambda self, query_texts: SimpleNamespace(queue=(), supporting_objects=()),
                ),
                patch.object(LongTermRetriever, "_select_graph_context_selection", lambda self, retrieval_text: None),
                patch.object(LongTermRetriever, "_render_graph_context_selection", lambda self, graph_selection, retrieval_text: None),
                patch.object(LongTermRetriever, "_render_durable_context", lambda self, objects: None),
                patch.object(LongTermRetriever, "_render_episodic_context", lambda self, entries: "episodic-context"),
                patch.object(LongTermRetriever, "_render_conflict_context", lambda self, conflicts: None),
                patch.object(LongTermRetriever, "_render_midterm_context", lambda self, packets: None),
                patch.object(LongTermRetriever, "_combine_query_texts", lambda self, query_texts: " ".join(query_texts)),
                patch.object(
                    LongTermRetriever,
                    "_build_subtext_context",
                    lambda self, query_text, retrieval_query_text, episodic_entries, graph_selection=None, unified_query_plan=None: f"subtext:{len(tuple(episodic_entries))}",
                ),
            ):
                context = retriever.build_context(
                    query=LongTermQueryProfile.from_text("Hallo"),
                    original_query_text="Hallo",
                )

        self.assertEqual(section_calls, [("Hallo",)])
        self.assertEqual(context.episodic_context, "episodic-context")
        self.assertEqual(context.subtext_context, "subtext:1")

    def test_load_context_inputs_carries_episodic_and_adaptive_sources_in_one_unified_plan(self) -> None:
        episode = LongTermMemoryObjectV1(
            memory_id="episode:corinna_called",
            kind="episode",
            summary='Conversation about "Corinna called earlier today."',
            details='User said: "Corinna called earlier today." Twinr answered: "I can keep that in mind."',
            source=_source("turn:episode"),
            status="active",
            confidence=1.0,
            attributes={
                "raw_transcript": "Corinna called earlier today.",
                "raw_response": "I can keep that in mind.",
                "person_ref": "person:corinna_maier",
            },
        )
        durable = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +15555551234.",
            details="Use the mobile number ending in 1234.",
            source=_source("turn:durable"),
            status="active",
            confidence=0.97,
            confirmed_by_user=True,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+15555551234",
            attributes={"person_ref": "person:corinna_maier", "support_count": 3},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(
                temp_dir,
                adaptive_policy_builder=LongTermAdaptivePolicyBuilder(),
            )
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(episode,),
                    durable_objects=(durable,),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            context_inputs = retriever._load_context_inputs(
                query_texts=("What is Corinna's phone number?",),
                retrieval_text="What is Corinna's phone number?",
            )

        self.assertEqual(
            [entry.entry_id for entry in context_inputs.episodic_entries],
            ["episode:corinna_called"],
        )
        self.assertTrue(context_inputs.adaptive_packets)
        self.assertIsNotNone(context_inputs.unified_query_plan)
        query_plan = context_inputs.unified_query_plan or {}
        self.assertEqual(
            query_plan["selected"]["episodic_entry_ids"],
            ["episode:corinna_called"],
        )
        self.assertEqual(
            query_plan["selected"]["adaptive_packet_ids"],
            [packet.packet_id for packet in context_inputs.adaptive_packets],
        )
        candidate_sources = {candidate["source"] for candidate in query_plan["candidates"]}
        self.assertIn("episodic", candidate_sources)
        self.assertIn("adaptive", candidate_sources)
        join_anchors = {
            item["anchor"]: item["sources"]
            for item in query_plan["join_anchors"]
        }
        self.assertEqual(
            join_anchors["person_ref:person:corinna_maier"],
            ["adaptive", "durable", "episodic"],
        )

    def test_build_context_still_attempts_subtext_when_no_episodic_matches_exist(self) -> None:
        subtext_calls: list[tuple[str | None, str, int]] = []

        def _select_context_object_sections(self, query_texts):
            del self
            del query_texts
            return [], ()

        def _build_subtext_context(
            self,
            query_text,
            retrieval_query_text,
            episodic_entries,
            graph_selection=None,
            unified_query_plan=None,
        ):
            del self
            del graph_selection
            del unified_query_plan
            subtext_calls.append((query_text, retrieval_query_text, len(tuple(episodic_entries))))
            return "unexpected"

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            with (
                patch.object(LongTermRetriever, "_normalize_query_text", lambda self, query, fallback_text=None: "Hallo"),
                patch.object(LongTermRetriever, "_query_text_variants", lambda self, query, fallback_text=None: ("Hallo",)),
                patch.object(LongTermRetriever, "_select_context_object_sections", _select_context_object_sections),
                patch.object(LongTermRetriever, "_select_midterm_packets", lambda self, query_texts: ()),
                patch.object(LongTermRetriever, "_build_adaptive_packets", lambda self, retrieval_text, durable_objects: ()),
                patch.object(
                    LongTermRetriever,
                    "_select_conflict_queue_selection_for_texts",
                    lambda self, query_texts: SimpleNamespace(queue=(), supporting_objects=()),
                ),
                patch.object(LongTermRetriever, "_select_graph_context_selection", lambda self, retrieval_text: None),
                patch.object(LongTermRetriever, "_render_graph_context_selection", lambda self, graph_selection, retrieval_text: None),
                patch.object(LongTermRetriever, "_render_durable_context", lambda self, objects: None),
                patch.object(LongTermRetriever, "_render_episodic_context", lambda self, entries: None),
                patch.object(LongTermRetriever, "_render_conflict_context", lambda self, conflicts: None),
                patch.object(LongTermRetriever, "_render_midterm_context", lambda self, packets: None),
                patch.object(LongTermRetriever, "_combine_query_texts", lambda self, query_texts: " ".join(query_texts)),
                patch.object(LongTermRetriever, "_build_subtext_context", _build_subtext_context),
            ):
                context = retriever.build_context(
                    query=LongTermQueryProfile.from_text("Hallo"),
                    original_query_text="Hallo",
                )

        self.assertEqual(subtext_calls, [("Hallo", "Hallo", 0)])
        self.assertEqual(context.subtext_context, "unexpected")

    def test_build_context_overlaps_local_sections_without_parallel_remote_searches(self) -> None:
        midterm_started = threading.Event()
        graph_selection_started = threading.Event()
        selector_threads: list[str] = []

        def _select_context_object_sections(self, query_texts):
            del self
            del query_texts
            selector_threads.append(threading.current_thread().name)
            return [], ()

        def _select_midterm_packets(self, query_texts):
            del self
            del query_texts
            midterm_started.set()
            if not graph_selection_started.wait(timeout=1.0):
                raise AssertionError("midterm retrieval did not overlap with graph retrieval")
            return ()

        def _select_graph_context_selection(self, retrieval_text):
            del self
            del retrieval_text
            graph_selection_started.set()
            if not midterm_started.wait(timeout=1.0):
                raise AssertionError("graph retrieval did not overlap with midterm retrieval")
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            with (
                patch.object(LongTermRetriever, "_normalize_query_text", lambda self, query, fallback_text=None: "Hallo"),
                patch.object(LongTermRetriever, "_query_text_variants", lambda self, query, fallback_text=None: ("Hallo",)),
                patch.object(LongTermRetriever, "_select_context_object_sections", _select_context_object_sections),
                patch.object(LongTermRetriever, "_select_midterm_packets", _select_midterm_packets),
                patch.object(LongTermRetriever, "_build_adaptive_packets", lambda self, retrieval_text, durable_objects: ()),
                patch.object(
                    LongTermRetriever,
                    "_select_conflict_queue_selection_for_texts",
                    lambda self, query_texts: SimpleNamespace(queue=(), supporting_objects=()),
                ),
                patch.object(LongTermRetriever, "_select_graph_context_selection", _select_graph_context_selection),
                patch.object(LongTermRetriever, "_render_graph_context_selection", lambda self, graph_selection, retrieval_text: None),
                patch.object(LongTermRetriever, "_render_durable_context", lambda self, objects: None),
                patch.object(LongTermRetriever, "_render_episodic_context", lambda self, entries: None),
                patch.object(LongTermRetriever, "_render_conflict_context", lambda self, conflicts: None),
                patch.object(LongTermRetriever, "_render_midterm_context", lambda self, packets: None),
                patch.object(LongTermRetriever, "_combine_query_texts", lambda self, query_texts: " ".join(query_texts)),
                patch.object(
                    LongTermRetriever,
                    "_build_subtext_context",
                    lambda self, query_text, retrieval_query_text, episodic_entries, graph_selection=None, unified_query_plan=None: None,
                ),
            ):
                context = retriever.build_context(
                    query=LongTermQueryProfile.from_text("Hallo"),
                    original_query_text="Hallo",
                )

        self.assertIsNone(context.episodic_context)
        self.assertIsNone(context.durable_context)
        self.assertEqual(selector_threads, ["MainThread"])

    def test_select_conflict_queue_respects_canonical_query_profile(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_OLD,
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_NEW,
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
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

    def test_build_context_merges_original_query_with_canonical_rewrite_for_same_language_memory(self) -> None:
        thermos_location = LongTermMemoryObjectV1(
            memory_id="fact:thermos_location_old",
            kind="fact",
            summary="Früher stand die rote Thermoskanne im Flurschrank.",
            details="Historische Ortsangabe zur roten Thermoskanne.",
            source=_source("turn:thermos"),
            status="active",
            confidence=0.98,
            confirmed_by_user=True,
            slot_key="object:red_thermos:location",
            value_key="hallway_cupboard",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            object_store.write_snapshot(
                objects=(thermos_location,),
                conflicts=(),
                archived_objects=(),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Wo stand früher meine rote Thermoskanne?",
                    canonical_english_text="Where did my red thermos flask used to be kept?",
                ),
                original_query_text="Wo stand früher meine rote Thermoskanne?",
                )

        self.assertIsNotNone(context.durable_context)
        self.assertIn("Flurschrank", context.durable_context or "")
        self.assertIn("Thermoskanne", context.durable_context or "")

    def test_build_context_ignores_numeric_distractors_across_query_variants(self) -> None:
        distractor_fourteen = LongTermMemoryObjectV1(
            memory_id="episode:distractor_14",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details='User said: "I talked about distractor topic 14 and a routine unrelated to buttons." Assistant answered: "Twinr answered distractor topic 14 in a calm way."',
            source=_source("turn:math_14"),
            status="active",
            confidence=1.0,
            attributes={
                "raw_transcript": "I talked about distractor topic 14 and a routine unrelated to buttons.",
                "raw_response": "Twinr answered distractor topic 14 in a calm way.",
            },
        )
        distractor_twenty_seven = LongTermMemoryObjectV1(
            memory_id="episode:distractor_27",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details='User said: "I talked about distractor topic 27 and a routine unrelated to buttons." Assistant answered: "Twinr answered distractor topic 27 in a calm way."',
            source=_source("turn:math_27"),
            status="active",
            confidence=1.0,
            attributes={
                "raw_transcript": "I talked about distractor topic 27 and a routine unrelated to buttons.",
                "raw_response": "Twinr answered distractor topic 27 in a calm way.",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            object_store.write_snapshot(
                objects=(distractor_fourteen, distractor_twenty_seven),
                conflicts=(),
                archived_objects=(),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Was ist 27 mal 14?",
                    canonical_english_text="27 14 multiplication",
                ),
                original_query_text="Was ist 27 mal 14?",
            )

        self.assertIsNone(context.episodic_context)

    def test_build_context_keeps_practical_support_for_connected_continuity_query(self) -> None:
        weather_episode = LongTermMemoryObjectV1(
            memory_id="episode:weather_morning",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details='User said: "After breakfast I usually ask Twinr about the weather before I start my day." Assistant answered: "I can keep that morning weather routine in mind."',
            source=_source("turn:weather"),
            status="active",
            confidence=1.0,
            attributes={
                "raw_transcript": "After breakfast I usually ask Twinr about the weather before I start my day.",
                "raw_response": "I can keep that morning weather routine in mind.",
                "daypart": "morning",
            },
        )
        button_pattern = LongTermMemoryObjectV1(
            memory_id="pattern:button:green:start_listening:morning",
            kind="summary",
            summary="The green button was used to start a conversation in the morning.",
            details="Low-confidence button usage pattern derived from a physical interaction event.",
            source=_source("event:button"),
            status="active",
            confidence=0.86,
            slot_key="pattern:button:green:start_listening:morning",
            value_key="green:start_listening:morning",
            attributes={
                "button": "green",
                "action": "start_listening",
                "daypart": "morning",
                "pattern_type": "interaction",
                "memory_domain": "interaction",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            object_store.write_snapshot(
                objects=(weather_episode, button_pattern),
                conflicts=(),
                archived_objects=(),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Was passt morgens gut zu meiner Wetterroutine mit Twinr?",
                    canonical_english_text="weather routine after breakfast morning twinr",
                ),
                original_query_text="Was passt morgens gut zu meiner Wetterroutine mit Twinr?",
            )

        self.assertIn("After breakfast I usually ask Twinr about the weather", context.episodic_context or "")
        self.assertIn("The green button was used to start a conversation in the morning.", context.durable_context or "")

    def test_build_subtext_context_uses_preselected_graph_payload_and_updates_unified_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _object_store, _prompt_context_store, graph_store, _midterm_store = self._make_retriever(temp_dir)
            graph_store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                role="Physiotherapist",
            )
            selection = retriever._select_graph_context_selection("Should I call Corinna today?")
            self.assertIsNotNone(selection)
            called_queries: list[str] = []
            original_build_from_selection = graph_store.build_subtext_payload_from_selection
            graph_store.build_subtext_payload = lambda query_text: (_ for _ in ()).throw(AssertionError(query_text))  # type: ignore[method-assign]

            def _build_from_selection(selection, *, query_text):
                called_queries.append(query_text)
                return original_build_from_selection(selection, query_text=query_text)

            graph_store.build_subtext_payload_from_selection = _build_from_selection  # type: ignore[method-assign]
            unified_query_plan = {"sources": {"graph_selected": True}}

            context = retriever._build_subtext_context(
                query_text="Soll ich Corinna heute noch anrufen?",
                retrieval_query_text="Should I call Corinna today?",
                episodic_entries=[],
                graph_selection=selection,
                unified_query_plan=unified_query_plan,
            )

        self.assertEqual(called_queries, ["Should I call Corinna today?"])
        self.assertIsNotNone(context)
        self.assertIn("social_context", context or "")
        self.assertEqual(unified_query_plan["subtext"]["graph_payload_source"], "unified_graph_selection")
        self.assertTrue(unified_query_plan["subtext"]["rendered"])

    def test_select_conflict_queue_merges_original_query_with_canonical_rewrite_for_same_language_conflicts(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:thermos_location_hallway",
            kind="fact",
            summary="Früher stand die rote Thermoskanne im Flurschrank.",
            details="Historische Ortsangabe zur roten Thermoskanne.",
            source=_source("turn:thermos_old"),
            status="active",
            confidence=0.95,
            slot_key="object:red_thermos:location",
            value_key="hallway_cupboard",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:thermos_location_kitchen",
            kind="fact",
            summary="Später stand die rote Thermoskanne in der Küche.",
            details="Abweichende Ortsangabe zur roten Thermoskanne.",
            source=_source("turn:thermos_new"),
            status="uncertain",
            confidence=0.9,
            slot_key="object:red_thermos:location",
            value_key="kitchen",
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="object:red_thermos:location",
            candidate_memory_id="fact:thermos_location_kitchen",
            existing_memory_ids=("fact:thermos_location_hallway",),
            question="Wo stand früher die rote Thermoskanne?",
            reason="Widersprüchliche Ortsangaben zur roten Thermoskanne.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thermos_conflict",
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
                    "Wo stand früher meine rote Thermoskanne?",
                    canonical_english_text="Where did my red thermos flask used to be kept?",
                )
            )

        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].slot_key, "object:red_thermos:location")
        self.assertIn("rote Thermoskanne", matching[0].question)

    def test_build_context_includes_adaptive_policy_packets_for_confirmed_memory_and_successful_usage(self) -> None:
        confirmed_preference = LongTermMemoryObjectV1(
            memory_id="fact:coffee_brand",
            kind="fact",
            summary="The user prefers Melitta coffee.",
            details="Use Melitta as the default coffee suggestion.",
            source=_source("turn:coffee"),
            status="active",
            confidence=0.99,
            confirmed_by_user=True,
            slot_key="preference:coffee:brand",
            value_key="Melitta",
            attributes={"fact_type": "preference", "support_count": 3},
        )
        routine_plan = LongTermMemoryObjectV1(
            memory_id="plan:morning_walk",
            kind="plan",
            summary="Go for a morning walk in the park.",
            details="The user often asks for simple walk-related suggestions.",
            source=_source("turn:walk"),
            status="active",
            confidence=0.94,
            slot_key="plan:morning_walk",
            value_key="park_walk",
            attributes={"support_count": 3},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            proactive_state_store = LongTermProactiveStateStore.from_config(_config(temp_dir))
            object.__setattr__(
                retriever,
                "adaptive_policy_builder",
                LongTermAdaptivePolicyBuilder(
                    proactive_state_store=proactive_state_store,
                ),
            )
            object_store.write_snapshot(
                objects=(confirmed_preference, routine_plan),
                conflicts=(),
                archived_objects=(),
            )
            candidate = LongTermProactiveCandidateV1(
                candidate_id="proactive:walk:coffee",
                kind="routine_check_in",
                summary="Offer one simple next step before the morning walk.",
                rationale="This suggestion supports the established walk routine.",
                source_memory_ids=(routine_plan.memory_id,),
                confidence=0.9,
            )
            proactive_state_store.mark_delivered(
                candidate=candidate,
                delivered_at=datetime(2026, 3, 18, 8, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                prompt_text="Would you like one simple coffee tip before the walk?",
            )
            proactive_state_store.mark_delivered(
                candidate=candidate,
                delivered_at=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                prompt_text="Shall I give you one short idea before the walk?",
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Which coffee should I get before my walk today?",
                    canonical_english_text="Which coffee should I get before my walk today?",
                ),
                original_query_text="Which coffee should I get before my walk today?",
            )

        self.assertIsNotNone(context.midterm_context)
        self.assertIn("adaptive_confirmed_memory_policy", context.midterm_context or "")
        self.assertIn("adaptive_delivery_policy", context.midterm_context or "")
        self.assertIn("Melitta coffee", context.midterm_context or "")

    def test_build_context_includes_confirmed_response_channel_policy_packet(self) -> None:
        confirmed_channel = LongTermMemoryObjectV1(
            memory_id="preference:response_channel:voice:weekday:morning",
            kind="summary",
            summary="The user confirmed that voice replies are preferred in the morning on weekdays.",
            details="Use voice first when the room context still clearly supports spoken delivery.",
            source=_source("turn:voice-pref"),
            status="active",
            confidence=0.96,
            confirmed_by_user=True,
            slot_key="preference:response_channel:weekday:morning",
            value_key="voice",
            attributes={
                "memory_class": "confirmed_preference",
                "preference_type": "response_channel",
                "preferred_channel": "voice",
                "weekday_class": "weekday",
                "daypart": "morning",
                "support_count": 3,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            object.__setattr__(
                retriever,
                "adaptive_policy_builder",
                LongTermAdaptivePolicyBuilder(
                    proactive_state_store=LongTermProactiveStateStore.from_config(_config(temp_dir)),
                ),
            )
            object_store.write_snapshot(
                objects=(confirmed_channel,),
                conflicts=(),
                archived_objects=(),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Should you answer me out loud this morning?",
                    canonical_english_text="Should you answer me out loud this morning?",
                ),
                original_query_text="Should you answer me out loud this morning?",
            )

        self.assertIn("adaptive_response_channel_policy", context.midterm_context or "")
        self.assertIn("voice replies are preferred", context.midterm_context or "")

    def test_build_context_includes_adaptive_avoidance_policy_after_repeated_skips(self) -> None:
        medication_plan = LongTermMemoryObjectV1(
            memory_id="plan:evening_medication",
            kind="plan",
            summary="Take the evening medication after dinner.",
            details="Keep medication reminders gentle and optional.",
            source=_source("turn:medication"),
            status="active",
            confidence=0.9,
            slot_key="plan:evening_medication",
            value_key="after_dinner",
            attributes={"support_count": 2},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, object_store, _prompt_context_store, _graph_store, _midterm_store = self._make_retriever(temp_dir)
            proactive_state_store = LongTermProactiveStateStore.from_config(_config(temp_dir))
            object.__setattr__(
                retriever,
                "adaptive_policy_builder",
                LongTermAdaptivePolicyBuilder(
                    proactive_state_store=proactive_state_store,
                ),
            )
            object_store.write_snapshot(
                objects=(medication_plan,),
                conflicts=(),
                archived_objects=(),
            )
            candidate = LongTermProactiveCandidateV1(
                candidate_id="proactive:medication:evening",
                kind="same_day_reminder",
                summary="Offer an evening medication reminder.",
                rationale="Supports the existing evening medication plan.",
                source_memory_ids=(medication_plan.memory_id,),
                confidence=0.87,
            )
            proactive_state_store.mark_skipped(
                candidate=candidate,
                skipped_at=datetime(2026, 3, 18, 18, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                reason="The user did not want a reminder just now.",
            )
            proactive_state_store.mark_skipped(
                candidate=candidate,
                skipped_at=datetime(2026, 3, 18, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                reason="The user already handled it alone.",
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Should I handle my evening medication now?",
                    canonical_english_text="Should I handle my evening medication now?",
                ),
                original_query_text="Should I handle my evening medication now?",
            )

        self.assertIsNotNone(context.midterm_context)
        self.assertIn("adaptive_avoidance_policy", context.midterm_context or "")
        self.assertIn("evening medication", context.midterm_context or "")


if __name__ == "__main__":
    unittest.main()
