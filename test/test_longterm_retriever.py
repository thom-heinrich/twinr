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
    LongTermMidtermStore,
    LongTermProactiveCandidateV1,
    LongTermProactiveStateStore,
    LongTermRetriever,
    LongTermSourceRefV1,
    LongTermStructuredStore,
)
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder
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
    def _make_retriever(
        self,
        temp_dir: str,
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
            summary="Corinna Maier can be reached at +491761234.",
            details="Use the mobile number ending in 1234.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            details="Use the office number ending in 8877.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
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
                phone="01761234",
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
        self.assertIn("+491761234", context.durable_context or "")
        self.assertIn("Corinna Maier", context.graph_context or "")
        self.assertIn("contact:person:corinna_maier:phone", context.conflict_context or "")
        self.assertNotIn("This local-only memory should not be used", context.episodic_context or "")

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

    def test_select_conflict_queue_respects_canonical_query_profile(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +491761234.",
            source=_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            source=_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
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
