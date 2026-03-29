from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
from typing import cast
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.intelligence.models import (
    WorldFeedSubscription,
    WorldInterestSignal,
    WorldIntelligenceState,
)
from twinr.agent.personality.models import (
    ContinuityThread,
    PersonalitySnapshot,
    PlaceFocus,
    WorldSignal,
)
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.service import PersonalityContextService
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
    LongTermSourceRefV1,
)
from twinr.proactive.runtime.display_reserve_flow import DisplayReserveCompanionFlow
from twinr.proactive.runtime.display_reserve_generation import DisplayReserveCopyGenerator


class _FakeObjectStore:
    def __init__(self, objects):
        self._objects = tuple(objects)

    def load_objects(self):
        return self._objects


class _FakeMidtermStore:
    def __init__(self, packets):
        self._packets = tuple(packets)

    def load_packets(self):
        return self._packets


class _FakeMemoryService:
    def __init__(self, *, proactive_candidates=(), objects=(), packets=()):
        self._plan = LongTermProactivePlanV1(candidates=tuple(proactive_candidates))
        self.object_store = _FakeObjectStore(objects)
        self.midterm_store = _FakeMidtermStore(packets)

    def select_conflict_queue(self, query_text, *, limit=None):
        del query_text, limit
        return ()

    def plan_proactive_candidates(self, *, now=None, live_facts=None):
        del now, live_facts
        return self._plan


class _FakePersonalityService:
    def __init__(self, *, snapshot=None, engagement_signals=()):
        self._snapshot = snapshot
        self._engagement_signals = tuple(engagement_signals)

    def load_snapshot(self, *, config, remote_state=None):
        del config, remote_state
        return self._snapshot

    def load_engagement_signals(self, *, config, remote_state=None):
        del config, remote_state
        return self._engagement_signals


class _FakeWorldStore:
    def __init__(self, *, subscriptions=(), state=None):
        self._subscriptions = tuple(subscriptions)
        self._state = state or WorldIntelligenceState()

    def load_subscriptions(self, *, config, remote_state=None):
        del config, remote_state
        return self._subscriptions

    def load_state(self, *, config, remote_state=None):
        del config, remote_state
        return self._state


class _NoopCopyGenerator:
    def rewrite_candidates(self, *, config, snapshot, candidates, local_now):
        del config, snapshot, local_now
        return tuple(candidates)


class _ExplodingCopyGenerator:
    def rewrite_candidates(self, *, config, snapshot, candidates, local_now):
        del config, snapshot, candidates, local_now
        raise AssertionError("copy generator should not run on raw candidate load")


class DisplayReserveCompanionFlowTests(unittest.TestCase):
    def test_flow_blends_personality_memory_reflection_and_history_learning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            now = datetime(2026, 3, 22, 18, 0, tzinfo=timezone.utc)
            history = DisplayAmbientImpulseHistoryStore.from_config(config)

            positive = history.append_exposure(
                source="world",
                topic_key="ai companions",
                title="AI companions",
                headline="AI companions",
                body="Da ist heute etwas spannend.",
                action="brief_update",
                attention_state="growing",
                shown_at=now - timedelta(days=1),
                expires_at=now - timedelta(days=1, minutes=-10),
                metadata={"candidate_family": "world"},
            )
            history.resolve_feedback(
                exposure_id=positive.exposure_id,
                response_status="engaged",
                response_sentiment="positive",
                response_at=now - timedelta(days=1) + timedelta(minutes=1),
                response_mode="voice_immediate_pickup",
                response_latency_seconds=8.0,
                response_turn_id="turn:1",
                response_target="AI companions",
                response_summary="Immediate pickup.",
            )

            negative = history.append_exposure(
                source="memory_follow_up",
                topic_key="janina",
                title="Janina",
                headline="Wie ist es damit weitergegangen?",
                body="Da fehlt mir noch etwas.",
                action="ask_one",
                attention_state="forming",
                shown_at=now - timedelta(hours=20),
                expires_at=now - timedelta(hours=19, minutes=50),
                metadata={"candidate_family": "memory_follow_up"},
            )
            history.resolve_feedback(
                exposure_id=negative.exposure_id,
                response_status="ignored",
                response_sentiment="neutral",
                response_at=now - timedelta(hours=16),
                response_mode="no_voice_pickup",
                response_latency_seconds=3600.0,
                response_turn_id="turn:2",
                response_target="Janina",
                response_summary="No pickup.",
            )

            personality_candidates = (
                AmbientDisplayImpulseCandidate(
                    topic_key="ai companions",
                    title="AI companions",
                    source="world",
                    action="brief_update",
                    attention_state="growing",
                    salience=0.46,
                    eyebrow="",
                    headline="AI companions",
                    body="Da ist heute etwas spannend.",
                    symbol="sparkles",
                    accent="info",
                    reason="personality",
                    candidate_family="world",
                ),
            )
            follow_up = LongTermProactiveCandidateV1(
                candidate_id="candidate:thread:janina:followup",
                kind="gentle_follow_up",
                summary="If relevant, gently follow up on: Janina und der Arzttermin.",
                rationale="Multiple long-term signals point to an ongoing life thread worth soft continuity.",
                confidence=0.74,
                sensitivity="normal",
            )
            reflection_summary = LongTermMemoryObjectV1(
                memory_id="thread:person:janina",
                kind="summary",
                summary="Ongoing thread about Janina: Arzttermin; neue Schule.",
                details="Reflected from multiple related long-term memory objects.",
                source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:3",)),
                status="active",
                confidence=0.84,
                sensitivity="normal",
                slot_key="thread:person:janina",
                value_key="person:janina",
                updated_at=now - timedelta(hours=5),
                attributes={
                    "summary_type": "thread",
                    "person_name": "Janina",
                    "memory_domain": "thread",
                },
            )
            reflection_packet = LongTermMidtermPacketV1(
                packet_id="midterm:user_preference_morning_coffee_melitta",
                kind="preference",
                summary="User likes to drink Melitta coffee in the morning.",
                details="This can help with relevant small talk about morning routines.",
                query_hints=("Melitta", "morning coffee"),
                sensitivity="normal",
                updated_at=now - timedelta(hours=3),
                attributes={
                    "display_anchor": "Dein Kaffee am Morgen",
                    "transcript_excerpt": "Mein Kaffee am Morgen ist mir wichtig.",
                },
            )
            fake_memory = _FakeMemoryService(
                proactive_candidates=(follow_up,),
                objects=(reflection_summary,),
                packets=(reflection_packet,),
            )
            flow = DisplayReserveCompanionFlow(
                personality_service=cast(PersonalityContextService, _FakePersonalityService()),
                world_store=cast(RemoteStateWorldIntelligenceStore, _FakeWorldStore()),
                copy_generator=cast(DisplayReserveCopyGenerator, _NoopCopyGenerator()),
            )

            with patch(
                "twinr.proactive.runtime.display_reserve_flow.build_ambient_display_impulse_candidates",
                return_value=personality_candidates,
            ), patch(
                "twinr.proactive.runtime.display_reserve_flow.LongTermMemoryService.from_config",
                return_value=fake_memory,
            ):
                candidates = flow.load_candidates(
                    config,
                    local_now=now,
                    max_items=5,
                )

        filtered_candidates = [candidate for candidate in candidates if candidate.source != "user_discovery"]
        semantic_topics = [candidate.semantic_key() for candidate in filtered_candidates]
        self.assertEqual(semantic_topics[0], "ai companions")
        self.assertIn("thread:person:janina", semantic_topics)
        self.assertIn("dein kaffee am morgen", semantic_topics)
        janina_candidate = next(
            candidate
            for candidate in candidates
            if candidate.source == "memory_follow_up"
        )
        self.assertLess(janina_candidate.salience, 0.8)
        ai_candidate = next(
            candidate
            for candidate in candidates
            if candidate.semantic_key() == "ai companions" and candidate.expansion_angle == "primary"
        )
        preference_candidate = next(
            candidate
            for candidate in candidates
            if candidate.semantic_key() == "dein kaffee am morgen" and candidate.expansion_angle == "primary"
        )
        self.assertIn("ambient_learning", ai_candidate.generation_context or {})
        self.assertEqual(preference_candidate.candidate_family, "reflection_preference")

    def test_flow_uses_world_seed_breadth_when_other_sources_are_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_candidate_limit=12)
            now = datetime(2026, 3, 22, 18, 0, tzinfo=timezone.utc)
            subscriptions = (
                WorldFeedSubscription(
                    subscription_id="feed:hamburg-local",
                    label="Hamburg Lokalpolitik",
                    feed_url="https://example.test/hamburg.xml",
                    scope="regional",
                    priority=0.84,
                    topics=("hamburg local politics",),
                ),
                WorldFeedSubscription(
                    subscription_id="feed:schwarzenbek-local",
                    label="Schwarzenbek / Herzogtum Lauenburg",
                    feed_url="https://example.test/schwarzenbek.xml",
                    scope="regional",
                    priority=0.81,
                    topics=("schwarzenbek civic life",),
                ),
                WorldFeedSubscription(
                    subscription_id="feed:mit-ai",
                    label="MIT AI",
                    feed_url="https://example.test/mit-ai.xml",
                    scope="topic",
                    priority=0.86,
                    topics=("AI companions", "agentic ai"),
                ),
                WorldFeedSubscription(
                    subscription_id="feed:bundestag-auswaertiges",
                    label="Bundestag Auswärtiges",
                    feed_url="https://example.test/foreign.xml",
                    scope="national",
                    priority=0.83,
                    topics=("world politics", "peace and diplomacy"),
                ),
            )
            flow = DisplayReserveCompanionFlow(
                personality_service=cast(PersonalityContextService, _FakePersonalityService()),
                world_store=cast(
                    RemoteStateWorldIntelligenceStore,
                    _FakeWorldStore(subscriptions=subscriptions),
                ),
                copy_generator=cast(DisplayReserveCopyGenerator, _NoopCopyGenerator()),
            )
            fake_memory = _FakeMemoryService()

            with patch(
                "twinr.proactive.runtime.display_reserve_flow.build_ambient_display_impulse_candidates",
                return_value=(),
            ), patch(
                "twinr.proactive.runtime.display_reserve_flow.LongTermMemoryService.from_config",
                return_value=fake_memory,
            ):
                candidates = flow.load_candidates(
                    config,
                    local_now=now,
                    max_items=12,
                )

        semantic_topics = {candidate.semantic_key() for candidate in candidates}
        self.assertIn("hamburg local politics", semantic_topics)
        self.assertIn("schwarzenbek civic life", semantic_topics)
        self.assertIn("agentic ai", semantic_topics)
        self.assertIn("peace and diplomacy", semantic_topics)
        self.assertEqual(len({candidate.topic_key for candidate in candidates}), len(candidates))
        self.assertGreaterEqual(len(candidates), 5)

    def test_flow_backfills_semantic_topics_from_snapshot_when_primary_sources_are_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_candidate_limit=12)
            now = datetime(2026, 3, 26, 9, 0, tzinfo=timezone.utc)
            subscriptions = (
                WorldFeedSubscription(
                    subscription_id="feed:hamburg-local",
                    label="Hamburg Lokalpolitik",
                    feed_url="https://example.test/hamburg.xml",
                    scope="regional",
                    priority=0.84,
                    topics=("hamburg local politics",),
                ),
                WorldFeedSubscription(
                    subscription_id="feed:mit-ai",
                    label="MIT AI",
                    feed_url="https://example.test/mit-ai.xml",
                    scope="topic",
                    priority=0.86,
                    topics=("agentic ai",),
                ),
            )
            snapshot = PersonalitySnapshot(
                continuity_threads=(
                    ContinuityThread(
                        title="Arzttermin gestern",
                        summary="Da war noch offen, was beim Termin herauskam.",
                        salience=0.81,
                    ),
                    ContinuityThread(
                        title="Janina und die neue Schule",
                        summary="Das bleibt als gemeinsamer Familienfaden relevant.",
                        salience=0.76,
                    ),
                    ContinuityThread(
                        title="Melitta Kaffee am Morgen",
                        summary="Das ist als kleine Alltagsroutine mehrfach aufgefallen.",
                        salience=0.68,
                    ),
                ),
                place_focuses=(
                    PlaceFocus(
                        name="Schwarzenbek",
                        summary="Der Ort taucht weiter als praktischer lokaler Bezug auf.",
                        geography="local",
                        salience=0.66,
                    ),
                ),
                world_signals=(
                    WorldSignal(
                        topic="OpenAI baut einen automatisierten Researcher",
                        summary="Der konkrete Produkt- und Forschungswinkel ist gerade auffaellig.",
                        source="world",
                        salience=0.74,
                        evidence_count=2,
                    ),
                    WorldSignal(
                        topic="Fragestunde im Bundestag",
                        summary="Die politische Lage wird dort gerade konkret verhandelt.",
                        source="regional_news",
                        salience=0.72,
                        evidence_count=2,
                    ),
                ),
            )
            engagement_signals = (
                WorldInterestSignal(
                    signal_id="signal:topic:arzttermin",
                    topic="Arzttermin gestern",
                    summary="Das bleibt ein relevanter offener Faden.",
                    salience=0.82,
                    engagement_score=0.71,
                    engagement_state="warm",
                ),
            )
            flow = DisplayReserveCompanionFlow(
                personality_service=cast(
                    PersonalityContextService,
                    _FakePersonalityService(
                        snapshot=snapshot,
                        engagement_signals=engagement_signals,
                    ),
                ),
                world_store=cast(
                    RemoteStateWorldIntelligenceStore,
                    _FakeWorldStore(subscriptions=subscriptions),
                ),
                copy_generator=cast(DisplayReserveCopyGenerator, _NoopCopyGenerator()),
            )
            fake_memory = _FakeMemoryService()

            with patch(
                "twinr.proactive.runtime.display_reserve_flow.build_ambient_display_impulse_candidates",
                return_value=(),
            ), patch(
                "twinr.proactive.runtime.display_reserve_flow.LongTermMemoryService.from_config",
                return_value=fake_memory,
            ):
                candidates = flow.load_candidates(
                    config,
                    local_now=now,
                    max_items=12,
                )

        semantic_topics = {candidate.semantic_key() for candidate in candidates}
        self.assertIn("hamburg local politics", semantic_topics)
        self.assertIn("agentic ai", semantic_topics)
        self.assertIn("arzttermin gestern", semantic_topics)
        self.assertIn("janina und die neue schule", semantic_topics)
        self.assertIn("openai baut einen automatisierten researcher", semantic_topics)
        self.assertIn("fragestunde im bundestag", semantic_topics)
        self.assertGreaterEqual(len(semantic_topics), 6)

    def test_load_raw_candidates_returns_selected_candidates_before_copy_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_candidate_limit=6)
            now = datetime(2026, 3, 22, 18, 0, tzinfo=timezone.utc)
            personality_candidates = (
                AmbientDisplayImpulseCandidate(
                    topic_key="ai companions",
                    title="AI companions",
                    source="world",
                    action="brief_update",
                    attention_state="growing",
                    salience=0.61,
                    eyebrow="",
                    headline="AI companions",
                    body="Da ist heute etwas spannend.",
                    symbol="sparkles",
                    accent="info",
                    reason="personality",
                    candidate_family="mindshare",
                ),
            )
            flow = DisplayReserveCompanionFlow(
                personality_service=cast(PersonalityContextService, _FakePersonalityService()),
                world_store=cast(RemoteStateWorldIntelligenceStore, _FakeWorldStore()),
                copy_generator=cast(DisplayReserveCopyGenerator, _ExplodingCopyGenerator()),
            )
            fake_memory = _FakeMemoryService()

            with patch(
                "twinr.proactive.runtime.display_reserve_flow.build_ambient_display_impulse_candidates",
                return_value=personality_candidates,
            ), patch(
                "twinr.proactive.runtime.display_reserve_flow.LongTermMemoryService.from_config",
                return_value=fake_memory,
            ):
                snapshot, candidates = flow.load_raw_candidates(
                    config,
                    local_now=now,
                    max_items=3,
                )

        self.assertIsNone(snapshot)
        primary = next(
            candidate
            for candidate in candidates
            if candidate.semantic_key() == "ai companions" and candidate.expansion_angle == "primary"
        )
        self.assertTrue(primary.topic_key.startswith("reserve_card::"))

    def test_flow_includes_user_discovery_candidate_when_setup_is_due(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_candidate_limit=8)
            now = datetime(2026, 3, 22, 18, 0, tzinfo=timezone.utc)
            fake_memory = _FakeMemoryService()
            flow = DisplayReserveCompanionFlow(
                personality_service=cast(PersonalityContextService, _FakePersonalityService()),
                world_store=cast(RemoteStateWorldIntelligenceStore, _FakeWorldStore()),
                copy_generator=cast(DisplayReserveCopyGenerator, _NoopCopyGenerator()),
            )

            with patch(
                "twinr.proactive.runtime.display_reserve_flow.build_ambient_display_impulse_candidates",
                return_value=(),
            ), patch(
                "twinr.proactive.runtime.display_reserve_flow.LongTermMemoryService.from_config",
                return_value=fake_memory,
            ):
                candidates = flow.load_candidates(
                    config,
                    local_now=now,
                    max_items=3,
                )

        discovery_candidate = next(candidate for candidate in candidates if candidate.source == "user_discovery")
        self.assertEqual(discovery_candidate.candidate_family, "user_discovery")
        self.assertIn("Hast du 15 Minuten", discovery_candidate.headline)


if __name__ == "__main__":
    unittest.main()
