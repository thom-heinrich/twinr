from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.intelligence.models import (
    WorldFeedSubscription,
    WorldIntelligenceState,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
    LongTermSourceRefV1,
)
from twinr.proactive.runtime.display_reserve_flow import DisplayReserveCompanionFlow


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
    def load_snapshot(self, *, config, remote_state=None):
        del config, remote_state
        return None

    def load_engagement_signals(self, *, config, remote_state=None):
        del config, remote_state
        return ()


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
            )
            fake_memory = _FakeMemoryService(
                proactive_candidates=(follow_up,),
                objects=(reflection_summary,),
                packets=(reflection_packet,),
            )
            flow = DisplayReserveCompanionFlow(
                personality_service=_FakePersonalityService(),
                world_store=_FakeWorldStore(),
                copy_generator=_NoopCopyGenerator(),
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

        topic_keys = [candidate.topic_key for candidate in candidates]
        self.assertEqual(topic_keys[0], "ai companions")
        self.assertIn("thread:person:janina", topic_keys)
        self.assertIn("morning coffee", topic_keys)
        janina_candidate = next(
            candidate
            for candidate in candidates
            if candidate.source == "memory_follow_up"
        )
        self.assertLess(janina_candidate.salience, 0.8)
        ai_candidate = next(candidate for candidate in candidates if candidate.topic_key == "ai companions")
        preference_candidate = next(candidate for candidate in candidates if candidate.topic_key == "morning coffee")
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
                personality_service=_FakePersonalityService(),
                world_store=_FakeWorldStore(subscriptions=subscriptions),
                copy_generator=_NoopCopyGenerator(),
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

        topic_keys = {candidate.topic_key for candidate in candidates}
        self.assertIn("hamburg local politics", topic_keys)
        self.assertIn("schwarzenbek civic life", topic_keys)
        self.assertIn("agentic ai", topic_keys)
        self.assertIn("peace and diplomacy", topic_keys)
        self.assertGreaterEqual(len(candidates), 5)


if __name__ == "__main__":
    unittest.main()
