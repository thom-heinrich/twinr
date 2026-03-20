"""Validate the structured agent-personality package and learning flow."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality import (
    BackgroundPersonalityEvolutionLoop,
    ConversationStyleProfile,
    DEFAULT_PERSONALITY_SNAPSHOT_KIND,
    ContinuityThread,
    HumorProfile,
    InteractionSignal,
    PersonalityLearningService,
    PersonalityDelta,
    PersonalityContextBuilder,
    PersonalityContextService,
    PersonalityEvolutionLoop,
    PersonalityEvolutionPolicy,
    PersonalitySignalExtractor,
    PersonalitySnapshot,
    PersonalityTrait,
    PlaceFocus,
    PlaceSignal,
    ReflectionDelta,
    RemoteStatePersonalityEvolutionStore,
    RelationshipSignal,
    RemoteStatePersonalitySnapshotStore,
    WorldSignal,
)
from twinr.agent.personality.self_expression import build_mindshare_items
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)


class _InMemorySnapshotStore:
    """Return a fixed snapshot for service-level tests."""

    def __init__(self, snapshot: PersonalitySnapshot | None) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, *, config: TwinrConfig, remote_state=None) -> PersonalitySnapshot | None:
        """Return the preconfigured snapshot and ignore runtime inputs."""

        del config
        del remote_state
        return self.snapshot


class _FakeRemoteState:
    """Mimic the remote snapshot adapter used by the store seam."""

    def __init__(self, payload: dict[str, object] | None) -> None:
        self.enabled = True
        self.payload = payload
        self.calls: list[str] = []
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        """Return the configured payload while recording the requested kind."""

        del local_path
        self.calls.append(snapshot_kind)
        saved_payload = self.snapshots.get(snapshot_kind)
        if saved_payload is not None:
            return dict(saved_payload)
        if self.payload is None:
            return None
        return dict(self.payload)

    def save_snapshot(self, *, snapshot_kind: str, payload):
        """Persist a snapshot payload in memory for round-trip tests."""

        self.snapshots[snapshot_kind] = dict(payload)


class AgentPersonalityTests(unittest.TestCase):
    def _source(self, event_id: str) -> LongTermSourceRefV1:
        return LongTermSourceRefV1(
            source_type="conversation_turn",
            event_ids=(event_id,),
            speaker="user",
            modality="voice",
        )

    def test_builder_preserves_legacy_sections_without_snapshot(self) -> None:
        builder = PersonalityContextBuilder()

        plan = builder.build_prompt_plan(
            legacy_sections=(
                ("SYSTEM", "System context"),
                ("PERSONALITY", "Style context"),
                ("USER", "User context"),
            ),
            snapshot=None,
        )

        self.assertEqual(
            plan.as_sections(),
            (
                ("SYSTEM", "System context"),
                ("PERSONALITY", "Style context"),
                ("USER", "User context"),
            ),
        )

    def test_builder_renders_structured_personality_layers_in_expected_order(self) -> None:
        builder = PersonalityContextBuilder()
        snapshot = PersonalitySnapshot(
            core_traits=(
                PersonalityTrait(
                    name="attentive companion",
                    summary="Stay warm, practical, and situationally aware.",
                    weight=0.92,
                ),
            ),
            style_profile=ConversationStyleProfile(
                verbosity=0.34,
                initiative=0.58,
            ),
            humor_profile=HumorProfile(
                style="light dry humor",
                summary="Use occasional gentle wit when the user is receptive.",
                intensity=0.38,
                boundaries=("never mocking", "never undercutting serious moments"),
            ),
            relationship_signals=(
                RelationshipSignal(
                    topic="local politics",
                    summary="The user wants Twinr to notice municipal decisions that affect daily life.",
                    salience=0.81,
                    source="conversation",
                ),
            ),
            continuity_threads=(
                ContinuityThread(
                    title="garden renovation",
                    summary="Track the user's spring planning and related errands.",
                    salience=0.74,
                    updated_at="2026-03-20T09:00:00+00:00",
                ),
            ),
            place_focuses=(
                PlaceFocus(
                    name="Hamburg region",
                    summary="Keep weather, transit, and local civic changes in view.",
                    geography="city_region",
                    salience=0.77,
                ),
            ),
            world_signals=(
                WorldSignal(
                    topic="energy prices",
                    summary="Rising utility costs matter because they shape the user's planning.",
                    region="Germany",
                    source="regional_news",
                    salience=0.64,
                    fresh_until="2026-03-21T00:00:00+00:00",
                ),
            ),
            reflection_deltas=(
                ReflectionDelta(
                    target="humor_band",
                    change="be a little more playful in relaxed turns",
                    reason="The user responded well to small observational jokes.",
                    confidence=0.72,
                ),
            ),
        )

        plan = builder.build_prompt_plan(
            legacy_sections=(
                ("SYSTEM", "System context"),
                ("PERSONALITY", "Style context"),
                ("USER", "User context"),
            ),
            snapshot=snapshot,
        )
        sections = plan.as_sections()

        self.assertEqual(
            [title for title, _content in sections],
            ["SYSTEM", "PERSONALITY", "USER", "MINDSHARE", "CONTINUITY", "PLACE", "WORLD", "REFLECTION"],
        )
        self.assertIn("Structured core character", dict(sections)["PERSONALITY"])
        self.assertIn("Evolving conversation style", dict(sections)["PERSONALITY"])
        self.assertIn("verbosity: concise", dict(sections)["PERSONALITY"])
        self.assertIn("initiative: gently proactive", dict(sections)["PERSONALITY"])
        self.assertIn("light dry humor", dict(sections)["PERSONALITY"])
        self.assertIn("Conversational self-expression", dict(sections)["PERSONALITY"])
        self.assertIn("local politics", dict(sections)["USER"])
        self.assertIn("Current companion mindshare", dict(sections)["MINDSHARE"])
        self.assertIn("Hamburg region", dict(sections)["MINDSHARE"])
        self.assertIn("garden renovation", dict(sections)["MINDSHARE"])
        self.assertIn("garden renovation", dict(sections)["CONTINUITY"])
        self.assertIn("Hamburg region", dict(sections)["PLACE"])
        self.assertIn("energy prices", dict(sections)["WORLD"])
        self.assertIn("humor_band", dict(sections)["REFLECTION"])

    def test_service_uses_injected_store_snapshot(self) -> None:
        snapshot = PersonalitySnapshot(
            core_traits=(
                PersonalityTrait(
                    name="situational awareness",
                    summary="Connect the current moment with the user's local context.",
                ),
            ),
        )
        service = PersonalityContextService(store=_InMemorySnapshotStore(snapshot))

        sections = service.build_static_sections(
            legacy_sections=(("PERSONALITY", "Style context"),),
            config=TwinrConfig(project_root="."),
        )

        self.assertEqual(sections[0][0], "PERSONALITY")
        self.assertIn("situational awareness", sections[0][1])

    def test_build_mindshare_items_uses_generic_scored_selection_not_place_first(self) -> None:
        snapshot = PersonalitySnapshot(
            generated_at="2026-03-20T20:35:00+00:00",
            relationship_signals=(
                RelationshipSignal(
                    topic="AI companions",
                    summary="Twinr should keep noticing long-term movement in companion design.",
                    salience=0.86,
                    source="conversation",
                ),
            ),
            continuity_threads=(
                ContinuityThread(
                    title="local democracy",
                    summary="Twinr has been following civic decisions that affect daily life.",
                    salience=0.91,
                    updated_at="2026-03-20T19:00:00+00:00",
                ),
            ),
            place_focuses=(
                PlaceFocus(
                    name="Schwarzenbek",
                    summary="Keep the immediate home context in view.",
                    geography="city",
                    salience=0.33,
                ),
                PlaceFocus(
                    name="Hamburg",
                    summary="Keep the nearby urban context in view.",
                    geography="city",
                    salience=0.31,
                ),
            ),
            world_signals=(
                WorldSignal(
                    topic="peace talks",
                    summary="De-escalation efforts matter for Twinr's broader situational awareness.",
                    source="situational_awareness",
                    salience=0.82,
                    fresh_until="2026-03-21T00:00:00+00:00",
                ),
            ),
        )

        items = build_mindshare_items(snapshot, max_items=2)
        titles = [item.title for item in items]

        self.assertEqual(len(items), 2)
        self.assertEqual(titles, ["local democracy", "AI companions"])
        self.assertNotIn("Schwarzenbek / Hamburg", titles)

    def test_remote_state_store_parses_snapshot_payload(self) -> None:
        payload = {
            "schema_version": 1,
            "generated_at": "2026-03-20T10:00:00+00:00",
            "core_traits": [
                {
                    "name": "attentive companion",
                    "summary": "Stay warm and practically useful.",
                    "weight": 0.88,
                }
            ],
            "style_profile": {
                "verbosity": 0.36,
                "initiative": 0.55,
            },
            "humor_profile": {
                "style": "gentle observational humor",
                "summary": "Use small, warm jokes when the moment allows it.",
                "intensity": 0.31,
            },
        }
        store = RemoteStatePersonalitySnapshotStore()
        remote_state = _FakeRemoteState(payload)

        snapshot = store.load_snapshot(
            config=TwinrConfig(project_root="."),
            remote_state=remote_state,
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(remote_state.calls, [DEFAULT_PERSONALITY_SNAPSHOT_KIND])
        self.assertEqual(snapshot.core_traits[0].name, "attentive companion")
        self.assertAlmostEqual(snapshot.style_profile.verbosity, 0.36)
        self.assertEqual(snapshot.humor_profile.style, "gentle observational humor")

    def test_evolution_store_round_trips_signals_and_deltas(self) -> None:
        remote_state = _FakeRemoteState(None)
        store = RemoteStatePersonalityEvolutionStore()
        config = TwinrConfig(project_root=".")
        interaction_signal = InteractionSignal(
            signal_id="signal:interaction:1",
            signal_kind="style_feedback",
            target="humor",
            summary="The user explicitly asked for a little more humor.",
            confidence=0.82,
            impact=0.6,
            evidence_count=2,
            source_event_ids=("turn:1", "turn:2"),
            delta_target="humor.intensity",
            delta_value=0.12,
            delta_summary="Increase humor slightly in relaxed turns.",
        )
        place_signal = PlaceSignal(
            signal_id="signal:place:1",
            place_name="Hamburg region",
            summary="Local transit and civic changes matter for the user.",
            geography="city_region",
            salience=0.79,
            confidence=0.75,
            evidence_count=2,
            source_event_ids=("turn:3", "turn:4"),
        )
        world_signal = WorldSignal(
            topic="Local election",
            summary="Municipal decisions affect day-to-day life for the user.",
            region="Hamburg",
            source="local_news",
            salience=0.73,
            fresh_until="2026-03-21T00:00:00+00:00",
            evidence_count=2,
            source_event_ids=("news:1", "news:2"),
        )
        delta = PersonalityDelta(
            delta_id="delta:1",
            target="humor.intensity",
            summary="Increase humor slightly in relaxed turns.",
            rationale="Repeated positive user feedback.",
            delta_value=0.1,
            confidence=0.79,
            support_count=2,
            source_signal_ids=("signal:interaction:1",),
            status="accepted",
        )

        store.save_interaction_signals(
            config=config,
            remote_state=remote_state,
            signals=(interaction_signal,),
        )
        store.save_place_signals(
            config=config,
            remote_state=remote_state,
            signals=(place_signal,),
        )
        store.save_world_signals(
            config=config,
            remote_state=remote_state,
            signals=(world_signal,),
        )
        store.save_personality_deltas(
            config=config,
            remote_state=remote_state,
            deltas=(delta,),
        )

        loaded_interaction = store.load_interaction_signals(
            config=config,
            remote_state=remote_state,
        )
        loaded_place = store.load_place_signals(
            config=config,
            remote_state=remote_state,
        )
        loaded_world = store.load_world_signals(
            config=config,
            remote_state=remote_state,
        )
        loaded_deltas = store.load_personality_deltas(
            config=config,
            remote_state=remote_state,
        )

        self.assertEqual(loaded_interaction[0].delta_target, "humor.intensity")
        self.assertEqual(loaded_place[0].place_name, "Hamburg region")
        self.assertEqual(loaded_world[0].topic, "Local election")
        self.assertEqual(loaded_deltas[0].status, "accepted")

    def test_evolution_loop_rejects_single_signal_personality_drift(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(min_support_count=2, max_humor_step=0.1)
        )
        snapshot = PersonalitySnapshot(
            humor_profile=HumorProfile(
                style="gentle observational humor",
                summary="Use gentle wit sparingly.",
                intensity=0.25,
            ),
        )

        result = loop.evolve(
            snapshot=snapshot,
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:interaction:solo",
                    signal_kind="style_feedback",
                    target="humor",
                    summary="One relaxed turn suggested more humor.",
                    confidence=0.91,
                    impact=0.8,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="humor.intensity",
                    delta_value=0.2,
                    delta_summary="Increase humor noticeably.",
                ),
            ),
            place_signals=(),
            world_signals=(),
        )

        self.assertEqual(result.accepted_deltas, ())
        self.assertEqual(result.snapshot.humor_profile.intensity, 0.25)
        self.assertEqual(result.rejected_deltas[0].status, "rejected")

    def test_evolution_loop_accepts_repeated_supported_humor_delta(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(min_support_count=2, max_humor_step=0.1)
        )
        snapshot = PersonalitySnapshot(
            humor_profile=HumorProfile(
                style="gentle observational humor",
                summary="Use gentle wit sparingly.",
                intensity=0.25,
            ),
        )

        result = loop.evolve(
            snapshot=snapshot,
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:interaction:1",
                    signal_kind="style_feedback",
                    target="humor",
                    summary="The user liked a slightly more playful tone.",
                    confidence=0.88,
                    impact=0.6,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="humor.intensity",
                    delta_value=0.2,
                    delta_summary="Increase humor slightly in relaxed turns.",
                ),
                InteractionSignal(
                    signal_id="signal:interaction:2",
                    signal_kind="style_feedback",
                    target="humor",
                    summary="Another relaxed turn confirmed the preference.",
                    confidence=0.84,
                    impact=0.5,
                    evidence_count=1,
                    source_event_ids=("turn:2",),
                    delta_target="humor.intensity",
                    delta_value=0.18,
                    delta_summary="Increase humor slightly in relaxed turns.",
                ),
            ),
            place_signals=(),
            world_signals=(),
        )

        self.assertEqual(len(result.accepted_deltas), 1)
        self.assertAlmostEqual(result.snapshot.humor_profile.intensity, 0.35, places=3)
        self.assertEqual(result.snapshot.personality_deltas[0].status, "accepted")

    def test_evolution_loop_accepts_repeated_style_and_topic_aversion_deltas(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(
                min_support_count=2,
                supported_delta_targets=(
                    "humor.intensity",
                    "style.verbosity",
                    "style.initiative",
                    "relationship.topic_affinity:",
                    "relationship.topic_aversion:",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        result = loop.evolve(
            snapshot=PersonalitySnapshot(
                style_profile=ConversationStyleProfile(
                    verbosity=0.5,
                    initiative=0.42,
                ),
            ),
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:verbosity:1",
                    signal_kind="verbosity_preference",
                    target="verbosity",
                    summary="The user prefers shorter answers.",
                    confidence=0.87,
                    impact=0.45,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="style.verbosity",
                    delta_value=-0.18,
                    delta_summary="Keep answers a bit more concise by default.",
                ),
                InteractionSignal(
                    signal_id="signal:verbosity:2",
                    signal_kind="verbosity_preference",
                    target="verbosity",
                    summary="The preference repeated in another turn.",
                    confidence=0.82,
                    impact=0.42,
                    evidence_count=1,
                    source_event_ids=("turn:2",),
                    delta_target="style.verbosity",
                    delta_value=-0.16,
                    delta_summary="Keep answers a bit more concise by default.",
                ),
                InteractionSignal(
                    signal_id="signal:initiative:1",
                    signal_kind="initiative_preference",
                    target="initiative",
                    summary="The user likes a little more initiative in follow-ups.",
                    confidence=0.84,
                    impact=0.35,
                    evidence_count=1,
                    source_event_ids=("turn:3",),
                    delta_target="style.initiative",
                    delta_value=0.16,
                    delta_summary="Take a slightly more proactive stance in relaxed turns.",
                ),
                InteractionSignal(
                    signal_id="signal:initiative:2",
                    signal_kind="initiative_preference",
                    target="initiative",
                    summary="Another turn confirmed the user welcomes small proactive follow-ups.",
                    confidence=0.8,
                    impact=0.33,
                    evidence_count=1,
                    source_event_ids=("turn:4",),
                    delta_target="style.initiative",
                    delta_value=0.14,
                    delta_summary="Take a slightly more proactive stance in relaxed turns.",
                ),
                InteractionSignal(
                    signal_id="signal:aversion:1",
                    signal_kind="topic_aversion",
                    target="celebrity gossip",
                    summary="The user does not want Twinr to dwell on celebrity gossip.",
                    confidence=0.86,
                    impact=-0.4,
                    evidence_count=1,
                    source_event_ids=("turn:5",),
                    delta_target="relationship.topic_aversion:celebrity gossip",
                    delta_value=0.18,
                    delta_summary="Avoid dwelling on celebrity gossip unless the user explicitly asks.",
                ),
                InteractionSignal(
                    signal_id="signal:aversion:2",
                    signal_kind="topic_aversion",
                    target="celebrity gossip",
                    summary="The aversion repeated in another turn.",
                    confidence=0.82,
                    impact=-0.38,
                    evidence_count=1,
                    source_event_ids=("turn:6",),
                    delta_target="relationship.topic_aversion:celebrity gossip",
                    delta_value=0.16,
                    delta_summary="Avoid dwelling on celebrity gossip unless the user explicitly asks.",
                ),
            ),
            place_signals=(),
            world_signals=(),
            continuity_threads=(),
        )

        self.assertEqual(len(result.accepted_deltas), 3)
        self.assertLess(result.snapshot.style_profile.verbosity, 0.5)
        self.assertGreater(result.snapshot.style_profile.initiative, 0.42)
        self.assertEqual(result.snapshot.relationship_signals[0].topic, "celebrity gossip")
        self.assertEqual(result.snapshot.relationship_signals[0].stance, "aversion")

    def test_evolution_loop_rejects_contradictory_topic_and_style_feedback(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(
                min_support_count=2,
                supported_delta_targets=(
                    "style.verbosity",
                    "relationship.topic_affinity:",
                    "relationship.topic_aversion:",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        result = loop.evolve(
            snapshot=PersonalitySnapshot(
                style_profile=ConversationStyleProfile(
                    verbosity=0.5,
                    initiative=0.42,
                ),
            ),
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:verbosity:short",
                    signal_kind="verbosity_preference",
                    target="verbosity",
                    summary="One memory says the user prefers short answers.",
                    confidence=0.8,
                    impact=0.3,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="style.verbosity",
                    delta_value=-0.2,
                    delta_summary="Keep answers shorter.",
                ),
                InteractionSignal(
                    signal_id="signal:verbosity:long",
                    signal_kind="verbosity_preference",
                    target="verbosity",
                    summary="Another memory says the user prefers more detail.",
                    confidence=0.82,
                    impact=0.32,
                    evidence_count=1,
                    source_event_ids=("turn:2",),
                    delta_target="style.verbosity",
                    delta_value=0.18,
                    delta_summary="Offer more detail by default.",
                ),
                InteractionSignal(
                    signal_id="signal:topic:like",
                    signal_kind="topic_affinity",
                    target="local politics",
                    summary="The user often asks about local politics.",
                    confidence=0.83,
                    impact=0.41,
                    evidence_count=2,
                    source_event_ids=("turn:3",),
                    delta_target="relationship.topic_affinity:local politics",
                    delta_value=0.18,
                    delta_summary="Treat local politics as a recurring interest.",
                ),
                InteractionSignal(
                    signal_id="signal:topic:avoid",
                    signal_kind="topic_aversion",
                    target="local politics",
                    summary="Another memory says the user does not want unsolicited local-politics updates.",
                    confidence=0.81,
                    impact=-0.39,
                    evidence_count=2,
                    source_event_ids=("turn:4",),
                    delta_target="relationship.topic_aversion:local politics",
                    delta_value=0.18,
                    delta_summary="Avoid unsolicited local-politics tangents unless the user asks.",
                ),
            ),
            place_signals=(),
            world_signals=(),
            continuity_threads=(),
        )

        self.assertEqual(result.accepted_deltas, ())
        rejected_targets = {delta.target for delta in result.rejected_deltas}
        self.assertEqual(result.snapshot.style_profile.verbosity, 0.5)
        self.assertEqual(result.snapshot.relationship_signals, ())
        self.assertIn("style.verbosity", rejected_targets)
        self.assertIn("relationship.topic_affinity:local politics", rejected_targets)
        self.assertIn("relationship.topic_aversion:local politics", rejected_targets)

    def test_evolution_loop_blocks_positive_humor_and_initiative_learning_in_sensitive_contexts(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(
                min_support_count=2,
                supported_delta_targets=("humor.intensity", "style.initiative"),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(
                verbosity=0.5,
                initiative=0.4,
            ),
            humor_profile=HumorProfile(
                style="gentle observational humor",
                summary="Use gentle wit sparingly.",
                intensity=0.25,
            ),
        )

        result = loop.evolve(
            snapshot=snapshot,
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:humor:sensitive:1",
                    signal_kind="humor_feedback",
                    target="humor",
                    summary="A serious health conversation included a brief laugh.",
                    confidence=0.84,
                    impact=0.2,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="humor.intensity",
                    delta_value=0.12,
                    delta_summary="Increase humor slightly.",
                    metadata={"sensitive_context": True},
                ),
                InteractionSignal(
                    signal_id="signal:humor:sensitive:2",
                    signal_kind="humor_feedback",
                    target="humor",
                    summary="Another sensitive turn should not reinforce humor.",
                    confidence=0.82,
                    impact=0.18,
                    evidence_count=1,
                    source_event_ids=("turn:2",),
                    delta_target="humor.intensity",
                    delta_value=0.1,
                    delta_summary="Increase humor slightly.",
                    metadata={"sensitive_context": True},
                ),
                InteractionSignal(
                    signal_id="signal:initiative:sensitive:1",
                    signal_kind="initiative_preference",
                    target="initiative",
                    summary="A distressed turn should not make Twinr more proactive.",
                    confidence=0.81,
                    impact=0.15,
                    evidence_count=1,
                    source_event_ids=("turn:3",),
                    delta_target="style.initiative",
                    delta_value=0.16,
                    delta_summary="Take more initiative.",
                    metadata={"sensitive_context": True},
                ),
                InteractionSignal(
                    signal_id="signal:initiative:sensitive:2",
                    signal_kind="initiative_preference",
                    target="initiative",
                    summary="A second sensitive turn should not reinforce initiative.",
                    confidence=0.8,
                    impact=0.14,
                    evidence_count=1,
                    source_event_ids=("turn:4",),
                    delta_target="style.initiative",
                    delta_value=0.14,
                    delta_summary="Take more initiative.",
                    metadata={"sensitive_context": True},
                ),
            ),
            place_signals=(),
            world_signals=(),
            continuity_threads=(),
        )

        self.assertEqual(result.accepted_deltas, ())
        rejected_targets = {delta.target for delta in result.rejected_deltas}
        self.assertEqual(result.snapshot.style_profile.initiative, 0.4)
        self.assertEqual(result.snapshot.humor_profile.intensity, 0.25)
        self.assertIn("humor.intensity", rejected_targets)
        self.assertIn("style.initiative", rejected_targets)

    def test_evolution_loop_prunes_expired_world_signals_and_decays_old_relationships(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(
                relationship_decay_days=30,
                relationship_decay_step=0.12,
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        snapshot = PersonalitySnapshot(
            relationship_signals=(
                RelationshipSignal(
                    topic="old topic",
                    summary="This used to matter to the user.",
                    salience=0.5,
                    source="personality_learning",
                    updated_at="2026-01-01T00:00:00+00:00",
                ),
                RelationshipSignal(
                    topic="recent topic",
                    summary="This came up recently.",
                    salience=0.5,
                    source="personality_learning",
                    updated_at="2026-03-15T00:00:00+00:00",
                ),
            ),
            continuity_threads=(
                ContinuityThread(
                    title="expired thread",
                    summary="This thread should have fallen out of view.",
                    salience=0.45,
                    updated_at="2026-02-10T00:00:00+00:00",
                    expires_at="2026-03-01T00:00:00+00:00",
                ),
            ),
            world_signals=(
                WorldSignal(
                    topic="Expired news",
                    summary="This signal is stale.",
                    region="Germany",
                    source="regional_news",
                    salience=0.52,
                    fresh_until="2026-03-18T00:00:00+00:00",
                    evidence_count=1,
                    source_event_ids=("news:expired",),
                ),
                WorldSignal(
                    topic="Fresh news",
                    summary="This signal is still current.",
                    region="Germany",
                    source="regional_news",
                    salience=0.56,
                    fresh_until="2026-03-21T00:00:00+00:00",
                    evidence_count=1,
                    source_event_ids=("news:fresh",),
                ),
            ),
        )

        result = loop.evolve(
            snapshot=snapshot,
            interaction_signals=(),
            place_signals=(),
            world_signals=(),
            continuity_threads=(),
        )

        by_topic = {item.topic: item for item in result.snapshot.relationship_signals}
        self.assertEqual([item.topic for item in result.snapshot.world_signals], ["Fresh news"])
        self.assertEqual(result.snapshot.continuity_threads, ())
        self.assertLess(by_topic["old topic"].salience, 0.5)
        self.assertAlmostEqual(by_topic["recent topic"].salience, 0.5, places=3)

    def test_evolution_loop_updates_world_context_without_mutating_personality(self) -> None:
        loop = PersonalityEvolutionLoop()
        snapshot = PersonalitySnapshot(
            humor_profile=HumorProfile(
                style="gentle observational humor",
                summary="Use gentle wit sparingly.",
                intensity=0.25,
            ),
        )

        result = loop.evolve(
            snapshot=snapshot,
            interaction_signals=(),
            place_signals=(),
            world_signals=(
                WorldSignal(
                    topic="Regional energy policy",
                    summary="Energy prices may affect the user's planning.",
                    region="Germany",
                    source="regional_news",
                    salience=0.77,
                    fresh_until="2026-03-21T00:00:00+00:00",
                    evidence_count=2,
                    source_event_ids=("news:1", "news:2"),
                ),
            ),
        )

        self.assertEqual(result.accepted_deltas, ())
        self.assertAlmostEqual(result.snapshot.humor_profile.intensity, 0.25, places=3)
        self.assertEqual(result.snapshot.world_signals[0].topic, "Regional energy policy")

    def test_signal_extractor_derives_topic_and_place_signals_from_consolidation(self) -> None:
        extractor = PersonalitySignalExtractor()
        turn = LongTermConversationTurn(
            transcript="Was ist gerade in der Hamburger Lokalpolitik los?",
            response="Ich schaue auf die aktuellen Haushaltsdebatten in Hamburg.",
        )
        consolidation = LongTermConsolidationResultV1(
            turn_id="turn:local_politics",
            occurred_at=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
            episodic_objects=(),
            durable_objects=(
                LongTermMemoryObjectV1(
                    memory_id="event:hamburg_budget_debate",
                    kind="event",
                    summary="The user asked about a budget debate in Hamburg city politics.",
                    source=self._source("turn:local_politics"),
                    status="active",
                    confidence=0.86,
                    attributes={
                        "topic": "local politics",
                        "action": "budget debate",
                        "place": "Hamburg",
                        "place_ref": "place:hamburg",
                        "event_domain": "civic",
                    },
                ),
            ),
            deferred_objects=(),
            conflicts=(),
            graph_edges=(),
        )

        batch = extractor.extract_from_consolidation(
            turn=turn,
            consolidation=consolidation,
        )

        self.assertEqual(len(batch.interaction_signals), 1)
        self.assertEqual(batch.interaction_signals[0].signal_kind, "topic_affinity")
        self.assertEqual(
            batch.interaction_signals[0].delta_target,
            "relationship.topic_affinity:local politics",
        )
        self.assertEqual(len(batch.place_signals), 1)
        self.assertEqual(batch.place_signals[0].place_name, "Hamburg")
        self.assertEqual(batch.continuity_threads, ())
        self.assertEqual(batch.world_signals, ())

    def test_signal_extractor_derives_style_aversion_and_continuity_signals_from_consolidation(self) -> None:
        extractor = PersonalitySignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        turn = LongTermConversationTurn(
            transcript="Bitte eher kurz antworten, aber sei manchmal ruhig etwas initiativer.",
            response="Ich halte mich kürzer und bringe gelegentlich einen hilfreichen nächsten Schritt mit.",
        )
        consolidation = LongTermConsolidationResultV1(
            turn_id="turn:style_signals",
            occurred_at=datetime(2026, 3, 20, 11, 30, tzinfo=timezone.utc),
            episodic_objects=(),
            durable_objects=(
                LongTermMemoryObjectV1(
                    memory_id="pref:verbosity:concise",
                    kind="fact",
                    summary="The user prefers concise answers.",
                    source=self._source("turn:style_signals"),
                    status="active",
                    confidence=0.87,
                    attributes={
                        "memory_domain": "preference",
                        "fact_type": "preference",
                        "preference_type": "verbosity",
                        "preference_value": "concise",
                        "support_count": 2,
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="pref:initiative:gentle",
                    kind="fact",
                    summary="The user likes gentle initiative in follow-ups.",
                    source=self._source("turn:style_signals"),
                    status="active",
                    confidence=0.83,
                    attributes={
                        "memory_domain": "preference",
                        "fact_type": "preference",
                        "preference_type": "initiative",
                        "preference_value": "more_proactive",
                        "support_count": 2,
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="feedback:humor:positive",
                    kind="fact",
                    summary="The user responded well to a small dry joke.",
                    source=self._source("turn:style_signals"),
                    status="active",
                    confidence=0.81,
                    attributes={
                        "memory_domain": "preference",
                        "fact_type": "feedback",
                        "feedback_target": "humor",
                        "feedback_polarity": "positive",
                        "support_count": 2,
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="pref:topic_aversion:celebrity_gossip",
                    kind="fact",
                    summary="The user does not enjoy celebrity gossip detours.",
                    source=self._source("turn:style_signals"),
                    status="active",
                    confidence=0.85,
                    attributes={
                        "memory_domain": "preference",
                        "fact_type": "preference",
                        "preference_type": "topic_aversion",
                        "topic": "celebrity gossip",
                        "support_count": 2,
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="thread:garden_renovation",
                    kind="summary",
                    summary="Ongoing thread about the user's garden renovation and related errands.",
                    source=self._source("turn:style_signals"),
                    status="active",
                    confidence=0.79,
                    attributes={
                        "memory_domain": "thread",
                        "summary_type": "thread",
                        "thread_title": "garden renovation",
                        "support_count": 3,
                    },
                ),
            ),
            deferred_objects=(),
            conflicts=(),
            graph_edges=(),
        )

        batch = extractor.extract_from_consolidation(
            turn=turn,
            consolidation=consolidation,
        )

        signals_by_kind = {signal.signal_kind: signal for signal in batch.interaction_signals}
        self.assertEqual(len(batch.interaction_signals), 4)
        self.assertEqual(signals_by_kind["verbosity_preference"].delta_target, "style.verbosity")
        self.assertLess(signals_by_kind["verbosity_preference"].delta_value, 0.0)
        self.assertEqual(signals_by_kind["initiative_preference"].delta_target, "style.initiative")
        self.assertGreater(signals_by_kind["initiative_preference"].delta_value, 0.0)
        self.assertEqual(signals_by_kind["humor_feedback"].delta_target, "humor.intensity")
        self.assertEqual(
            signals_by_kind["topic_aversion"].delta_target,
            "relationship.topic_aversion:celebrity gossip",
        )
        self.assertEqual(len(batch.continuity_threads), 1)
        self.assertEqual(batch.continuity_threads[0].title, "garden renovation")
        self.assertIn("garden renovation", batch.continuity_threads[0].summary)

    def test_signal_extractor_derives_world_and_place_signals_from_search_tool_history(self) -> None:
        extractor = PersonalitySignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        tool_call = AgentToolCall(
            name="search_live_info",
            call_id="call:search:1",
            arguments={
                "question": "What changed in Hamburg local politics today?",
                "location_hint": "Hamburg",
                "date_context": "today",
            },
        )
        tool_result = AgentToolResult(
            call_id="call:search:1",
            name="search_live_info",
            output={
                "status": "ok",
                "answer": "Hamburg approved a transit budget update at today's council meeting.",
                "sources": ("https://example.com/hamburg",),
                "used_web_search": True,
                "response_id": "resp_search_1",
            },
            serialized_output='{"status":"ok"}',
        )

        batch = extractor.extract_from_tool_history(
            tool_calls=(tool_call,),
            tool_results=(tool_result,),
        )

        self.assertEqual(batch.interaction_signals, ())
        self.assertEqual(len(batch.place_signals), 1)
        self.assertEqual(batch.place_signals[0].place_name, "Hamburg")
        self.assertEqual(len(batch.world_signals), 1)
        self.assertEqual(batch.continuity_threads, ())
        self.assertEqual(batch.world_signals[0].region, "Hamburg")
        self.assertEqual(batch.world_signals[0].source, "live_search")
        self.assertEqual(batch.world_signals[0].source_event_ids, ("call:search:1", "resp_search_1"))

    def test_evolution_loop_accepts_repeated_topic_affinity_delta(self) -> None:
        loop = PersonalityEvolutionLoop(
            policy=PersonalityEvolutionPolicy(
                min_support_count=2,
                supported_delta_targets=(
                    "humor.intensity",
                    "relationship.topic_affinity:",
                ),
            )
        )

        result = loop.evolve(
            snapshot=PersonalitySnapshot(),
            interaction_signals=(
                InteractionSignal(
                    signal_id="signal:topic:1",
                    signal_kind="topic_affinity",
                    target="local politics",
                    summary="Local politics keeps coming up in the user's questions.",
                    confidence=0.82,
                    impact=0.45,
                    evidence_count=1,
                    source_event_ids=("turn:1",),
                    delta_target="relationship.topic_affinity:local politics",
                    delta_value=0.18,
                    delta_summary="Treat local politics as a recurring user interest.",
                ),
                InteractionSignal(
                    signal_id="signal:topic:2",
                    signal_kind="topic_affinity",
                    target="local politics",
                    summary="The topic repeated in another recent turn.",
                    confidence=0.78,
                    impact=0.4,
                    evidence_count=1,
                    source_event_ids=("turn:2",),
                    delta_target="relationship.topic_affinity:local politics",
                    delta_value=0.16,
                    delta_summary="Treat local politics as a recurring user interest.",
                ),
            ),
            place_signals=(),
            world_signals=(),
        )

        self.assertEqual(len(result.accepted_deltas), 1)
        self.assertEqual(result.snapshot.relationship_signals[0].topic, "local politics")
        self.assertGreater(result.snapshot.relationship_signals[0].salience, 0.15)

    def test_learning_service_records_conversation_and_tool_signals(self) -> None:
        remote_state = _FakeRemoteState(None)
        config = TwinrConfig(project_root=".")
        learning = PersonalityLearningService(
            extractor=PersonalitySignalExtractor(
                now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
            ),
            background_loop=BackgroundPersonalityEvolutionLoop(
                config=config,
                remote_state=remote_state,
                evolution_store=RemoteStatePersonalityEvolutionStore(),
                snapshot_store=RemoteStatePersonalitySnapshotStore(),
                evolution_loop=PersonalityEvolutionLoop(
                    policy=PersonalityEvolutionPolicy(
                        min_support_count=2,
                        supported_delta_targets=(
                            "humor.intensity",
                            "style.verbosity",
                            "relationship.topic_affinity:",
                        ),
                    )
                ),
            ),
        )
        turn = LongTermConversationTurn(
            transcript="Was ist gerade in der Hamburger Lokalpolitik los?",
            response="Ich behalte die Haushaltsdebatte im Blick.",
        )
        consolidation = LongTermConsolidationResultV1(
            turn_id="turn:learn:1",
            occurred_at=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
            episodic_objects=(),
            durable_objects=(
                LongTermMemoryObjectV1(
                    memory_id="event:hamburg_budget_debate",
                    kind="event",
                    summary="The user asked about a budget debate in Hamburg city politics.",
                    source=self._source("turn:learn:1"),
                    status="active",
                    confidence=0.86,
                    attributes={
                        "topic": "local politics",
                        "place": "Hamburg",
                        "place_ref": "place:hamburg",
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="pref:verbosity:concise",
                    kind="fact",
                    summary="The user prefers concise answers.",
                    source=self._source("turn:learn:1"),
                    status="active",
                    confidence=0.84,
                    attributes={
                        "memory_domain": "preference",
                        "fact_type": "preference",
                        "preference_type": "verbosity",
                        "preference_value": "concise",
                        "support_count": 2,
                    },
                ),
                LongTermMemoryObjectV1(
                    memory_id="thread:hamburg_budget",
                    kind="summary",
                    summary="Ongoing thread about Hamburg budget updates and transit changes.",
                    source=self._source("turn:learn:1"),
                    status="active",
                    confidence=0.8,
                    attributes={
                        "memory_domain": "thread",
                        "summary_type": "thread",
                        "thread_title": "hamburg budget updates",
                        "support_count": 3,
                    },
                ),
            ),
            deferred_objects=(),
            conflicts=(),
            graph_edges=(),
        )

        learning.record_conversation_consolidation(
            turn=turn,
            consolidation=consolidation,
        )
        result = learning.record_tool_history(
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id="call:search:1",
                    arguments={
                        "question": "What changed in Hamburg local politics today?",
                        "location_hint": "Hamburg",
                    },
                ),
            ),
            tool_results=(
                AgentToolResult(
                    call_id="call:search:1",
                    name="search_live_info",
                    output={
                        "status": "ok",
                        "answer": "Hamburg approved a transit budget update at today's council meeting.",
                        "sources": ("https://example.com/hamburg",),
                        "used_web_search": True,
                    },
                    serialized_output='{"status":"ok"}',
                ),
            ),
        )

        self.assertEqual(result.snapshot.place_focuses[0].name, "Hamburg")
        self.assertEqual(result.snapshot.world_signals[0].region, "Hamburg")
        self.assertLess(result.snapshot.style_profile.verbosity, 0.5)
        self.assertEqual(result.snapshot.continuity_threads[0].title, "hamburg budget updates")
        self.assertIn("agent_personality_world_signals_v1", remote_state.snapshots)

    def test_background_loop_persists_snapshot_and_deltas(self) -> None:
        remote_state = _FakeRemoteState(None)
        config = TwinrConfig(project_root=".")
        background_loop = BackgroundPersonalityEvolutionLoop(
            config=config,
            remote_state=remote_state,
            evolution_store=RemoteStatePersonalityEvolutionStore(),
            snapshot_store=RemoteStatePersonalitySnapshotStore(),
            evolution_loop=PersonalityEvolutionLoop(
                policy=PersonalityEvolutionPolicy(min_support_count=2, max_humor_step=0.1)
            ),
        )

        background_loop.enqueue_interaction_signal(
            InteractionSignal(
                signal_id="signal:interaction:1",
                signal_kind="style_feedback",
                target="humor",
                summary="The user liked slightly more humor.",
                confidence=0.88,
                impact=0.6,
                evidence_count=1,
                source_event_ids=("turn:1",),
                delta_target="humor.intensity",
                delta_value=0.2,
                delta_summary="Increase humor slightly in relaxed turns.",
            )
        )
        background_loop.enqueue_interaction_signal(
            InteractionSignal(
                signal_id="signal:interaction:2",
                signal_kind="style_feedback",
                target="humor",
                summary="The preference repeated in another turn.",
                confidence=0.85,
                impact=0.5,
                evidence_count=1,
                source_event_ids=("turn:2",),
                delta_target="humor.intensity",
                delta_value=0.18,
                delta_summary="Increase humor slightly in relaxed turns.",
            )
        )
        background_loop.enqueue_place_signal(
            PlaceSignal(
                signal_id="signal:place:1",
                place_name="Hamburg region",
                summary="Keep local changes in view.",
                geography="city_region",
                salience=0.74,
                confidence=0.78,
                evidence_count=2,
                source_event_ids=("turn:3", "turn:4"),
            )
        )
        background_loop.enqueue_world_signal(
            WorldSignal(
                topic="Regional energy policy",
                summary="Energy prices may affect the user's planning.",
                region="Germany",
                source="regional_news",
                salience=0.77,
                fresh_until="2026-03-21T00:00:00+00:00",
                evidence_count=2,
                source_event_ids=("news:1", "news:2"),
            )
        )

        result = background_loop.process_pending()

        self.assertEqual(len(result.accepted_deltas), 1)
        self.assertIn("agent_personality_context_v1", remote_state.snapshots)
        self.assertIn("agent_personality_interaction_signals_v1", remote_state.snapshots)
        self.assertAlmostEqual(result.snapshot.humor_profile.intensity, 0.35, places=3)
        self.assertEqual(result.snapshot.place_focuses[0].name, "Hamburg region")

    def test_background_loop_accumulates_repeated_support_without_double_applying(self) -> None:
        remote_state = _FakeRemoteState(None)
        config = TwinrConfig(project_root=".")
        background_loop = BackgroundPersonalityEvolutionLoop(
            config=config,
            remote_state=remote_state,
            evolution_store=RemoteStatePersonalityEvolutionStore(),
            snapshot_store=RemoteStatePersonalitySnapshotStore(),
            evolution_loop=PersonalityEvolutionLoop(
                policy=PersonalityEvolutionPolicy(min_support_count=2, max_humor_step=0.1)
            ),
        )

        background_loop.enqueue_interaction_signal(
            InteractionSignal(
                signal_id="signal:interaction:first",
                signal_kind="style_feedback",
                target="humor",
                summary="One relaxed turn suggested slightly more humor.",
                confidence=0.9,
                impact=0.5,
                evidence_count=1,
                source_event_ids=("turn:1",),
                delta_target="humor.intensity",
                delta_value=0.2,
                delta_summary="Increase humor slightly in relaxed turns.",
            )
        )
        first_result = background_loop.process_pending()

        self.assertEqual(first_result.accepted_deltas, ())
        self.assertEqual(first_result.rejected_deltas[0].status, "rejected")

        background_loop.enqueue_interaction_signal(
            InteractionSignal(
                signal_id="signal:interaction:second",
                signal_kind="style_feedback",
                target="humor",
                summary="A second relaxed turn confirmed the preference.",
                confidence=0.85,
                impact=0.6,
                evidence_count=1,
                source_event_ids=("turn:2",),
                delta_target="humor.intensity",
                delta_value=0.18,
                delta_summary="Increase humor slightly in relaxed turns.",
            )
        )
        second_result = background_loop.process_pending()

        self.assertEqual(len(second_result.accepted_deltas), 1)
        self.assertAlmostEqual(second_result.snapshot.humor_profile.intensity, 0.35, places=3)

        third_result = background_loop.process_pending()

        self.assertEqual(third_result.accepted_deltas, ())
        self.assertAlmostEqual(third_result.snapshot.humor_profile.intensity, 0.35, places=3)


if __name__ == "__main__":
    unittest.main()
