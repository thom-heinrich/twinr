"""Validate RSS-backed world-intelligence storage, discovery, and refresh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality import (
    BackgroundPersonalityEvolutionLoop,
    PersonalityEvolutionLoop,
    PersonalityLearningService,
    PersonalitySignalExtractor,
    PersonalitySnapshot,
    RemoteStatePersonalityEvolutionStore,
    RemoteStatePersonalitySnapshotStore,
    WorldInterestSignal,
    WorldInterestSignalExtractor,
)
from twinr.agent.personality.intelligence import (
    DEFAULT_WORLD_INTELLIGENCE_STATE_KIND,
    DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND,
    RemoteStateWorldIntelligenceStore,
    WorldFeedItem,
    WorldFeedSubscription,
    WorldIntelligenceConfigRequest,
    WorldIntelligenceService,
    WorldIntelligenceState,
)
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)


class _FakeRemoteState:
    """Keep remote snapshot payloads in memory for deterministic tests."""

    def __init__(self) -> None:
        self.enabled = True
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        """Return one saved snapshot payload when present."""

        del local_path
        payload = self.snapshots.get(snapshot_kind)
        if payload is None:
            return None
        return dict(payload)

    def save_snapshot(self, *, snapshot_kind: str, payload):
        """Persist one snapshot payload in memory."""

        self.snapshots[snapshot_kind] = dict(payload)


@dataclass(frozen=True, slots=True)
class _FakeSearchResult:
    """Expose the minimal metadata surface used by discovery."""

    sources: tuple[str, ...]
    answer: str = "Discovered candidate sources."
    used_web_search: bool = True
    response_id: str | None = "resp_world_discovery_1"
    request_id: str | None = "req_world_discovery_1"


class _FakeSearchBackend:
    """Record discovery queries and return deterministic source URLs."""

    def __init__(self, *, sources: tuple[str, ...]) -> None:
        self.sources = sources
        self.calls: list[tuple[str, str | None, str | None]] = []

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> _FakeSearchResult:
        """Return the configured source pages for one discovery prompt."""

        del conversation
        self.calls.append((question, location_hint, date_context))
        return _FakeSearchResult(sources=self.sources)


@dataclass(frozen=True, slots=True)
class _FetchedDocument:
    """Represent one fetched HTML or XML document for tests."""

    url: str
    text: str
    content_type: str


class WorldIntelligenceTests(unittest.TestCase):
    def _source(self, event_id: str) -> LongTermSourceRefV1:
        return LongTermSourceRefV1(
            source_type="conversation_turn",
            event_ids=(event_id,),
            speaker="user",
            modality="voice",
        )

    def test_store_round_trips_subscriptions_and_state(self) -> None:
        remote_state = _FakeRemoteState()
        store = RemoteStateWorldIntelligenceStore()
        config = TwinrConfig(project_root=".")
        subscription = WorldFeedSubscription(
            subscription_id="feed:hamburg_local",
            label="Hamburg local politics",
            feed_url="https://example.com/hamburg/rss.xml",
            scope="local",
            region="Hamburg",
            topics=("local politics", "transit"),
            priority=0.84,
            refresh_interval_hours=72,
            created_by="installer",
            created_at="2026-03-20T08:00:00+00:00",
        )
        state = WorldIntelligenceState(
            last_discovered_at="2026-03-20T08:00:00+00:00",
            last_refreshed_at="2026-03-20T10:00:00+00:00",
            discovery_interval_hours=336,
            last_discovery_query="hamburg local politics rss",
        )

        store.save_subscriptions(
            config=config,
            subscriptions=(subscription,),
            remote_state=remote_state,
        )
        store.save_state(
            config=config,
            state=state,
            remote_state=remote_state,
        )

        loaded_subscriptions = store.load_subscriptions(
            config=config,
            remote_state=remote_state,
        )
        loaded_state = store.load_state(
            config=config,
            remote_state=remote_state,
        )

        self.assertEqual(remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND]["schema_version"], 1)
        self.assertEqual(remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_STATE_KIND]["schema_version"], 1)
        self.assertEqual(loaded_subscriptions[0].label, "Hamburg local politics")
        self.assertEqual(loaded_subscriptions[0].topics, ("local politics", "transit"))
        self.assertEqual(loaded_state.last_discovery_query, "hamburg local politics rss")

    def test_service_discovers_feeds_from_search_sources_and_persists_subscriptions(self) -> None:
        remote_state = _FakeRemoteState()
        search_backend = _FakeSearchBackend(sources=("https://example.com/hamburg",))
        config = TwinrConfig(project_root=".")
        documents = {
            "https://example.com/hamburg": _FetchedDocument(
                url="https://example.com/hamburg",
                content_type="text/html; charset=utf-8",
                text="""
                <html>
                  <head>
                    <title>Hamburg News</title>
                    <link rel="alternate" type="application/rss+xml" title="Hamburg RSS" href="/feeds/local.xml">
                  </head>
                </html>
                """,
            ),
        }

        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            page_loader=lambda url: documents[url],
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        result = service.configure(
            request=WorldIntelligenceConfigRequest(
                action="discover",
                query="Find RSS feeds for Hamburg local politics.",
                label="Hamburg local politics",
                location_hint="Hamburg",
                region="Hamburg",
                topics=("local politics",),
                scope="local",
                priority=0.82,
                auto_subscribe=True,
                created_by="tool",
            ),
            search_backend=search_backend,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.discovered_feed_urls, ("https://example.com/feeds/local.xml",))
        self.assertEqual(result.subscriptions[0].feed_url, "https://example.com/feeds/local.xml")
        self.assertEqual(result.subscriptions[0].label, "Hamburg local politics")
        self.assertEqual(search_backend.calls[0][1], "Hamburg")
        self.assertIn(DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND, remote_state.snapshots)
        self.assertIn(DEFAULT_WORLD_INTELLIGENCE_STATE_KIND, remote_state.snapshots)

    def test_service_refreshes_due_feeds_into_world_signals_and_threads(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="Hamburg News",
                    title="Transit budget approved by city council",
                    link="https://example.com/hamburg/transit-budget",
                    published_at="2026-03-20T06:30:00+00:00",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:hamburg_local",
                    label="Hamburg local politics",
                    feed_url="https://example.com/hamburg/rss.xml",
                    scope="local",
                    region="Hamburg",
                    topics=("local politics", "transit"),
                    priority=0.86,
                    refresh_interval_hours=72,
                    created_by="installer",
                    created_at="2026-03-20T08:00:00+00:00",
                ),
            ),
        )

        refresh = service.maybe_refresh()

        self.assertEqual(refresh.status, "refreshed")
        self.assertEqual(refresh.refreshed_subscription_ids, ("feed:hamburg_local",))
        self.assertEqual(refresh.world_signals[0].topic, "Transit budget approved by city council")
        self.assertEqual(refresh.world_signals[0].region, "Hamburg")
        self.assertEqual(refresh.world_signals[0].source, "Hamburg News")
        self.assertEqual(refresh.continuity_threads[0].title, "Hamburg local politics")
        persisted = service.store.load_subscriptions(config=config, remote_state=remote_state)
        self.assertEqual(
            persisted[0].last_item_ids,
            ("https://example.com/hamburg/transit-budget",),
        )
        self.assertIsNotNone(persisted[0].last_refreshed_at)

    def test_service_skips_refresh_when_subscription_is_not_due(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (_ for _ in ()).throw(AssertionError("should not fetch")),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:hamburg_local",
                    label="Hamburg local politics",
                    feed_url="https://example.com/hamburg/rss.xml",
                    scope="local",
                    region="Hamburg",
                    topics=("local politics",),
                    priority=0.8,
                    refresh_interval_hours=72,
                    last_checked_at="2026-03-20T10:00:00+00:00",
                    last_refreshed_at="2026-03-20T10:00:00+00:00",
                ),
            ),
        )

        refresh = service.maybe_refresh()

        self.assertEqual(refresh.status, "skipped")
        self.assertFalse(refresh.refreshed)
        self.assertEqual(refresh.world_signals, ())
        self.assertEqual(refresh.continuity_threads, ())

    def test_learning_service_configure_world_intelligence_commits_refresh_into_snapshot(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        learning = PersonalityLearningService(
            extractor=PersonalitySignalExtractor(),
            background_loop=BackgroundPersonalityEvolutionLoop(
                config=config,
                remote_state=remote_state,
                evolution_store=RemoteStatePersonalityEvolutionStore(),
                snapshot_store=RemoteStatePersonalitySnapshotStore(),
                evolution_loop=PersonalityEvolutionLoop(),
            ),
            world_intelligence=WorldIntelligenceService(
                config=config,
                remote_state=remote_state,
                feed_reader=lambda feed_url, *, max_items, timeout_s: (
                    WorldFeedItem(
                        feed_url=feed_url,
                        source="Hamburg News",
                        title="Harbor ferry schedule changes next week",
                        link="https://example.com/hamburg/ferry",
                        published_at="2026-03-20T07:00:00+00:00",
                    ),
                ),
                now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
            ),
        )

        result = learning.configure_world_intelligence(
            request=WorldIntelligenceConfigRequest(
                action="subscribe",
                label="Hamburg local mobility",
                feed_urls=("https://example.com/hamburg/rss.xml",),
                region="Hamburg",
                topics=("mobility", "transit"),
                scope="local",
                priority=0.83,
                refresh_interval_hours=72,
                refresh_after_change=True,
                created_by="installer",
            ),
        )
        snapshot = PersonalitySnapshot.from_payload(
            remote_state.snapshots["agent_personality_context_v1"]
        )

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.refresh)
        self.assertEqual(result.refresh.status, "refreshed")
        self.assertEqual(snapshot.world_signals[0].topic, "Harbor ferry schedule changes next week")
        self.assertEqual(snapshot.continuity_threads[0].title, "Hamburg local mobility")

    def test_interest_signals_round_trip_through_state_and_recalibrate_feeds(self) -> None:
        remote_state = _FakeRemoteState()
        search_backend = _FakeSearchBackend(sources=("https://example.com/hamburg",))
        config = TwinrConfig(project_root=".")
        documents = {
            "https://example.com/hamburg": _FetchedDocument(
                url="https://example.com/hamburg",
                content_type="text/html; charset=utf-8",
                text="""
                <html>
                  <head>
                    <link rel="alternate" type="application/rss+xml" title="Hamburg RSS" href="/feeds/local.xml">
                  </head>
                </html>
                """,
            ),
        }
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            page_loader=lambda url: documents[url],
            feed_reader=lambda feed_url, *, max_items, timeout_s: (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="Hamburg News",
                    title="New district transit funding package advances",
                    link="https://example.com/hamburg/transit-package",
                    published_at="2026-03-20T06:30:00+00:00",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        state = service.record_interest_signals(
            signals=(
                WorldInterestSignal(
                    signal_id="interest:conversation:hamburg_local_politics",
                    topic="local politics",
                    summary="Recurring conversation interest in Hamburg local politics.",
                    region="Hamburg",
                    scope="local",
                    salience=0.84,
                    confidence=0.82,
                    evidence_count=3,
                    updated_at="2026-03-20T11:00:00+00:00",
                ),
            ),
        )
        refresh = service.maybe_refresh(search_backend=search_backend)
        updated_state = service.store.load_state(config=config, remote_state=remote_state)
        subscriptions = service.store.load_subscriptions(config=config, remote_state=remote_state)

        self.assertEqual(len(state.interest_signals), 1)
        self.assertEqual(refresh.status, "refreshed")
        self.assertEqual(subscriptions[0].feed_url, "https://example.com/feeds/local.xml")
        self.assertEqual(updated_state.last_recalibrated_at, "2026-03-20T12:00:00+00:00")
        self.assertEqual(updated_state.last_discovery_query, "Find RSS or Atom feeds for local politics relevant to Hamburg.")
        self.assertEqual(refresh.world_signals[0].topic, "New district transit funding package advances")

    def test_refresh_builds_and_updates_situational_awareness_threads(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        call_count = {"value": 0}

        def _feed_reader(feed_url: str, *, max_items: int, timeout_s: float):
            del max_items
            del timeout_s
            call_count["value"] += 1
            if call_count["value"] == 1:
                return (
                    WorldFeedItem(
                        feed_url=feed_url,
                        source="Hamburg News",
                        title="Transit budget approved by city council",
                        link="https://example.com/hamburg/transit-budget",
                        published_at="2026-03-20T06:30:00+00:00",
                    ),
                )
            return (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="Hamburg News",
                    title="Harbor ferry timetable expands on weekends",
                    link="https://example.com/hamburg/ferry-weekend",
                    published_at="2026-03-21T06:30:00+00:00",
                ),
            )

        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=_feed_reader,
            now_provider=lambda: datetime(2026, 3, 20 + call_count["value"], 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:hamburg_local",
                    label="Hamburg local politics",
                    feed_url="https://example.com/hamburg/rss.xml",
                    scope="local",
                    region="Hamburg",
                    topics=("local politics", "transit"),
                    priority=0.86,
                    refresh_interval_hours=72,
                    created_by="installer",
                    created_at="2026-03-20T08:00:00+00:00",
                ),
            ),
        )

        first_refresh = service.maybe_refresh()
        second_refresh = service.maybe_refresh(force=True)
        updated_state = service.store.load_state(config=config, remote_state=remote_state)

        self.assertEqual(first_refresh.awareness_threads[0].title, "Hamburg local politics")
        self.assertEqual(second_refresh.awareness_threads[0].update_count, 2)
        self.assertIn("Transit budget approved by city council", second_refresh.awareness_threads[0].recent_titles)
        self.assertIn("Harbor ferry timetable expands on weekends", second_refresh.awareness_threads[0].recent_titles)
        self.assertEqual(updated_state.awareness_threads[0].update_count, 2)
        self.assertEqual(second_refresh.world_signals[1].source, "situational_awareness")

    def test_world_interest_extractor_derives_calibration_from_conversation_and_tools(self) -> None:
        extractor = WorldInterestSignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        personality_batch = PersonalitySignalExtractor().extract_from_consolidation(
            turn=LongTermConversationTurn(
                transcript="Was ist gerade in der Hamburger Lokalpolitik los?",
                response="Ich behalte die Haushaltsdebatte im Blick.",
            ),
            consolidation=LongTermConsolidationResultV1(
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
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        conversation_batch = extractor.extract_from_personality_batch(
            turn_id="turn:learn:1",
            batch=personality_batch,
            occurred_at=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
        )
        tool_batch = extractor.extract_from_tool_history(
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
                        "answer": "Hamburg approved a transit budget update.",
                        "response_id": "resp_search_1",
                    },
                    serialized_output='{"status":"ok"}',
                ),
            ),
        )

        self.assertEqual(conversation_batch.interest_signals[0].region, "Hamburg")
        self.assertEqual(conversation_batch.interest_signals[0].scope, "local")
        self.assertEqual(tool_batch.interest_signals[0].topic, "What changed in Hamburg local politics today?")
        self.assertFalse(tool_batch.interest_signals[0].explicit)

    def test_learning_service_records_conversation_interest_signals_in_world_intelligence_state(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        learning = PersonalityLearningService.from_config(config, remote_state=remote_state)

        learning.record_conversation_consolidation(
            turn=LongTermConversationTurn(
                transcript="Was ist gerade in der Hamburger Lokalpolitik los?",
                response="Ich behalte die Haushaltsdebatte im Blick.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:learn:world-interest",
                occurred_at=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="event:hamburg_budget_debate",
                        kind="event",
                        summary="The user asked about a budget debate in Hamburg city politics.",
                        source=self._source("turn:learn:world-interest"),
                        status="active",
                        confidence=0.86,
                        attributes={
                            "topic": "local politics",
                            "place": "Hamburg",
                            "place_ref": "place:hamburg",
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        state = WorldIntelligenceState.from_payload(
            remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_STATE_KIND]
        )
        self.assertEqual(len(state.interest_signals), 1)
        self.assertEqual(state.interest_signals[0].topic, "local politics")
        self.assertEqual(state.interest_signals[0].region, "Hamburg")


if __name__ == "__main__":
    unittest.main()
