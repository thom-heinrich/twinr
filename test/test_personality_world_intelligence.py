"""Validate RSS-backed world-intelligence storage, discovery, and refresh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality import (
    BackgroundPersonalityEvolutionLoop,
    PersonalityEvolutionLoop,
    PersonalityLearningService,
    PersonalitySignalExtractor,
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
            interest_signals=(
                WorldInterestSignal(
                    signal_id="interest:hamburg_local",
                    topic="local politics",
                    summary="The user keeps engaging with Hamburg local politics.",
                    region="Hamburg",
                    scope="local",
                    salience=0.84,
                    confidence=0.82,
                    engagement_score=0.91,
                    engagement_state="resonant",
                    evidence_count=3,
                    engagement_count=5,
                    positive_signal_count=4,
                    exposure_count=4,
                    ongoing_interest_score=0.94,
                    ongoing_interest="active",
                    co_attention_score=0.66,
                    co_attention_state="forming",
                    co_attention_count=1,
                    updated_at="2026-03-20T09:30:00+00:00",
                ),
            ),
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

        self.assertEqual(remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND]["schema_version"], 2)
        self.assertEqual(remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_STATE_KIND]["schema_version"], 2)
        self.assertEqual(loaded_subscriptions[0].label, "Hamburg local politics")
        self.assertEqual(loaded_subscriptions[0].topics, ("local politics", "transit"))
        self.assertEqual(loaded_state.last_discovery_query, "hamburg local politics rss")
        self.assertAlmostEqual(loaded_state.interest_signals[0].engagement_score, 0.91)
        self.assertEqual(loaded_state.interest_signals[0].engagement_count, 5)
        self.assertEqual(loaded_state.interest_signals[0].engagement_state, "resonant")
        self.assertEqual(loaded_state.interest_signals[0].positive_signal_count, 4)
        self.assertEqual(loaded_state.interest_signals[0].ongoing_interest, "active")
        self.assertAlmostEqual(loaded_state.interest_signals[0].ongoing_interest_score, 0.94)
        self.assertEqual(loaded_state.interest_signals[0].co_attention_state, "forming")
        self.assertAlmostEqual(loaded_state.interest_signals[0].co_attention_score, 0.66)
        self.assertEqual(loaded_state.interest_signals[0].co_attention_count, 1)

    def test_legacy_interest_signals_backfill_engagement_defaults(self) -> None:
        state = WorldIntelligenceState.from_payload(
            {
                "schema_version": 1,
                "interest_signals": [
                    {
                        "signal_id": "interest:legacy:ai_companions",
                        "topic": "AI companions",
                        "summary": "Legacy seed without explicit engagement fields.",
                        "scope": "topic",
                        "salience": 0.86,
                        "confidence": 0.84,
                        "evidence_count": 3,
                    }
                ],
            }
        )

        self.assertGreater(state.interest_signals[0].engagement_score, 0.7)
        self.assertEqual(state.interest_signals[0].engagement_count, 3)
        self.assertEqual(state.interest_signals[0].engagement_state, "resonant")
        self.assertEqual(state.interest_signals[0].ongoing_interest, "active")

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
            "https://example.com/feeds/local.xml": _FetchedDocument(
                url="https://example.com/feeds/local.xml",
                content_type="application/rss+xml",
                text="<?xml version='1.0'?><rss version='2.0'><channel><title>Hamburg RSS</title></channel></rss>",
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

    def test_service_discovery_skips_failing_source_pages_and_keeps_other_candidates(self) -> None:
        remote_state = _FakeRemoteState()
        search_backend = _FakeSearchBackend(
            sources=(
                "https://example.com/too-large",
                "https://example.com/hamburg",
            )
        )
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
            "https://example.com/feeds/local.xml": _FetchedDocument(
                url="https://example.com/feeds/local.xml",
                content_type="application/rss+xml",
                text="<?xml version='1.0'?><rss version='2.0'><channel><title>Hamburg RSS</title></channel></rss>",
            ),
        }

        def _page_loader(url: str) -> _FetchedDocument:
            if url == "https://example.com/too-large":
                raise ValueError("world_intelligence_document_too_large")
            return documents[url]

        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            page_loader=_page_loader,
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
        snapshot = RemoteStatePersonalitySnapshotStore().load_snapshot(
            config=config,
            remote_state=remote_state,
        )
        assert snapshot is not None

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.refresh)
        self.assertEqual(result.refresh.status, "refreshed")
        self.assertIn(
            "Harbor ferry schedule changes next week",
            {signal.topic for signal in result.refresh.world_signals},
        )
        self.assertEqual(snapshot.world_signals[0].topic, "Hamburg local mobility")
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
            "https://example.com/feeds/local.xml": _FetchedDocument(
                url="https://example.com/feeds/local.xml",
                content_type="application/rss+xml",
                text="<?xml version='1.0'?><rss version='2.0'><channel><title>Hamburg RSS</title></channel></rss>",
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
                    engagement_score=0.84,
                    evidence_count=3,
                    engagement_count=3,
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
        self.assertEqual(updated_state.last_recalibrated_at, "2026-03-20T12:00:00Z")
        self.assertEqual(
            updated_state.last_discovery_query,
            "Find RSS, Atom, or JSON feeds for local politics relevant to Hamburg.",
        )
        self.assertEqual(refresh.world_signals[0].topic, "New district transit funding package advances")
        self.assertAlmostEqual(updated_state.interest_signals[0].engagement_score, 0.84)

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
                            "support_count": 2,
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
        self.assertGreater(conversation_batch.interest_signals[0].engagement_score, 0.6)
        self.assertEqual(tool_batch.interest_signals[0].topic, "What changed in Hamburg local politics today?")
        self.assertGreater(tool_batch.interest_signals[0].engagement_score, conversation_batch.interest_signals[0].engagement_score)
        self.assertEqual(tool_batch.interest_signals[0].engagement_count, 2)
        self.assertFalse(tool_batch.interest_signals[0].explicit)

    def test_world_interest_extractor_promotes_explicit_topic_follow_up_into_stronger_interest(self) -> None:
        extractor = WorldInterestSignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        personality_batch = PersonalitySignalExtractor().extract_from_consolidation(
            turn=LongTermConversationTurn(
                transcript="Erzaehl mir mehr ueber AI companions, darueber will ich oefter sprechen.",
                response="Ich bleibe bei dem Thema und halte weitere Entwicklungen im Blick.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:follow-up:1",
                occurred_at=datetime(2026, 3, 20, 11, 30, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pref:topic_follow_up:ai_companions",
                        kind="fact",
                        summary="The user explicitly asked to keep talking about AI companions.",
                        source=self._source("turn:follow-up:1"),
                        status="active",
                        confidence=0.91,
                        attributes={
                            "memory_domain": "preference",
                            "fact_type": "preference",
                            "preference_type": "topic_follow_up",
                            "topic": "AI companions",
                            "support_count": 2,
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        conversation_batch = extractor.extract_from_personality_batch(
            turn_id="turn:follow-up:1",
            batch=personality_batch,
            occurred_at=datetime(2026, 3, 20, 11, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(len(conversation_batch.interest_signals), 1)
        signal = conversation_batch.interest_signals[0]
        self.assertEqual(signal.topic, "AI companions")
        self.assertTrue(signal.explicit)
        self.assertGreaterEqual(signal.engagement_score, 0.88)
        self.assertGreaterEqual(signal.salience, 0.75)
        self.assertEqual(signal.engagement_count, 3)
        self.assertEqual(signal.engagement_state, "resonant")
        self.assertTrue(signal.signal_id.startswith("interest:engagement:"))

    def test_world_interest_extractor_derives_exposure_aware_cooling_state(self) -> None:
        extractor = WorldInterestSignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        personality_batch = PersonalitySignalExtractor().extract_from_consolidation(
            turn=LongTermConversationTurn(
                transcript="Dann lieber etwas anderes.",
                response="Alles klar, ich lasse das Thema erstmal ruhen.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:cooling:1",
                occurred_at=datetime(2026, 3, 20, 11, 30, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pattern:topic_non_reengagement:ai_companions",
                        kind="pattern",
                        summary="AI companions did not draw the user back in after repeated exposure.",
                        source=self._source("turn:cooling:1"),
                        status="active",
                        confidence=0.8,
                        attributes={
                            "memory_domain": "pattern",
                            "pattern_type": "topic_non_reengagement",
                            "topic": "AI companions",
                            "exposure_count": 3,
                            "non_reengagement_count": 2,
                            "exposure_aware": True,
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        conversation_batch = extractor.extract_from_personality_batch(
            turn_id="turn:cooling:1",
            batch=personality_batch,
            occurred_at=datetime(2026, 3, 20, 11, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(len(conversation_batch.interest_signals), 1)
        signal = conversation_batch.interest_signals[0]
        self.assertEqual(signal.topic, "AI companions")
        self.assertEqual(signal.engagement_state, "cooling")
        self.assertEqual(signal.positive_signal_count, 0)
        self.assertEqual(signal.exposure_count, 3)
        self.assertEqual(signal.non_reengagement_count, 2)
        self.assertLess(signal.engagement_score, 0.3)

    def test_world_interest_extractor_promotes_explicit_topic_aversion_into_avoid(self) -> None:
        extractor = WorldInterestSignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        personality_batch = PersonalitySignalExtractor().extract_from_consolidation(
            turn=LongTermConversationTurn(
                transcript="Bitte keine Promi-Geschichten mehr.",
                response="Verstanden, ich lasse das Thema weg.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:avoid:1",
                occurred_at=datetime(2026, 3, 20, 11, 40, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pref:topic_aversion:celebrity_gossip",
                        kind="fact",
                        summary="The user clearly does not want celebrity gossip.",
                        source=self._source("turn:avoid:1"),
                        status="active",
                        confidence=0.88,
                        attributes={
                            "memory_domain": "preference",
                            "fact_type": "preference",
                            "preference_type": "topic_aversion",
                            "topic": "celebrity gossip",
                            "support_count": 2,
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        conversation_batch = extractor.extract_from_personality_batch(
            turn_id="turn:avoid:1",
            batch=personality_batch,
            occurred_at=datetime(2026, 3, 20, 11, 40, tzinfo=timezone.utc),
        )

        self.assertEqual(len(conversation_batch.interest_signals), 1)
        signal = conversation_batch.interest_signals[0]
        self.assertEqual(signal.topic, "celebrity gossip")
        self.assertEqual(signal.engagement_state, "avoid")
        self.assertEqual(signal.positive_signal_count, 0)
        self.assertGreaterEqual(signal.deflection_count, 2)
        self.assertTrue(signal.explicit)

    def test_world_interest_extractor_derives_follow_through_cooling_from_topic_switch(self) -> None:
        extractor = WorldInterestSignalExtractor(
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        )
        personality_batch = PersonalitySignalExtractor().extract_from_consolidation(
            turn=LongTermConversationTurn(
                transcript="Erzaehl mir mehr ueber Weltpolitik und Diplomatie.",
                response="Ich bleibe bei Diplomatie und globalen Entwicklungen.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:switch:1",
                occurred_at=datetime(2026, 3, 20, 11, 50, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pref:topic_follow_up:world_politics",
                        kind="fact",
                        summary="The user wanted to stay with world politics.",
                        source=self._source("turn:switch:1"),
                        status="active",
                        confidence=0.9,
                        attributes={
                            "memory_domain": "preference",
                            "fact_type": "preference",
                            "preference_type": "topic_follow_up",
                            "topic": "world politics",
                            "support_count": 2,
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        conversation_batch = extractor.extract_from_personality_batch(
            turn_id="turn:switch:1",
            batch=personality_batch,
            occurred_at=datetime(2026, 3, 20, 11, 50, tzinfo=timezone.utc),
            existing_interest_signals=(
                WorldInterestSignal(
                    signal_id="interest:ai_companions",
                    topic="AI companions",
                    summary="This topic had been strongly engaging the user before.",
                    salience=0.8,
                    confidence=0.86,
                    engagement_score=0.92,
                    engagement_state="resonant",
                    evidence_count=3,
                    engagement_count=5,
                    positive_signal_count=4,
                    exposure_count=4,
                    updated_at="2026-03-20T10:30:00+00:00",
                ),
            ),
        )

        signals_by_topic = {signal.topic: signal for signal in conversation_batch.interest_signals}
        self.assertIn("world politics", signals_by_topic)
        self.assertIn("AI companions", signals_by_topic)
        self.assertEqual(signals_by_topic["AI companions"].non_reengagement_count, 1)
        self.assertEqual(signals_by_topic["AI companions"].positive_signal_count, 0)
        self.assertLess(signals_by_topic["AI companions"].engagement_score, 0.5)

    def test_recalibration_tunes_existing_subscription_from_engagement(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:ai_companions",
                    label="AI companions",
                    feed_url="https://example.com/ai/rss.xml",
                    topics=("AI companions",),
                    priority=0.52,
                    refresh_interval_hours=96,
                    created_by="installer",
                    created_at="2026-03-20T08:00:00+00:00",
                    last_checked_at="2026-03-20T10:30:00+00:00",
                    last_refreshed_at="2026-03-20T10:30:00+00:00",
                ),
            ),
        )
        service.store.save_state(
            config=config,
            remote_state=remote_state,
            state=WorldIntelligenceState(
                last_recalibrated_at="2026-03-01T08:00:00+00:00",
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions",
                        topic="AI companions",
                        summary="The user repeatedly asks follow-up questions about AI companions.",
                        salience=0.78,
                        confidence=0.86,
                        engagement_score=0.94,
                        evidence_count=3,
                        engagement_count=5,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        refresh = service.maybe_refresh(search_backend=None)
        subscriptions = service.store.load_subscriptions(config=config, remote_state=remote_state)
        state = service.store.load_state(config=config, remote_state=remote_state)

        self.assertEqual(refresh.status, "skipped")
        self.assertEqual(len(subscriptions), 1)
        self.assertGreaterEqual(subscriptions[0].priority, 0.84)
        self.assertEqual(subscriptions[0].refresh_interval_hours, 24)
        self.assertAlmostEqual(subscriptions[0].base_priority, 0.52)
        self.assertEqual(subscriptions[0].base_refresh_interval_hours, 96)
        self.assertEqual(state.interest_signals[0].ongoing_interest, "active")
        self.assertEqual(state.interest_signals[0].co_attention_state, "forming")
        self.assertGreaterEqual(state.interest_signals[0].co_attention_count, 1)

    def test_recalibration_prefers_active_interest_for_new_feed_discovery(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        candidates = service._select_recalibration_candidates(
            subscriptions=(),
            state=WorldIntelligenceState(
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions",
                        topic="AI companions",
                        summary="This topic keeps drawing the user back in.",
                        salience=0.74,
                        confidence=0.84,
                        engagement_score=0.8,
                        engagement_state="warm",
                        ongoing_interest="active",
                        ongoing_interest_score=0.9,
                        evidence_count=3,
                        engagement_count=4,
                        positive_signal_count=3,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                    WorldInterestSignal(
                        signal_id="interest:world_politics",
                        topic="world politics",
                        summary="This topic remains relevant but less sticky.",
                        salience=0.88,
                        confidence=0.86,
                        engagement_score=0.86,
                        engagement_state="resonant",
                        ongoing_interest="growing",
                        ongoing_interest_score=0.72,
                        evidence_count=3,
                        engagement_count=3,
                        positive_signal_count=2,
                        updated_at="2026-03-20T11:40:00+00:00",
                    ),
                ),
            ),
        )

        self.assertEqual(tuple(item.topic for item in candidates), ("AI companions", "world politics"))

    def test_recalibration_ignores_cooling_interest_for_new_feed_discovery(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            page_loader=lambda url: (_ for _ in ()).throw(AssertionError("should not discover cooling topics")),
            feed_reader=lambda feed_url, *, max_items, timeout_s: (),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_state(
            config=config,
            remote_state=remote_state,
            state=WorldIntelligenceState(
                last_recalibrated_at="2026-03-01T08:00:00+00:00",
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions:cooling",
                        topic="AI companions",
                        summary="The topic has cooled after repeated exposure without follow-up.",
                        salience=0.5,
                        confidence=0.78,
                        engagement_score=0.22,
                        engagement_state="cooling",
                        evidence_count=2,
                        engagement_count=0,
                        positive_signal_count=0,
                        exposure_count=3,
                        non_reengagement_count=2,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        refresh = service.maybe_refresh(search_backend=_FakeSearchBackend(sources=("https://example.com/ai",)))
        subscriptions = service.store.load_subscriptions(config=config, remote_state=remote_state)

        self.assertEqual(refresh.status, "skipped")
        self.assertEqual(subscriptions, ())

    def test_recalibration_ignores_live_search_tool_interest_for_new_feed_discovery(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )

        candidates = service._select_recalibration_candidates(
            subscriptions=(),
            state=WorldIntelligenceState(
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:tool:call:search:1",
                        topic="What changed in Hamburg local politics today?",
                        summary="Recent live search suggests the user wanted fresh situational awareness about Hamburg.",
                        region="Hamburg",
                        scope="local",
                        salience=0.72,
                        confidence=0.7,
                        engagement_score=0.82,
                        engagement_state="warm",
                        ongoing_interest="active",
                        ongoing_interest_score=0.88,
                        evidence_count=1,
                        engagement_count=2,
                        positive_signal_count=2,
                        updated_at="2026-03-20T11:40:00+00:00",
                    ),
                    WorldInterestSignal(
                        signal_id="interest:conversation:hamburg_local_politics",
                        topic="local politics",
                        summary="Recurring conversation interest in Hamburg local politics.",
                        region="Hamburg",
                        scope="local",
                        salience=0.74,
                        confidence=0.82,
                        engagement_score=0.8,
                        engagement_state="warm",
                        ongoing_interest="active",
                        ongoing_interest_score=0.9,
                        evidence_count=3,
                        engagement_count=4,
                        positive_signal_count=3,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        self.assertEqual(tuple(item.signal_id for item in candidates), ("interest:conversation:hamburg_local_politics",))

    def test_recalibration_relaxes_subscription_back_to_baseline_after_engagement_decays(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (),
            now_provider=lambda: datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:ai_companions",
                    label="AI companions",
                    feed_url="https://example.com/ai/rss.xml",
                    topics=("AI companions",),
                    priority=0.89,
                    base_priority=0.52,
                    refresh_interval_hours=24,
                    base_refresh_interval_hours=96,
                    created_by="installer",
                    created_at="2026-03-20T08:00:00+00:00",
                    last_checked_at="2026-04-28T10:30:00+00:00",
                    last_refreshed_at="2026-04-28T10:30:00+00:00",
                ),
            ),
        )
        service.store.save_state(
            config=config,
            remote_state=remote_state,
            state=WorldIntelligenceState(
                last_recalibrated_at="2026-03-20T08:00:00+00:00",
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions",
                        topic="AI companions",
                        summary="This topic used to pull the user back in repeatedly.",
                        salience=0.84,
                        confidence=0.86,
                        engagement_score=0.92,
                        evidence_count=4,
                        engagement_count=5,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        refresh = service.maybe_refresh(search_backend=None)
        subscriptions = service.store.load_subscriptions(config=config, remote_state=remote_state)
        state = service.store.load_state(config=config, remote_state=remote_state)

        self.assertEqual(refresh.status, "skipped")
        self.assertEqual(len(subscriptions), 1)
        self.assertAlmostEqual(subscriptions[0].priority, 0.52)
        self.assertEqual(subscriptions[0].refresh_interval_hours, 96)
        self.assertEqual(subscriptions[0].base_refresh_interval_hours, 96)
        self.assertLess(state.interest_signals[0].engagement_score, 0.6)
        self.assertLess(state.interest_signals[0].salience, 0.7)
        self.assertLess(state.interest_signals[0].engagement_count, 5)
        self.assertIn(state.interest_signals[0].engagement_state, {"warm", "uncertain"})
        self.assertEqual(state.interest_signals[0].ongoing_interest, "peripheral")
        self.assertEqual(state.interest_signals[0].co_attention_state, "latent")

    def test_refresh_promotes_active_interest_into_shared_co_attention_thread(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="AI Companion News",
                    title="Companion models now hold longer-lived memory threads",
                    link="https://example.com/ai/companion-memory",
                    published_at="2026-03-20T11:00:00+00:00",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:ai_companions",
                    label="AI companions",
                    feed_url="https://example.com/ai/rss.xml",
                    topics=("AI companions",),
                    priority=0.86,
                    refresh_interval_hours=24,
                    created_by="reflection",
                    created_at="2026-03-20T08:00:00+00:00",
                ),
            ),
        )
        service.store.save_state(
            config=config,
            remote_state=remote_state,
            state=WorldIntelligenceState(
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions",
                        topic="AI companions",
                        summary="The user keeps asking to return to AI companions.",
                        salience=0.81,
                        confidence=0.88,
                        engagement_score=0.94,
                        engagement_state="resonant",
                        ongoing_interest="active",
                        ongoing_interest_score=0.92,
                        evidence_count=4,
                        engagement_count=5,
                        positive_signal_count=4,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        refresh = service.maybe_refresh()
        state = service.store.load_state(config=config, remote_state=remote_state)

        self.assertEqual(refresh.status, "refreshed")
        self.assertEqual(state.interest_signals[0].ongoing_interest, "active")
        self.assertEqual(state.interest_signals[0].co_attention_state, "shared_thread")
        self.assertGreaterEqual(state.interest_signals[0].co_attention_count, 2)
        self.assertGreaterEqual(state.interest_signals[0].co_attention_score, 0.8)

    def test_repeated_refresh_without_new_items_does_not_keep_increasing_co_attention(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        service = WorldIntelligenceService(
            config=config,
            remote_state=remote_state,
            feed_reader=lambda feed_url, *, max_items, timeout_s: (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="AI Companion News",
                    title="Companion models now hold longer-lived memory threads",
                    link="https://example.com/ai/companion-memory",
                    published_at="2026-03-20T11:00:00+00:00",
                ),
            ),
            now_provider=lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        )
        service.store.save_subscriptions(
            config=config,
            remote_state=remote_state,
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:ai_companions",
                    label="AI companions",
                    feed_url="https://example.com/ai/rss.xml",
                    topics=("AI companions",),
                    priority=0.86,
                    refresh_interval_hours=24,
                    created_by="reflection",
                    created_at="2026-03-20T08:00:00+00:00",
                ),
            ),
        )
        service.store.save_state(
            config=config,
            remote_state=remote_state,
            state=WorldIntelligenceState(
                interest_signals=(
                    WorldInterestSignal(
                        signal_id="interest:ai_companions",
                        topic="AI companions",
                        summary="This topic is already a strong ongoing focus.",
                        salience=0.82,
                        confidence=0.86,
                        engagement_score=0.93,
                        engagement_state="resonant",
                        ongoing_interest="active",
                        ongoing_interest_score=0.9,
                        co_attention_state="forming",
                        co_attention_count=1,
                        evidence_count=4,
                        engagement_count=5,
                        positive_signal_count=4,
                        updated_at="2026-03-20T11:30:00+00:00",
                    ),
                ),
            ),
        )

        first_refresh = service.maybe_refresh(force=True)
        first_state = service.store.load_state(config=config, remote_state=remote_state)
        second_refresh = service.maybe_refresh(force=True)
        second_state = service.store.load_state(config=config, remote_state=remote_state)

        self.assertEqual(first_refresh.status, "refreshed")
        self.assertEqual(second_refresh.status, "refreshed")
        self.assertNotEqual(first_refresh.continuity_threads, ())
        self.assertEqual(second_refresh.continuity_threads, ())
        self.assertEqual(second_refresh.world_signals, ())
        self.assertEqual(
            second_state.interest_signals[0].co_attention_count,
            first_state.interest_signals[0].co_attention_count,
        )
        self.assertEqual(
            second_state.interest_signals[0].co_attention_state,
            first_state.interest_signals[0].co_attention_state,
        )

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
                            "support_count": 2,
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

    def test_learning_service_accumulates_cross_session_cooling_after_topic_switches(self) -> None:
        remote_state = _FakeRemoteState()
        config = TwinrConfig(project_root=".")
        learning = PersonalityLearningService.from_config(config, remote_state=remote_state)

        learning.record_conversation_consolidation(
            turn=LongTermConversationTurn(
                transcript="Erzaehl mir mehr ueber AI companions.",
                response="Ich bleibe gern bei AI companions und halte Entwicklungen im Blick.",
            ),
            consolidation=LongTermConsolidationResultV1(
                turn_id="turn:session:1",
                occurred_at=datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pref:topic_follow_up:ai_companions",
                        kind="fact",
                        summary="The user explicitly wanted to continue with AI companions.",
                        source=self._source("turn:session:1"),
                        status="active",
                        confidence=0.92,
                        attributes={
                            "memory_domain": "preference",
                            "fact_type": "preference",
                            "preference_type": "topic_follow_up",
                            "topic": "AI companions",
                            "support_count": 2,
                        },
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            ),
        )

        for index in range(2, 5):
            learning.record_conversation_consolidation(
                turn=LongTermConversationTurn(
                    transcript="Was gibt es Neues in der Weltpolitik?",
                    response="Ich schaue auf Diplomatie und internationale Entwicklungen.",
                ),
                consolidation=LongTermConsolidationResultV1(
                    turn_id=f"turn:session:{index}",
                    occurred_at=datetime(2026, 3, 20, 10 + index, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id=f"pref:topic_follow_up:world_politics:{index}",
                            kind="fact",
                            summary="The user wanted to continue with world politics.",
                            source=self._source(f"turn:session:{index}"),
                            status="active",
                            confidence=0.88,
                            attributes={
                                "memory_domain": "preference",
                                "fact_type": "preference",
                                "preference_type": "topic_follow_up",
                                "topic": "world politics",
                                "support_count": 2,
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
        signals_by_topic = {signal.topic: signal for signal in state.interest_signals}

        self.assertIn("AI companions", signals_by_topic)
        self.assertIn("world politics", signals_by_topic)
        self.assertGreaterEqual(signals_by_topic["AI companions"].non_reengagement_count, 1)
        self.assertGreaterEqual(signals_by_topic["AI companions"].exposure_count, 3)
        self.assertIn(signals_by_topic["AI companions"].engagement_state, {"cooling", "avoid"})
        self.assertIn(signals_by_topic["world politics"].engagement_state, {"resonant", "warm"})


if __name__ == "__main__":
    unittest.main()
