from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.news_ticker import (
    DisplayNewsTickerFetcher,
    DisplayNewsTickerItem,
    DisplayNewsTickerRuntime,
    DisplayNewsTickerSnapshot,
    DisplayNewsTickerStore,
    DisplayWorldIntelligenceNewsTickerSource,
)
from twinr.display.news_ticker_sources import DisplayWorldIntelligenceTickerFeedResolver
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError


class DisplayNewsTickerTests(unittest.TestCase):
    def test_fetcher_parses_rss_feed_items(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Tagesschau</title>
    <item>
      <title>First Headline</title>
      <link>https://example.com/one</link>
      <pubDate>Wed, 18 Mar 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Second Headline</title>
      <link>https://example.com/two</link>
      <pubDate>Wed, 18 Mar 2026 11:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
        fetcher = DisplayNewsTickerFetcher(feed_urls=("https://example.com/rss.xml",))

        items = fetcher._parse_feed(payload, feed_url="https://example.com/rss.xml")

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].source, "Tagesschau")
        self.assertEqual(items[0].title, "First Headline")
        self.assertEqual(items[0].link, "https://example.com/one")
        self.assertEqual(items[0].published_at, "2026-03-18T12:00:00+00:00")

    def test_fetcher_parses_atom_feed_items(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>heise online</title>
  <entry>
    <title>Atom Headline</title>
    <link href="https://example.com/atom" />
    <updated>2026-03-18T12:30:00Z</updated>
  </entry>
</feed>
"""
        fetcher = DisplayNewsTickerFetcher(feed_urls=("https://example.com/atom.xml",))

        items = fetcher._parse_feed(payload, feed_url="https://example.com/atom.xml")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "heise online")
        self.assertEqual(items[0].title, "Atom Headline")
        self.assertEqual(items[0].link, "https://example.com/atom")
        self.assertEqual(items[0].published_at, "2026-03-18T12:30:00+00:00")

    def test_runtime_rotates_cached_items_by_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DisplayNewsTickerStore(Path(temp_dir) / "display_news_ticker.json")
            store.save(
                DisplayNewsTickerSnapshot(
                    captured_at="2026-03-18T12:00:00+00:00",
                    items=(
                        DisplayNewsTickerItem(title="First Headline", source="Tagesschau"),
                        DisplayNewsTickerItem(title="Second Headline", source="Tagesschau"),
                    ),
                    feed_urls=("https://example.com/rss.xml",),
                )
            )
            runtime = DisplayNewsTickerRuntime(
                enabled=True,
                store=store,
                fetcher=None,
                rotation_interval_s=10.0,
            )

            first = runtime.current_text(now=datetime(2026, 3, 18, 12, 0, 1, tzinfo=timezone.utc))
            second = runtime.current_text(now=datetime(2026, 3, 18, 12, 0, 11, tzinfo=timezone.utc))

        self.assertEqual(first, "First Headline")
        self.assertEqual(second, "Second Headline")

    def test_runtime_uses_loading_placeholder_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = DisplayNewsTickerRuntime(
                enabled=True,
                store=DisplayNewsTickerStore(Path(temp_dir) / "display_news_ticker.json"),
                fetcher=None,
            )

            text = runtime.current_text(now=datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc))

        self.assertEqual(text, "Loading headlines...")

    def test_runtime_uses_unavailable_placeholder_for_cached_error_without_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DisplayNewsTickerStore(Path(temp_dir) / "display_news_ticker.json")
            store.save(
                DisplayNewsTickerSnapshot(
                    captured_at="2026-03-18T12:00:00+00:00",
                    items=(),
                    feed_urls=("https://example.com/rss.xml",),
                    last_error="https://example.com/rss.xml: TimeoutError",
                )
            )
            runtime = DisplayNewsTickerRuntime(
                enabled=True,
                store=store,
                fetcher=None,
            )

            text = runtime.current_text(now=datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc))

        self.assertEqual(text, "Headlines unavailable.")

    def test_world_intelligence_feed_resolver_prefers_active_high_priority_subscriptions(self) -> None:
        remote_state = mock.Mock()
        remote_state.enabled = True
        current_records = mock.Mock()
        current_records.probe_current_head_result.return_value = ("found", {"items_count": 3})
        current_records.load_collection_payloads.return_value = (
            {
                "subscription_id": "feed:one",
                "label": "Low",
                "feed_url": "https://example.com/low.xml",
                "priority": 0.3,
                "active": True,
            },
            {
                "subscription_id": "feed:two",
                "label": "Inactive",
                "feed_url": "https://example.com/off.xml",
                "priority": 0.95,
                "active": False,
            },
            {
                "subscription_id": "feed:three",
                "label": "High",
                "feed_url": "https://example.com/high.xml",
                "priority": 0.9,
                "active": True,
            },
        )

        with (
            mock.patch(
                "twinr.display.news_ticker_sources._clone_remote_state_with_capped_read_timeout",
                return_value=remote_state,
            ) as clone_remote_state,
            mock.patch(
                "twinr.display.news_ticker_sources.LongTermRemoteCurrentRecordStore",
                return_value=current_records,
            ),
        ):
            resolver = DisplayWorldIntelligenceTickerFeedResolver(
                config=TwinrConfig(openai_api_key="sk-test"),
                remote_state=remote_state,
                max_feed_urls=4,
                remote_read_timeout_s=3.5,
            )

            self.assertEqual(
                resolver.resolve_feed_urls(),
                ("https://example.com/high.xml", "https://example.com/low.xml"),
            )

        clone_remote_state.assert_called_once_with(
            config=resolver.config,
            remote_state=remote_state,
            timeout_s=3.5,
        )
        current_records.probe_current_head_result.assert_called_once_with(
            snapshot_kind="agent_world_intelligence_subscriptions_v1",
            fast_fail=True,
        )
        remote_state.load_snapshot.assert_not_called()

    def test_resolver_rejects_legacy_snapshot_blob_when_current_head_missing(self) -> None:
        remote_state = mock.Mock()
        remote_state.enabled = True
        current_records = mock.Mock()
        current_records.probe_current_head_result.return_value = ("not_found", None)
        current_records.probe_legacy_collection_head.return_value = {"items_count": 2}

        with (
            mock.patch(
                "twinr.display.news_ticker_sources._clone_remote_state_with_capped_read_timeout",
                return_value=remote_state,
            ),
            mock.patch(
                "twinr.display.news_ticker_sources.LongTermRemoteCurrentRecordStore",
                return_value=current_records,
            ),
        ):
            resolver = DisplayWorldIntelligenceTickerFeedResolver(
                config=TwinrConfig(openai_api_key="sk-test"),
                remote_state=remote_state,
                max_feed_urls=4,
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                resolver.resolve_feed_urls()

        current_records.probe_legacy_collection_head.assert_called_once_with(
            snapshot_kind="agent_world_intelligence_subscriptions_v1",
            prefer_metadata_only=True,
        )
        remote_state.load_snapshot.assert_not_called()

    def test_world_intelligence_news_ticker_source_fetches_from_resolved_feed_urls(self) -> None:
        resolver = mock.Mock()
        resolver.resolve_feed_urls.return_value = (
            "https://example.com/one.xml",
            "https://example.com/two.xml",
        )
        source = DisplayWorldIntelligenceNewsTickerSource(
            resolver=resolver,
            timeout_s=3.5,
            max_items=5,
        )

        with mock.patch.object(DisplayNewsTickerFetcher, "fetch", autospec=True) as fetch:
            fetch.return_value = DisplayNewsTickerSnapshot(
                captured_at="2026-03-22T12:00:00+00:00",
                items=(DisplayNewsTickerItem(title="Headline", source="Feed"),),
                feed_urls=("https://example.com/one.xml", "https://example.com/two.xml"),
            )

            snapshot = source.fetch(now=datetime(2026, 3, 22, 12, 0, 0, tzinfo=timezone.utc))

        self.assertEqual(len(snapshot.items), 1)
        resolver.resolve_feed_urls.assert_called_once_with()
        fetch.assert_called_once()

    def test_resolver_migrates_legacy_ticker_feed_urls_when_world_pool_is_empty(self) -> None:
        remote_state = mock.Mock()
        remote_state.enabled = True
        current_records = mock.Mock()
        current_records.probe_current_head_result.return_value = ("not_found", None)
        current_records.probe_legacy_collection_head.return_value = None
        world_store = mock.Mock()

        with (
            mock.patch(
                "twinr.display.news_ticker_sources._clone_remote_state_with_capped_read_timeout",
                return_value=remote_state,
            ),
            mock.patch(
                "twinr.display.news_ticker_sources.LongTermRemoteCurrentRecordStore",
                return_value=current_records,
            ),
            mock.patch(
                "twinr.display.news_ticker_sources.RemoteStateWorldIntelligenceStore",
                return_value=world_store,
            ),
        ):
            resolver = DisplayWorldIntelligenceTickerFeedResolver(
                config=TwinrConfig(
                    openai_api_key="sk-test",
                    display_news_ticker_legacy_feed_urls=(
                        "https://example.com/a.rss",
                        "https://example.com/b.atom",
                    ),
                ),
                remote_state=remote_state,
                max_feed_urls=4,
            )

            feed_urls = resolver.resolve_feed_urls()

        self.assertEqual(
            feed_urls,
            ("https://example.com/a.rss", "https://example.com/b.atom"),
        )
        world_store.save_subscriptions.assert_called_once()
        current_records.probe_legacy_collection_head.assert_called_once_with(
            snapshot_kind="agent_world_intelligence_subscriptions_v1",
            prefer_metadata_only=True,
        )
        remote_state.save_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
