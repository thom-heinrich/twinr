from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.news_ticker import (
    DisplayNewsTickerFetcher,
    DisplayNewsTickerItem,
    DisplayNewsTickerRuntime,
    DisplayNewsTickerSnapshot,
    DisplayNewsTickerStore,
)


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


if __name__ == "__main__":
    unittest.main()
