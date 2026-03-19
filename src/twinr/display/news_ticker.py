"""Fetch and cache HDMI news-ticker headlines from RSS or Atom feeds.

The HDMI renderer only needs one calm, readable ticker line. This module keeps
network access, XML parsing, caching, and headline rotation separate from the
display loop so the visible screen can stay responsive on the Raspberry Pi.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import email.utils
import html
import json
from pathlib import Path
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_STORE_PATH = "artifacts/stores/ops/display_news_ticker.json"
_DEFAULT_USER_AGENT = "TwinrNewsTicker/1.0"
_MAX_FEED_BYTES = 512 * 1024
_MAX_TITLE_CHARS = 220


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _normalize_now(value: datetime | None) -> datetime:
    """Return one aware UTC datetime."""

    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _local_name(tag: object) -> str:
    """Return the local XML element name without any namespace prefix."""

    text = str(tag or "")
    if "}" in text:
        return text.rsplit("}", 1)[-1]
    return text


def _clean_text(value: str | None) -> str:
    """Normalize one free-form headline into bounded single-line copy."""

    if not value:
        return ""
    compact = " ".join(html.unescape(value).split())
    if len(compact) <= _MAX_TITLE_CHARS:
        return compact
    return compact[: _MAX_TITLE_CHARS - 1].rstrip() + "…"


def _child_text(element: ET.Element | None, *names: str) -> str:
    """Return the first non-empty direct-child text matching one of the names."""

    if element is None:
        return ""
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag).lower() not in wanted:
            continue
        text = _clean_text("".join(child.itertext()))
        if text:
            return text
    return ""


def _child_elements(element: ET.Element | None, name: str) -> tuple[ET.Element, ...]:
    """Return all direct children with one matching local name."""

    if element is None:
        return ()
    wanted = name.lower()
    return tuple(child for child in element if _local_name(child.tag).lower() == wanted)


def _parse_timestamp(value: str | None) -> str | None:
    """Normalize one RSS or Atom timestamp to UTC ISO-8601 text."""

    compact = _clean_text(value)
    if not compact:
        return None
    parsed: datetime | None = None
    try:
        parsed = email.utils.parsedate_to_datetime(compact)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(compact.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _feed_source(feed_url: str, feed_title: str) -> str:
    """Return one short human-friendly source label for a feed."""

    clean_title = _clean_text(feed_title)
    if clean_title:
        return clean_title
    host = urllib.parse.urlparse(feed_url).netloc.strip()
    if host.startswith("www."):
        host = host[4:]
    return host or "Feed"


@dataclass(frozen=True, slots=True)
class DisplayNewsTickerItem:
    """Store one cached headline item for the HDMI news ticker."""

    title: str
    source: str
    link: str | None = None
    published_at: str | None = None

    def display_text(self, *, include_source: bool) -> str:
        """Return the user-facing single-line ticker copy."""

        title = _clean_text(self.title)
        source = _clean_text(self.source)
        if include_source and source:
            return f"{source} · {title}"
        return title


@dataclass(frozen=True, slots=True)
class DisplayNewsTickerSnapshot:
    """Represent one cached news-ticker headline bundle."""

    captured_at: str
    items: tuple[DisplayNewsTickerItem, ...]
    feed_urls: tuple[str, ...] = ()
    last_error: str | None = None

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "DisplayNewsTickerSnapshot":
        """Load one snapshot from JSON-compatible storage data."""

        raw_items = payload.get("items")
        items: list[DisplayNewsTickerItem] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                title = _clean_text(str(raw_item.get("title") or ""))
                if not title:
                    continue
                items.append(
                    DisplayNewsTickerItem(
                        title=title,
                        source=_clean_text(str(raw_item.get("source") or "")) or "Feed",
                        link=_clean_text(str(raw_item.get("link") or "")) or None,
                        published_at=_parse_timestamp(raw_item.get("published_at") if isinstance(raw_item.get("published_at"), str) else None),
                    )
                )
        raw_urls = payload.get("feed_urls")
        urls = tuple(str(value).strip() for value in raw_urls if str(value).strip()) if isinstance(raw_urls, list) else ()
        captured_at = _parse_timestamp(payload.get("captured_at") if isinstance(payload.get("captured_at"), str) else None)
        return cls(
            captured_at=captured_at or _utc_now().isoformat(),
            items=tuple(items),
            feed_urls=urls,
            last_error=_clean_text(str(payload.get("last_error") or "")) or None,
        )

    def to_json(self) -> dict[str, object]:
        """Return one JSON-compatible storage payload."""

        return {
            "captured_at": self.captured_at,
            "feed_urls": list(self.feed_urls),
            "last_error": self.last_error,
            "items": [
                {
                    "title": item.title,
                    "source": item.source,
                    "link": item.link,
                    "published_at": item.published_at,
                }
                for item in self.items
            ],
        }


@dataclass(slots=True)
class DisplayNewsTickerStore:
    """Persist cached HDMI news headlines on a runtime-writable path."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayNewsTickerStore":
        """Build the store path from Twinr configuration."""

        return cls(Path(config.project_root) / config.display_news_ticker_store_path)

    def load(self) -> DisplayNewsTickerSnapshot | None:
        """Load the cached headline bundle if the file exists and is readable."""

        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return DisplayNewsTickerSnapshot.from_json(payload)

    def save(self, snapshot: DisplayNewsTickerSnapshot) -> None:
        """Persist one headline bundle atomically."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(snapshot.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
        temp_path.replace(self.path)


@dataclass(slots=True)
class DisplayNewsTickerFetcher:
    """Fetch and parse bounded headline bundles from RSS or Atom URLs."""

    feed_urls: tuple[str, ...]
    timeout_s: float = 4.0
    max_items: int = 12
    user_agent: str = _DEFAULT_USER_AGENT

    def fetch(self, *, now: datetime | None = None) -> DisplayNewsTickerSnapshot:
        """Fetch the current headline bundle from all configured feeds."""

        captured_at = _normalize_now(now)
        collected: list[DisplayNewsTickerItem] = []
        errors: list[str] = []
        for feed_url in self.feed_urls:
            try:
                payload = self._download_feed(feed_url)
                collected.extend(self._parse_feed(payload, feed_url=feed_url))
            except Exception as exc:
                errors.append(f"{feed_url}: {type(exc).__name__}")
        deduped = self._dedupe_items(collected)
        return DisplayNewsTickerSnapshot(
            captured_at=captured_at.isoformat(),
            items=tuple(deduped[: max(1, int(self.max_items))]),
            feed_urls=self.feed_urls,
            last_error=" | ".join(errors[:3]) or None,
        )

    def _download_feed(self, feed_url: str) -> bytes:
        request = urllib.request.Request(feed_url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=max(0.5, float(self.timeout_s))) as response:
            payload = response.read(_MAX_FEED_BYTES + 1)
        if len(payload) > _MAX_FEED_BYTES:
            raise ValueError("feed_payload_too_large")
        return payload

    def _parse_feed(self, payload: bytes, *, feed_url: str) -> tuple[DisplayNewsTickerItem, ...]:
        root = ET.fromstring(payload)
        root_name = _local_name(root.tag).lower()
        if root_name == "rss":
            return self._parse_rss_feed(root, feed_url=feed_url)
        if root_name == "feed":
            return self._parse_atom_feed(root, feed_url=feed_url)
        raise ValueError("unsupported_feed_format")

    def _parse_rss_feed(self, root: ET.Element, *, feed_url: str) -> tuple[DisplayNewsTickerItem, ...]:
        channel = next((child for child in root if _local_name(child.tag).lower() == "channel"), None)
        source = _feed_source(feed_url, _child_text(channel, "title"))
        items: list[DisplayNewsTickerItem] = []
        for item in _child_elements(channel, "item"):
            title = _child_text(item, "title")
            if not title:
                continue
            items.append(
                DisplayNewsTickerItem(
                    title=title,
                    source=source,
                    link=_child_text(item, "link") or None,
                    published_at=_parse_timestamp(_child_text(item, "pubDate", "published", "updated")),
                )
            )
        return tuple(items)

    def _parse_atom_feed(self, root: ET.Element, *, feed_url: str) -> tuple[DisplayNewsTickerItem, ...]:
        source = _feed_source(feed_url, _child_text(root, "title"))
        items: list[DisplayNewsTickerItem] = []
        for entry in _child_elements(root, "entry"):
            title = _child_text(entry, "title")
            if not title:
                continue
            items.append(
                DisplayNewsTickerItem(
                    title=title,
                    source=source,
                    link=self._atom_entry_link(entry),
                    published_at=_parse_timestamp(_child_text(entry, "updated", "published")),
                )
            )
        return tuple(items)

    def _atom_entry_link(self, entry: ET.Element) -> str | None:
        for child in entry:
            if _local_name(child.tag).lower() != "link":
                continue
            href = _clean_text(child.attrib.get("href"))
            if href:
                return href
        return None

    def _dedupe_items(self, items: Sequence[DisplayNewsTickerItem]) -> tuple[DisplayNewsTickerItem, ...]:
        deduped: list[DisplayNewsTickerItem] = []
        seen: set[str] = set()
        for item in items:
            key = item.title.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return tuple(deduped)


@dataclass(slots=True)
class DisplayNewsTickerRuntime:
    """Serve calm, readable HDMI ticker text from cached headline bundles."""

    enabled: bool
    store: DisplayNewsTickerStore
    fetcher: DisplayNewsTickerFetcher | None = None
    refresh_interval_s: float = 600.0
    rotation_interval_s: float = 12.0
    emit: Callable[[str], None] | None = None
    _snapshot: DisplayNewsTickerSnapshot | None = field(default=None, init=False, repr=False)
    _refresh_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _refresh_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "DisplayNewsTickerRuntime":
        """Build one ticker runtime from Twinr configuration."""

        feed_urls = tuple(url for url in config.display_news_ticker_feed_urls if str(url).strip())
        return cls(
            enabled=bool(config.display_news_ticker_enabled and feed_urls),
            store=DisplayNewsTickerStore.from_config(config),
            fetcher=DisplayNewsTickerFetcher(
                feed_urls=feed_urls,
                timeout_s=config.display_news_ticker_timeout_s,
                max_items=config.display_news_ticker_max_items,
            )
            if feed_urls
            else None,
            refresh_interval_s=config.display_news_ticker_refresh_interval_s,
            rotation_interval_s=config.display_news_ticker_rotation_interval_s,
            emit=emit,
        )

    def current_text(self, *, now: datetime | None = None) -> str | None:
        """Return the currently visible ticker line, or ``None`` when disabled."""

        if not self.enabled:
            return None
        safe_now = _normalize_now(now)
        self._ensure_snapshot_loaded()
        self._ensure_refresh_async(now=safe_now)
        snapshot = self._snapshot
        if snapshot is None or not snapshot.items:
            if snapshot is not None and snapshot.last_error:
                return "Headlines unavailable."
            return "Loading headlines..."
        rotation_seconds = max(4.0, float(self.rotation_interval_s))
        index = int(safe_now.timestamp() // rotation_seconds) % len(snapshot.items)
        visible_sources = {item.source for item in snapshot.items if item.source}
        return snapshot.items[index].display_text(include_source=len(visible_sources) > 1)

    def _ensure_snapshot_loaded(self) -> None:
        if self._snapshot is not None:
            return
        try:
            self._snapshot = self.store.load()
        except Exception as exc:
            self._safe_emit(f"display_news_ticker_cache_load_failed={type(exc).__name__}")

    def _ensure_refresh_async(self, *, now: datetime) -> None:
        fetcher = self.fetcher
        if fetcher is None:
            return
        thread = self._refresh_thread
        if thread is not None and thread.is_alive():
            return
        snapshot = self._snapshot
        if snapshot is not None and not self._is_snapshot_stale(snapshot, now=now):
            return
        with self._refresh_lock:
            thread = self._refresh_thread
            if thread is not None and thread.is_alive():
                return
            self._refresh_thread = threading.Thread(
                target=self._refresh_once,
                name="twinr-display-news-ticker",
                daemon=True,
            )
            self._refresh_thread.start()

    def _is_snapshot_stale(self, snapshot: DisplayNewsTickerSnapshot, *, now: datetime) -> bool:
        captured_at = _parse_timestamp(snapshot.captured_at)
        if captured_at is None:
            return True
        try:
            parsed = datetime.fromisoformat(captured_at)
        except ValueError:
            return True
        age_s = (now - _normalize_now(parsed)).total_seconds()
        return age_s >= max(30.0, float(self.refresh_interval_s))

    def _refresh_once(self) -> None:
        fetcher = self.fetcher
        if fetcher is None:
            return
        previous = self._snapshot
        if previous is None:
            try:
                previous = self.store.load()
            except Exception:
                previous = None
        try:
            snapshot = fetcher.fetch()
        except Exception as exc:
            self._safe_emit(f"display_news_ticker_refresh_failed={type(exc).__name__}")
            return
        if not snapshot.items and previous is not None and previous.items:
            snapshot = DisplayNewsTickerSnapshot(
                captured_at=snapshot.captured_at,
                items=previous.items,
                feed_urls=snapshot.feed_urls,
                last_error=snapshot.last_error or previous.last_error,
            )
        self._snapshot = snapshot
        try:
            self.store.save(snapshot)
        except Exception as exc:
            self._safe_emit(f"display_news_ticker_cache_save_failed={type(exc).__name__}")

    def _safe_emit(self, line: str) -> None:
        if self.emit is None:
            return
        try:
            self.emit(line)
        except Exception:
            return
