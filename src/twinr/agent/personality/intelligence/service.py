"""Manage RSS-backed place/world intelligence for the personality system.

This service owns three bounded responsibilities:

- persist and mutate the curated subscription set that Twinr uses for calm
  place/world awareness
- discover RSS/Atom feeds from explicit source pages returned by the web-search
  backend when the installer or a recalibration pass asks for new sources
- refresh due feeds and convert fresh items into world signals plus continuity
  threads that the background personality loop can commit

The service does not rewrite core character. It only emits contextual world and
continuity updates that higher layers may merge through existing policy gates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import hashlib
from typing import Any
from urllib.parse import urljoin, urlparse
import urllib.request
import xml.etree.ElementTree as ET

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldFeedItem,
    WorldFeedSubscription,
    WorldIntelligenceConfigRequest,
    WorldIntelligenceConfigResult,
    WorldInterestSignal,
    WorldIntelligenceRefreshResult,
    WorldIntelligenceState,
)
from twinr.agent.personality.intelligence.store import (
    RemoteStateWorldIntelligenceStore,
    WorldIntelligenceStore,
)
from twinr.agent.personality.models import ContinuityThread, WorldSignal
from twinr.display.news_ticker import DisplayNewsTickerFetcher
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_SUPPORTED_FEED_MIME_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/feed+json",
    }
)
_MAX_DISCOVERY_BYTES = 512 * 1024


def _utcnow() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    """Render one aware datetime as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    """Parse one ISO timestamp or return ``None`` for blank input."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_text(value: object | None) -> str:
    """Normalize one free-form string into a trimmed single line."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _token_set(value: object | None) -> frozenset[str]:
    """Normalize one label into a comparable set of lowercase tokens."""

    normalized = _clean_text(value).casefold()
    for separator in (",", ".", ";", ":", "/", "-", "_", "(", ")", "[", "]"):
        normalized = normalized.replace(separator, " ")
    return frozenset(token for token in normalized.split() if token)


def _host_label(url: str) -> str:
    """Return one short label derived from a URL host."""

    host = urlparse(url).netloc.strip().casefold()
    if host.startswith("www."):
        host = host[4:]
    return host or "Feed"


def _subscription_id(feed_url: str) -> str:
    """Return a stable subscription id for one feed URL."""

    digest = hashlib.sha1(feed_url.encode("utf-8")).hexdigest()[:12]
    return f"feed:{digest}"


def _feed_item_id(item: WorldFeedItem) -> str:
    """Return a stable item identifier for one fetched feed item."""

    link = _clean_text(item.link)
    if link:
        return link
    published_at = _clean_text(item.published_at)
    if published_at:
        return f"{item.feed_url}::{published_at}::{item.title}"
    return f"{item.feed_url}::{item.title}"


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp one numeric value into an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


@dataclass(frozen=True, slots=True)
class _FetchedDocument:
    """Represent one fetched HTML or feed document."""

    url: str
    text: str
    content_type: str


class _FeedAutodiscoveryParser(HTMLParser):
    """Extract RSS/Atom alternate links from one HTML document."""

    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.discovered: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one alternate feed link when the tag describes one."""

        if str(tag).strip().casefold() != "link":
            return
        normalized = {str(key).strip().casefold(): (value or "") for key, value in attrs if key}
        rel_tokens = {token.strip().casefold() for token in normalized.get("rel", "").split() if token.strip()}
        if "alternate" not in rel_tokens:
            return
        mime_type = normalized.get("type", "").strip().casefold()
        if mime_type not in _SUPPORTED_FEED_MIME_TYPES:
            return
        href = normalized.get("href", "").strip()
        if not href:
            return
        resolved = urljoin(self.base_url, href)
        title = _clean_text(normalized.get("title")) or None
        self.discovered.append((resolved, title))


def _default_page_loader(url: str) -> _FetchedDocument:
    """Fetch one HTML or XML document used for feed autodiscovery."""

    request = urllib.request.Request(url, headers={"User-Agent": "TwinrWorldIntelligence/1.0"})
    with urllib.request.urlopen(request, timeout=5.0) as response:
        payload = response.read(_MAX_DISCOVERY_BYTES + 1)
        content_type = str(response.headers.get("Content-Type") or "application/octet-stream")
    if len(payload) > _MAX_DISCOVERY_BYTES:
        raise ValueError("world_intelligence_document_too_large")
    charset = "utf-8"
    content_type_parts = [part.strip() for part in content_type.split(";") if part.strip()]
    for part in content_type_parts[1:]:
        lower = part.casefold()
        if lower.startswith("charset="):
            charset = part.split("=", 1)[-1].strip() or "utf-8"
            break
    try:
        text = payload.decode(charset, errors="replace")
    except LookupError:
        text = payload.decode("utf-8", errors="replace")
    return _FetchedDocument(url=url, text=text, content_type=content_type_parts[0] if content_type_parts else content_type)


def _default_feed_reader(
    feed_url: str,
    *,
    max_items: int,
    timeout_s: float,
) -> tuple[WorldFeedItem, ...]:
    """Fetch one RSS or Atom URL through the bounded display feed parser."""

    snapshot = DisplayNewsTickerFetcher(
        feed_urls=(feed_url,),
        timeout_s=timeout_s,
        max_items=max_items,
    ).fetch()
    if snapshot.last_error and not snapshot.items:
        raise RuntimeError(snapshot.last_error)
    return tuple(
        WorldFeedItem(
            feed_url=feed_url,
            source=item.source,
            title=item.title,
            link=item.link,
            published_at=item.published_at,
        )
        for item in snapshot.items
    )


class WorldIntelligenceService:
    """Persist RSS subscriptions and convert refreshes into personality context."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        store: WorldIntelligenceStore | None = None,
        now_provider: Callable[[], datetime] = _utcnow,
        page_loader: Callable[[str], Any] = _default_page_loader,
        feed_reader: Callable[..., tuple[WorldFeedItem, ...]] = _default_feed_reader,
        default_refresh_interval_hours: int = 72,
        default_freshness_hours: int = 96,
        feed_timeout_s: float = 4.0,
        max_items_per_refresh: int = 4,
        max_signals_per_subscription: int = 2,
        max_retained_item_ids: int = 24,
        max_interest_signals: int = 48,
        max_awareness_threads: int = 24,
        max_recent_titles_per_thread: int = 4,
        max_supporting_item_ids_per_thread: int = 24,
        awareness_review_days: int = 14,
    ) -> None:
        """Store the bounded collaborators used by the intelligence loop."""

        self.config = config
        self.remote_state = remote_state
        self.store = store or RemoteStateWorldIntelligenceStore()
        self.now_provider = now_provider
        self.page_loader = page_loader
        self.feed_reader = feed_reader
        self.default_refresh_interval_hours = max(24, int(default_refresh_interval_hours))
        self.default_freshness_hours = max(24, int(default_freshness_hours))
        self.feed_timeout_s = max(0.5, float(feed_timeout_s))
        self.max_items_per_refresh = max(1, int(max_items_per_refresh))
        self.max_signals_per_subscription = max(1, int(max_signals_per_subscription))
        self.max_retained_item_ids = max(4, int(max_retained_item_ids))
        self.max_interest_signals = max(8, int(max_interest_signals))
        self.max_awareness_threads = max(4, int(max_awareness_threads))
        self.max_recent_titles_per_thread = max(1, int(max_recent_titles_per_thread))
        self.max_supporting_item_ids_per_thread = max(8, int(max_supporting_item_ids_per_thread))
        self.awareness_review_days = max(3, int(awareness_review_days))

    def configure(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None = None,
    ) -> WorldIntelligenceConfigResult:
        """Apply one explicit subscription/discovery request."""

        subscriptions = list(
            self.store.load_subscriptions(config=self.config, remote_state=self.remote_state)
        )
        state = self.store.load_state(config=self.config, remote_state=self.remote_state)
        now = self.now_provider()
        now_iso = _isoformat(now)
        discovered_feed_urls: tuple[str, ...] = ()

        if request.action == "discover":
            discovered_feed_urls = self._discover_feed_urls(request=request, search_backend=search_backend)
            state = replace(
                state,
                last_discovered_at=now_iso,
                last_discovery_query=request.query or self._discovery_question(request),
            )
            if request.auto_subscribe and discovered_feed_urls:
                subscriptions = self._upsert_subscriptions(
                    existing=subscriptions,
                    request=request,
                    feed_urls=discovered_feed_urls,
                    now_iso=now_iso,
                )
        elif request.action == "subscribe":
            subscriptions = self._upsert_subscriptions(
                existing=subscriptions,
                request=request,
                feed_urls=request.feed_urls,
                now_iso=now_iso,
            )
        elif request.action == "deactivate":
            subscriptions = self._deactivate_subscriptions(
                existing=subscriptions,
                request=request,
                now_iso=now_iso,
            )
        elif request.action == "list":
            pass
        elif request.action == "refresh_now":
            pass

        self.store.save_subscriptions(
            config=self.config,
            subscriptions=subscriptions,
            remote_state=self.remote_state,
        )
        self.store.save_state(
            config=self.config,
            state=state,
            remote_state=self.remote_state,
        )

        refresh_result: WorldIntelligenceRefreshResult | None = None
        if request.action == "refresh_now" or request.refresh_after_change:
            refresh_result = self.maybe_refresh(force=True)
            subscriptions = list(refresh_result.subscriptions)

        return WorldIntelligenceConfigResult(
            status="ok",
            action=request.action,
            subscriptions=tuple(subscriptions),
            discovered_feed_urls=discovered_feed_urls,
            refresh=refresh_result,
        )

    def record_interest_signals(
        self,
        *,
        signals: Sequence[WorldInterestSignal],
    ) -> WorldIntelligenceState:
        """Persist slowly learned topic/region interests for later recalibration."""

        if not signals:
            return self.store.load_state(config=self.config, remote_state=self.remote_state)
        state = self.store.load_state(config=self.config, remote_state=self.remote_state)
        merged = self._merge_interest_signals(
            existing=state.interest_signals,
            incoming=signals,
        )
        updated_state = replace(
            state,
            interest_signals=merged,
        )
        self.store.save_state(
            config=self.config,
            state=updated_state,
            remote_state=self.remote_state,
        )
        return updated_state

    def maybe_refresh(
        self,
        *,
        force: bool = False,
        search_backend: object | None = None,
    ) -> WorldIntelligenceRefreshResult:
        """Recalibrate if due, then refresh due feed subscriptions."""

        subscriptions = list(
            self.store.load_subscriptions(config=self.config, remote_state=self.remote_state)
        )
        state = self.store.load_state(config=self.config, remote_state=self.remote_state)
        now = self.now_provider()
        now_iso = _isoformat(now)
        subscriptions, state = self._maybe_recalibrate(
            subscriptions=subscriptions,
            state=state,
            now=now,
            force=force,
            search_backend=search_backend,
        )
        due_subscriptions = [
            subscription
            for subscription in subscriptions
            if subscription.active and (force or self._is_due(subscription, now=now))
        ]
        if not due_subscriptions:
            return WorldIntelligenceRefreshResult(
                status="skipped",
                refreshed=False,
                subscriptions=tuple(subscriptions),
                checked_at=now_iso,
            )

        refreshed_ids: list[str] = []
        errors: list[str] = []
        world_signals: list[WorldSignal] = []
        continuity_threads: list[ContinuityThread] = []
        updated_awareness_threads = tuple(state.awareness_threads)
        updated_subscriptions: list[WorldFeedSubscription] = []

        for subscription in subscriptions:
            if subscription.subscription_id not in {item.subscription_id for item in due_subscriptions}:
                updated_subscriptions.append(subscription)
                continue

            try:
                items = tuple(
                    self.feed_reader(
                        subscription.feed_url,
                        max_items=self.max_items_per_refresh,
                        timeout_s=self.feed_timeout_s,
                    )
                )
                unseen_items, retained_item_ids = self._unseen_items(subscription=subscription, items=items)
                world_signals.extend(
                    self._build_world_signals(
                        subscription=subscription,
                        items=unseen_items[: self.max_signals_per_subscription],
                        now=now,
                    )
                )
                if unseen_items:
                    awareness_thread = self._merge_awareness_thread(
                        existing_threads=updated_awareness_threads,
                        subscription=subscription,
                        items=unseen_items[: self.max_signals_per_subscription],
                        now=now,
                    )
                    updated_awareness_threads = self._upsert_awareness_thread(
                        existing=updated_awareness_threads,
                        thread=awareness_thread,
                    )
                    world_signals.append(
                        self._build_awareness_world_signal(
                            thread=awareness_thread,
                            now=now,
                        )
                    )
                    continuity_threads.append(
                        self._build_continuity_thread(
                            thread=awareness_thread,
                            now=now,
                        )
                    )
                updated_subscriptions.append(
                    replace(
                        subscription,
                        updated_at=now_iso,
                        last_checked_at=now_iso,
                        last_refreshed_at=now_iso,
                        last_error=None,
                        last_item_ids=retained_item_ids,
                    )
                )
                refreshed_ids.append(subscription.subscription_id)
            except Exception as exc:
                errors.append(f"{subscription.subscription_id}: {type(exc).__name__}")
                updated_subscriptions.append(
                    replace(
                        subscription,
                        updated_at=now_iso,
                        last_checked_at=now_iso,
                        last_error=type(exc).__name__,
                    )
                )

        self.store.save_subscriptions(
            config=self.config,
            subscriptions=updated_subscriptions,
            remote_state=self.remote_state,
        )
        self.store.save_state(
            config=self.config,
            state=replace(
                state,
                last_refreshed_at=now_iso,
                awareness_threads=updated_awareness_threads,
            ),
            remote_state=self.remote_state,
        )

        status = "refreshed"
        if errors and not refreshed_ids:
            status = "error"
        return WorldIntelligenceRefreshResult(
            status=status,
            refreshed=bool(refreshed_ids),
            subscriptions=tuple(updated_subscriptions),
            world_signals=tuple(world_signals),
            continuity_threads=tuple(continuity_threads),
            awareness_threads=updated_awareness_threads,
            refreshed_subscription_ids=tuple(refreshed_ids),
            errors=tuple(errors),
            checked_at=now_iso,
        )

    def _discover_feed_urls(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None,
    ) -> tuple[str, ...]:
        """Discover feed URLs from explicit source pages returned by web search."""

        if search_backend is None:
            raise RuntimeError("discover requires a search backend")
        search = getattr(search_backend, "search_live_info_with_metadata", None)
        if not callable(search):
            raise RuntimeError("discover requires search_live_info_with_metadata")
        result = search(
            request.query or self._discovery_question(request),
            conversation=None,
            location_hint=request.location_hint or request.region,
            date_context=None,
        )
        raw_sources = getattr(result, "sources", ())
        source_urls = tuple(
            _clean_text(source)
            for source in raw_sources
            if _clean_text(source)
        )
        discovered: list[str] = []
        seen: set[str] = set()
        for source_url in source_urls:
            for feed_url in self._discover_feeds_from_source(source_url):
                if feed_url in seen:
                    continue
                seen.add(feed_url)
                discovered.append(feed_url)
        return tuple(discovered)

    def _discovery_question(self, request: WorldIntelligenceConfigRequest) -> str:
        """Build one calm discovery question for the live web backend."""

        parts = ["Find RSS or Atom feeds"]
        if request.topics:
            parts.append(f"for {', '.join(request.topics)}")
        if request.location_hint or request.region:
            parts.append(f"relevant to {request.location_hint or request.region}")
        return " ".join(parts).strip() + "."

    def _discover_feeds_from_source(self, source_url: str) -> tuple[str, ...]:
        """Extract feed URLs from one source page or direct feed URL."""

        document = self.page_loader(source_url)
        url = _clean_text(getattr(document, "url", None)) or source_url
        content_type = _clean_text(getattr(document, "content_type", None)).casefold()
        text = str(getattr(document, "text", ""))
        if self._looks_like_feed_document(text=text, content_type=content_type):
            return (url,)
        parser = _FeedAutodiscoveryParser(base_url=url)
        parser.feed(text)
        ordered: list[str] = []
        seen: set[str] = set()
        for feed_url, _title in parser.discovered:
            if feed_url in seen:
                continue
            seen.add(feed_url)
            ordered.append(feed_url)
        return tuple(ordered)

    def _looks_like_feed_document(self, *, text: str, content_type: str) -> bool:
        """Return whether one fetched document is already a feed."""

        normalized_type = content_type.casefold()
        if normalized_type in _SUPPORTED_FEED_MIME_TYPES:
            return True
        if "xml" not in normalized_type and not text.lstrip().startswith("<"):
            return False
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return False
        local_name = str(root.tag).rsplit("}", 1)[-1].casefold()
        return local_name in {"rss", "feed"}

    def _upsert_subscriptions(
        self,
        *,
        existing: Sequence[WorldFeedSubscription],
        request: WorldIntelligenceConfigRequest,
        feed_urls: Sequence[str],
        now_iso: str,
    ) -> list[WorldFeedSubscription]:
        """Create or update subscriptions for one set of feed URLs."""

        ordered: list[WorldFeedSubscription] = []
        normalized_feed_urls = [_clean_text(url) for url in feed_urls if _clean_text(url)]
        updated_feed_urls = set(normalized_feed_urls)

        for subscription in existing:
            if subscription.feed_url not in updated_feed_urls:
                ordered.append(subscription)
                continue
            ordered.append(
                replace(
                    subscription,
                    label=request.label or subscription.label,
                    scope=request.scope,
                    region=request.region or request.location_hint or subscription.region,
                    topics=request.topics or subscription.topics,
                    priority=request.priority,
                    active=True,
                    refresh_interval_hours=request.refresh_interval_hours or subscription.refresh_interval_hours,
                    updated_at=now_iso,
                    created_by=subscription.created_by,
                )
            )

        existing_feed_urls = {subscription.feed_url for subscription in existing}
        for feed_url in normalized_feed_urls:
            if feed_url in existing_feed_urls:
                continue
            ordered.append(
                WorldFeedSubscription(
                    subscription_id=_subscription_id(feed_url),
                    label=request.label or _host_label(feed_url),
                    feed_url=feed_url,
                    scope=request.scope,
                    region=request.region or request.location_hint,
                    topics=request.topics,
                    priority=request.priority,
                    active=True,
                    refresh_interval_hours=request.refresh_interval_hours or self.default_refresh_interval_hours,
                    created_by=request.created_by,
                    created_at=now_iso,
                    updated_at=now_iso,
                )
            )

        return ordered

    def _deactivate_subscriptions(
        self,
        *,
        existing: Sequence[WorldFeedSubscription],
        request: WorldIntelligenceConfigRequest,
        now_iso: str,
    ) -> list[WorldFeedSubscription]:
        """Deactivate matching subscriptions by id or feed URL."""

        refs = {_clean_text(item) for item in request.subscription_refs if _clean_text(item)}
        urls = {_clean_text(item) for item in request.feed_urls if _clean_text(item)}
        updated: list[WorldFeedSubscription] = []
        for subscription in existing:
            if subscription.subscription_id in refs or subscription.feed_url in urls:
                updated.append(
                    replace(
                        subscription,
                        active=False,
                        updated_at=now_iso,
                    )
                )
            else:
                updated.append(subscription)
        return updated

    def _is_due(self, subscription: WorldFeedSubscription, *, now: datetime) -> bool:
        """Return whether one subscription is due for a bounded refresh."""

        checked_at = _parse_iso(subscription.last_checked_at)
        if checked_at is None:
            return True
        due_at = checked_at + timedelta(hours=max(24, int(subscription.refresh_interval_hours)))
        return now >= due_at

    def _maybe_recalibrate(
        self,
        *,
        subscriptions: Sequence[WorldFeedSubscription],
        state: WorldIntelligenceState,
        now: datetime,
        force: bool,
        search_backend: object | None,
    ) -> tuple[list[WorldFeedSubscription], WorldIntelligenceState]:
        """Discover new feeds from durable interest signals when recalibration is due."""

        if search_backend is None:
            return list(subscriptions), state
        if not force and not self._recalibration_due(state=state, now=now):
            return list(subscriptions), state
        now_iso = _isoformat(now)
        current_subscriptions = list(subscriptions)
        updated_state = replace(
            state,
            last_recalibrated_at=now_iso,
        )
        for interest_signal in self._select_recalibration_candidates(
            subscriptions=current_subscriptions,
            state=state,
        ):
            request = WorldIntelligenceConfigRequest(
                action="discover",
                query=self._interest_discovery_question(interest_signal),
                label=self._interest_label(interest_signal),
                location_hint=interest_signal.region,
                region=interest_signal.region,
                topics=(interest_signal.topic,),
                scope=interest_signal.scope,
                priority=interest_signal.salience,
                auto_subscribe=True,
                refresh_after_change=False,
                created_by="reflection",
            )
            discovered_feed_urls = self._discover_feed_urls(
                request=request,
                search_backend=search_backend,
            )
            if not discovered_feed_urls:
                continue
            current_subscriptions = self._upsert_subscriptions(
                existing=current_subscriptions,
                request=request,
                feed_urls=discovered_feed_urls,
                now_iso=now_iso,
            )
            updated_state = replace(
                updated_state,
                last_discovered_at=now_iso,
                last_discovery_query=request.query,
            )
        self.store.save_subscriptions(
            config=self.config,
            subscriptions=current_subscriptions,
            remote_state=self.remote_state,
        )
        self.store.save_state(
            config=self.config,
            state=updated_state,
            remote_state=self.remote_state,
        )
        return current_subscriptions, updated_state

    def _recalibration_due(self, *, state: WorldIntelligenceState, now: datetime) -> bool:
        """Return whether the slow RSS recalibration cadence is due."""

        recalibrated_at = _parse_iso(state.last_recalibrated_at)
        if recalibrated_at is None:
            return True
        due_at = recalibrated_at + timedelta(hours=max(168, int(state.recalibration_interval_hours)))
        return now >= due_at

    def _select_recalibration_candidates(
        self,
        *,
        subscriptions: Sequence[WorldFeedSubscription],
        state: WorldIntelligenceState,
    ) -> tuple[WorldInterestSignal, ...]:
        """Choose the strongest uncovered interest signals for feed discovery."""

        ranked = sorted(
            state.interest_signals,
            key=lambda item: (item.explicit, item.salience, item.confidence, item.evidence_count, item.updated_at or ""),
            reverse=True,
        )
        selected: list[WorldInterestSignal] = []
        for signal in ranked:
            if any(self._subscription_covers_interest(subscription, signal) for subscription in subscriptions if subscription.active):
                continue
            selected.append(signal)
            if len(selected) >= 2:
                break
        return tuple(selected)

    def _subscription_covers_interest(
        self,
        subscription: WorldFeedSubscription,
        signal: WorldInterestSignal,
    ) -> bool:
        """Return whether one active subscription already covers one learned interest."""

        if signal.region and subscription.region:
            if _clean_text(signal.region).casefold() != _clean_text(subscription.region).casefold():
                return False
        signal_tokens = _token_set(signal.topic)
        candidate_topics = tuple(subscription.topics) or (subscription.label,)
        for candidate in candidate_topics:
            if signal_tokens & _token_set(candidate):
                return True
        return False

    def _interest_label(self, signal: WorldInterestSignal) -> str:
        """Build one human-readable subscription label from a learned interest."""

        if signal.region:
            return f"{signal.region} {signal.topic}".strip()
        return signal.topic

    def _interest_discovery_question(self, signal: WorldInterestSignal) -> str:
        """Build one calm discovery question for a learned interest."""

        parts = ["Find RSS or Atom feeds"]
        parts.append(f"for {signal.topic}")
        if signal.region:
            parts.append(f"relevant to {signal.region}")
        return " ".join(parts).strip() + "."

    def _merge_interest_signals(
        self,
        *,
        existing: Sequence[WorldInterestSignal],
        incoming: Sequence[WorldInterestSignal],
    ) -> tuple[WorldInterestSignal, ...]:
        """Merge calibration signals by topic/region/scope while keeping bounded history."""

        merged: dict[tuple[str, str, str], WorldInterestSignal] = {}
        for signal in existing:
            merged[self._interest_key(signal)] = signal
        for signal in incoming:
            key = self._interest_key(signal)
            current = merged.get(key)
            if current is None:
                merged[key] = signal
                continue
            merged[key] = WorldInterestSignal(
                signal_id=current.signal_id,
                topic=signal.topic,
                summary=signal.summary,
                region=signal.region or current.region,
                scope=signal.scope,
                salience=_clamp(max(current.salience, signal.salience) + 0.04, minimum=0.0, maximum=1.0),
                confidence=_clamp(max(current.confidence, signal.confidence), minimum=0.0, maximum=1.0),
                evidence_count=current.evidence_count + signal.evidence_count,
                explicit=current.explicit or signal.explicit,
                source_event_ids=tuple(
                    dict.fromkeys((*current.source_event_ids, *signal.source_event_ids))
                ),
                updated_at=signal.updated_at or current.updated_at,
            )
        ranked = sorted(
            merged.values(),
            key=lambda item: (item.explicit, item.salience, item.confidence, item.evidence_count, item.updated_at or ""),
            reverse=True,
        )
        return tuple(ranked[: self.max_interest_signals])

    def _interest_key(self, signal: WorldInterestSignal) -> tuple[str, str, str]:
        """Return one merge key for durable interest signals."""

        return (
            _clean_text(signal.topic).casefold(),
            _clean_text(signal.region).casefold(),
            _clean_text(signal.scope).casefold(),
        )

    def _unseen_items(
        self,
        *,
        subscription: WorldFeedSubscription,
        items: Sequence[WorldFeedItem],
    ) -> tuple[tuple[WorldFeedItem, ...], tuple[str, ...]]:
        """Return unseen items plus the bounded retained item-id history."""

        existing_ids = tuple(subscription.last_item_ids)
        seen_ids = set(existing_ids)
        unseen: list[WorldFeedItem] = []
        latest_ids: list[str] = []
        for item in items:
            item_id = _feed_item_id(item)
            latest_ids.append(item_id)
            if item_id in seen_ids:
                continue
            unseen.append(item)
        retained: list[str] = []
        for item_id in latest_ids + list(existing_ids):
            if item_id in retained:
                continue
            retained.append(item_id)
            if len(retained) >= self.max_retained_item_ids:
                break
        return tuple(unseen), tuple(retained)

    def _merge_awareness_thread(
        self,
        *,
        existing_threads: Sequence[SituationalAwarenessThread],
        subscription: WorldFeedSubscription,
        items: Sequence[WorldFeedItem],
        now: datetime,
    ) -> SituationalAwarenessThread:
        """Condense refreshed feed items into one slower-moving awareness thread."""

        existing = next(
            (
                thread
                for thread in existing_threads
                if thread.thread_id == self._awareness_thread_id(subscription)
            ),
            None,
        )
        recent_titles = tuple(
            dict.fromkeys(
                [
                    *(_clean_text(item.title) for item in items if _clean_text(item.title)),
                    *(existing.recent_titles if existing is not None else ()),
                ]
            )
        )[: self.max_recent_titles_per_thread]
        source_labels = tuple(
            dict.fromkeys(
                [
                    *(existing.source_labels if existing is not None else ()),
                    *(item.source or _host_label(subscription.feed_url) for item in items),
                ]
            )
        )
        supporting_item_ids = tuple(
            dict.fromkeys(
                [
                    *(_feed_item_id(item) for item in items),
                    *(existing.supporting_item_ids if existing is not None else ()),
                ]
            )
        )[: self.max_supporting_item_ids_per_thread]
        update_count = (existing.update_count if existing is not None else 0) + len(items)
        salience = _clamp(
            max(subscription.priority * 0.75, (existing.salience if existing is not None else 0.0) + 0.05),
            minimum=0.25,
            maximum=0.96,
        )
        return SituationalAwarenessThread(
            thread_id=self._awareness_thread_id(subscription),
            title=subscription.label,
            summary=self._awareness_summary(
                label=subscription.label,
                region=subscription.region,
                recent_titles=recent_titles,
                update_count=update_count,
            ),
            topic=", ".join(subscription.topics) if subscription.topics else subscription.label,
            region=subscription.region,
            scope=subscription.scope,
            salience=salience,
            update_count=max(1, update_count),
            recent_titles=recent_titles,
            source_labels=source_labels,
            supporting_item_ids=supporting_item_ids,
            updated_at=_isoformat(now),
            review_at=_isoformat(now + timedelta(days=self.awareness_review_days)),
        )

    def _upsert_awareness_thread(
        self,
        *,
        existing: Sequence[SituationalAwarenessThread],
        thread: SituationalAwarenessThread,
    ) -> tuple[SituationalAwarenessThread, ...]:
        """Insert or replace one awareness thread while keeping the set bounded."""

        merged: list[SituationalAwarenessThread] = []
        seen = False
        for current in existing:
            if current.thread_id == thread.thread_id:
                merged.append(thread)
                seen = True
                continue
            merged.append(current)
        if not seen:
            merged.append(thread)
        ranked = sorted(
            merged,
            key=lambda item: (item.salience, item.update_count, item.updated_at or ""),
            reverse=True,
        )
        return tuple(ranked[: self.max_awareness_threads])

    def _awareness_thread_id(self, subscription: WorldFeedSubscription) -> str:
        """Return one stable awareness-thread id for a subscription theme."""

        seed = "::".join(
            (
                _clean_text(subscription.scope).casefold() or "topic",
                _clean_text(subscription.region).casefold(),
                _clean_text(",".join(subscription.topics) if subscription.topics else subscription.label).casefold(),
            )
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"awareness:{digest}"

    def _awareness_summary(
        self,
        *,
        label: str,
        region: str | None,
        recent_titles: Sequence[str],
        update_count: int,
    ) -> str:
        """Render one calm summary for a condensed situational-awareness thread."""

        headline_part = ""
        if recent_titles:
            headline_part = f" Recent signals include: {'; '.join(recent_titles[:2])}."
        if region:
            return (
                f"Keep a calm watch on {label} around {region}; "
                f"{update_count} relevant RSS update(s) are being tracked."
                f"{headline_part}"
            ).strip()
        return (
            f"Keep a calm watch on {label}; "
            f"{update_count} relevant RSS update(s) are being tracked."
            f"{headline_part}"
        ).strip()

    def _build_world_signals(
        self,
        *,
        subscription: WorldFeedSubscription,
        items: Sequence[WorldFeedItem],
        now: datetime,
    ) -> tuple[WorldSignal, ...]:
        """Convert unseen feed items into fresh world signals."""

        fresh_until = _isoformat(now + timedelta(hours=self.default_freshness_hours))
        signals: list[WorldSignal] = []
        for item in items:
            summary = f"Relevant to {subscription.label}."
            if item.source and item.source != subscription.label:
                summary = f"Relevant to {subscription.label}; source {item.source}."
            signals.append(
                WorldSignal(
                    topic=item.title,
                    summary=summary,
                    region=subscription.region,
                    source=item.source or _host_label(subscription.feed_url),
                    salience=_clamp(subscription.priority * 0.9, minimum=0.25, maximum=0.95),
                    fresh_until=fresh_until,
                    evidence_count=1,
                    source_event_ids=tuple(
                        item_id
                        for item_id in (
                            subscription.subscription_id,
                            _feed_item_id(item),
                        )
                        if item_id
                    ),
                )
            )
        return tuple(signals)

    def _build_awareness_world_signal(
        self,
        *,
        thread: SituationalAwarenessThread,
        now: datetime,
    ) -> WorldSignal:
        """Convert one condensed awareness thread into prompt-facing world context."""

        return WorldSignal(
            topic=thread.title,
            summary=thread.summary,
            region=thread.region,
            source="situational_awareness",
            salience=_clamp(thread.salience, minimum=0.25, maximum=0.95),
            fresh_until=_isoformat(now + timedelta(days=max(2, self.awareness_review_days))),
            evidence_count=max(1, thread.update_count),
            source_event_ids=thread.supporting_item_ids[:6],
        )

    def _build_continuity_thread(
        self,
        *,
        thread: SituationalAwarenessThread,
        now: datetime,
    ) -> ContinuityThread:
        """Convert one awareness thread into calm continuity context."""

        return ContinuityThread(
            title=thread.title,
            summary=thread.summary,
            salience=_clamp(thread.salience, minimum=0.25, maximum=0.92),
            updated_at=thread.updated_at or _isoformat(now),
            expires_at=thread.review_at or _isoformat(now + timedelta(days=self.awareness_review_days)),
        )
