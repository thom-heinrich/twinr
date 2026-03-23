"""Resolve calm HDMI news-ticker source feeds from Twinr world intelligence.

The HDMI ticker should not drift into a second independent news universe.
This module keeps the read-only source-resolution step separate from ticker
fetch/rotation so the bottom bar can reuse the same persisted RSS
subscriptions that Twinr already follows for its world-intelligence layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_DEFAULT_MAX_FEED_URLS = 8
_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND = "agent_world_intelligence_subscriptions_v1"


def _clean_text(value: object | None) -> str:
    """Normalize one string-ish value into a trimmed single line."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _utcnow_iso() -> str:
    """Return the current UTC wall clock as ISO-8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _subscription_id(feed_url: str) -> str:
    """Return a stable compatibility subscription id for one feed URL."""

    digest = hashlib.sha1(feed_url.encode("utf-8")).hexdigest()[:12]
    return f"feed:{digest}"


def _host_label(feed_url: str) -> str:
    """Return one short feed label from a URL host."""

    normalized = _clean_text(feed_url)
    if "://" in normalized:
        normalized = normalized.split("://", 1)[-1]
    host = normalized.split("/", 1)[0].strip().casefold()
    if host.startswith("www."):
        host = host[4:]
    return host or "Feed"


def _normalize_float(value: object | None, *, default: float) -> float:
    """Normalize one optional numeric value into a bounded float."""

    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _mapping_items(value: object | None) -> tuple[Mapping[str, object], ...]:
    """Normalize one raw payload field into mapping items only."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("world-intelligence subscription snapshot items must be a sequence.")
    items: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("world-intelligence subscription snapshot items must be mappings.")
        items.append(item)
    return tuple(items)


def _subscription_sort_key(subscription: Mapping[str, object]) -> tuple[float, str, str]:
    """Rank subscriptions for ticker coverage without topic-specific rules.

    The bottom line should prefer feeds that Twinr currently considers more
    important while staying generic: higher priority feeds win, then more
    recently refreshed or updated subscriptions, then a stable URL tie-breaker.
    """

    recency_hint = _clean_text(
        subscription.get("last_refreshed_at")
        or subscription.get("updated_at")
        or subscription.get("created_at")
    )
    return (
        _normalize_float(subscription.get("priority"), default=0.0),
        recency_hint,
        _clean_text(subscription.get("feed_url")),
    )


@dataclass(slots=True)
class DisplayWorldIntelligenceTickerFeedResolver:
    """Resolve active feed URLs from the persisted world-intelligence pool."""

    config: TwinrConfig
    remote_state: LongTermRemoteStateStore | None = None
    max_feed_urls: int = _DEFAULT_MAX_FEED_URLS
    subscriptions_snapshot_kind: str = _WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> "DisplayWorldIntelligenceTickerFeedResolver":
        """Build the resolver from Twinr configuration."""

        return cls(
            config=config,
            remote_state=remote_state,
            max_feed_urls=_DEFAULT_MAX_FEED_URLS,
        )

    def resolve_feed_urls(self) -> tuple[str, ...]:
        """Return the bounded active feed URLs Twinr already follows."""

        remote_state = self.remote_state or LongTermRemoteStateStore.from_config(self.config)
        if not getattr(remote_state, "enabled", False):
            return ()
        payload = remote_state.load_snapshot(
            snapshot_kind=self.subscriptions_snapshot_kind,
        )
        if payload is None:
            migrated = self._maybe_migrate_legacy_feed_urls(remote_state=remote_state)
            if migrated:
                return migrated
            return ()
        if not isinstance(payload, Mapping):
            raise ValueError("world-intelligence subscription snapshot must decode to a mapping payload.")
        subscriptions = _mapping_items(payload.get("items"))
        if not subscriptions:
            migrated = self._maybe_migrate_legacy_feed_urls(remote_state=remote_state)
            if migrated:
                return migrated
        active = sorted(
            (
                subscription
                for subscription in subscriptions
                if bool(subscription.get("active", True)) and _clean_text(subscription.get("feed_url"))
            ),
            key=_subscription_sort_key,
            reverse=True,
        )
        ordered: list[str] = []
        seen: set[str] = set()
        for subscription in active:
            feed_url = _clean_text(subscription.get("feed_url"))
            if not feed_url or feed_url in seen:
                continue
            seen.add(feed_url)
            ordered.append(feed_url)
            if len(ordered) >= max(1, int(self.max_feed_urls)):
                break
        return tuple(ordered)

    def _maybe_migrate_legacy_feed_urls(
        self,
        *,
        remote_state: LongTermRemoteStateStore,
    ) -> tuple[str, ...]:
        """Import one legacy static ticker feed list into world intelligence once.

        Older Pi environments may still carry `TWINR_DISPLAY_NEWS_TICKER_FEED_URLS`.
        The new ticker source model should not keep reading that static list
        forever, but a one-way migration prevents existing devices from losing
        the bottom ticker the moment the shared source pool becomes
        authoritative.
        """

        legacy_urls = tuple(
            dict.fromkeys(
                _clean_text(url)
                for url in self.config.display_news_ticker_legacy_feed_urls
                if _clean_text(url)
            )
        )
        if not legacy_urls:
            return ()
        now_iso = _utcnow_iso()
        remote_state.save_snapshot(
            snapshot_kind=self.subscriptions_snapshot_kind,
            payload={
                "schema_version": 1,
                "items": [
                    {
                        "subscription_id": _subscription_id(feed_url),
                        "label": _host_label(feed_url),
                        "feed_url": feed_url,
                        "scope": "topic",
                        "topics": [],
                        "priority": 0.6,
                        "base_priority": 0.6,
                        "active": True,
                        "refresh_interval_hours": 72,
                        "base_refresh_interval_hours": 72,
                        "created_by": "legacy_display_news_ticker",
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "last_item_ids": [],
                    }
                    for feed_url in legacy_urls
                ],
            },
        )
        return legacy_urls[: max(1, int(self.max_feed_urls))]
