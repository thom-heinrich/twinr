"""Resolve calm HDMI news-ticker source feeds from Twinr world intelligence.

The HDMI ticker should not drift into a second independent news universe.
This module keeps the read-only source-resolution step separate from ticker
fetch/rotation so the bottom bar can reuse the same persisted RSS
subscriptions that Twinr already follows for its world-intelligence layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import math
import socket
from urllib.parse import urlsplit, urlunsplit

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

# CHANGELOG: 2026-03-28
# BUG-1: Fixed incorrect recency ordering caused by lexicographic sorting of ISO timestamps with different UTC offsets.
# BUG-2: Fixed repeated legacy re-migration when an authoritative snapshot existed but its items list was empty, which could resurrect removed feeds.
# BUG-3: Fixed resolver crashes on malformed snapshots and remote-state read/write failures by degrading gracefully and salvaging valid items.
# BUG-4: Fixed duplicate feed selection for semantically equivalent URLs (case/default-port/fragment variants).
# SEC-1: Reject non-http(s), credential-bearing, localhost/private/link-local/multicast/reserved feed URLs before returning or persisting them.
# IMP-1: Added canonical URL normalization and typed resolved-feed metadata via resolve_feed_entries().
# IMP-2: Added deterministic schema-aware normalization with in-process last-good caching for edge-resilient Raspberry Pi deployments.

_DEFAULT_MAX_FEED_URLS = 8
_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND = "agent_world_intelligence_subscriptions_v1"

_ALLOWED_FEED_URL_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORT_BY_SCHEME = {"http": 80, "https": 443}
_LOCAL_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost6",
        "localhost.localdomain",
        "localhost6.localdomain6",
    }
)
_LOCAL_HOST_SUFFIXES = (
    ".arpa",
    ".home",
    ".internal",
    ".lan",
    ".local",
    ".localdomain",
)
_MAX_FEED_URL_LENGTH = 2048
_MIN_RECENCY_EPOCH = float("-inf")
_TRUE_TEXT = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TEXT = frozenset({"0", "false", "no", "n", "off"})


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
    if not normalized:
        return "Feed"
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        parsed = None
    host = parsed.hostname if parsed is not None else ""
    if not host:
        if "://" in normalized:
            normalized = normalized.split("://", 1)[-1]
        host = normalized.split("/", 1)[0]
    candidate = host.strip().casefold().rstrip(".")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    try:
        candidate = _canonicalize_hostname(candidate)
    except ValueError:
        return candidate or "Feed"
    ip_literal = _parse_ip_literal(candidate)
    if ip_literal is not None:
        candidate = str(ip_literal)
    return candidate or "Feed"


def _normalize_float(value: object | None, *, default: float) -> float:
    """Normalize one optional numeric value into a finite float."""

    try:
        candidate = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _normalize_bool(value: object | None, *, default: bool) -> bool:
    """Normalize one optional truthy-ish value into a bool."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return bool(value)
    text = _clean_text(value).casefold()
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    return default


def _normalize_int(value: object | None, *, default: int, minimum: int) -> int:
    """Normalize one optional integer-ish value into a bounded int."""

    try:
        normalized = int(value) if value is not None else default
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _parse_iso_datetime(value: object | None) -> datetime | None:
    """Best-effort parse of one snapshot timestamp into an aware UTC datetime."""

    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_epoch(*values: object | None) -> float:
    """Return one comparable UTC timestamp epoch from the first parseable value."""

    for value in values:
        parsed = _parse_iso_datetime(value)
        if parsed is not None:
            return parsed.timestamp()
    return _MIN_RECENCY_EPOCH


def _first_timestamp_text(*values: object | None) -> str:
    """Return the first valid ISO timestamp normalized to UTC text."""

    for value in values:
        parsed = _parse_iso_datetime(value)
        if parsed is not None:
            return parsed.isoformat()
    return ""


def _canonicalize_hostname(hostname: str) -> str:
    """Return one lowercase ASCII hostname suitable for comparisons."""

    normalized = _clean_text(hostname).casefold().rstrip(".")
    if not normalized:
        raise ValueError("feed URL hostname is required.")
    if "%" in normalized:
        raise ValueError("feed URL hostname must not contain a zone identifier.")
    try:
        return normalized.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("feed URL hostname is not valid IDNA.") from exc


def _parse_ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse strict or legacy textual IP literal host forms."""

    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    if ":" in hostname:
        return None
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _is_disallowed_feed_host(hostname: str) -> bool:
    """Return whether one host is unsafe for outbound feed fetching."""

    canonical = _canonicalize_hostname(hostname)
    if canonical in _LOCAL_HOSTNAMES:
        return True
    if canonical.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    ip_literal = _parse_ip_literal(canonical)
    if ip_literal is not None:
        return not ip_literal.is_global
    if "." not in canonical:
        return True
    return False


def _normalize_feed_url(
    feed_url: object | None,
    *,
    allow_private_hosts: bool = False,
) -> str:
    """Normalize and validate one feed URL for safe public fetching."""

    raw = _clean_text(feed_url)
    if not raw:
        return ""
    if len(raw) > _MAX_FEED_URL_LENGTH:
        raise ValueError("feed URL is unreasonably long.")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise ValueError("feed URL is malformed.") from exc

    scheme = parsed.scheme.casefold()
    if scheme not in _ALLOWED_FEED_URL_SCHEMES:
        # BREAKING: only public HTTP(S) feeds are accepted now; file/gopher/data/etc.
        # are rejected because later fetch stages would make them practically SSRF-prone.
        raise ValueError("feed URL scheme must be http or https.")

    if parsed.username or parsed.password:
        # BREAKING: inline credentials are rejected to avoid secret persistence/leakage
        # and to keep ticker sources limited to public, reproducible feed URLs.
        raise ValueError("feed URL must not embed credentials.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("feed URL hostname is required.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("feed URL port is invalid.") from exc

    canonical_host = _canonicalize_hostname(hostname)
    ip_literal = _parse_ip_literal(canonical_host)
    if ip_literal is not None:
        canonical_host = str(ip_literal)
    if not allow_private_hosts and _is_disallowed_feed_host(canonical_host):
        # BREAKING: localhost/private/link-local/single-label/local-suffix hosts are
        # rejected by default so the resolver cannot hand internal-network targets
        # to the fetcher unless a caller explicitly opts in.
        raise ValueError("feed URL host is not a safe public host.")

    host_for_netloc = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    default_port = _DEFAULT_PORT_BY_SCHEME[scheme]
    netloc = host_for_netloc if port in (None, default_port) else f"{host_for_netloc}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


@dataclass(frozen=True, slots=True)
class ResolvedWorldIntelligenceFeed:
    """One normalized feed source exposed by the resolver."""

    subscription_id: str
    feed_url: str
    label: str
    priority: float
    recency_epoch: float
    refreshed_at: str


@dataclass(slots=True)
class DisplayWorldIntelligenceTickerFeedResolver:
    """Resolve active feed URLs from the persisted world-intelligence pool."""

    config: TwinrConfig
    remote_state: LongTermRemoteStateStore | None = None
    max_feed_urls: int = _DEFAULT_MAX_FEED_URLS
    subscriptions_snapshot_kind: str = _WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND
    allow_private_hosts: bool = False
    _last_good_entries: tuple[ResolvedWorldIntelligenceFeed, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

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

    def resolve_feed_entries(self) -> tuple[ResolvedWorldIntelligenceFeed, ...]:
        """Return bounded normalized feed metadata Twinr already follows."""

        remote_state = self._get_remote_state()
        if remote_state is None or not getattr(remote_state, "enabled", False):
            return self._last_good_entries

        try:
            payload = remote_state.load_snapshot(
                snapshot_kind=self.subscriptions_snapshot_kind,
            )
        except Exception:
            return self._last_good_entries

        if payload is None:
            migrated = self._maybe_migrate_legacy_feed_urls(remote_state=remote_state)
            if migrated:
                self._last_good_entries = migrated
                return migrated
            self._last_good_entries = ()
            return ()

        resolved, authoritative = self._resolve_entries_from_payload(payload)
        if authoritative:
            self._last_good_entries = resolved
            return resolved

        return self._last_good_entries

    def resolve_feed_urls(self) -> tuple[str, ...]:
        """Return the bounded active feed URLs Twinr already follows."""

        return tuple(entry.feed_url for entry in self.resolve_feed_entries())

    def _get_remote_state(self) -> LongTermRemoteStateStore | None:
        """Return one reusable remote-state handle if available."""

        if self.remote_state is not None:
            return self.remote_state
        try:
            self.remote_state = LongTermRemoteStateStore.from_config(self.config)
        except Exception:
            self.remote_state = None
        return self.remote_state

    def _resolve_entries_from_payload(
        self,
        payload: object,
    ) -> tuple[tuple[ResolvedWorldIntelligenceFeed, ...], bool]:
        """Normalize one snapshot payload into resolved feed entries."""

        if not isinstance(payload, Mapping):
            return (), False

        raw_items = payload.get("items")
        if raw_items is None:
            return (), False
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            return (), False

        active: list[ResolvedWorldIntelligenceFeed] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            normalized = self._normalize_subscription(raw_item)
            if normalized is None:
                continue
            active.append(normalized)

        limit = _normalize_int(self.max_feed_urls, default=_DEFAULT_MAX_FEED_URLS, minimum=1)
        ordered: list[ResolvedWorldIntelligenceFeed] = []
        seen_urls: set[str] = set()
        for subscription in sorted(
            active,
            key=lambda item: (-item.priority, -item.recency_epoch, item.feed_url),
        ):
            if subscription.feed_url in seen_urls:
                continue
            seen_urls.add(subscription.feed_url)
            ordered.append(subscription)
            if len(ordered) >= limit:
                break
        return tuple(ordered), True

    def _normalize_subscription(
        self,
        subscription: Mapping[str, object],
    ) -> ResolvedWorldIntelligenceFeed | None:
        """Best-effort normalize one subscription mapping."""

        if not _normalize_bool(subscription.get("active", True), default=True):
            return None

        try:
            feed_url = _normalize_feed_url(
                subscription.get("feed_url"),
                allow_private_hosts=self.allow_private_hosts,
            )
        except ValueError:
            return None
        if not feed_url:
            return None

        refreshed_at = _first_timestamp_text(
            subscription.get("last_refreshed_at"),
            subscription.get("updated_at"),
            subscription.get("created_at"),
        )
        recency_epoch = _timestamp_epoch(
            subscription.get("last_refreshed_at"),
            subscription.get("updated_at"),
            subscription.get("created_at"),
        )
        label = _clean_text(subscription.get("label")) or _host_label(feed_url)
        subscription_id = _clean_text(subscription.get("subscription_id")) or _subscription_id(feed_url)
        priority = _normalize_float(subscription.get("priority"), default=0.0)
        return ResolvedWorldIntelligenceFeed(
            subscription_id=subscription_id,
            feed_url=feed_url,
            label=label,
            priority=priority,
            recency_epoch=recency_epoch,
            refreshed_at=refreshed_at,
        )

    def _normalized_legacy_feed_urls(self) -> tuple[str, ...]:
        """Return validated legacy ticker URLs in canonical form."""

        legacy_urls: list[str] = []
        seen: set[str] = set()
        for raw_url in getattr(self.config, "display_news_ticker_legacy_feed_urls", ()):
            try:
                feed_url = _normalize_feed_url(
                    raw_url,
                    allow_private_hosts=self.allow_private_hosts,
                )
            except ValueError:
                continue
            if not feed_url or feed_url in seen:
                continue
            seen.add(feed_url)
            legacy_urls.append(feed_url)

        limit = _normalize_int(self.max_feed_urls, default=_DEFAULT_MAX_FEED_URLS, minimum=1)
        return tuple(legacy_urls[:limit])

    def _maybe_migrate_legacy_feed_urls(
        self,
        *,
        remote_state: LongTermRemoteStateStore,
    ) -> tuple[ResolvedWorldIntelligenceFeed, ...]:
        """Import one legacy static ticker feed list into world intelligence once.

        Older Pi environments may still carry `TWINR_DISPLAY_NEWS_TICKER_FEED_URLS`.
        The new ticker source model should not keep reading that static list
        forever, but a one-way migration prevents existing devices from losing
        the bottom ticker the moment the shared source pool becomes
        authoritative.
        """

        legacy_urls = self._normalized_legacy_feed_urls()
        if not legacy_urls:
            return ()
        now_iso = _utcnow_iso()
        payload = {
            "schema_version": 2,
            "migrated_at": now_iso,
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
                    "last_refreshed_at": now_iso,
                    "last_item_ids": [],
                }
                for feed_url in legacy_urls
            ],
        }
        try:
            remote_state.save_snapshot(
                snapshot_kind=self.subscriptions_snapshot_kind,
                payload=payload,
            )
        except Exception:
            pass
        return tuple(
            ResolvedWorldIntelligenceFeed(
                subscription_id=_subscription_id(feed_url),
                feed_url=feed_url,
                label=_host_label(feed_url),
                priority=0.6,
                recency_epoch=_timestamp_epoch(now_iso),
                refreshed_at=now_iso,
            )
            for feed_url in legacy_urls
        )