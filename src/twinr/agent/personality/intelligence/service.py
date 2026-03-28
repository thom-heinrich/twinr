# CHANGELOG: 2026-03-27
# BUG-1: Fixed false JSON Feed support; discovery now accepts JSON Feed 1.1 advertised as application/feed+json or application/json and refresh parses JSON Feed items correctly.
# BUG-2: Fixed interest-merge state loss; co_attention_score is now preserved during merges and decayed state is persisted even when refresh skips because nothing is due.
# BUG-3: Fixed feed lifecycle bugs; permanent redirects update stored feed URLs and 410 Gone deactivates dead subscriptions instead of polling them forever.
# SEC-1: Blocked SSRF/file-scheme/local-network fetches by default for discovery and refresh, with per-hop redirect revalidation to protect Raspberry Pi home-network deployments.
# SEC-2: Replaced unsafe XML sniff parsing with defusedxml and bounded streamed HTTP reads to reduce XML-bomb and oversized-response risk from untrusted feeds.
# IMP-1: Upgraded transport to pooled HTTPX with strict timeouts, resource limits, trust_env=False, HTTP/2 support, and in-process conditional GET caching.
# IMP-2: Added bounded parallel refresh, stable item IDs from feed-native ids/guid, title/source sanitization, and URL canonicalization to cut duplicate signals and latency.

"""Manage RSS-backed place/world intelligence for the personality system.

This service owns three bounded responsibilities:

- persist and mutate the curated subscription set that Twinr uses for calm
  place/world awareness
- discover RSS/Atom/JSON feeds from explicit source pages returned by the
  web-search backend when the installer or a recalibration pass asks for new
  sources
- refresh due feeds and convert fresh items into world signals plus continuity
  threads that the background personality loop can commit

The service does not rewrite core character. It only emits contextual world and
continuity updates that higher layers may merge through existing policy gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import hashlib
import ipaddress
import json
import socket
import threading
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

# BREAKING: This upgraded implementation depends on maintained parsing/network
# libraries that are lightweight enough for Raspberry Pi 4 deployments.
try:
    import feedparser
except ImportError as exc:  # pragma: no cover - dependency contract
    raise RuntimeError(
        "WorldIntelligenceService requires feedparser>=6.0.12. "
        "Install with: pip install feedparser>=6.0.12"
    ) from exc

try:
    import httpx
except ImportError as exc:  # pragma: no cover - dependency contract
    raise RuntimeError(
        "WorldIntelligenceService requires httpx>=0.28. "
        "Install with: pip install httpx>=0.28"
    ) from exc

try:
    from defusedxml import ElementTree as SafeET
except ImportError as exc:  # pragma: no cover - dependency contract
    raise RuntimeError(
        "WorldIntelligenceService requires defusedxml>=0.7.1. "
        "Install with: pip install defusedxml>=0.7.1"
    ) from exc

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
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


_DEFAULT_USER_AGENT = "TwinrWorldIntelligence/2026.03"
_SUPPORTED_FEED_MIME_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/feed+json",
        "application/json",
        "application/xml",
        "text/xml",
    }
)
_SUPPORTED_DISCOVERY_LINK_MIME_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/feed+json",
        "application/json",
        "application/xml",
        "text/xml",
    }
)
_MAX_DISCOVERY_BYTES = 512 * 1024
_MAX_FEED_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 5
_DEFAULT_CONNECT_TIMEOUT_S = 2.0
_DEFAULT_POOL_TIMEOUT_S = 2.0
_DEFAULT_MAX_PARALLEL_REFRESHES = 4
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 8
_DEFAULT_KEEPALIVE_EXPIRY_S = 30.0
_PRIVATE_HOST_LABELS = frozenset({"localhost", "localhost.localdomain"})


def _http2_enabled(requested: bool) -> bool:
    """Enable HTTP/2 only when the optional h2 dependency is available."""

    if not requested:
        return False
    try:
        import h2  # noqa: F401  # pylint: disable=import-error
    except Exception as exc:
        raise RuntimeError(
            "HTTP/2 support requires the optional h2 package "
            "(for example `pip install httpx[http2]` or `pip install h2<5,>=3`)."
        ) from exc
    return True


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
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_http_datetime(value: str | None) -> datetime | None:
    """Parse one RFC-2822/RFC-7231 HTTP datetime value."""

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_http_datetime(value: datetime | None) -> str | None:
    """Format one datetime for HTTP conditional headers."""

    if value is None:
        return None
    return format_datetime(value.astimezone(timezone.utc), usegmt=True)


def _clean_text(value: object | None) -> str:
    """Normalize one free-form string into a trimmed single line."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _truncate_text(value: object | None, *, maximum: int) -> str:
    """Return one bounded single-line text value."""

    cleaned = _clean_text(value)
    if len(cleaned) <= maximum:
        return cleaned
    if maximum <= 1:
        return cleaned[:maximum]
    return cleaned[: maximum - 1].rstrip() + "…"


class _PlainTextExtractor(HTMLParser):
    """Extract readable text from small HTML fragments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return _clean_text(" ".join(self.parts))


def _to_plain_text(value: object | None, *, maximum: int = 280) -> str:
    """Convert one short HTML-ish fragment into bounded plain text."""

    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    if "<" in cleaned or "&" in cleaned:
        parser = _PlainTextExtractor()
        try:
            parser.feed(cleaned)
            parser.close()
            extracted = parser.text()
            if extracted:
                cleaned = extracted
            else:
                cleaned = unescape(cleaned)
        except Exception:
            cleaned = unescape(cleaned)
    return _truncate_text(cleaned, maximum=maximum)


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
    return _truncate_text(host or "Feed", maximum=80)


def _canonicalize_http_url(url: str) -> str:
    """Normalize one HTTP(S) URL into a stable canonical text form."""

    cleaned = _clean_text(url)
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("world_intelligence_unsupported_url_scheme")
    if parsed.username or parsed.password:
        raise ValueError("world_intelligence_url_userinfo_not_allowed")
    hostname = (parsed.hostname or "").strip().casefold()
    if not hostname:
        raise ValueError("world_intelligence_url_missing_host")
    port = parsed.port
    host_port = hostname
    if ":" in hostname and not hostname.startswith("["):
        host_port = f"[{hostname}]"
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host_port = f"{host_port}:{port}"
    path = parsed.path or "/"
    return urlunparse((scheme, host_port, path, "", parsed.query, ""))


def _normalize_candidate_url(url: object | None) -> str:
    """Normalize one externally supplied URL candidate or raise."""

    return _canonicalize_http_url(_clean_text(url))


def _subscription_id(feed_url: str) -> str:
    """Return a stable subscription id for one feed URL."""

    digest = hashlib.sha256(feed_url.encode("utf-8")).hexdigest()[:12]
    return f"feed:{digest}"


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp one numeric value into an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _signal_ongoing_interest_score(signal: WorldInterestSignal) -> float:
    """Return one normalized ongoing-interest score for ranking math."""

    return 0.0 if signal.ongoing_interest_score is None else signal.ongoing_interest_score


def _signal_co_attention_score(signal: WorldInterestSignal) -> float:
    """Return one normalized co-attention score for ranking math."""

    return 0.0 if signal.co_attention_score is None else signal.co_attention_score


def _subscription_base_priority(subscription: WorldFeedSubscription) -> float:
    """Return the baseline priority Twinr should relax back toward."""

    return subscription.priority if subscription.base_priority is None else subscription.base_priority


def _subscription_base_refresh_interval_hours(subscription: WorldFeedSubscription) -> int:
    """Return the baseline refresh cadence Twinr should relax back toward."""

    if subscription.base_refresh_interval_hours is None:
        return subscription.refresh_interval_hours
    return subscription.base_refresh_interval_hours


def _media_type(content_type: str | None) -> str:
    """Return the lower-cased media type without parameters."""

    raw = _clean_text(content_type)
    if not raw:
        return "application/octet-stream"
    return raw.split(";", 1)[0].strip().casefold() or "application/octet-stream"


def _extract_charset(content_type: str | None) -> str | None:
    """Extract one charset parameter from a content-type header."""

    raw = _clean_text(content_type)
    if not raw:
        return None
    for part in raw.split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().casefold() == "charset":
            charset = value.strip().strip('"').strip("'")
            return charset or None
    return None


def _decode_text(payload: bytes, content_type: str | None) -> str:
    """Decode one HTTP payload into text with a conservative fallback."""

    charset = _extract_charset(content_type) or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _is_json_feed_document(text: str) -> bool:
    """Return whether one text document looks like JSON Feed."""

    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(document, dict):
        return False
    version = _clean_text(document.get("version"))
    return version.startswith("https://jsonfeed.org/version/")


def _looks_like_feed_document(*, text: str, content_type: str) -> bool:
    """Return whether one fetched document is already a feed."""

    media_type = _media_type(content_type)
    if media_type in {"application/feed+json", "application/json"}:
        return _is_json_feed_document(text)
    if media_type in _SUPPORTED_FEED_MIME_TYPES:
        if _is_json_feed_document(text):
            return True
    stripped = text.lstrip()
    if not stripped.startswith("<"):
        return False
    try:
        root = SafeET.fromstring(text)
    except Exception:
        return False
    local_name = str(root.tag).rsplit("}", 1)[-1].casefold()
    return local_name in {"rss", "feed", "rdf"}


def _struct_time_to_datetime(value: Any) -> datetime | None:
    """Convert one feedparser date tuple into UTC datetime."""

    if not value:
        return None
    try:
        return datetime(
            int(value.tm_year),
            int(value.tm_mon),
            int(value.tm_mday),
            int(value.tm_hour),
            int(value.tm_min),
            int(value.tm_sec),
            tzinfo=timezone.utc,
        )
    except Exception:
        return None


def _parse_json_datetime(value: object | None) -> datetime | None:
    """Parse one JSON Feed datetime string into UTC."""

    if value is None:
        return None
    return _parse_iso(_clean_text(value))


def _candidate_entry_link(entry: Any) -> str:
    """Return one best-effort canonical entry URL."""

    direct_link = _clean_text(getattr(entry, "get", lambda _key, _default=None: None)("link", None))
    if direct_link:
        try:
            return _canonicalize_http_url(direct_link)
        except ValueError:
            return direct_link
    links = getattr(entry, "get", lambda _key, _default=None: None)("links", None) or ()
    for link in links:
        href = _clean_text(getattr(link, "get", lambda _key, _default=None: None)("href", None))
        if not href:
            continue
        try:
            return _canonicalize_http_url(href)
        except ValueError:
            return href
    return ""


def _safe_item_fingerprint(
    *,
    feed_url: str,
    native_id: str | None,
    link: str | None,
    published_at: str | None,
    title: str | None,
) -> str:
    """Return one stable item identifier."""

    item_id = _clean_text(native_id)
    if item_id:
        return item_id
    normalized_link = _clean_text(link)
    if normalized_link:
        return normalized_link
    published = _clean_text(published_at)
    headline = _clean_text(title)
    if published:
        return f"{feed_url}::{published}::{headline}"
    return f"{feed_url}::{headline}"


def _ip_is_private_or_special(address: Any) -> bool:
    """Return whether one IP should be blocked for outbound fetches."""

    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _resolve_host_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve one host into a bounded tuple of IP addresses."""

    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = []
    for _family, _socktype, _proto, _canonname, sockaddr in infos:
        ip_text = _clean_text(sockaddr[0])
        if ip_text and ip_text not in addresses:
            addresses.append(ip_text)
    return tuple(addresses)


@dataclass(frozen=True, slots=True)
class _FetchedDocument:
    """Represent one fetched HTML or feed document."""

    url: str
    text: str
    content_type: str
    status_code: int


@dataclass(frozen=True, slots=True)
class _FetchedPayload:
    """Represent one fetched binary payload plus response metadata."""

    url: str
    payload: bytes
    content_type: str
    status_code: int
    headers: Mapping[str, str]
    not_modified: bool = False
    gone: bool = False
    permanently_moved_to: str | None = None


@dataclass(frozen=True, slots=True)
class _ParsedFeedItem:
    """Represent one parsed feed item plus the stable id used for dedupe."""

    stable_id: str
    item: WorldFeedItem
    published_dt: datetime | None


@dataclass(frozen=True, slots=True)
class _FeedReadResult:
    """Represent one bounded feed refresh attempt."""

    items: tuple[_ParsedFeedItem, ...]
    final_feed_url: str
    status_code: int
    not_modified: bool = False
    gone: bool = False
    permanently_moved_to: str | None = None


@dataclass(frozen=True, slots=True)
class _ConditionalRequestState:
    """Persist in-process conditional request headers for one feed URL."""

    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True, slots=True)
class _FetchedSubscriptionResult:
    """Represent one due-subscription fetch result before thread merging."""

    subscription: WorldFeedSubscription
    unseen_items: tuple[_ParsedFeedItem, ...]
    refreshed: bool
    had_new_items: bool
    error: str | None = None


class _FeedAutodiscoveryParser(HTMLParser):
    """Extract RSS/Atom/JSON Feed alternate links from one HTML document."""

    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.discovered: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one alternate feed link when the tag describes one."""

        normalized_tag = str(tag).strip().casefold()
        normalized = {str(key).strip().casefold(): (value or "") for key, value in attrs if key}
        href = normalized.get("href", "").strip()
        if not href:
            return
        rel_tokens = {
            token.strip().casefold()
            for token in normalized.get("rel", "").split()
            if token.strip()
        }
        mime_type = normalized.get("type", "").strip().casefold()
        if normalized_tag == "link":
            if not rel_tokens.intersection({"alternate", "feed"}):
                return
            if mime_type and mime_type not in _SUPPORTED_DISCOVERY_LINK_MIME_TYPES:
                return
        elif normalized_tag == "a":
            if mime_type and mime_type not in _SUPPORTED_DISCOVERY_LINK_MIME_TYPES:
                return
            lowered_href = href.casefold()
            if not (
                rel_tokens.intersection({"alternate", "feed"})
                or lowered_href.endswith(
                    (
                        ".rss",
                        ".atom",
                        ".xml",
                        ".rdf",
                        "/feed",
                        "/feed/",
                        "/rss",
                        "/rss/",
                        "/atom",
                        "/atom/",
                        "/index.xml",
                        "/feed.xml",
                        "/rss.xml",
                        "/atom.xml",
                        "/feed.json",
                        "/index.json",
                    )
                )
            ):
                return
        else:
            return
        resolved = urljoin(self.base_url, href)
        title = _to_plain_text(normalized.get("title"), maximum=120) or None
        self.discovered.append((resolved, title))


class WorldIntelligenceService:
    """Persist feed subscriptions and convert refreshes into personality context."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        store: WorldIntelligenceStore | None = None,
        now_provider: Callable[[], datetime] = _utcnow,
        page_loader: Callable[[str], Any] | None = None,
        feed_reader: Callable[..., Any] | None = None,
        default_refresh_interval_hours: int = 72,
        default_freshness_hours: int = 96,
        page_timeout_s: float = 5.0,
        feed_timeout_s: float = 4.0,
        max_items_per_refresh: int = 4,
        max_signals_per_subscription: int = 2,
        max_retained_item_ids: int = 24,
        max_interest_signals: int = 48,
        max_awareness_threads: int = 24,
        max_recent_titles_per_thread: int = 4,
        max_supporting_item_ids_per_thread: int = 24,
        awareness_review_days: int = 14,
        interest_decay_grace_days: int = 7,
        interest_decay_step_days: int = 7,
        interest_decay_engagement_step: float = 0.08,
        interest_decay_salience_step: float = 0.05,
        interest_retention_days: int = 84,
        max_parallel_refreshes: int = _DEFAULT_MAX_PARALLEL_REFRESHES,
        allow_private_network_hosts: bool = False,
        max_document_bytes: int = _MAX_DISCOVERY_BYTES,
        max_feed_bytes: int = _MAX_FEED_BYTES,
        max_redirects: int = _MAX_REDIRECTS,
        # HTTPX documents HTTP/2 as opt-in and notes HTTP/1.1 is the more
        # robust default transport. Twinr prefers stable defaults and only opts
        # in when the caller requests HTTP/2 explicitly.
        http2: bool = False,
        http_connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        http_pool_timeout_s: float = _DEFAULT_POOL_TIMEOUT_S,
    ) -> None:
        """Store the bounded collaborators used by the intelligence loop."""

        self.config = config
        self.remote_state = remote_state
        self.store = store or RemoteStateWorldIntelligenceStore()
        self.now_provider = now_provider

        self.default_refresh_interval_hours = max(24, int(default_refresh_interval_hours))
        self.default_freshness_hours = max(24, int(default_freshness_hours))
        self.page_timeout_s = max(0.5, float(page_timeout_s))
        self.feed_timeout_s = max(0.5, float(feed_timeout_s))
        self.max_items_per_refresh = max(1, int(max_items_per_refresh))
        self.max_signals_per_subscription = max(1, int(max_signals_per_subscription))
        self.max_retained_item_ids = max(4, int(max_retained_item_ids))
        self.max_interest_signals = max(8, int(max_interest_signals))
        self.max_awareness_threads = max(4, int(max_awareness_threads))
        self.max_recent_titles_per_thread = max(1, int(max_recent_titles_per_thread))
        self.max_supporting_item_ids_per_thread = max(8, int(max_supporting_item_ids_per_thread))
        self.awareness_review_days = max(3, int(awareness_review_days))
        self.interest_decay_grace_days = max(1, int(interest_decay_grace_days))
        self.interest_decay_step_days = max(1, int(interest_decay_step_days))
        self.interest_decay_engagement_step = _clamp(
            float(interest_decay_engagement_step),
            minimum=0.01,
            maximum=0.25,
        )
        self.interest_decay_salience_step = _clamp(
            float(interest_decay_salience_step),
            minimum=0.01,
            maximum=0.2,
        )
        self.interest_retention_days = max(self.interest_decay_grace_days, int(interest_retention_days))

        self.max_parallel_refreshes = max(1, int(max_parallel_refreshes))
        # BREAKING: outbound discovery/refresh now blocks private-network targets by default.
        # Set allow_private_network_hosts=True only for intentionally trusted local feeds.
        self.allow_private_network_hosts = bool(allow_private_network_hosts)
        self.max_document_bytes = max(64 * 1024, int(max_document_bytes))
        self.max_feed_bytes = max(128 * 1024, int(max_feed_bytes))
        self.max_redirects = max(0, int(max_redirects))

        max_connections = max(4, self.max_parallel_refreshes * 2)
        keepalive_connections = max(2, min(max_connections, _DEFAULT_MAX_KEEPALIVE_CONNECTIONS))
        self._http_client = httpx.Client(
            http2=_http2_enabled(bool(http2)),
            follow_redirects=False,
            verify=True,
            trust_env=False,
            timeout=httpx.Timeout(
                timeout=max(self.page_timeout_s, self.feed_timeout_s),
                connect=max(0.25, float(http_connect_timeout_s)),
                pool=max(0.25, float(http_pool_timeout_s)),
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=keepalive_connections,
                keepalive_expiry=_DEFAULT_KEEPALIVE_EXPIRY_S,
            ),
            headers={"User-Agent": _DEFAULT_USER_AGENT},
        )
        self._conditional_headers_by_feed_url: dict[str, _ConditionalRequestState] = {}
        self._conditional_headers_lock = threading.Lock()

        self.page_loader = page_loader or self._default_page_loader
        self.feed_reader = feed_reader or self._default_feed_reader

    def close(self) -> None:
        """Close the pooled HTTP client."""

        self._http_client.close()

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
        state = replace(
            state,
            interest_signals=self._decay_interest_signals(
                signals=state.interest_signals,
                now=now,
            ),
        )
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
        elif request.action in {"list", "refresh_now"}:
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
            refresh_result = self.maybe_refresh(force=True, search_backend=search_backend)
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
        now = self.now_provider()
        merged = self._merge_interest_signals(
            existing=self._decay_interest_signals(
                signals=state.interest_signals,
                now=now,
            ),
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
        allow_recalibration: bool = True,
    ) -> WorldIntelligenceRefreshResult:
        """Recalibrate if allowed and due, then refresh due feed subscriptions."""

        subscriptions = list(
            self.store.load_subscriptions(config=self.config, remote_state=self.remote_state)
        )
        state = self.store.load_state(config=self.config, remote_state=self.remote_state)
        now = self.now_provider()
        now_iso = _isoformat(now)
        state = replace(
            state,
            interest_signals=self._decay_interest_signals(
                signals=state.interest_signals,
                now=now,
            ),
        )
        if allow_recalibration:
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
            return WorldIntelligenceRefreshResult(
                status="skipped",
                refreshed=False,
                subscriptions=tuple(subscriptions),
                awareness_threads=tuple(state.awareness_threads),
                checked_at=now_iso,
            )

        fetched_results = self._fetch_due_subscriptions(
            due_subscriptions=due_subscriptions,
            now_iso=now_iso,
        )
        fetched_by_id = {result.subscription.subscription_id: result for result in fetched_results}

        refreshed_ids: list[str] = []
        errors: list[str] = []
        world_signals: list[WorldSignal] = []
        continuity_threads: list[ContinuityThread] = []
        updated_awareness_threads = tuple(state.awareness_threads)
        updated_subscriptions: list[WorldFeedSubscription] = []
        subscriptions_with_new_items: set[str] = set()

        for subscription in subscriptions:
            result = fetched_by_id.get(subscription.subscription_id)
            if result is None:
                updated_subscriptions.append(subscription)
                continue

            updated_subscription = result.subscription
            updated_subscriptions.append(updated_subscription)

            if result.error:
                errors.append(f"{subscription.subscription_id}: {result.error}")
                continue

            if result.refreshed:
                refreshed_ids.append(updated_subscription.subscription_id)

            if not result.had_new_items:
                continue

            subscriptions_with_new_items.add(updated_subscription.subscription_id)
            world_signals.extend(
                self._build_world_signals(
                    subscription=updated_subscription,
                    items=result.unseen_items[: self.max_signals_per_subscription],
                    now=now,
                )
            )
            awareness_thread = self._merge_awareness_thread(
                existing_threads=updated_awareness_threads,
                subscription=updated_subscription,
                items=result.unseen_items[: self.max_signals_per_subscription],
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

        synchronized_state = self._synchronize_interest_policy_state(
            state=replace(
                state,
                awareness_threads=updated_awareness_threads,
            ),
            subscriptions=updated_subscriptions,
            shared_evidence_subscription_ids=tuple(subscriptions_with_new_items),
        )

        self.store.save_subscriptions(
            config=self.config,
            subscriptions=updated_subscriptions,
            remote_state=self.remote_state,
        )
        self.store.save_state(
            config=self.config,
            state=replace(
                synchronized_state,
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

    # ------------------------- Network / parsing -------------------------

    def _validated_public_url(self, url: str) -> str:
        """Canonicalize and validate one outbound fetch URL."""

        normalized = _normalize_candidate_url(url)
        if self.allow_private_network_hosts:
            return normalized

        parsed = urlparse(normalized)
        host = (parsed.hostname or "").casefold()
        if host in _PRIVATE_HOST_LABELS:
            raise ValueError("world_intelligence_private_host_blocked")

        host_ip: Any | None
        try:
            host_ip = ipaddress.ip_address(host)
        except ValueError:
            host_ip = None

        if host_ip is not None:
            if _ip_is_private_or_special(host_ip):
                raise ValueError("world_intelligence_private_host_blocked")
            return normalized

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            resolved_addresses = _resolve_host_addresses(host, port)
        except socket.gaierror as exc:
            raise ValueError("world_intelligence_host_resolution_failed") from exc
        if not resolved_addresses:
            raise ValueError("world_intelligence_host_resolution_failed")

        for address_text in resolved_addresses:
            address = ipaddress.ip_address(address_text)
            if _ip_is_private_or_special(address):
                raise ValueError("world_intelligence_private_host_blocked")
        return normalized

    def _bounded_read(self, response: httpx.Response, *, max_bytes: int) -> bytes:
        """Read one HTTP response body with a hard byte cap."""

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("world_intelligence_document_too_large")
            chunks.append(chunk)
        return b"".join(chunks)

    def _fetch_payload(
        self,
        url: str,
        *,
        accept: str,
        timeout_s: float,
        max_bytes: int,
        request_headers: Mapping[str, str] | None = None,
    ) -> _FetchedPayload:
        """Fetch one URL through the pooled HTTP client with bounded redirects."""

        current_url = self._validated_public_url(url)
        permanent_target: str | None = None
        headers = dict(request_headers or {})
        headers["Accept"] = accept

        for _hop in range(self.max_redirects + 1):
            with self._http_client.stream(
                "GET",
                current_url,
                headers=headers,
                timeout=httpx.Timeout(
                    timeout=timeout_s,
                    connect=min(timeout_s, _DEFAULT_CONNECT_TIMEOUT_S),
                    pool=min(timeout_s, _DEFAULT_POOL_TIMEOUT_S),
                ),
            ) as response:
                status_code = int(response.status_code)
                response_headers = {key: value for key, value in response.headers.items()}

                if status_code in {301, 302, 303, 307, 308}:
                    location = _clean_text(response.headers.get("Location"))
                    if not location:
                        raise RuntimeError("world_intelligence_redirect_missing_location")
                    next_url = self._validated_public_url(urljoin(current_url, location))
                    if status_code in {301, 308}:
                        permanent_target = next_url
                    current_url = next_url
                    headers = {"Accept": accept}
                    continue

                content_length = _clean_text(response.headers.get("Content-Length"))
                if content_length.isdigit() and int(content_length) > max_bytes:
                    raise ValueError("world_intelligence_document_too_large")

                if status_code == 304:
                    return _FetchedPayload(
                        url=current_url,
                        payload=b"",
                        content_type=_clean_text(response.headers.get("Content-Type")),
                        status_code=status_code,
                        headers=response_headers,
                        not_modified=True,
                        permanently_moved_to=permanent_target,
                    )
                if status_code == 410:
                    return _FetchedPayload(
                        url=current_url,
                        payload=b"",
                        content_type=_clean_text(response.headers.get("Content-Type")),
                        status_code=status_code,
                        headers=response_headers,
                        gone=True,
                        permanently_moved_to=permanent_target,
                    )

                response.raise_for_status()
                payload = self._bounded_read(response, max_bytes=max_bytes)
                return _FetchedPayload(
                    url=current_url,
                    payload=payload,
                    content_type=_clean_text(response.headers.get("Content-Type")),
                    status_code=status_code,
                    headers=response_headers,
                    permanently_moved_to=permanent_target,
                )

        raise RuntimeError("world_intelligence_too_many_redirects")

    def _default_page_loader(self, url: str) -> _FetchedDocument:
        """Fetch one HTML or feed document used for autodiscovery."""

        fetched = self._fetch_payload(
            self._validated_public_url(url),
            accept=(
                "text/html,application/xhtml+xml,"
                "application/feed+json,application/rss+xml,application/atom+xml,"
                "application/json,application/xml,text/xml;q=0.9,*/*;q=0.1"
            ),
            timeout_s=self.page_timeout_s,
            max_bytes=self.max_document_bytes,
        )
        return _FetchedDocument(
            url=fetched.url,
            text=_decode_text(fetched.payload, fetched.content_type),
            content_type=fetched.content_type,
            status_code=fetched.status_code,
        )

    def _conditional_headers_for_feed(self, feed_url: str) -> dict[str, str]:
        """Return conditional GET headers for one feed URL."""

        normalized = self._validated_public_url(feed_url)
        with self._conditional_headers_lock:
            cached = self._conditional_headers_by_feed_url.get(normalized)
        if cached is None:
            return {}
        headers: dict[str, str] = {}
        if cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        return headers

    def _remember_feed_conditional_headers(
        self,
        *,
        requested_feed_url: str,
        final_feed_url: str,
        headers: Mapping[str, str],
    ) -> None:
        """Persist conditional headers for one feed URL in process memory."""

        etag = _clean_text(headers.get("etag"))
        last_modified = _clean_text(headers.get("last-modified"))
        if not etag and not last_modified:
            return
        state = _ConditionalRequestState(
            etag=etag or None,
            last_modified=last_modified or None,
        )
        requested = self._validated_public_url(requested_feed_url)
        final = self._validated_public_url(final_feed_url)
        with self._conditional_headers_lock:
            self._conditional_headers_by_feed_url[requested] = state
            self._conditional_headers_by_feed_url[final] = state

    def _move_conditional_headers(self, *, old_url: str, new_url: str) -> None:
        """Move any remembered conditional headers onto a new canonical URL."""

        normalized_old = self._validated_public_url(old_url)
        normalized_new = self._validated_public_url(new_url)
        with self._conditional_headers_lock:
            cached = self._conditional_headers_by_feed_url.pop(normalized_old, None)
            if cached is not None:
                self._conditional_headers_by_feed_url[normalized_new] = cached

    def _parse_xml_feed_items(
        self,
        *,
        feed_url: str,
        payload: bytes,
        content_type: str,
    ) -> tuple[_ParsedFeedItem, ...]:
        """Parse one RSS/Atom payload into bounded feed items."""

        parsed = feedparser.parse(
            payload,
            response_headers={"content-type": content_type},
            sanitize_html=True,
            resolve_relative_uris=True,
        )
        if not getattr(parsed, "entries", None) and not _clean_text(parsed.get("feed", {}).get("title")):
            raise ValueError("world_intelligence_unparseable_feed")

        source_label = _to_plain_text(parsed.get("feed", {}).get("title"), maximum=120) or _host_label(feed_url)
        entries = []
        for raw_entry in parsed.entries:
            published_dt = (
                _struct_time_to_datetime(raw_entry.get("published_parsed"))
                or _struct_time_to_datetime(raw_entry.get("updated_parsed"))
                or _struct_time_to_datetime(raw_entry.get("created_parsed"))
            )
            published_at = _isoformat(published_dt) if published_dt is not None else _clean_text(
                raw_entry.get("published")
                or raw_entry.get("updated")
                or raw_entry.get("created")
            )
            link = _candidate_entry_link(raw_entry)
            title = (
                _to_plain_text(raw_entry.get("title"), maximum=280)
                or _to_plain_text(raw_entry.get("summary"), maximum=280)
                or _to_plain_text(link, maximum=280)
                or "Untitled"
            )
            stable_id = _safe_item_fingerprint(
                feed_url=feed_url,
                native_id=_clean_text(raw_entry.get("id") or raw_entry.get("guid")),
                link=link,
                published_at=published_at,
                title=title,
            )
            entries.append(
                _ParsedFeedItem(
                    stable_id=stable_id,
                    item=WorldFeedItem(
                        feed_url=feed_url,
                        source=source_label,
                        title=title,
                        link=link,
                        published_at=published_at,
                    ),
                    published_dt=published_dt,
                )
            )

        ranked = sorted(
            enumerate(entries),
            key=lambda pair: (
                pair[1].published_dt is not None,
                pair[1].published_dt or datetime.min.replace(tzinfo=timezone.utc),
                -pair[0],
            ),
            reverse=True,
        )
        return tuple(item for _index, item in ranked[: self.max_items_per_refresh])

    def _parse_json_feed_items(
        self,
        *,
        feed_url: str,
        text: str,
    ) -> tuple[_ParsedFeedItem, ...]:
        """Parse one JSON Feed document into bounded feed items."""

        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("world_intelligence_unparseable_json_feed") from exc
        if not isinstance(document, dict):
            raise ValueError("world_intelligence_unparseable_json_feed")
        version = _clean_text(document.get("version"))
        if not version.startswith("https://jsonfeed.org/version/"):
            raise ValueError("world_intelligence_unparseable_json_feed")

        source_label = _to_plain_text(document.get("title"), maximum=120) or _host_label(feed_url)

        items = []
        raw_items = document.get("items") or []
        if not isinstance(raw_items, list):
            raise ValueError("world_intelligence_unparseable_json_feed")

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            published_dt = (
                _parse_json_datetime(raw_item.get("date_published"))
                or _parse_json_datetime(raw_item.get("date_modified"))
            )
            published_at = _isoformat(published_dt) if published_dt is not None else _clean_text(
                raw_item.get("date_published") or raw_item.get("date_modified")
            )

            link = ""
            for candidate in (
                raw_item.get("url"),
                raw_item.get("external_url"),
            ):
                candidate_text = _clean_text(candidate)
                if not candidate_text:
                    continue
                try:
                    link = _canonicalize_http_url(candidate_text)
                except ValueError:
                    link = candidate_text
                break

            title = (
                _to_plain_text(raw_item.get("title"), maximum=280)
                or _to_plain_text(raw_item.get("summary"), maximum=280)
                or _to_plain_text(raw_item.get("content_text"), maximum=280)
                or _to_plain_text(raw_item.get("content_html"), maximum=280)
                or _to_plain_text(link, maximum=280)
                or "Untitled"
            )
            stable_id = _safe_item_fingerprint(
                feed_url=feed_url,
                native_id=_clean_text(raw_item.get("id")),
                link=link,
                published_at=published_at,
                title=title,
            )
            items.append(
                _ParsedFeedItem(
                    stable_id=stable_id,
                    item=WorldFeedItem(
                        feed_url=feed_url,
                        source=source_label,
                        title=title,
                        link=link,
                        published_at=published_at,
                    ),
                    published_dt=published_dt,
                )
            )

        ranked = sorted(
            enumerate(items),
            key=lambda pair: (
                pair[1].published_dt is not None,
                pair[1].published_dt or datetime.min.replace(tzinfo=timezone.utc),
                -pair[0],
            ),
            reverse=True,
        )
        return tuple(item for _index, item in ranked[: self.max_items_per_refresh])

    def _default_feed_reader(
        self,
        feed_url: str,
        *,
        max_items: int,
        timeout_s: float,
    ) -> _FeedReadResult:
        """Fetch one RSS/Atom/JSON Feed URL through the safe pooled client."""

        normalized_feed_url = self._validated_public_url(feed_url)
        conditional_headers = self._conditional_headers_for_feed(normalized_feed_url)
        fetched = self._fetch_payload(
            normalized_feed_url,
            accept=(
                "application/feed+json,application/rss+xml,application/atom+xml,"
                "application/xml,text/xml,application/json;q=0.95,*/*;q=0.1"
            ),
            timeout_s=timeout_s,
            max_bytes=self.max_feed_bytes,
            request_headers=conditional_headers,
        )

        final_feed_url = fetched.permanently_moved_to or fetched.url

        if fetched.not_modified:
            if fetched.permanently_moved_to and fetched.permanently_moved_to != normalized_feed_url:
                self._move_conditional_headers(
                    old_url=normalized_feed_url,
                    new_url=fetched.permanently_moved_to,
                )
            return _FeedReadResult(
                items=(),
                final_feed_url=final_feed_url,
                status_code=fetched.status_code,
                not_modified=True,
                permanently_moved_to=fetched.permanently_moved_to,
            )

        if fetched.gone:
            return _FeedReadResult(
                items=(),
                final_feed_url=final_feed_url,
                status_code=fetched.status_code,
                gone=True,
                permanently_moved_to=fetched.permanently_moved_to,
            )

        self._remember_feed_conditional_headers(
            requested_feed_url=normalized_feed_url,
            final_feed_url=final_feed_url,
            headers=fetched.headers,
        )

        media_type = _media_type(fetched.content_type)
        text = _decode_text(fetched.payload, fetched.content_type)
        if media_type in {"application/feed+json", "application/json"} or _is_json_feed_document(text):
            items = self._parse_json_feed_items(feed_url=final_feed_url, text=text)
        else:
            items = self._parse_xml_feed_items(
                feed_url=final_feed_url,
                payload=fetched.payload,
                content_type=fetched.content_type or "application/xml",
            )

        return _FeedReadResult(
            items=tuple(items[: max(1, int(max_items))]),
            final_feed_url=final_feed_url,
            status_code=fetched.status_code,
            permanently_moved_to=fetched.permanently_moved_to,
        )

    def _normalize_feed_reader_result(
        self,
        *,
        subscription: WorldFeedSubscription,
        raw_result: Any,
    ) -> _FeedReadResult:
        """Adapt legacy injectable feed readers to the upgraded internal format."""

        if isinstance(raw_result, _FeedReadResult):
            return raw_result

        legacy_items = tuple(raw_result or ())
        parsed_items: list[_ParsedFeedItem] = []
        for item in legacy_items:
            if isinstance(item, _ParsedFeedItem):
                parsed_items.append(item)
                continue
            stable_id = _safe_item_fingerprint(
                feed_url=subscription.feed_url,
                native_id=None,
                link=_clean_text(getattr(item, "link", None)),
                published_at=_clean_text(getattr(item, "published_at", None)),
                title=_clean_text(getattr(item, "title", None)),
            )
            parsed_items.append(
                _ParsedFeedItem(
                    stable_id=stable_id,
                    item=WorldFeedItem(
                        feed_url=_clean_text(getattr(item, "feed_url", None)) or subscription.feed_url,
                        source=_to_plain_text(getattr(item, "source", None), maximum=120),
                        title=_to_plain_text(getattr(item, "title", None), maximum=280),
                        link=_clean_text(getattr(item, "link", None)),
                        published_at=_clean_text(getattr(item, "published_at", None)),
                    ),
                    published_dt=_parse_iso(_clean_text(getattr(item, "published_at", None))),
                )
            )
        return _FeedReadResult(
            items=tuple(parsed_items[: self.max_items_per_refresh]),
            final_feed_url=subscription.feed_url,
            status_code=200,
        )

    def _fetch_one_due_subscription(
        self,
        *,
        subscription: WorldFeedSubscription,
        now_iso: str,
    ) -> _FetchedSubscriptionResult:
        """Fetch one due subscription and return the updated subscription record."""

        try:
            raw_result = self.feed_reader(
                subscription.feed_url,
                max_items=self.max_items_per_refresh,
                timeout_s=self.feed_timeout_s,
            )
            feed_result = self._normalize_feed_reader_result(
                subscription=subscription,
                raw_result=raw_result,
            )

            updated_feed_url = subscription.feed_url
            if feed_result.permanently_moved_to:
                updated_feed_url = self._validated_public_url(feed_result.permanently_moved_to)

            if feed_result.gone:
                return _FetchedSubscriptionResult(
                    subscription=replace(
                        subscription,
                        active=False,
                        feed_url=updated_feed_url,
                        updated_at=now_iso,
                        last_checked_at=now_iso,
                        last_refreshed_at=now_iso,
                        last_error="410_Gone",
                    ),
                    unseen_items=(),
                    refreshed=True,
                    had_new_items=False,
                )

            if feed_result.not_modified:
                return _FetchedSubscriptionResult(
                    subscription=replace(
                        subscription,
                        feed_url=updated_feed_url,
                        updated_at=now_iso,
                        last_checked_at=now_iso,
                        last_refreshed_at=now_iso,
                        last_error=None,
                    ),
                    unseen_items=(),
                    refreshed=True,
                    had_new_items=False,
                )

            unseen_items, retained_item_ids = self._unseen_items(
                subscription=subscription,
                items=feed_result.items,
            )
            return _FetchedSubscriptionResult(
                subscription=replace(
                    subscription,
                    feed_url=updated_feed_url,
                    updated_at=now_iso,
                    last_checked_at=now_iso,
                    last_refreshed_at=now_iso,
                    last_error=None,
                    last_item_ids=retained_item_ids,
                ),
                unseen_items=unseen_items,
                refreshed=True,
                had_new_items=bool(unseen_items),
            )
        except Exception as exc:
            return _FetchedSubscriptionResult(
                subscription=replace(
                    subscription,
                    updated_at=now_iso,
                    last_checked_at=now_iso,
                    last_error=f"{type(exc).__name__}:{_truncate_text(str(exc), maximum=120)}",
                ),
                unseen_items=(),
                refreshed=False,
                had_new_items=False,
                error=type(exc).__name__,
            )

    def _fetch_due_subscriptions(
        self,
        *,
        due_subscriptions: Sequence[WorldFeedSubscription],
        now_iso: str,
    ) -> tuple[_FetchedSubscriptionResult, ...]:
        """Fetch due subscriptions concurrently with a bounded worker pool."""

        if len(due_subscriptions) <= 1 or self.max_parallel_refreshes <= 1:
            return tuple(
                self._fetch_one_due_subscription(subscription=subscription, now_iso=now_iso)
                for subscription in due_subscriptions
            )

        results_by_id: dict[str, _FetchedSubscriptionResult] = {}
        max_workers = min(self.max_parallel_refreshes, len(due_subscriptions))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="twinr-feed") as pool:
            futures = {
                pool.submit(
                    self._fetch_one_due_subscription,
                    subscription=subscription,
                    now_iso=now_iso,
                ): subscription.subscription_id
                for subscription in due_subscriptions
            }
            for future in as_completed(futures):
                subscription_id = futures[future]
                try:
                    results_by_id[subscription_id] = future.result()
                except Exception as exc:  # pragma: no cover - executor safety net
                    original = next(
                        item for item in due_subscriptions if item.subscription_id == subscription_id
                    )
                    results_by_id[subscription_id] = _FetchedSubscriptionResult(
                        subscription=replace(
                            original,
                            updated_at=now_iso,
                            last_checked_at=now_iso,
                            last_error=f"{type(exc).__name__}:{_truncate_text(str(exc), maximum=120)}",
                        ),
                        unseen_items=(),
                        refreshed=False,
                        had_new_items=False,
                        error=type(exc).__name__,
                    )
        return tuple(results_by_id[subscription.subscription_id] for subscription in due_subscriptions)

    # ------------------------- Discovery -------------------------

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
        source_urls = []
        for source in raw_sources:
            candidate = _clean_text(source)
            if not candidate:
                continue
            try:
                source_urls.append(self._validated_public_url(candidate))
            except ValueError:
                continue

        discovered: list[str] = []
        seen: set[str] = set()
        for source_url in source_urls:
            try:
                discovered_from_source = self._discover_feeds_from_source(source_url)
            except Exception:
                continue
            for feed_url in discovered_from_source:
                if feed_url in seen:
                    continue
                seen.add(feed_url)
                discovered.append(feed_url)
        return tuple(discovered)

    def _discovery_question(self, request: WorldIntelligenceConfigRequest) -> str:
        """Build one calm discovery question for the live web backend."""

        parts = ["Find RSS, Atom, or JSON feeds"]
        if request.topics:
            parts.append(f"for {', '.join(request.topics)}")
        if request.location_hint or request.region:
            parts.append(f"relevant to {request.location_hint or request.region}")
        return " ".join(parts).strip() + "."

    def _discover_feeds_from_source(self, source_url: str) -> tuple[str, ...]:
        """Extract feed URLs from one source page or direct feed URL."""

        safe_source_url = self._validated_public_url(source_url)
        document = self.page_loader(safe_source_url)
        url = _clean_text(getattr(document, "url", None)) or safe_source_url
        content_type = _clean_text(getattr(document, "content_type", None)).casefold()
        text = str(getattr(document, "text", ""))
        if _looks_like_feed_document(text=text, content_type=content_type):
            return (self._validated_public_url(url),)
        parser = _FeedAutodiscoveryParser(base_url=url)
        parser.feed(text)
        parser.close()
        ordered: list[str] = []
        seen: set[str] = set()
        for feed_url, _title in parser.discovered:
            try:
                normalized = self._validated_public_url(feed_url)
            except ValueError:
                continue
            if normalized in seen:
                continue
            try:
                candidate_document = self.page_loader(normalized)
                candidate_content_type = _clean_text(
                    getattr(candidate_document, "content_type", None)
                ).casefold()
                candidate_text = str(getattr(candidate_document, "text", ""))
                if not _looks_like_feed_document(
                    text=candidate_text,
                    content_type=candidate_content_type,
                ):
                    continue
            except Exception:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return tuple(ordered)

    # ------------------------- Subscription mutation -------------------------

    def _upsert_subscriptions(
        self,
        *,
        existing: Sequence[WorldFeedSubscription],
        request: WorldIntelligenceConfigRequest,
        feed_urls: Sequence[str],
        now_iso: str,
    ) -> list[WorldFeedSubscription]:
        """Create or update subscriptions for one set of feed URLs."""

        normalized_feed_urls: list[str] = []
        for url in feed_urls:
            candidate = _clean_text(url)
            if not candidate:
                continue
            normalized_feed_urls.append(self._validated_public_url(candidate))

        ordered: list[WorldFeedSubscription] = []
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
                    base_priority=request.priority,
                    active=True,
                    refresh_interval_hours=request.refresh_interval_hours or subscription.refresh_interval_hours,
                    base_refresh_interval_hours=request.refresh_interval_hours or subscription.base_refresh_interval_hours,
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
                    base_priority=request.priority,
                    active=True,
                    refresh_interval_hours=request.refresh_interval_hours or self.default_refresh_interval_hours,
                    base_refresh_interval_hours=request.refresh_interval_hours or self.default_refresh_interval_hours,
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
        urls = set()
        for item in request.feed_urls:
            candidate = _clean_text(item)
            if not candidate:
                continue
            try:
                urls.add(self._validated_public_url(candidate))
            except ValueError:
                urls.add(candidate)

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

    # ------------------------- Recalibration / interests -------------------------

    def _maybe_recalibrate(
        self,
        *,
        subscriptions: Sequence[WorldFeedSubscription],
        state: WorldIntelligenceState,
        now: datetime,
        force: bool,
        search_backend: object | None,
    ) -> tuple[list[WorldFeedSubscription], WorldIntelligenceState]:
        """Discover new feeds from durable interest signals when due."""

        if not force and not self._recalibration_due(state=state, now=now):
            return list(subscriptions), state
        now_iso = _isoformat(now)
        current_subscriptions = self._apply_interest_policy_to_subscriptions(
            subscriptions=subscriptions,
            state=state,
            now_iso=now_iso,
        )
        updated_state = replace(
            self._synchronize_interest_policy_state(
                state=state,
                subscriptions=current_subscriptions,
                shared_evidence_subscription_ids=(),
            ),
            last_recalibrated_at=now_iso,
        )
        if search_backend is None:
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

        for interest_signal in self._select_recalibration_candidates(
            subscriptions=current_subscriptions,
            state=updated_state,
        ):
            request = WorldIntelligenceConfigRequest(
                action="discover",
                query=self._interest_discovery_question(interest_signal),
                label=self._interest_label(interest_signal),
                location_hint=interest_signal.region,
                region=interest_signal.region,
                topics=(interest_signal.topic,),
                scope=interest_signal.scope,
                priority=self._interest_priority(interest_signal),
                refresh_interval_hours=self._interest_refresh_interval_hours(interest_signal),
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
                self._synchronize_interest_policy_state(
                    state=updated_state,
                    subscriptions=current_subscriptions,
                    shared_evidence_subscription_ids=(),
                ),
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
        """Return whether the slow feed recalibration cadence is due."""

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
            (
                item
                for item in state.interest_signals
                if self._signal_can_seed_feed_discovery(item)
                and item.engagement_state not in {"cooling", "avoid"}
                and (
                    item.ongoing_interest in {"active", "growing"}
                    or (item.explicit and item.engagement_score >= 0.72)
                )
            ),
            key=lambda item: (
                self._ongoing_interest_rank(item.ongoing_interest),
                self._engagement_state_rank(item.engagement_state),
                self._co_attention_state_rank(item.co_attention_state),
                item.explicit,
                _signal_ongoing_interest_score(item),
                item.engagement_score,
                item.engagement_count,
                item.salience,
                item.confidence,
                item.evidence_count,
                item.updated_at or "",
            ),
            reverse=True,
        )
        selected: list[WorldInterestSignal] = []
        for signal in ranked:
            if any(
                self._subscription_covers_interest(subscription, signal)
                for subscription in subscriptions
                if subscription.active
            ):
                continue
            selected.append(signal)
            if len(selected) >= 2:
                break
        return tuple(selected)

    def _signal_can_seed_feed_discovery(self, signal: WorldInterestSignal) -> bool:
        """Return whether one learned interest should drive durable feed discovery."""

        return not signal.signal_id.startswith("interest:tool:")

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

        parts = ["Find RSS, Atom, or JSON feeds"]
        parts.append(f"for {signal.topic}")
        if signal.region:
            parts.append(f"relevant to {signal.region}")
        return " ".join(parts).strip() + "."

    def _interest_priority(self, signal: WorldInterestSignal) -> float:
        """Map one learned interest to a bounded subscription priority."""

        if signal.ongoing_interest == "active":
            return _clamp(
                max(
                    signal.salience * 0.84,
                    (signal.engagement_score * 0.52)
                    + (_signal_ongoing_interest_score(signal) * 0.44),
                ),
                minimum=0.52,
                maximum=0.98,
            )
        if signal.ongoing_interest == "growing":
            return _clamp(
                max(
                    signal.salience * 0.74,
                    (signal.engagement_score * 0.46)
                    + (_signal_ongoing_interest_score(signal) * 0.34),
                ),
                minimum=0.4,
                maximum=0.9,
            )
        if signal.engagement_state == "resonant":
            return _clamp(
                max(signal.salience, (signal.salience * 0.74) + (signal.engagement_score * 0.5)),
                minimum=0.45,
                maximum=0.97,
            )
        if signal.engagement_state == "warm":
            return _clamp(
                max(signal.salience * 0.92, (signal.salience * 0.7) + (signal.engagement_score * 0.4)),
                minimum=0.4,
                maximum=0.9,
            )
        if signal.engagement_state == "uncertain":
            return _clamp(signal.salience * 0.88, minimum=0.35, maximum=0.78)
        if signal.engagement_state == "cooling":
            return _clamp(signal.salience * 0.65, minimum=0.28, maximum=0.62)
        if signal.engagement_state == "avoid":
            return _clamp(signal.salience * 0.5, minimum=0.18, maximum=0.48)
        return _clamp(
            max(signal.salience, (signal.salience * 0.7) + (signal.engagement_score * 0.45)),
            minimum=0.4,
            maximum=0.97,
        )

    def _interest_refresh_interval_hours(self, signal: WorldInterestSignal) -> int:
        """Map one learned interest to a calm but engagement-aware refresh cadence."""

        if signal.engagement_state == "avoid":
            return max(self.default_refresh_interval_hours, 168)
        if signal.engagement_state == "cooling":
            return max(self.default_refresh_interval_hours, 120)
        if signal.ongoing_interest == "active" and signal.engagement_state in {"resonant", "warm"}:
            return 24
        if signal.ongoing_interest == "growing":
            return 48
        if signal.engagement_state == "uncertain":
            return self.default_refresh_interval_hours
        if signal.engagement_state == "resonant" and (signal.explicit or signal.engagement_score >= 0.9):
            return 24
        if signal.engagement_state == "resonant":
            return 48
        if signal.engagement_state == "warm":
            return 48 if signal.engagement_score >= 0.78 or signal.engagement_count >= 4 else 72
        if signal.explicit or signal.engagement_score >= 0.9:
            return 24
        if signal.engagement_score >= 0.78 or signal.engagement_count >= 4:
            return 48
        if signal.engagement_score >= 0.64 or signal.engagement_count >= 2:
            return 72
        return self.default_refresh_interval_hours

    def _apply_interest_policy_to_subscriptions(
        self,
        *,
        subscriptions: Sequence[WorldFeedSubscription],
        state: WorldIntelligenceState,
        now_iso: str,
    ) -> list[WorldFeedSubscription]:
        """Tune covered subscriptions from durable engagement evidence."""

        tuned: list[WorldFeedSubscription] = []
        for subscription in subscriptions:
            base_priority = _subscription_base_priority(subscription)
            base_refresh_interval_hours = _subscription_base_refresh_interval_hours(subscription)
            matching = [
                signal
                for signal in state.interest_signals
                if signal.engagement_state != "avoid"
                and (
                    signal.ongoing_interest in {"active", "growing"}
                    or signal.engagement_state in {"resonant", "warm"}
                )
                and signal.engagement_score >= 0.52
                and self._subscription_covers_interest(subscription, signal)
            ]
            if not matching:
                tuned.append(
                    replace(
                        subscription,
                        priority=base_priority,
                        refresh_interval_hours=base_refresh_interval_hours,
                        updated_at=now_iso,
                    )
                )
                continue
            strongest = max(
                matching,
                key=lambda item: (
                    self._ongoing_interest_rank(item.ongoing_interest),
                    item.explicit,
                    self._co_attention_state_rank(item.co_attention_state),
                    _signal_ongoing_interest_score(item),
                    item.engagement_score,
                    item.engagement_count,
                    item.salience,
                    item.updated_at or "",
                ),
            )
            tuned.append(
                replace(
                    subscription,
                    priority=max(base_priority, self._interest_priority(strongest)),
                    refresh_interval_hours=min(
                        base_refresh_interval_hours,
                        self._interest_refresh_interval_hours(strongest),
                    ),
                    updated_at=now_iso,
                )
            )
        return tuned

    def _decay_interest_signals(
        self,
        *,
        signals: Sequence[WorldInterestSignal],
        now: datetime,
    ) -> tuple[WorldInterestSignal, ...]:
        """Decay stale engagement so topic boosts do not stick forever."""

        decayed: list[WorldInterestSignal] = []
        for signal in signals:
            updated_at = _parse_iso(signal.updated_at)
            if updated_at is None:
                decayed.append(signal)
                continue
            age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
            if age_days <= float(self.interest_decay_grace_days):
                decayed.append(signal)
                continue
            decay_steps = int(
                (age_days - float(self.interest_decay_grace_days))
                // float(self.interest_decay_step_days)
            ) + 1
            min_engagement = 0.42 if signal.explicit else 0.2
            min_salience = 0.38 if signal.explicit else 0.16
            ongoing_interest_floor = (
                0.42
                if signal.explicit and signal.engagement_state not in {"cooling", "avoid"}
                else 0.04
            )
            co_attention_floor = 0.0
            decayed_signal = WorldInterestSignal(
                signal_id=signal.signal_id,
                topic=signal.topic,
                summary=signal.summary,
                region=signal.region,
                scope=signal.scope,
                salience=_clamp(
                    signal.salience - (decay_steps * self.interest_decay_salience_step),
                    minimum=min_salience,
                    maximum=1.0,
                ),
                confidence=signal.confidence,
                engagement_score=_clamp(
                    signal.engagement_score - (decay_steps * self.interest_decay_engagement_step),
                    minimum=min_engagement,
                    maximum=1.0,
                ),
                ongoing_interest_score=_clamp(
                    _signal_ongoing_interest_score(signal)
                    - (decay_steps * max(self.interest_decay_engagement_step, 0.08))
                    - (signal.non_reengagement_count * 0.03)
                    - (signal.deflection_count * 0.06),
                    minimum=ongoing_interest_floor,
                    maximum=1.0,
                ),
                co_attention_score=_clamp(
                    _signal_co_attention_score(signal)
                    - (decay_steps * max(self.interest_decay_engagement_step, 0.09))
                    - (
                        0.06
                        if signal.engagement_state == "cooling"
                        else 0.12
                        if signal.engagement_state == "avoid"
                        else 0.0
                    ),
                    minimum=co_attention_floor,
                    maximum=1.0,
                ),
                co_attention_count=max(
                    0,
                    signal.co_attention_count
                    - max(1, decay_steps // 2)
                    - (1 if signal.engagement_state in {"cooling", "avoid"} else 0),
                ),
                evidence_count=max(1, signal.evidence_count - max(0, decay_steps // 2)),
                engagement_count=max(0, signal.engagement_count - decay_steps),
                positive_signal_count=max(0, signal.positive_signal_count - max(1, decay_steps // 2)),
                exposure_count=max(0, signal.exposure_count - max(0, decay_steps // 3)),
                non_reengagement_count=max(0, signal.non_reengagement_count - max(1, decay_steps // 2)),
                deflection_count=max(
                    0,
                    signal.deflection_count - max(0, decay_steps // (3 if signal.explicit else 2)),
                ),
                explicit=signal.explicit,
                source_event_ids=signal.source_event_ids,
                updated_at=signal.updated_at,
            )
            if (
                not signal.explicit
                and age_days >= float(self.interest_retention_days)
                and decayed_signal.engagement_score <= 0.24
                and decayed_signal.salience <= 0.2
                and decayed_signal.ongoing_interest == "peripheral"
                and decayed_signal.co_attention_count == 0
                and decayed_signal.engagement_state in {"uncertain", "cooling"}
            ):
                continue
            decayed.append(decayed_signal)
        ranked = sorted(
            decayed,
            key=lambda item: (
                self._ongoing_interest_rank(item.ongoing_interest),
                self._engagement_state_rank(item.engagement_state),
                self._co_attention_state_rank(item.co_attention_state),
                item.explicit,
                    _signal_ongoing_interest_score(item),
                item.engagement_score,
                item.engagement_count,
                item.salience,
                item.confidence,
                item.evidence_count,
                item.updated_at or "",
            ),
            reverse=True,
        )
        return tuple(ranked[: self.max_interest_signals])

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
            incoming_negative = signal.non_reengagement_count > 0 or signal.deflection_count > 0
            incoming_positive_bonus = min(
                0.14,
                (signal.positive_signal_count * 0.03) + (signal.engagement_count * 0.015),
            )
            incoming_negative_penalty = min(
                0.36,
                (signal.non_reengagement_count * 0.08) + (signal.deflection_count * 0.18),
            )
            existing_negative_penalty = min(
                0.2,
                (current.non_reengagement_count * 0.04) + (current.deflection_count * 0.1),
            )
            merged_score = (
                _clamp(
                    current.engagement_score - incoming_negative_penalty,
                    minimum=0.0,
                    maximum=1.0,
                )
                if incoming_negative
                else _clamp(
                    max(current.engagement_score, signal.engagement_score)
                    + incoming_positive_bonus
                    - existing_negative_penalty,
                    minimum=0.0,
                    maximum=1.0,
                )
            )
            merged_co_attention_count = (
                max(
                    0,
                    current.co_attention_count
                    - (
                        2
                        if signal.deflection_count > 0
                        else 1
                        if signal.non_reengagement_count > 0
                        else 0
                    ),
                )
                if incoming_negative
                else max(current.co_attention_count, signal.co_attention_count)
            )
            merged_co_attention_score = (
                _clamp(
                    min(
                        _signal_co_attention_score(current),
                        _signal_co_attention_score(signal),
                        merged_score + 0.12,
                    )
                    - 0.12
                    - (signal.non_reengagement_count * 0.05)
                    - (signal.deflection_count * 0.08),
                    minimum=0.0,
                    maximum=1.0,
                )
                if incoming_negative
                else _clamp(
                    max(
                        _signal_co_attention_score(current),
                        _signal_co_attention_score(signal),
                    )
                    + min(0.12, incoming_positive_bonus)
                    - (existing_negative_penalty * 0.45),
                    minimum=0.0,
                    maximum=1.0,
                )
            )
            merged_ongoing_interest_score = (
                _clamp(
                    min(
                        _signal_ongoing_interest_score(current),
                        _signal_ongoing_interest_score(signal),
                        merged_score + 0.12,
                    )
                    - 0.12
                    - (signal.non_reengagement_count * 0.05)
                    - (signal.deflection_count * 0.08),
                    minimum=0.0,
                    maximum=1.0,
                )
                if incoming_negative
                else _clamp(
                    max(
                        _signal_ongoing_interest_score(current),
                        _signal_ongoing_interest_score(signal),
                    )
                    + incoming_positive_bonus
                    - (existing_negative_penalty * 0.55),
                    minimum=0.0,
                    maximum=1.0,
                )
            )
            merged[key] = WorldInterestSignal(
                signal_id=current.signal_id,
                topic=signal.topic,
                summary=signal.summary,
                region=signal.region or current.region,
                scope=signal.scope,
                salience=_clamp(
                    max(current.salience, signal.salience) + (0.04 if not incoming_negative else 0.0),
                    minimum=0.0,
                    maximum=1.0,
                ),
                confidence=_clamp(max(current.confidence, signal.confidence), minimum=0.0, maximum=1.0),
                engagement_score=merged_score,
                ongoing_interest_score=merged_ongoing_interest_score,
                co_attention_score=merged_co_attention_score,
                co_attention_count=merged_co_attention_count,
                evidence_count=current.evidence_count + signal.evidence_count,
                engagement_count=current.engagement_count + signal.engagement_count,
                positive_signal_count=current.positive_signal_count + signal.positive_signal_count,
                exposure_count=current.exposure_count + signal.exposure_count,
                non_reengagement_count=current.non_reengagement_count + signal.non_reengagement_count,
                deflection_count=current.deflection_count + signal.deflection_count,
                explicit=current.explicit or signal.explicit,
                source_event_ids=tuple(
                    dict.fromkeys((*current.source_event_ids, *signal.source_event_ids))
                ),
                updated_at=signal.updated_at or current.updated_at,
            )
        ranked = sorted(
            merged.values(),
            key=lambda item: (
                self._ongoing_interest_rank(item.ongoing_interest),
                self._engagement_state_rank(item.engagement_state),
                self._co_attention_state_rank(item.co_attention_state),
                item.explicit,
                _signal_ongoing_interest_score(item),
                item.engagement_score,
                item.engagement_count,
                item.salience,
                item.confidence,
                item.evidence_count,
                item.updated_at or "",
            ),
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

    def _engagement_state_rank(self, state: str | None) -> int:
        """Map one engagement state onto a stable ranking priority."""

        normalized = _clean_text(state).casefold()
        if normalized == "resonant":
            return 4
        if normalized == "warm":
            return 3
        if normalized == "uncertain":
            return 2
        if normalized == "cooling":
            return 1
        if normalized == "avoid":
            return 0
        return 2

    def _ongoing_interest_rank(self, state: str | None) -> int:
        """Map one ongoing-interest state onto a stable ranking priority."""

        normalized = _clean_text(state).casefold()
        if normalized == "active":
            return 2
        if normalized == "growing":
            return 1
        return 0

    def _co_attention_state_rank(self, state: str | None) -> int:
        """Map one co-attention state onto a stable ranking priority."""

        normalized = _clean_text(state).casefold()
        if normalized == "shared_thread":
            return 2
        if normalized == "forming":
            return 1
        return 0

    def _synchronize_interest_policy_state(
        self,
        *,
        state: WorldIntelligenceState,
        subscriptions: Sequence[WorldFeedSubscription],
        shared_evidence_subscription_ids: Sequence[str],
    ) -> WorldIntelligenceState:
        """Project coverage and new shared evidence back onto durable interests."""

        synchronized = self._synchronize_interest_signals(
            signals=state.interest_signals,
            subscriptions=subscriptions,
            awareness_threads=state.awareness_threads,
            shared_evidence_subscription_ids=shared_evidence_subscription_ids,
        )
        return replace(
            state,
            interest_signals=synchronized,
        )

    def _synchronize_interest_signals(
        self,
        *,
        signals: Sequence[WorldInterestSignal],
        subscriptions: Sequence[WorldFeedSubscription],
        awareness_threads: Sequence[SituationalAwarenessThread],
        shared_evidence_subscription_ids: Sequence[str],
    ) -> tuple[WorldInterestSignal, ...]:
        """Fold feed coverage back into durable interest and co-attention state."""

        refreshed_ids = {
            subscription_id
            for subscription_id in shared_evidence_subscription_ids
            if _clean_text(subscription_id)
        }
        synchronized: list[WorldInterestSignal] = []
        for signal in signals:
            covering_subscriptions = tuple(
                subscription
                for subscription in subscriptions
                if subscription.active and self._subscription_covers_interest(subscription, signal)
            )
            matching_thread = self._matching_awareness_thread(
                signal=signal,
                awareness_threads=awareness_threads,
            )
            has_coverage = bool(covering_subscriptions)
            has_shared_refresh = bool(
                matching_thread is not None
                and any(subscription.subscription_id in refreshed_ids for subscription in covering_subscriptions)
            )
            co_attention_count = signal.co_attention_count
            if signal.engagement_state in {"cooling", "avoid"} or signal.ongoing_interest == "peripheral":
                co_attention_count = max(
                    0,
                    co_attention_count
                    - (
                        2
                        if signal.engagement_state == "avoid"
                        else 1
                        if signal.engagement_state == "cooling"
                        else 0
                    ),
                )
                if not has_coverage:
                    co_attention_count = 0
            else:
                if has_coverage:
                    co_attention_count = max(co_attention_count, 1)
                if has_shared_refresh:
                    co_attention_count = min(6, co_attention_count + 1)

            ongoing_interest_score = _signal_ongoing_interest_score(signal)
            co_attention_score = _signal_co_attention_score(signal)
            if signal.engagement_state not in {"cooling", "avoid"}:
                if has_coverage:
                    ongoing_interest_score = _clamp(
                        ongoing_interest_score + 0.04,
                        minimum=0.0,
                        maximum=1.0,
                    )
                    co_attention_score = _clamp(
                        co_attention_score + 0.08,
                        minimum=0.0,
                        maximum=1.0,
                    )
                if has_shared_refresh:
                    ongoing_interest_score = _clamp(
                        ongoing_interest_score + 0.06,
                        minimum=0.0,
                        maximum=1.0,
                    )
                    co_attention_score = _clamp(
                        co_attention_score + 0.18,
                        minimum=0.0,
                        maximum=1.0,
                    )
            else:
                ongoing_interest_score = min(
                    ongoing_interest_score,
                    0.48 if signal.engagement_state == "cooling" else 0.2,
                )
                co_attention_score = min(
                    co_attention_score,
                    0.26 if signal.engagement_state == "cooling" else 0.08,
                )

            synchronized.append(
                WorldInterestSignal(
                    signal_id=signal.signal_id,
                    topic=signal.topic,
                    summary=signal.summary,
                    region=signal.region,
                    scope=signal.scope,
                    salience=signal.salience,
                    confidence=signal.confidence,
                    engagement_score=signal.engagement_score,
                    engagement_state=signal.engagement_state,
                    ongoing_interest_score=ongoing_interest_score,
                    co_attention_score=co_attention_score,
                    co_attention_count=co_attention_count,
                    evidence_count=signal.evidence_count,
                    engagement_count=signal.engagement_count,
                    positive_signal_count=signal.positive_signal_count,
                    exposure_count=signal.exposure_count,
                    non_reengagement_count=signal.non_reengagement_count,
                    deflection_count=signal.deflection_count,
                    explicit=signal.explicit,
                    source_event_ids=signal.source_event_ids,
                    updated_at=signal.updated_at,
                )
            )
        ranked = sorted(
            synchronized,
            key=lambda item: (
                self._ongoing_interest_rank(item.ongoing_interest),
                self._engagement_state_rank(item.engagement_state),
                self._co_attention_state_rank(item.co_attention_state),
                _signal_ongoing_interest_score(item),
                item.engagement_score,
                item.salience,
                item.updated_at or "",
            ),
            reverse=True,
        )
        return tuple(ranked[: self.max_interest_signals])

    def _matching_awareness_thread(
        self,
        *,
        signal: WorldInterestSignal,
        awareness_threads: Sequence[SituationalAwarenessThread],
    ) -> SituationalAwarenessThread | None:
        """Return the strongest awareness thread that structurally covers one interest."""

        signal_tokens = _token_set(signal.topic)
        if not signal_tokens:
            return None
        ranked = sorted(
            awareness_threads,
            key=lambda item: (item.salience, item.update_count, item.updated_at or "", item.title),
            reverse=True,
        )
        for thread in ranked:
            if signal.region and thread.region:
                if _clean_text(signal.region).casefold() != _clean_text(thread.region).casefold():
                    continue
            candidate_tokens = _token_set(thread.topic) | _token_set(thread.title)
            if signal_tokens & candidate_tokens:
                return thread
        return None

    # ------------------------- Feed item / awareness handling -------------------------

    def _unseen_items(
        self,
        *,
        subscription: WorldFeedSubscription,
        items: Sequence[_ParsedFeedItem],
    ) -> tuple[tuple[_ParsedFeedItem, ...], tuple[str, ...]]:
        """Return unseen items plus the bounded retained item-id history."""

        existing_ids = tuple(subscription.last_item_ids)
        seen_ids = set(existing_ids)
        unseen: list[_ParsedFeedItem] = []
        latest_ids: list[str] = []
        for parsed_item in items:
            item_id = _clean_text(parsed_item.stable_id)
            latest_ids.append(item_id)
            if item_id in seen_ids:
                continue
            unseen.append(parsed_item)
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
        items: Sequence[_ParsedFeedItem],
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
                    *(_to_plain_text(item.item.title, maximum=180) for item in items if _clean_text(item.item.title)),
                    *(existing.recent_titles if existing is not None else ()),
                ]
            )
        )[: self.max_recent_titles_per_thread]
        source_labels = tuple(
            dict.fromkeys(
                [
                    *(existing.source_labels if existing is not None else ()),
                    *(item.item.source or _host_label(subscription.feed_url) for item in items),
                ]
            )
        )
        supporting_item_ids = tuple(
            dict.fromkeys(
                [
                    *(_clean_text(item.stable_id) for item in items),
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
            title=_truncate_text(subscription.label, maximum=120),
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
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
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
                f"{update_count} relevant feed update(s) are being tracked."
                f"{headline_part}"
            ).strip()
        return (
            f"Keep a calm watch on {label}; "
            f"{update_count} relevant feed update(s) are being tracked."
            f"{headline_part}"
        ).strip()

    def _build_world_signals(
        self,
        *,
        subscription: WorldFeedSubscription,
        items: Sequence[_ParsedFeedItem],
        now: datetime,
    ) -> tuple[WorldSignal, ...]:
        """Convert unseen feed items into fresh world signals."""

        fresh_until = _isoformat(now + timedelta(hours=self.default_freshness_hours))
        signals: list[WorldSignal] = []
        for parsed_item in items:
            item = parsed_item.item
            summary = f"Relevant to {subscription.label}."
            if item.source and item.source != subscription.label:
                summary = f"Relevant to {subscription.label}; source {item.source}."
            signals.append(
                WorldSignal(
                    topic=_to_plain_text(item.title, maximum=280),
                    summary=summary,
                    region=subscription.region,
                    source=_to_plain_text(item.source or _host_label(subscription.feed_url), maximum=120),
                    salience=_clamp(subscription.priority * 0.9, minimum=0.25, maximum=0.95),
                    fresh_until=fresh_until,
                    evidence_count=1,
                    source_event_ids=tuple(
                        item_id
                        for item_id in (
                            subscription.subscription_id,
                            _clean_text(parsed_item.stable_id),
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
