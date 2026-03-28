# CHANGELOG: 2026-03-27
# BUG-1: Fixed co_attention_state so cooling topics can no longer remain shared_thread purely from stale historical score.
# BUG-2: Fixed optimistic WorldInterestSignal defaults and legacy backfill so fresh topics without positive evidence no longer auto-classify as warm/growing.
# BUG-3: Nested payload collections are now coerced/validated on construction, preventing late crashes when callers pass parsed mappings directly instead of dataclass instances.
# SEC-1: Restricted feed/source URLs to safe http/https endpoints, rejected embedded credentials and blocked local/private hosts by default to reduce practical SSRF/LFI risk on Raspberry Pi deployments.
# SEC-2: Rejected control characters and bounded large persisted text/list fields to reduce prompt/state bloat and log/header poisoning risk from malformed feed data.
# IMP-1: Added canonical URL normalization, bounded deduplicated tuple handling, and RFC3339 timestamp normalization for stable persistence and prompt use.
# IMP-2: Added frontier RSS runtime metadata (ETag, Last-Modified, HTTP status, redirect target, retry/backoff state, content fingerprints) so refreshers can use conditional GETs and adaptive polling.
# IMP-3: Added full JSON-safe serializers/deserializers for request/result models plus optional msgspec-backed JSON helpers for Pi-friendly fast persistence.

"""Define persistent RSS/world-intelligence models for evolving personality.

The world-intelligence layer keeps three concerns explicit:

- durable feed subscriptions chosen by the installer, the user, or future
  reflection/recalibration passes
- refresh/discovery runtime state that limits how often Twinr looks for new
  sources or polls existing feeds
- slowly learned calibration signals that summarize which topics and regions
  deserve ongoing source coverage
- reflected situational-awareness threads that condense repeated feed updates
  into calmer place/world context over days or weeks
- bounded refresh outputs that can be converted into prompt-time world and
  continuity context without rewriting core character

2026 frontier notes reflected in this module:

- persistent feed state carries HTTP cache metadata (ETag/Last-Modified/status)
  and redirect/gone signals so refreshers can use conditional requests and adapt
  polling instead of re-downloading unchanged feeds
- externally supplied URLs are normalized and constrained early because feed
  subscriptions are a real SSRF boundary in deployed assistants
- prompt-facing persisted fields are bounded and deduplicated to keep storage,
  memory, and Pi-class CPU usage predictable over long runtimes
- all public request/result/state models round-trip cleanly through payload
  mappings and optional JSON helpers
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, time as _time, timezone
from email.utils import parsedate_to_datetime
from ipaddress import ip_address
from typing import Callable, Protocol, TypeVar
from urllib.parse import SplitResult, urlsplit, urlunsplit

try:  # Optional fast path for Pi-friendly serialization.
    import msgspec as _msgspec  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency.
    _msgspec = None

from twinr.agent.personality._payload_utils import (
    clean_text as _clean_text,
    mapping_items as _mapping_items,
    normalize_float as _normalize_float,
    normalize_int as _normalize_int,
    normalize_string_tuple as _normalize_string_tuple,
    optional_text as _optional_text,
    required_mapping_text as _required_mapping_text,
)

DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND = "agent_world_intelligence_subscriptions_v1"
DEFAULT_WORLD_INTELLIGENCE_STATE_KIND = "agent_world_intelligence_state_v1"
DEFAULT_WORLD_INTELLIGENCE_SCHEMA_VERSION = 2

_ALLOWED_WORLD_ACTIONS = frozenset(
    {"list", "subscribe", "discover", "deactivate", "refresh_now"}
)
_ALLOWED_WORLD_SCOPES = frozenset({"local", "regional", "national", "global", "topic"})
_ALLOWED_ENGAGEMENT_STATES = frozenset({"resonant", "warm", "uncertain", "cooling", "avoid"})
_ALLOWED_ONGOING_INTEREST_STATES = frozenset({"active", "growing", "peripheral"})
_ALLOWED_CO_ATTENTION_STATES = frozenset({"latent", "forming", "shared_thread"})
_ALLOWED_REMOTE_URL_SCHEMES = frozenset({"http", "https"})

_DEFAULT_REFRESH_INTERVAL_HOURS = 24
_DEFAULT_DISCOVERY_INTERVAL_HOURS = 336
_DEFAULT_RECALIBRATION_INTERVAL_HOURS = 336
_DEFAULT_FETCH_TIMEOUT_SECONDS = 20

_MAX_ID_LENGTH = 160
_MAX_SHORT_TEXT_LENGTH = 160
_MAX_LABEL_LENGTH = 256
_MAX_TITLE_LENGTH = 512
_MAX_SUMMARY_LENGTH = 4096
_MAX_QUERY_LENGTH = 1024
_MAX_ERROR_LENGTH = 512
_MAX_URL_LENGTH = 2048
_MAX_ETAG_LENGTH = 512
_MAX_HTTP_HEADER_VALUE_LENGTH = 512
_MAX_TEXT_LIST_ITEM_LENGTH = 256
_MAX_LAST_ITEM_IDS = 128
_MAX_LAST_ITEM_FINGERPRINTS = 128
_MAX_TOPICS = 32
_MAX_FEED_URLS = 32
_MAX_SUBSCRIPTION_REFS = 64
_MAX_RECENT_TITLES = 12
_MAX_SOURCE_LABELS = 24
_MAX_SOURCE_EVENT_IDS = 128
_MAX_SUPPORTING_ITEM_IDS = 128
_MAX_WORLD_SIGNALS = 64
_MAX_CONTINUITY_THREADS = 64
_MAX_ERRORS = 32

_PRIVATE_HOST_SUFFIXES = (
    ".local",
    ".localdomain",
    ".internal",
    ".home",
    ".lan",
)
_EXPLICIT_LOCAL_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "0",
        "0.0.0.0",
        "::1",
        "[::1]",
    }
)


def _csv_env_tuple(name: str) -> tuple[str, ...]:
    """Parse one simple comma-separated environment variable."""

    raw = os.getenv(name, "")
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        cleaned = _clean_text(part).casefold()
        if not cleaned or cleaned in seen:
            continue
        values.append(cleaned)
        seen.add(cleaned)
    return tuple(values)


_DEFAULT_ALLOWED_FEED_HOSTS = _csv_env_tuple("TWINR_WORLD_INTELLIGENCE_ALLOWED_FEED_HOSTS")
_ALLOW_PRIVATE_FEED_HOSTS = (
    _clean_text(os.getenv("TWINR_WORLD_INTELLIGENCE_ALLOW_PRIVATE_FEED_HOSTS", ""))
    .casefold()
    in {"1", "true", "yes", "on"}
)


class _SupportsPayload(Protocol):
    def to_payload(self) -> dict[str, object]: ...


_T = TypeVar("_T")


def _reject_control_chars(text: str, *, field_name: str) -> str:
    """Reject ASCII control chars that can poison logs, headers, or payloads."""

    for char in text:
        codepoint = ord(char)
        if codepoint < 32 or codepoint == 127:
            raise ValueError(f"{field_name} contains control characters.")
    return text


def _bounded_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
    truncate: bool = False,
) -> str:
    """Bound one already-normalized text value to predictable storage size."""

    text = _reject_control_chars(_clean_text(value), field_name=field_name)
    if not text:
        raise ValueError(f"{field_name} is required.")
    if len(text) <= max_length:
        return text
    if not truncate:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length}.")
    if max_length <= 1:
        return text[:max_length]
    trimmed = text[: max_length - 1].rstrip()
    return f"{trimmed}…" if trimmed else text[:max_length]


def _bounded_required_text(value: object, *, field_name: str, max_length: int) -> str:
    return _bounded_text(
        _required_mapping_text({field_name: value}, field_name=field_name),
        field_name=field_name,
        max_length=max_length,
    )


def _bounded_required_mapping_text(
    payload: Mapping[str, object],
    *,
    field_name: str,
    max_length: int,
    aliases: tuple[str, ...] = (),
    truncate: bool = False,
) -> str:
    return _bounded_text(
        _required_mapping_text(payload, field_name=field_name, aliases=aliases),
        field_name=field_name,
        max_length=max_length,
        truncate=truncate,
    )


def _bounded_optional_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
    truncate: bool = False,
) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return _bounded_text(text, field_name=field_name, max_length=max_length, truncate=truncate)


def _bounded_score(value: object, *, field_name: str, default: float) -> float:
    """Normalize one score and clamp to [0.0, 1.0] for stable ranking logic."""

    normalized = _normalize_float(value, field_name=field_name, default=default)
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def _bounded_string_tuple(
    value: object,
    *,
    field_name: str,
    max_items: int,
    max_item_length: int = _MAX_TEXT_LIST_ITEM_LENGTH,
    truncate: bool = True,
    dedupe: bool = True,
) -> tuple[str, ...]:
    """Normalize, dedupe, and cap one tuple of strings."""

    normalized = _normalize_string_tuple(value, field_name=field_name)
    result: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        bounded = _bounded_text(
            item,
            field_name=field_name,
            max_length=max_item_length,
            truncate=truncate,
        )
        key = bounded.casefold()
        if dedupe and key in seen:
            continue
        if dedupe:
            seen.add(key)
        result.append(bounded)
        if len(result) >= max_items:
            break
    return tuple(result)


def _parse_datetime_like(value: object, *, field_name: str) -> datetime:
    """Parse one date/datetime string into a timezone-aware datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, _date):
        parsed = datetime.combine(value, _time.min, tzinfo=timezone.utc)
    else:
        text = _bounded_text(str(value), field_name=field_name, max_length=128)
        iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
        parsed = None
        try:
            parsed = datetime.fromisoformat(iso_candidate)
        except ValueError:
            try:
                parsed_date = _date.fromisoformat(text)
            except ValueError:
                parsed_date = None
            if parsed_date is not None:
                parsed = datetime.combine(parsed_date, _time.min, tzinfo=timezone.utc)
        if parsed is None:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                parsed = None
        if parsed is None:
            raise ValueError(f"{field_name} must be ISO-8601/RFC3339 or RFC-2822 style text.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_timestamp(value: object, *, field_name: str) -> str | None:
    """Normalize one optional timestamp to canonical UTC RFC3339 text."""

    text = _optional_text(value)
    if text is None:
        return None
    parsed = _parse_datetime_like(text, field_name=field_name)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _allowed_feed_host_configured(host: str) -> bool:
    host_cf = host.casefold()
    if not _DEFAULT_ALLOWED_FEED_HOSTS:
        return False
    return any(
        host_cf == allowed or host_cf.endswith(f".{allowed}")
        for allowed in _DEFAULT_ALLOWED_FEED_HOSTS
    )


def _host_is_private_or_local(host: str) -> bool:
    host_cf = host.casefold()
    if host_cf in _EXPLICIT_LOCAL_HOSTS:
        return True
    if any(host_cf.endswith(suffix) for suffix in _PRIVATE_HOST_SUFFIXES):
        return True
    try:
        parsed_ip = ip_address(host_cf)
    except ValueError:
        return False
    return bool(
        parsed_ip.is_private
        or parsed_ip.is_loopback
        or parsed_ip.is_link_local
        or parsed_ip.is_reserved
        or parsed_ip.is_multicast
        or parsed_ip.is_unspecified
    )


# BREAKING: feed/source URLs are now normalized and restricted to remote http/https endpoints
# by default. Private/local hosts require an explicit allow-list or override env flag.
def _canonicalize_url(
    value: object,
    *,
    field_name: str,
    allow_private_host: bool,
) -> str:
    """Normalize one fetchable URL and block unsafe targets by default."""

    text = _bounded_required_text(value, field_name=field_name, max_length=_MAX_URL_LENGTH)
    parsed = urlsplit(text)
    scheme = parsed.scheme.casefold()
    if scheme not in _ALLOWED_REMOTE_URL_SCHEMES:
        raise ValueError(f"{field_name} must use one of {sorted(_ALLOWED_REMOTE_URL_SCHEMES)}.")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"{field_name} must include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} must not embed credentials.")
    host = parsed.hostname
    assert host is not None  # guarded above for type-checkers.
    try:
        host_ascii = host.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise ValueError(f"{field_name} contains an invalid hostname.") from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"{field_name} contains an invalid port.")
    if not allow_private_host and not _ALLOW_PRIVATE_FEED_HOSTS and not _allowed_feed_host_configured(host_ascii):
        if _host_is_private_or_local(host_ascii):
            raise ValueError(
                f"{field_name} points to a private/local host. Configure TWINR_WORLD_INTELLIGENCE_ALLOWED_FEED_HOSTS "
                "or TWINR_WORLD_INTELLIGENCE_ALLOW_PRIVATE_FEED_HOSTS=1 to allow this deliberately."
            )
    if ":" in host_ascii and not host_ascii.startswith("["):
        canonical_host = f"[{host_ascii}]"
    else:
        canonical_host = host_ascii
    default_port = 80 if scheme == "http" else 443
    netloc = canonical_host if port in (None, default_port) else f"{canonical_host}:{port}"
    path = parsed.path or "/"
    canonical = SplitResult(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=parsed.query,
        fragment="",
    )
    return urlunsplit(canonical)


def _canonicalize_optional_url(
    value: object,
    *,
    field_name: str,
    allow_private_host: bool,
) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return _canonicalize_url(text, field_name=field_name, allow_private_host=allow_private_host)


def _normalize_http_status(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    status = _normalize_int(value, field_name=field_name, default=0, minimum=0)
    if status != 0 and not (100 <= status <= 599):
        raise ValueError(f"{field_name} must be a valid HTTP status code.")
    return None if status == 0 else status


def _normalize_jsonish_sequence(value: object, *, field_name: str, max_items: int) -> tuple[object, ...]:
    """Accept one bounded tuple/list of payload-ish objects."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must be a sequence.")
    try:
        raw_items = tuple(value)
    except TypeError as exc:  # pragma: no cover - defensive.
        raise ValueError(f"{field_name} must be a sequence.") from exc
    result: list[object] = []
    for item in raw_items[:max_items]:
        if isinstance(item, Mapping):
            result.append(dict(item))
        elif hasattr(item, "to_payload"):
            result.append(item)
        else:
            result.append(item)
    return tuple(result)


def _serialize_payloadish(item: object) -> object:
    if hasattr(item, "to_payload") and callable(getattr(item, "to_payload")):
        return getattr(item, "to_payload")()
    if isinstance(item, Mapping):
        return dict(item)
    return item


def _coerce_tuple_of(
    value: object,
    *,
    field_name: str,
    item_type: type[_T],
    builder: Callable[[Mapping[str, object]], _T],
) -> tuple[_T, ...]:
    """Accept already-built dataclasses or mapping payloads for nested fields."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must be a sequence of {item_type.__name__}.")
    try:
        raw_items = tuple(value)
    except TypeError as exc:  # pragma: no cover - defensive.
        raise ValueError(f"{field_name} must be a sequence of {item_type.__name__}.") from exc
    result: list[_T] = []
    for item in raw_items:
        if isinstance(item, item_type):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(builder(item))
        else:
            raise ValueError(f"{field_name} items must be {item_type.__name__} or mapping payloads.")
    return tuple(result)


def _ensure_unique_attr(items: tuple[object, ...], *, field_name: str, attr_name: str) -> tuple[object, ...]:
    """Reject duplicate stable identifiers that would make updates ambiguous."""

    seen: set[str] = set()
    for item in items:
        key = _clean_text(getattr(item, attr_name)).casefold()
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate {attr_name} values.")
        seen.add(key)
    return items


def _legacy_default_engagement_score(payload: Mapping[str, object]) -> float:
    """Derive one bounded engagement default for older stored interest signals.

    Older snapshots may not yet carry explicit engagement fields. In that case
    we backfill them from the already persisted salience/confidence/evidence
    structure so the policy can become useful immediately without requiring a
    manual reseed.
    """

    salience = _bounded_score(payload.get("salience"), field_name="salience", default=0.5)
    confidence = _bounded_score(payload.get("confidence"), field_name="confidence", default=0.5)
    evidence_count = _normalize_int(
        payload.get("evidence_count"),
        field_name="evidence_count",
        default=0,
        minimum=0,
    )
    explicit = bool(payload.get("explicit", False))
    base = 0.38 + (salience * 0.24) + (confidence * 0.22) + (min(evidence_count, 4) * 0.05)
    if explicit:
        base = max(base, 0.86)
    return _bounded_score(base, field_name="engagement_score", default=0.5)


def _derive_engagement_state(
    *,
    engagement_score: float,
    engagement_count: int,
    positive_signal_count: int,
    exposure_count: int,
    non_reengagement_count: int,
    deflection_count: int,
    explicit: bool,
) -> str:
    """Classify one topic's current engagement state from bounded evidence.

    The states intentionally separate steady positive pull from evidence of
    cooling:

    - ``resonant``: the topic repeatedly draws the user back in
    - ``warm``: the topic is still a healthy ongoing interest
    - ``uncertain``: there is too little evidence either way
    - ``cooling``: repeated exposure is not converting into renewed uptake
    - ``avoid``: explicit or repeated stronger deflection says Twinr should back off
    """

    if deflection_count >= 2:
        return "avoid"
    if deflection_count >= 1 and explicit and positive_signal_count == 0:
        return "avoid"
    if deflection_count >= 1:
        return "cooling"
    if (
        exposure_count >= 2
        and non_reengagement_count >= 2
        and non_reengagement_count >= max(1, positive_signal_count)
    ):
        return "cooling"
    if explicit and exposure_count >= 2 and non_reengagement_count >= 1 and positive_signal_count == 0:
        return "cooling"
    if engagement_score >= 0.86 or engagement_count >= 4 or positive_signal_count >= 3:
        return "resonant"
    if engagement_score >= 0.62 or engagement_count >= 2 or positive_signal_count >= 1 or explicit:
        return "warm"
    return "uncertain"


def _derive_ongoing_interest_score(
    *,
    salience: float,
    engagement_score: float,
    engagement_state: str | None,
    engagement_count: int,
    positive_signal_count: int,
    non_reengagement_count: int,
    deflection_count: int,
    explicit: bool,
) -> float:
    """Summarize whether one topic keeps feeling worth following over time."""

    normalized_state = _clean_text(engagement_state).casefold()
    positive_pull = (
        (engagement_score * 0.7)
        + (salience * 0.18)
        + (min(max(positive_signal_count, 0), 4) * 0.04)
        + (min(max(engagement_count, 0), 5) * 0.03)
    )
    if explicit and positive_signal_count > 0 and deflection_count == 0:
        positive_pull = max(positive_pull, 0.76)
    penalty = (non_reengagement_count * 0.11) + (deflection_count * 0.18)
    if normalized_state == "cooling":
        penalty += 0.18
    elif normalized_state == "avoid":
        penalty += 0.38
    return _bounded_score(
        positive_pull - penalty,
        field_name="ongoing_interest_score",
        default=0.5,
    )


def _derive_ongoing_interest_state(
    *,
    ongoing_interest_score: float,
    engagement_state: str | None,
    engagement_count: int,
    positive_signal_count: int,
    non_reengagement_count: int,
    deflection_count: int,
) -> str:
    """Classify how alive one topic currently feels in Twinr's attention."""

    normalized_state = _clean_text(engagement_state).casefold()
    if normalized_state == "avoid" or deflection_count >= 1:
        return "peripheral"
    if normalized_state == "cooling" and (non_reengagement_count >= 1 or ongoing_interest_score < 0.62):
        return "peripheral"
    if (
        normalized_state == "uncertain"
        and positive_signal_count == 0
        and engagement_count <= 1
        and ongoing_interest_score < 0.68
    ):
        return "peripheral"
    if (
        ongoing_interest_score >= 0.84
        or (normalized_state == "resonant" and positive_signal_count >= 2)
        or engagement_count >= 5
    ):
        return "active"
    if (
        ongoing_interest_score >= 0.58
        or normalized_state in {"warm", "resonant"}
        or positive_signal_count >= 1
        or engagement_count >= 2
    ):
        return "growing"
    return "peripheral"


def _derive_co_attention_score(
    *,
    ongoing_interest_score: float,
    ongoing_interest: str,
    engagement_state: str | None,
    co_attention_count: int,
    non_reengagement_count: int,
    deflection_count: int,
) -> float:
    """Summarize whether a topic has become a shared ongoing thread."""

    normalized_state = _clean_text(engagement_state).casefold()
    base = (ongoing_interest_score * 0.66) + (min(max(co_attention_count, 0), 4) * 0.12)
    if ongoing_interest == "active":
        base += 0.12
    elif ongoing_interest == "growing":
        base += 0.05
    penalty = (non_reengagement_count * 0.05) + (deflection_count * 0.12)
    if normalized_state == "cooling":
        penalty += 0.18
    elif normalized_state == "avoid":
        penalty += 0.36
    return _bounded_score(
        base - penalty,
        field_name="co_attention_score",
        default=0.0,
    )


def _derive_co_attention_state(
    *,
    co_attention_score: float,
    co_attention_count: int,
    ongoing_interest: str,
    engagement_state: str | None,
) -> str:
    """Classify whether Twinr and the user now share an ongoing thread."""

    normalized_state = _clean_text(engagement_state).casefold()
    if normalized_state in {"cooling", "avoid"}:
        return "latent"
    if (
        co_attention_count >= 2
        and ongoing_interest == "active"
        and co_attention_score >= 0.78
    ):
        return "shared_thread"
    if co_attention_score >= 0.86 and ongoing_interest != "peripheral":
        return "shared_thread"
    if (
        co_attention_count >= 1
        and ongoing_interest in {"active", "growing"}
    ):
        return "forming"
    return "latent"


@dataclass(frozen=True, slots=True)
class WorldFeedItem:
    """Describe one fetched RSS or Atom item before prompt conversion."""

    feed_url: str
    source: str
    title: str
    link: str | None = None
    published_at: str | None = None
    item_id: str | None = None
    content_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Normalize one fetched feed item."""

        object.__setattr__(
            self,
            "feed_url",
            _canonicalize_url(self.feed_url, field_name="feed_url", allow_private_host=False),
        )
        object.__setattr__(
            self,
            "source",
            _bounded_required_text(self.source, field_name="source", max_length=_MAX_LABEL_LENGTH),
        )
        object.__setattr__(
            self,
            "title",
            _bounded_required_text(self.title, field_name="title", max_length=_MAX_TITLE_LENGTH),
        )
        object.__setattr__(
            self,
            "link",
            _canonicalize_optional_url(self.link, field_name="link", allow_private_host=False),
        )
        object.__setattr__(self, "published_at", _normalize_timestamp(self.published_at, field_name="published_at"))
        object.__setattr__(
            self,
            "item_id",
            _bounded_optional_text(self.item_id, field_name="item_id", max_length=_MAX_ID_LENGTH),
        )
        object.__setattr__(
            self,
            "content_fingerprint",
            _bounded_optional_text(
                self.content_fingerprint,
                field_name="content_fingerprint",
                max_length=_MAX_ID_LENGTH,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the feed item into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "feed_url": self.feed_url,
            "source": self.source,
            "title": self.title,
        }
        if self.link is not None:
            payload["link"] = self.link
        if self.published_at is not None:
            payload["published_at"] = self.published_at
        if self.item_id is not None:
            payload["item_id"] = self.item_id
        if self.content_fingerprint is not None:
            payload["content_fingerprint"] = self.content_fingerprint
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldFeedItem":
        """Build a fetched feed item from one payload mapping."""

        return cls(
            feed_url=_bounded_required_mapping_text(payload, field_name="feed_url", max_length=_MAX_URL_LENGTH),
            source=_bounded_required_mapping_text(payload, field_name="source", max_length=_MAX_LABEL_LENGTH),
            title=_bounded_required_mapping_text(payload, field_name="title", max_length=_MAX_TITLE_LENGTH),
            link=payload.get("link"),
            published_at=payload.get("published_at"),
            item_id=payload.get("item_id") or payload.get("guid"),
            content_fingerprint=payload.get("content_fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class WorldFeedSubscription:
    """Persist one ongoing RSS/Atom subscription for world awareness."""

    subscription_id: str
    label: str
    feed_url: str
    scope: str = "topic"
    region: str | None = None
    topics: tuple[str, ...] = ()
    priority: float = 0.6
    base_priority: float | None = None
    active: bool = True
    refresh_interval_hours: int = _DEFAULT_REFRESH_INTERVAL_HOURS
    base_refresh_interval_hours: int | None = None
    source_page_url: str | None = None
    source_title: str | None = None
    created_by: str = "installer"
    created_at: str | None = None
    updated_at: str | None = None
    last_checked_at: str | None = None
    last_refreshed_at: str | None = None
    last_error: str | None = None
    last_item_ids: tuple[str, ...] = ()
    last_item_fingerprints: tuple[str, ...] = ()
    http_etag: str | None = None
    http_last_modified: str | None = None
    last_http_status: int | None = None
    redirected_feed_url: str | None = None
    next_refresh_after: str | None = None
    backoff_until: str | None = None
    consecutive_error_count: int = 0
    fetch_timeout_seconds: int = _DEFAULT_FETCH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Normalize one persisted feed subscription."""

        object.__setattr__(
            self,
            "subscription_id",
            _bounded_required_text(self.subscription_id, field_name="subscription_id", max_length=_MAX_ID_LENGTH),
        )
        object.__setattr__(self, "label", _bounded_required_text(self.label, field_name="label", max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(
            self,
            "feed_url",
            _canonicalize_url(self.feed_url, field_name="feed_url", allow_private_host=False),
        )
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(
            self,
            "region",
            _bounded_optional_text(self.region, field_name="region", max_length=_MAX_SHORT_TEXT_LENGTH),
        )
        object.__setattr__(
            self,
            "topics",
            _bounded_string_tuple(self.topics, field_name="topics", max_items=_MAX_TOPICS),
        )
        normalized_priority = _bounded_score(self.priority, field_name="priority", default=0.6)
        object.__setattr__(self, "priority", normalized_priority)
        object.__setattr__(
            self,
            "base_priority",
            _bounded_score(
                normalized_priority if self.base_priority is None else self.base_priority,
                field_name="base_priority",
                default=normalized_priority,
            ),
        )
        object.__setattr__(self, "active", bool(self.active))
        normalized_refresh_interval_hours = _normalize_int(
            self.refresh_interval_hours,
            field_name="refresh_interval_hours",
            default=_DEFAULT_REFRESH_INTERVAL_HOURS,
            minimum=1,
        )
        object.__setattr__(
            self,
            "refresh_interval_hours",
            normalized_refresh_interval_hours,
        )
        object.__setattr__(
            self,
            "base_refresh_interval_hours",
            _normalize_int(
                normalized_refresh_interval_hours
                if self.base_refresh_interval_hours is None
                else self.base_refresh_interval_hours,
                field_name="base_refresh_interval_hours",
                default=normalized_refresh_interval_hours,
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "source_page_url",
            _canonicalize_optional_url(self.source_page_url, field_name="source_page_url", allow_private_host=False),
        )
        object.__setattr__(
            self,
            "source_title",
            _bounded_optional_text(
                self.source_title,
                field_name="source_title",
                max_length=_MAX_LABEL_LENGTH,
                truncate=True,
            ),
        )
        object.__setattr__(
            self,
            "created_by",
            _bounded_required_text(self.created_by, field_name="created_by", max_length=_MAX_SHORT_TEXT_LENGTH),
        )
        object.__setattr__(self, "created_at", _normalize_timestamp(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "last_checked_at",
            _normalize_timestamp(self.last_checked_at, field_name="last_checked_at"),
        )
        object.__setattr__(
            self,
            "last_refreshed_at",
            _normalize_timestamp(self.last_refreshed_at, field_name="last_refreshed_at"),
        )
        object.__setattr__(
            self,
            "last_error",
            _bounded_optional_text(
                self.last_error,
                field_name="last_error",
                max_length=_MAX_ERROR_LENGTH,
                truncate=True,
            ),
        )
        object.__setattr__(
            self,
            "last_item_ids",
            _bounded_string_tuple(
                self.last_item_ids,
                field_name="last_item_ids",
                max_items=_MAX_LAST_ITEM_IDS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "last_item_fingerprints",
            _bounded_string_tuple(
                self.last_item_fingerprints,
                field_name="last_item_fingerprints",
                max_items=_MAX_LAST_ITEM_FINGERPRINTS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "http_etag",
            _bounded_optional_text(self.http_etag, field_name="http_etag", max_length=_MAX_ETAG_LENGTH),
        )
        object.__setattr__(
            self,
            "http_last_modified",
            _bounded_optional_text(
                self.http_last_modified,
                field_name="http_last_modified",
                max_length=_MAX_HTTP_HEADER_VALUE_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "last_http_status",
            _normalize_http_status(self.last_http_status, field_name="last_http_status"),
        )
        object.__setattr__(
            self,
            "redirected_feed_url",
            _canonicalize_optional_url(
                self.redirected_feed_url,
                field_name="redirected_feed_url",
                allow_private_host=False,
            ),
        )
        object.__setattr__(
            self,
            "next_refresh_after",
            _normalize_timestamp(self.next_refresh_after, field_name="next_refresh_after"),
        )
        object.__setattr__(
            self,
            "backoff_until",
            _normalize_timestamp(self.backoff_until, field_name="backoff_until"),
        )
        object.__setattr__(
            self,
            "consecutive_error_count",
            _normalize_int(
                self.consecutive_error_count,
                field_name="consecutive_error_count",
                default=0,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "fetch_timeout_seconds",
            _normalize_int(
                self.fetch_timeout_seconds,
                field_name="fetch_timeout_seconds",
                default=_DEFAULT_FETCH_TIMEOUT_SECONDS,
                minimum=1,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the subscription into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "subscription_id": self.subscription_id,
            "label": self.label,
            "feed_url": self.feed_url,
            "scope": self.scope,
            "topics": list(self.topics),
            "priority": self.priority,
            "base_priority": self.base_priority,
            "active": self.active,
            "refresh_interval_hours": self.refresh_interval_hours,
            "base_refresh_interval_hours": self.base_refresh_interval_hours,
            "created_by": self.created_by,
            "last_item_ids": list(self.last_item_ids),
            "last_item_fingerprints": list(self.last_item_fingerprints),
            "consecutive_error_count": self.consecutive_error_count,
            "fetch_timeout_seconds": self.fetch_timeout_seconds,
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.source_page_url is not None:
            payload["source_page_url"] = self.source_page_url
        if self.source_title is not None:
            payload["source_title"] = self.source_title
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.last_checked_at is not None:
            payload["last_checked_at"] = self.last_checked_at
        if self.last_refreshed_at is not None:
            payload["last_refreshed_at"] = self.last_refreshed_at
        if self.last_error is not None:
            payload["last_error"] = self.last_error
        if self.http_etag is not None:
            payload["http_etag"] = self.http_etag
        if self.http_last_modified is not None:
            payload["http_last_modified"] = self.http_last_modified
        if self.last_http_status is not None:
            payload["last_http_status"] = self.last_http_status
        if self.redirected_feed_url is not None:
            payload["redirected_feed_url"] = self.redirected_feed_url
        if self.next_refresh_after is not None:
            payload["next_refresh_after"] = self.next_refresh_after
        if self.backoff_until is not None:
            payload["backoff_until"] = self.backoff_until
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldFeedSubscription":
        """Build a subscription from one snapshot payload item."""

        return cls(
            subscription_id=_bounded_required_mapping_text(
                payload,
                field_name="subscription_id",
                max_length=_MAX_ID_LENGTH,
                aliases=("id",),
            ),
            label=_bounded_required_mapping_text(
                payload,
                field_name="label",
                max_length=_MAX_LABEL_LENGTH,
                aliases=("title",),
            ),
            feed_url=_bounded_required_mapping_text(payload, field_name="feed_url", max_length=_MAX_URL_LENGTH),
            scope=payload.get("scope", "topic"),
            region=payload.get("region"),
            topics=payload.get("topics"),
            priority=payload.get("priority"),
            base_priority=payload.get("base_priority"),
            active=payload.get("active", True),
            refresh_interval_hours=payload.get("refresh_interval_hours", _DEFAULT_REFRESH_INTERVAL_HOURS),
            base_refresh_interval_hours=payload.get("base_refresh_interval_hours"),
            source_page_url=payload.get("source_page_url"),
            source_title=payload.get("source_title"),
            created_by=_clean_text(payload.get("created_by")) or "installer",
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            last_checked_at=payload.get("last_checked_at"),
            last_refreshed_at=payload.get("last_refreshed_at"),
            last_error=payload.get("last_error"),
            last_item_ids=payload.get("last_item_ids"),
            last_item_fingerprints=payload.get("last_item_fingerprints"),
            http_etag=payload.get("http_etag") or payload.get("etag"),
            http_last_modified=payload.get("http_last_modified") or payload.get("last_modified"),
            last_http_status=payload.get("last_http_status") or payload.get("status"),
            redirected_feed_url=payload.get("redirected_feed_url"),
            next_refresh_after=payload.get("next_refresh_after"),
            backoff_until=payload.get("backoff_until"),
            consecutive_error_count=payload.get("consecutive_error_count", 0),
            fetch_timeout_seconds=payload.get("fetch_timeout_seconds", _DEFAULT_FETCH_TIMEOUT_SECONDS),
        )


@dataclass(frozen=True, slots=True)
class WorldInterestSignal:
    """Capture one slowly learned topic/region interest for source calibration.

    ``engagement_score`` remains the compact bounded numeric summary used by
    ranking paths, while ``engagement_state`` captures the coarser Twinr policy
    stance:

    - ``resonant`` / ``warm`` mean the topic still actively pulls the user in
    - ``uncertain`` means Twinr should stay lightweight and observant
    - ``cooling`` means repeated exposure is not turning into renewed uptake
    - ``avoid`` means Twinr should actively back off

    Two slower derivative layers make the policy reusable across prompting and
    RSS calibration:

    - ``ongoing_interest`` says whether the topic currently feels active,
      growing, or merely peripheral in Twinr's continuing attention
    - ``co_attention`` says whether the topic is still latent, is becoming a
      shared thread, or has become a durable shared running topic between the
      user and Twinr's world-intelligence layer
    """

    signal_id: str
    topic: str
    summary: str
    region: str | None = None
    scope: str = "topic"
    salience: float = 0.5
    confidence: float = 0.5
    engagement_score: float = 0.5
    engagement_state: str | None = None
    ongoing_interest_score: float | None = None
    ongoing_interest: str | None = None
    co_attention_score: float | None = None
    co_attention_state: str | None = None
    co_attention_count: int = 0
    evidence_count: int = 1
    # BREAKING: fresh signals now start neutral instead of implicitly warm.
    engagement_count: int = 0
    positive_signal_count: int = 0
    exposure_count: int = 0
    non_reengagement_count: int = 0
    deflection_count: int = 0
    explicit: bool = False
    source_event_ids: tuple[str, ...] = ()
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one world-interest signal."""

        object.__setattr__(
            self,
            "signal_id",
            _bounded_required_text(self.signal_id, field_name="signal_id", max_length=_MAX_ID_LENGTH),
        )
        object.__setattr__(self, "topic", _bounded_required_text(self.topic, field_name="topic", max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(
            self,
            "summary",
            _bounded_required_text(self.summary, field_name="summary", max_length=_MAX_SUMMARY_LENGTH),
        )
        object.__setattr__(
            self,
            "region",
            _bounded_optional_text(self.region, field_name="region", max_length=_MAX_SHORT_TEXT_LENGTH),
        )
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "salience", _bounded_score(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "confidence", _bounded_score(self.confidence, field_name="confidence", default=0.5))
        object.__setattr__(
            self,
            "engagement_score",
            _bounded_score(self.engagement_score, field_name="engagement_score", default=0.5),
        )
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "engagement_count",
            _normalize_int(self.engagement_count, field_name="engagement_count", default=0, minimum=0),
        )
        object.__setattr__(
            self,
            "positive_signal_count",
            _normalize_int(self.positive_signal_count, field_name="positive_signal_count", default=0, minimum=0),
        )
        object.__setattr__(
            self,
            "exposure_count",
            _normalize_int(self.exposure_count, field_name="exposure_count", default=0, minimum=0),
        )
        object.__setattr__(
            self,
            "non_reengagement_count",
            _normalize_int(
                self.non_reengagement_count,
                field_name="non_reengagement_count",
                default=0,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "deflection_count",
            _normalize_int(self.deflection_count, field_name="deflection_count", default=0, minimum=0),
        )
        object.__setattr__(self, "explicit", bool(self.explicit))
        normalized_state = _optional_text(self.engagement_state)
        if normalized_state is not None:
            normalized_state = normalized_state.casefold()
            if normalized_state not in _ALLOWED_ENGAGEMENT_STATES:
                raise ValueError(f"engagement_state must be one of {sorted(_ALLOWED_ENGAGEMENT_STATES)}.")
        else:
            normalized_state = _derive_engagement_state(
                engagement_score=self.engagement_score,
                engagement_count=self.engagement_count,
                positive_signal_count=self.positive_signal_count,
                exposure_count=self.exposure_count,
                non_reengagement_count=self.non_reengagement_count,
                deflection_count=self.deflection_count,
                explicit=self.explicit,
            )
        object.__setattr__(self, "engagement_state", normalized_state)
        derived_ongoing_interest_score = _derive_ongoing_interest_score(
            salience=self.salience,
            engagement_score=self.engagement_score,
            engagement_state=normalized_state,
            engagement_count=self.engagement_count,
            positive_signal_count=self.positive_signal_count,
            non_reengagement_count=self.non_reengagement_count,
            deflection_count=self.deflection_count,
            explicit=self.explicit,
        )
        object.__setattr__(
            self,
            "ongoing_interest_score",
            _bounded_score(
                derived_ongoing_interest_score if self.ongoing_interest_score is None else self.ongoing_interest_score,
                field_name="ongoing_interest_score",
                default=derived_ongoing_interest_score,
            ),
        )
        normalized_ongoing_interest = _optional_text(self.ongoing_interest)
        if normalized_ongoing_interest is not None:
            normalized_ongoing_interest = normalized_ongoing_interest.casefold()
            if normalized_ongoing_interest not in _ALLOWED_ONGOING_INTEREST_STATES:
                raise ValueError(
                    f"ongoing_interest must be one of {sorted(_ALLOWED_ONGOING_INTEREST_STATES)}."
                )
        else:
            normalized_ongoing_interest = _derive_ongoing_interest_state(
                ongoing_interest_score=self.ongoing_interest_score,
                engagement_state=normalized_state,
                engagement_count=self.engagement_count,
                positive_signal_count=self.positive_signal_count,
                non_reengagement_count=self.non_reengagement_count,
                deflection_count=self.deflection_count,
            )
        object.__setattr__(self, "ongoing_interest", normalized_ongoing_interest)
        object.__setattr__(
            self,
            "co_attention_count",
            _normalize_int(
                self.co_attention_count,
                field_name="co_attention_count",
                default=0,
                minimum=0,
            ),
        )
        derived_co_attention_score = _derive_co_attention_score(
            ongoing_interest_score=self.ongoing_interest_score,
            ongoing_interest=self.ongoing_interest,
            engagement_state=normalized_state,
            co_attention_count=self.co_attention_count,
            non_reengagement_count=self.non_reengagement_count,
            deflection_count=self.deflection_count,
        )
        object.__setattr__(
            self,
            "co_attention_score",
            _bounded_score(
                derived_co_attention_score if self.co_attention_score is None else self.co_attention_score,
                field_name="co_attention_score",
                default=derived_co_attention_score,
            ),
        )
        normalized_co_attention_state = _optional_text(self.co_attention_state)
        if normalized_co_attention_state is not None:
            normalized_co_attention_state = normalized_co_attention_state.casefold()
            if normalized_co_attention_state not in _ALLOWED_CO_ATTENTION_STATES:
                raise ValueError(
                    f"co_attention_state must be one of {sorted(_ALLOWED_CO_ATTENTION_STATES)}."
                )
        else:
            normalized_co_attention_state = _derive_co_attention_state(
                co_attention_score=self.co_attention_score,
                co_attention_count=self.co_attention_count,
                ongoing_interest=self.ongoing_interest,
                engagement_state=normalized_state,
            )
        object.__setattr__(self, "co_attention_state", normalized_co_attention_state)
        object.__setattr__(
            self,
            "source_event_ids",
            _bounded_string_tuple(
                self.source_event_ids,
                field_name="source_event_ids",
                max_items=_MAX_SOURCE_EVENT_IDS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))

    def to_payload(self) -> dict[str, object]:
        """Serialize one calibration signal into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "topic": self.topic,
            "summary": self.summary,
            "scope": self.scope,
            "salience": self.salience,
            "confidence": self.confidence,
            "engagement_score": self.engagement_score,
            "engagement_state": self.engagement_state,
            "ongoing_interest_score": self.ongoing_interest_score,
            "ongoing_interest": self.ongoing_interest,
            "co_attention_score": self.co_attention_score,
            "co_attention_state": self.co_attention_state,
            "co_attention_count": self.co_attention_count,
            "evidence_count": self.evidence_count,
            "engagement_count": self.engagement_count,
            "positive_signal_count": self.positive_signal_count,
            "exposure_count": self.exposure_count,
            "non_reengagement_count": self.non_reengagement_count,
            "deflection_count": self.deflection_count,
            "explicit": self.explicit,
            "source_event_ids": list(self.source_event_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldInterestSignal":
        """Build one calibration signal from stored payload data."""

        legacy_evidence_count = payload.get("evidence_count")
        legacy_engagement_count = (
            payload.get("engagement_count")
            if "engagement_count" in payload
            else legacy_evidence_count
        )
        positive_signal_count = (
            payload.get("positive_signal_count")
            if "positive_signal_count" in payload
            else (legacy_engagement_count if legacy_engagement_count is not None else 0)
        )
        engagement_score = (
            payload.get("engagement_score")
            if "engagement_score" in payload
            else _legacy_default_engagement_score(payload)
        )
        engagement_count = (
            payload.get("engagement_count")
            if "engagement_count" in payload
            else (legacy_engagement_count if legacy_engagement_count is not None else 0)
        )
        exposure_count = payload.get("exposure_count")
        if exposure_count is None:
            fallback_positive_count = _normalize_int(
                positive_signal_count,
                field_name="positive_signal_count",
                default=0,
                minimum=0,
            )
            exposure_count = max(0, fallback_positive_count)
        return cls(
            signal_id=_bounded_required_mapping_text(
                payload,
                field_name="signal_id",
                max_length=_MAX_ID_LENGTH,
                aliases=("id",),
            ),
            topic=_bounded_required_mapping_text(payload, field_name="topic", max_length=_MAX_LABEL_LENGTH),
            summary=_bounded_required_mapping_text(
                payload,
                field_name="summary",
                max_length=_MAX_SUMMARY_LENGTH,
                aliases=("description",),
            ),
            region=payload.get("region"),
            scope=payload.get("scope", "topic"),
            salience=payload.get("salience"),
            confidence=payload.get("confidence"),
            engagement_score=engagement_score,
            engagement_state=payload.get("engagement_state"),
            ongoing_interest_score=payload.get("ongoing_interest_score"),
            ongoing_interest=payload.get("ongoing_interest"),
            co_attention_score=payload.get("co_attention_score"),
            co_attention_state=payload.get("co_attention_state"),
            co_attention_count=payload.get("co_attention_count", 0),
            evidence_count=payload.get("evidence_count"),
            engagement_count=engagement_count,
            positive_signal_count=positive_signal_count,
            exposure_count=exposure_count,
            non_reengagement_count=payload.get("non_reengagement_count", 0),
            deflection_count=payload.get("deflection_count", 0),
            explicit=payload.get("explicit", False),
            source_event_ids=payload.get("source_event_ids"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True, slots=True)
class SituationalAwarenessThread:
    """Persist one condensed place/world thread built from repeated feed updates."""

    thread_id: str
    title: str
    summary: str
    topic: str
    region: str | None = None
    scope: str = "topic"
    salience: float = 0.5
    update_count: int = 1
    recent_titles: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    supporting_item_ids: tuple[str, ...] = ()
    updated_at: str | None = None
    review_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one awareness thread payload."""

        object.__setattr__(self, "thread_id", _bounded_required_text(self.thread_id, field_name="thread_id", max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "title", _bounded_required_text(self.title, field_name="title", max_length=_MAX_TITLE_LENGTH))
        object.__setattr__(
            self,
            "summary",
            _bounded_required_text(self.summary, field_name="summary", max_length=_MAX_SUMMARY_LENGTH),
        )
        object.__setattr__(self, "topic", _bounded_required_text(self.topic, field_name="topic", max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(
            self,
            "region",
            _bounded_optional_text(self.region, field_name="region", max_length=_MAX_SHORT_TEXT_LENGTH),
        )
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "salience", _bounded_score(self.salience, field_name="salience", default=0.5))
        object.__setattr__(
            self,
            "update_count",
            _normalize_int(self.update_count, field_name="update_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "recent_titles",
            _bounded_string_tuple(
                self.recent_titles,
                field_name="recent_titles",
                max_items=_MAX_RECENT_TITLES,
                max_item_length=_MAX_TITLE_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "source_labels",
            _bounded_string_tuple(
                self.source_labels,
                field_name="source_labels",
                max_items=_MAX_SOURCE_LABELS,
                max_item_length=_MAX_LABEL_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "supporting_item_ids",
            _bounded_string_tuple(
                self.supporting_item_ids,
                field_name="supporting_item_ids",
                max_items=_MAX_SUPPORTING_ITEM_IDS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(self, "review_at", _normalize_timestamp(self.review_at, field_name="review_at"))

    def to_payload(self) -> dict[str, object]:
        """Serialize one awareness thread into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "thread_id": self.thread_id,
            "title": self.title,
            "summary": self.summary,
            "topic": self.topic,
            "scope": self.scope,
            "salience": self.salience,
            "update_count": self.update_count,
            "recent_titles": list(self.recent_titles),
            "source_labels": list(self.source_labels),
            "supporting_item_ids": list(self.supporting_item_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.review_at is not None:
            payload["review_at"] = self.review_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SituationalAwarenessThread":
        """Build one awareness thread from stored payload data."""

        return cls(
            thread_id=_bounded_required_mapping_text(
                payload,
                field_name="thread_id",
                max_length=_MAX_ID_LENGTH,
                aliases=("id",),
            ),
            title=_bounded_required_mapping_text(
                payload,
                field_name="title",
                max_length=_MAX_TITLE_LENGTH,
                aliases=("name",),
            ),
            summary=_bounded_required_mapping_text(
                payload,
                field_name="summary",
                max_length=_MAX_SUMMARY_LENGTH,
                aliases=("description",),
            ),
            topic=_bounded_required_mapping_text(payload, field_name="topic", max_length=_MAX_LABEL_LENGTH),
            region=payload.get("region"),
            scope=payload.get("scope", "topic"),
            salience=payload.get("salience"),
            update_count=payload.get("update_count"),
            recent_titles=payload.get("recent_titles"),
            source_labels=payload.get("source_labels"),
            supporting_item_ids=payload.get("supporting_item_ids"),
            updated_at=payload.get("updated_at"),
            review_at=payload.get("review_at"),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceState:
    """Persist global timing, calibration, and awareness state."""

    schema_version: int = DEFAULT_WORLD_INTELLIGENCE_SCHEMA_VERSION
    last_discovered_at: str | None = None
    last_refreshed_at: str | None = None
    last_recalibrated_at: str | None = None
    discovery_interval_hours: int = _DEFAULT_DISCOVERY_INTERVAL_HOURS
    recalibration_interval_hours: int = _DEFAULT_RECALIBRATION_INTERVAL_HOURS
    last_discovery_query: str | None = None
    interest_signals: tuple[WorldInterestSignal, ...] = ()
    awareness_threads: tuple[SituationalAwarenessThread, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the global state snapshot."""

        schema_version = _normalize_int(
            self.schema_version,
            field_name="schema_version",
            default=DEFAULT_WORLD_INTELLIGENCE_SCHEMA_VERSION,
            minimum=1,
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "last_discovered_at", _normalize_timestamp(self.last_discovered_at, field_name="last_discovered_at"))
        object.__setattr__(self, "last_refreshed_at", _normalize_timestamp(self.last_refreshed_at, field_name="last_refreshed_at"))
        object.__setattr__(
            self,
            "last_recalibrated_at",
            _normalize_timestamp(self.last_recalibrated_at, field_name="last_recalibrated_at"),
        )
        object.__setattr__(
            self,
            "discovery_interval_hours",
            _normalize_int(
                self.discovery_interval_hours,
                field_name="discovery_interval_hours",
                default=_DEFAULT_DISCOVERY_INTERVAL_HOURS,
                minimum=24,
            ),
        )
        object.__setattr__(
            self,
            "recalibration_interval_hours",
            _normalize_int(
                self.recalibration_interval_hours,
                field_name="recalibration_interval_hours",
                default=_DEFAULT_RECALIBRATION_INTERVAL_HOURS,
                minimum=24,
            ),
        )
        object.__setattr__(
            self,
            "last_discovery_query",
            _bounded_optional_text(
                self.last_discovery_query,
                field_name="last_discovery_query",
                max_length=_MAX_QUERY_LENGTH,
                truncate=True,
            ),
        )
        interest_signals = _coerce_tuple_of(
            self.interest_signals,
            field_name="interest_signals",
            item_type=WorldInterestSignal,
            builder=WorldInterestSignal.from_payload,
        )
        awareness_threads = _coerce_tuple_of(
            self.awareness_threads,
            field_name="awareness_threads",
            item_type=SituationalAwarenessThread,
            builder=SituationalAwarenessThread.from_payload,
        )
        object.__setattr__(
            self,
            "interest_signals",
            _ensure_unique_attr(interest_signals, field_name="interest_signals", attr_name="signal_id"),
        )
        object.__setattr__(
            self,
            "awareness_threads",
            _ensure_unique_attr(awareness_threads, field_name="awareness_threads", attr_name="thread_id"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the state snapshot into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "discovery_interval_hours": self.discovery_interval_hours,
            "recalibration_interval_hours": self.recalibration_interval_hours,
            "interest_signals": [item.to_payload() for item in self.interest_signals],
            "awareness_threads": [item.to_payload() for item in self.awareness_threads],
        }
        if self.last_discovered_at is not None:
            payload["last_discovered_at"] = self.last_discovered_at
        if self.last_refreshed_at is not None:
            payload["last_refreshed_at"] = self.last_refreshed_at
        if self.last_recalibrated_at is not None:
            payload["last_recalibrated_at"] = self.last_recalibrated_at
        if self.last_discovery_query is not None:
            payload["last_discovery_query"] = self.last_discovery_query
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldIntelligenceState":
        """Build a global state snapshot from one payload mapping."""

        return cls(
            schema_version=payload.get("schema_version", DEFAULT_WORLD_INTELLIGENCE_SCHEMA_VERSION),
            last_discovered_at=payload.get("last_discovered_at"),
            last_refreshed_at=payload.get("last_refreshed_at"),
            last_recalibrated_at=payload.get("last_recalibrated_at"),
            discovery_interval_hours=payload.get("discovery_interval_hours"),
            recalibration_interval_hours=payload.get("recalibration_interval_hours"),
            last_discovery_query=payload.get("last_discovery_query"),
            interest_signals=tuple(
                WorldInterestSignal.from_payload(item)
                for item in _mapping_items(payload.get("interest_signals"), field_name="interest_signals")
            ),
            awareness_threads=tuple(
                SituationalAwarenessThread.from_payload(item)
                for item in _mapping_items(payload.get("awareness_threads"), field_name="awareness_threads")
            ),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceConfigRequest:
    """Describe one installer/tool request that changes world intelligence."""

    action: str
    query: str | None = None
    label: str | None = None
    location_hint: str | None = None
    region: str | None = None
    topics: tuple[str, ...] = ()
    feed_urls: tuple[str, ...] = ()
    subscription_refs: tuple[str, ...] = ()
    scope: str = "topic"
    priority: float = 0.6
    refresh_interval_hours: int = _DEFAULT_REFRESH_INTERVAL_HOURS
    auto_subscribe: bool = True
    refresh_after_change: bool = False
    created_by: str = "tool"

    def __post_init__(self) -> None:
        """Normalize one config request before execution."""

        normalized_action = _clean_text(self.action).casefold()
        if normalized_action not in _ALLOWED_WORLD_ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ALLOWED_WORLD_ACTIONS)}.")
        object.__setattr__(self, "action", normalized_action)
        object.__setattr__(
            self,
            "query",
            _bounded_optional_text(self.query, field_name="query", max_length=_MAX_QUERY_LENGTH, truncate=True),
        )
        object.__setattr__(
            self,
            "label",
            _bounded_optional_text(self.label, field_name="label", max_length=_MAX_LABEL_LENGTH, truncate=True),
        )
        object.__setattr__(
            self,
            "location_hint",
            _bounded_optional_text(
                self.location_hint,
                field_name="location_hint",
                max_length=_MAX_LABEL_LENGTH,
                truncate=True,
            ),
        )
        object.__setattr__(
            self,
            "region",
            _bounded_optional_text(self.region, field_name="region", max_length=_MAX_SHORT_TEXT_LENGTH),
        )
        object.__setattr__(
            self,
            "topics",
            _bounded_string_tuple(self.topics, field_name="topics", max_items=_MAX_TOPICS),
        )
        canonical_feed_url_items = _bounded_string_tuple(
            self.feed_urls,
            field_name="feed_urls",
            max_items=_MAX_FEED_URLS,
            max_item_length=_MAX_URL_LENGTH,
            truncate=False,
        )
        canonical_feed_urls: list[str] = []
        seen_feed_urls: set[str] = set()
        for item in canonical_feed_url_items:
            canonical = _canonicalize_url(item, field_name="feed_urls", allow_private_host=False)
            if canonical in seen_feed_urls:
                continue
            canonical_feed_urls.append(canonical)
            seen_feed_urls.add(canonical)
        object.__setattr__(self, "feed_urls", tuple(canonical_feed_urls))
        object.__setattr__(
            self,
            "subscription_refs",
            _bounded_string_tuple(
                self.subscription_refs,
                field_name="subscription_refs",
                max_items=_MAX_SUBSCRIPTION_REFS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "priority", _bounded_score(self.priority, field_name="priority", default=0.6))
        object.__setattr__(
            self,
            "refresh_interval_hours",
            _normalize_int(
                self.refresh_interval_hours,
                field_name="refresh_interval_hours",
                default=_DEFAULT_REFRESH_INTERVAL_HOURS,
                minimum=1,
            ),
        )
        object.__setattr__(self, "auto_subscribe", bool(self.auto_subscribe))
        object.__setattr__(self, "refresh_after_change", bool(self.refresh_after_change))
        object.__setattr__(
            self,
            "created_by",
            _bounded_required_text(self.created_by, field_name="created_by", max_length=_MAX_SHORT_TEXT_LENGTH),
        )

        if self.action == "subscribe" and not self.feed_urls and self.query is None:
            raise ValueError("subscribe requires feed_urls or query.")
        if self.action == "discover" and self.query is None and not self.topics and self.region is None and self.location_hint is None:
            raise ValueError("discover requires query, topics, region, or location_hint.")
        if self.action in {"deactivate", "refresh_now"} and not self.subscription_refs and not self.feed_urls:
            raise ValueError(f"{self.action} requires subscription_refs or feed_urls.")

    def to_payload(self) -> dict[str, object]:
        """Serialize the config request into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "action": self.action,
            "topics": list(self.topics),
            "feed_urls": list(self.feed_urls),
            "subscription_refs": list(self.subscription_refs),
            "scope": self.scope,
            "priority": self.priority,
            "refresh_interval_hours": self.refresh_interval_hours,
            "auto_subscribe": self.auto_subscribe,
            "refresh_after_change": self.refresh_after_change,
            "created_by": self.created_by,
        }
        if self.query is not None:
            payload["query"] = self.query
        if self.label is not None:
            payload["label"] = self.label
        if self.location_hint is not None:
            payload["location_hint"] = self.location_hint
        if self.region is not None:
            payload["region"] = self.region
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldIntelligenceConfigRequest":
        """Build one config request from a payload mapping."""

        return cls(
            action=_bounded_required_mapping_text(payload, field_name="action", max_length=_MAX_SHORT_TEXT_LENGTH),
            query=payload.get("query"),
            label=payload.get("label"),
            location_hint=payload.get("location_hint"),
            region=payload.get("region"),
            topics=payload.get("topics"),
            feed_urls=payload.get("feed_urls"),
            subscription_refs=payload.get("subscription_refs"),
            scope=payload.get("scope", "topic"),
            priority=payload.get("priority", 0.6),
            refresh_interval_hours=payload.get("refresh_interval_hours", _DEFAULT_REFRESH_INTERVAL_HOURS),
            auto_subscribe=payload.get("auto_subscribe", True),
            refresh_after_change=payload.get("refresh_after_change", False),
            created_by=payload.get("created_by", "tool"),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceRefreshResult:
    """Return the bounded outcome of one feed refresh pass."""

    status: str
    refreshed: bool
    subscriptions: tuple[WorldFeedSubscription, ...] = ()
    world_signals: tuple[object, ...] = ()
    continuity_threads: tuple[object, ...] = ()
    awareness_threads: tuple[SituationalAwarenessThread, ...] = ()
    refreshed_subscription_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    checked_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one refresh result."""

        object.__setattr__(self, "status", _bounded_required_text(self.status, field_name="status", max_length=_MAX_SHORT_TEXT_LENGTH))
        object.__setattr__(self, "refreshed", bool(self.refreshed))
        subscriptions = _coerce_tuple_of(
            self.subscriptions,
            field_name="subscriptions",
            item_type=WorldFeedSubscription,
            builder=WorldFeedSubscription.from_payload,
        )
        awareness_threads = _coerce_tuple_of(
            self.awareness_threads,
            field_name="awareness_threads",
            item_type=SituationalAwarenessThread,
            builder=SituationalAwarenessThread.from_payload,
        )
        object.__setattr__(
            self,
            "subscriptions",
            _ensure_unique_attr(subscriptions, field_name="subscriptions", attr_name="subscription_id"),
        )
        object.__setattr__(
            self,
            "world_signals",
            _normalize_jsonish_sequence(self.world_signals, field_name="world_signals", max_items=_MAX_WORLD_SIGNALS),
        )
        object.__setattr__(
            self,
            "continuity_threads",
            _normalize_jsonish_sequence(
                self.continuity_threads,
                field_name="continuity_threads",
                max_items=_MAX_CONTINUITY_THREADS,
            ),
        )
        object.__setattr__(
            self,
            "awareness_threads",
            _ensure_unique_attr(awareness_threads, field_name="awareness_threads", attr_name="thread_id"),
        )
        object.__setattr__(
            self,
            "refreshed_subscription_ids",
            _bounded_string_tuple(
                self.refreshed_subscription_ids,
                field_name="refreshed_subscription_ids",
                max_items=_MAX_SUBSCRIPTION_REFS,
                max_item_length=_MAX_ID_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "errors",
            _bounded_string_tuple(
                self.errors,
                field_name="errors",
                max_items=_MAX_ERRORS,
                max_item_length=_MAX_ERROR_LENGTH,
            ),
        )
        object.__setattr__(self, "checked_at", _normalize_timestamp(self.checked_at, field_name="checked_at"))

    def to_payload(self) -> dict[str, object]:
        """Serialize one refresh result into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "status": self.status,
            "refreshed": self.refreshed,
            "subscriptions": [item.to_payload() for item in self.subscriptions],
            "world_signals": [_serialize_payloadish(item) for item in self.world_signals],
            "continuity_threads": [_serialize_payloadish(item) for item in self.continuity_threads],
            "awareness_threads": [item.to_payload() for item in self.awareness_threads],
            "refreshed_subscription_ids": list(self.refreshed_subscription_ids),
            "errors": list(self.errors),
        }
        if self.checked_at is not None:
            payload["checked_at"] = self.checked_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldIntelligenceRefreshResult":
        """Build one refresh result from a payload mapping."""

        return cls(
            status=_bounded_required_mapping_text(payload, field_name="status", max_length=_MAX_SHORT_TEXT_LENGTH),
            refreshed=bool(payload.get("refreshed", False)),
            subscriptions=tuple(
                WorldFeedSubscription.from_payload(item)
                for item in _mapping_items(payload.get("subscriptions"), field_name="subscriptions")
            ),
            world_signals=_normalize_jsonish_sequence(payload.get("world_signals"), field_name="world_signals", max_items=_MAX_WORLD_SIGNALS),
            continuity_threads=_normalize_jsonish_sequence(
                payload.get("continuity_threads"),
                field_name="continuity_threads",
                max_items=_MAX_CONTINUITY_THREADS,
            ),
            awareness_threads=tuple(
                SituationalAwarenessThread.from_payload(item)
                for item in _mapping_items(payload.get("awareness_threads"), field_name="awareness_threads")
            ),
            refreshed_subscription_ids=payload.get("refreshed_subscription_ids"),
            errors=payload.get("errors"),
            checked_at=payload.get("checked_at"),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceConfigResult:
    """Return the outcome of one subscription/discovery mutation request."""

    status: str
    action: str
    subscriptions: tuple[WorldFeedSubscription, ...] = ()
    discovered_feed_urls: tuple[str, ...] = ()
    message: str | None = None
    refresh: WorldIntelligenceRefreshResult | None = None

    def __post_init__(self) -> None:
        """Normalize one config result."""

        object.__setattr__(self, "status", _bounded_required_text(self.status, field_name="status", max_length=_MAX_SHORT_TEXT_LENGTH))
        object.__setattr__(self, "action", _bounded_required_text(self.action, field_name="action", max_length=_MAX_SHORT_TEXT_LENGTH))
        subscriptions = _coerce_tuple_of(
            self.subscriptions,
            field_name="subscriptions",
            item_type=WorldFeedSubscription,
            builder=WorldFeedSubscription.from_payload,
        )
        object.__setattr__(
            self,
            "subscriptions",
            _ensure_unique_attr(subscriptions, field_name="subscriptions", attr_name="subscription_id"),
        )
        discovered_feed_url_items = _bounded_string_tuple(
            self.discovered_feed_urls,
            field_name="discovered_feed_urls",
            max_items=_MAX_FEED_URLS,
            max_item_length=_MAX_URL_LENGTH,
            truncate=False,
        )
        discovered_feed_urls: list[str] = []
        seen_discovered_feed_urls: set[str] = set()
        for item in discovered_feed_url_items:
            canonical = _canonicalize_url(item, field_name="discovered_feed_urls", allow_private_host=False)
            if canonical in seen_discovered_feed_urls:
                continue
            discovered_feed_urls.append(canonical)
            seen_discovered_feed_urls.add(canonical)
        object.__setattr__(self, "discovered_feed_urls", tuple(discovered_feed_urls))
        object.__setattr__(
            self,
            "message",
            _bounded_optional_text(self.message, field_name="message", max_length=_MAX_SUMMARY_LENGTH, truncate=True),
        )
        refresh = self.refresh
        if isinstance(refresh, Mapping):
            refresh = WorldIntelligenceRefreshResult.from_payload(refresh)
        if refresh is not None and not isinstance(refresh, WorldIntelligenceRefreshResult):
            raise ValueError("refresh must be a WorldIntelligenceRefreshResult.")
        object.__setattr__(self, "refresh", refresh)

    def to_payload(self) -> dict[str, object]:
        """Serialize one config result into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "status": self.status,
            "action": self.action,
            "subscriptions": [item.to_payload() for item in self.subscriptions],
            "discovered_feed_urls": list(self.discovered_feed_urls),
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.refresh is not None:
            payload["refresh"] = self.refresh.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldIntelligenceConfigResult":
        """Build one config result from a payload mapping."""

        return cls(
            status=_bounded_required_mapping_text(payload, field_name="status", max_length=_MAX_SHORT_TEXT_LENGTH),
            action=_bounded_required_mapping_text(payload, field_name="action", max_length=_MAX_SHORT_TEXT_LENGTH),
            subscriptions=tuple(
                WorldFeedSubscription.from_payload(item)
                for item in _mapping_items(payload.get("subscriptions"), field_name="subscriptions")
            ),
            discovered_feed_urls=payload.get("discovered_feed_urls"),
            message=payload.get("message"),
            refresh=(
                WorldIntelligenceRefreshResult.from_payload(payload["refresh"])
                if isinstance(payload.get("refresh"), Mapping)
                else payload.get("refresh")
            ),
        )


def payload_to_json_bytes(value: _SupportsPayload | Mapping[str, object]) -> bytes:
    """Serialize one payload-bearing object with an optional msgspec fast path."""

    payload = value.to_payload() if hasattr(value, "to_payload") else dict(value)
    if _msgspec is not None:
        return _msgspec.json.encode(payload)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def payload_from_json_bytes(data: bytes | bytearray | memoryview | str) -> dict[str, object]:
    """Parse one JSON payload with an optional msgspec fast path."""

    if _msgspec is not None:
        decoded = _msgspec.json.decode(data)
    else:
        if isinstance(data, (bytes, bytearray, memoryview)):
            decoded = json.loads(bytes(data).decode("utf-8"))
        else:
            decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise ValueError("JSON payload must decode to a mapping.")
    return dict(decoded)


def world_intelligence_state_from_json(data: bytes | bytearray | memoryview | str) -> WorldIntelligenceState:
    """Decode a WorldIntelligenceState from JSON text/bytes."""

    return WorldIntelligenceState.from_payload(payload_from_json_bytes(data))


def world_intelligence_config_request_from_json(
    data: bytes | bytearray | memoryview | str,
) -> WorldIntelligenceConfigRequest:
    """Decode a WorldIntelligenceConfigRequest from JSON text/bytes."""

    return WorldIntelligenceConfigRequest.from_payload(payload_from_json_bytes(data))


def world_intelligence_config_result_from_json(
    data: bytes | bytearray | memoryview | str,
) -> WorldIntelligenceConfigResult:
    """Decode a WorldIntelligenceConfigResult from JSON text/bytes."""

    return WorldIntelligenceConfigResult.from_payload(payload_from_json_bytes(data))


__all__ = [
    "DEFAULT_WORLD_INTELLIGENCE_SCHEMA_VERSION",
    "DEFAULT_WORLD_INTELLIGENCE_STATE_KIND",
    "DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND",
    "SituationalAwarenessThread",
    "WorldFeedItem",
    "WorldFeedSubscription",
    "WorldInterestSignal",
    "WorldIntelligenceConfigRequest",
    "WorldIntelligenceConfigResult",
    "WorldIntelligenceRefreshResult",
    "WorldIntelligenceState",
    "payload_from_json_bytes",
    "payload_to_json_bytes",
    "world_intelligence_config_request_from_json",
    "world_intelligence_config_result_from_json",
    "world_intelligence_state_from_json",
]
