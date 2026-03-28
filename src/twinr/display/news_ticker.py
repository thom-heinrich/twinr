"""Fetch and cache HDMI news-ticker headlines from Twinr's source pool.

The HDMI renderer only needs one calm, readable ticker line. This module keeps
network access, feed parsing, source resolution, caching, and headline rotation
separate from the display loop so the visible screen can stay responsive on the
Raspberry Pi while the ticker still reflects sources Twinr actually follows.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Added exponential retry backoff plus persisted error snapshots so network failures no longer trigger refresh-thread storms and permanent "Loading headlines..." loops.
# BUG-2: Added per-feed conditional HTTP revalidation (ETag / Last-Modified) plus per-feed item caches so 304 responses reuse cached headlines correctly instead of silently dropping feed content.
# SEC-1: Restricted feed fetching to public HTTP(S) endpoints and validated redirect targets to block file:// reads and practical SSRF into localhost / RFC1918 / link-local addresses on Raspberry Pi deployments.
# SEC-2: Hardened XML intake with payload-size caps, gzip/deflate decompression caps, DTD / ENTITY rejection, and an Expat minimum-version gate for known XML parser issues.
# IMP-1: Upgraded parsing to prefer feedparser 6.x when installed for robust RSS/Atom normalization, relative-link resolution, malformed-feed tolerance, and broader date handling, while keeping a stdlib fallback.
# IMP-2: Added bounded parallel feed fetch, source-diverse ranking, fsync-backed atomic cache writes, and richer per-feed cache metadata for lower latency and better Pi resilience.

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import email.utils
import gzip
import html
import inspect
import io
import ipaddress
import json
import os
from pathlib import Path
import pyexpat
import re
import socket
import tempfile
import threading
import time
from time import struct_time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib

from typing import TYPE_CHECKING, Any, Protocol, overload

from twinr.display.news_ticker_sources import (
    DisplayWorldIntelligenceTickerFeedResolver,
)

try:
    import feedparser  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    feedparser = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_STORE_PATH = "artifacts/stores/ops/display_news_ticker.json"
_DEFAULT_USER_AGENT = "TwinrNewsTicker/2.0"
_DEFAULT_ACCEPT = "application/atom+xml, application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1"
_DEFAULT_ACCEPT_ENCODING = "gzip, deflate"
_MAX_FEED_BYTES = 512 * 1024
_MAX_STORE_BYTES = 2 * 1024 * 1024
_MAX_TITLE_CHARS = 220
_MAX_CONCURRENT_FETCHES = 4
_MAX_FEED_ERRORS = 3
_MAX_ITEMS_PER_FEED_CACHE = 24
_MIN_REFRESH_RETRY_S = 15.0
_MAX_REFRESH_RETRY_S = 300.0
_MIN_SAFE_EXPAT = (2, 7, 2)
_MIN_FEEDPARSER_VERSION = (6, 0, 12)
_SCHEMA_VERSION = 2

_WHITESPACE_RE = re.compile(r"\s+")
_DEDUPE_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_XML_DANGEROUS_TOKEN_RE = re.compile(rb"<!DOCTYPE\b|<!ENTITY\b", flags=re.IGNORECASE)


class DisplayNewsTickerSource(Protocol):
    """Describe one bounded source that can build ticker snapshots."""

    def fetch(
        self,
        *,
        now: datetime | None = None,
        previous_snapshot: "DisplayNewsTickerSnapshot | None" = None,
    ) -> "DisplayNewsTickerSnapshot":
        """Return one freshly fetched headline snapshot."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_now(value: datetime | None) -> datetime:
    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", value)]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _assert_safe_expat() -> None:
    # BREAKING: refuse to parse remote XML when the linked Expat is older than 2.7.2,
    # because Python's XML security docs call out older Expat versions as vulnerable.
    if _version_tuple(pyexpat.EXPAT_VERSION) < _MIN_SAFE_EXPAT:
        raise RuntimeError(f"unsafe_expat_{pyexpat.EXPAT_VERSION}")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    compact = _WHITESPACE_RE.sub(" ", html.unescape(value)).strip()
    if len(compact) <= _MAX_TITLE_CHARS:
        return compact
    return compact[: _MAX_TITLE_CHARS - 1].rstrip() + "…"


def _parse_timestamp(value: str | None) -> str | None:
    compact = _clean_text(value)
    if not compact:
        return None
    parsed: datetime | None = None
    try:
        parsed = email.utils.parsedate_to_datetime(compact)
    except (TypeError, ValueError, IndexError, OverflowError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(compact.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_struct_time(value: struct_time | tuple[int, ...] | None) -> str | None:
    if value is None:
        return None
    try:
        parts = tuple(int(part) for part in value[:6])  # type: ignore[index]
    except (TypeError, ValueError):
        return None
    if len(parts) != 6:
        return None
    try:
        return datetime(*parts, tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _timestamp_sort_value(value: str | None) -> float:
    if not value:
        return float("-inf")
    parsed = _parse_timestamp(value)
    if parsed is None:
        return float("-inf")
    try:
        return datetime.fromisoformat(parsed).timestamp()
    except ValueError:
        return float("-inf")


def _dedupe_key(title: str) -> str:
    return _DEDUPE_RE.sub(" ", title.casefold()).strip()


def _feed_source(feed_url: str, feed_title: str) -> str:
    clean_title = _clean_text(feed_title)
    if clean_title:
        return clean_title
    host = urllib.parse.urlsplit(feed_url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host or "Feed"


def _clean_url(value: str | None) -> str | None:
    if not value:
        return None
    compact = _clean_text(value)
    if not compact:
        return None
    return compact


def _canonical_feed_url(feed_url: str) -> str:
    compact = _clean_text(feed_url)
    parsed = urllib.parse.urlsplit(compact)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("unsupported_feed_scheme")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("missing_feed_host")
    if parsed.username or parsed.password:
        # BREAKING: userinfo in feed URLs is rejected to avoid credential leakage and
        # accidental authenticated requests to private infrastructure from the Pi.
        raise ValueError("feed_credentials_forbidden")
    path = parsed.path or "/"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def _is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def _resolved_ip_set(host: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: OrderedDict[str, ipaddress.IPv4Address | ipaddress.IPv6Address] = OrderedDict()
    for _, _, _, _, sockaddr in infos:
        raw_address = sockaddr[0]
        address = ipaddress.ip_address(raw_address)
        addresses[str(address)] = address
    if not addresses:
        raise ValueError("feed_host_resolution_failed")
    return tuple(addresses.values())


def _validate_public_feed_url(feed_url: str) -> str:
    canonical = _canonical_feed_url(feed_url)
    parsed = urllib.parse.urlsplit(canonical)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolved_ip_set(parsed.hostname or "", port)
    if any(not _is_public_ip(address) for address in addresses):
        raise ValueError("non_public_feed_host")
    return canonical


def _guard_xml_payload(payload: bytes) -> None:
    _assert_safe_expat()
    if len(payload) > _MAX_FEED_BYTES:
        raise ValueError("feed_payload_too_large")
    if _XML_DANGEROUS_TOKEN_RE.search(payload):
        raise ValueError("xml_dtd_forbidden")


def _local_name(tag: object) -> str:
    text = str(tag or "")
    if "}" in text:
        return text.rsplit("}", 1)[-1]
    return text


def _child_text(element: ET.Element | None, *names: str) -> str:
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
    if element is None:
        return ()
    wanted = name.lower()
    return tuple(child for child in element if _local_name(child.tag).lower() == wanted)


@dataclass(frozen=True, slots=True)
class DisplayNewsTickerItem:
    title: str
    source: str
    link: str | None = None
    published_at: str | None = None
    feed_url: str | None = None

    def display_text(self, *, include_source: bool) -> str:
        title = _clean_text(self.title)
        source = _clean_text(self.source)
        if include_source and source:
            return f"{source} · {title}"
        return title


@dataclass(frozen=True, slots=True)
class DisplayNewsTickerFeedState:
    url: str
    resolved_url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    last_checked_at: str | None = None
    last_success_at: str | None = None
    consecutive_failures: int = 0
    last_error: str | None = None

    @classmethod
    def from_json(cls, url: str, payload: Mapping[str, object]) -> "DisplayNewsTickerFeedState":
        return cls(
            url=_clean_text(url),
            resolved_url=_clean_url(str(payload.get("resolved_url") or "")),
            etag=_clean_text(str(payload.get("etag") or "")) or None,
            last_modified=_clean_text(str(payload.get("last_modified") or "")) or None,
            last_checked_at=_parse_timestamp(payload.get("last_checked_at") if isinstance(payload.get("last_checked_at"), str) else None),
            last_success_at=_parse_timestamp(payload.get("last_success_at") if isinstance(payload.get("last_success_at"), str) else None),
            consecutive_failures=max(0, int(payload.get("consecutive_failures") or 0)),
            last_error=_clean_text(str(payload.get("last_error") or "")) or None,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "resolved_url": self.resolved_url,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "last_checked_at": self.last_checked_at,
            "last_success_at": self.last_success_at,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class DisplayNewsTickerSnapshot:
    captured_at: str
    items: tuple[DisplayNewsTickerItem, ...]
    feed_urls: tuple[str, ...] = ()
    last_error: str | None = None
    feed_states: dict[str, DisplayNewsTickerFeedState] = field(default_factory=dict)
    feed_item_cache: dict[str, tuple[DisplayNewsTickerItem, ...]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "DisplayNewsTickerSnapshot":
        raw_items = payload.get("items")
        items = cls._decode_items(raw_items)
        raw_urls = payload.get("feed_urls")
        urls = tuple(
            value
            for value in (
                _clean_url(str(raw_value))
                for raw_value in raw_urls
            )
            if value
        ) if isinstance(raw_urls, list) else ()

        raw_feed_states = payload.get("feed_states")
        feed_states: dict[str, DisplayNewsTickerFeedState] = {}
        if isinstance(raw_feed_states, dict):
            for url, raw_state in raw_feed_states.items():
                if not isinstance(url, str) or not isinstance(raw_state, dict):
                    continue
                state = DisplayNewsTickerFeedState.from_json(url, raw_state)
                if state.url:
                    feed_states[state.url] = state

        raw_feed_item_cache = payload.get("feed_item_cache")
        feed_item_cache: dict[str, tuple[DisplayNewsTickerItem, ...]] = {}
        if isinstance(raw_feed_item_cache, dict):
            for url, raw_feed_items in raw_feed_item_cache.items():
                if not isinstance(url, str):
                    continue
                canonical_url = _clean_url(url)
                if not canonical_url:
                    continue
                feed_item_cache[canonical_url] = cls._decode_items(raw_feed_items)

        captured_at = _parse_timestamp(payload.get("captured_at") if isinstance(payload.get("captured_at"), str) else None)
        return cls(
            captured_at=captured_at or _utc_now().isoformat(),
            items=items,
            feed_urls=urls,
            last_error=_clean_text(str(payload.get("last_error") or "")) or None,
            feed_states=feed_states,
            feed_item_cache=feed_item_cache,
        )

    @staticmethod
    def _decode_items(raw_items: object) -> tuple[DisplayNewsTickerItem, ...]:
        items: list[DisplayNewsTickerItem] = []
        if not isinstance(raw_items, list):
            return ()
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
                    link=_clean_url(str(raw_item.get("link") or "")),
                    published_at=_parse_timestamp(raw_item.get("published_at") if isinstance(raw_item.get("published_at"), str) else None),
                    feed_url=_clean_url(str(raw_item.get("feed_url") or "")),
                )
            )
        return tuple(items)

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "captured_at": self.captured_at,
            "feed_urls": list(self.feed_urls),
            "last_error": self.last_error,
            "feed_states": {
                url: state.to_json()
                for url, state in sorted(self.feed_states.items())
            },
            "feed_item_cache": {
                url: [
                    {
                        "title": item.title,
                        "source": item.source,
                        "link": item.link,
                        "published_at": item.published_at,
                        "feed_url": item.feed_url,
                    }
                    for item in items
                ]
                for url, items in sorted(self.feed_item_cache.items())
            },
            "items": [
                {
                    "title": item.title,
                    "source": item.source,
                    "link": item.link,
                    "published_at": item.published_at,
                    "feed_url": item.feed_url,
                }
                for item in self.items
            ],
        }


@dataclass(slots=True)
class DisplayNewsTickerStore:
    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayNewsTickerStore":
        return cls(Path(config.project_root) / config.display_news_ticker_store_path)

    def load(self) -> DisplayNewsTickerSnapshot | None:
        if not self.path.exists():
            return None
        try:
            if self.path.stat().st_size > _MAX_STORE_BYTES:
                raise ValueError("display_news_ticker_cache_too_large")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict):
            return None
        return DisplayNewsTickerSnapshot.from_json(payload)

    def save(self, snapshot: DisplayNewsTickerSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot.to_json(), ensure_ascii=False, indent=2)
        temp_file_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_file_path = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_file_path).replace(self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_file_path is not None:
                try:
                    if Path(temp_file_path).exists():
                        Path(temp_file_path).unlink()
                except OSError:
                    pass


@dataclass(frozen=True, slots=True)
class _DownloadedFeed:
    url: str
    headers: dict[str, str]
    payload: bytes


@dataclass(frozen=True, slots=True)
class _FetchedFeedResult:
    feed_url: str
    items: tuple[DisplayNewsTickerItem, ...]
    state: DisplayNewsTickerFeedState
    error: str | None = None


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], str]) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urljoin(req.full_url, newurl)
        safe_target = self._validator(target)
        return super().redirect_request(req, fp, code, msg, headers, safe_target)


@dataclass(slots=True)
class DisplayNewsTickerFetcher:
    feed_urls: tuple[str, ...]
    timeout_s: float = 4.0
    max_items: int = 12
    user_agent: str = _DEFAULT_USER_AGENT
    max_workers: int = _MAX_CONCURRENT_FETCHES

    def fetch(
        self,
        *,
        now: datetime | None = None,
        previous_snapshot: DisplayNewsTickerSnapshot | None = None,
    ) -> DisplayNewsTickerSnapshot:
        captured_at = _normalize_now(now)
        requested_urls = self._normalized_feed_urls()
        if not requested_urls:
            return DisplayNewsTickerSnapshot(
                captured_at=captured_at.isoformat(),
                items=(),
                feed_urls=(),
                last_error=None,
                feed_states={},
                feed_item_cache={},
            )

        previous_states = previous_snapshot.feed_states if previous_snapshot is not None else {}
        results: list[_FetchedFeedResult] = []
        errors: list[str] = []
        max_workers = max(1, min(int(self.max_workers), _MAX_CONCURRENT_FETCHES, len(requested_urls)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="twinr-news-feed") as executor:
            futures = {
                executor.submit(
                    self._fetch_one_feed,
                    feed_url,
                    previous_snapshot,
                    previous_states.get(feed_url),
                    captured_at,
                ): feed_url
                for feed_url in requested_urls
            }
            for future in as_completed(futures):
                feed_url = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append(f"{feed_url}: {type(exc).__name__}")
                    previous_state = previous_states.get(feed_url)
                    results.append(
                        _FetchedFeedResult(
                            feed_url=feed_url,
                            items=self._items_for_feed(previous_snapshot, feed_url),
                            state=DisplayNewsTickerFeedState(
                                url=feed_url,
                                resolved_url=previous_state.resolved_url if previous_state else None,
                                etag=previous_state.etag if previous_state else None,
                                last_modified=previous_state.last_modified if previous_state else None,
                                last_checked_at=captured_at.isoformat(),
                                last_success_at=previous_state.last_success_at if previous_state else None,
                                consecutive_failures=(previous_state.consecutive_failures if previous_state else 0) + 1,
                                last_error=type(exc).__name__,
                            ),
                            error=type(exc).__name__,
                        )
                    )
                    continue
                if result.error:
                    errors.append(f"{feed_url}: {result.error}")
                results.append(result)

        ordered_results = {result.feed_url: result for result in results}
        flattened_items: list[DisplayNewsTickerItem] = []
        feed_states: dict[str, DisplayNewsTickerFeedState] = {}
        feed_item_cache: dict[str, tuple[DisplayNewsTickerItem, ...]] = {}
        for feed_url in requested_urls:
            result = ordered_results.get(feed_url)
            if result is None:
                continue
            flattened_items.extend(result.items)
            feed_states[feed_url] = result.state
            feed_item_cache[feed_url] = result.items

        ranked_items = self._rank_items(flattened_items)
        limit = max(1, int(self.max_items))
        return DisplayNewsTickerSnapshot(
            captured_at=captured_at.isoformat(),
            items=tuple(ranked_items[:limit]),
            feed_urls=requested_urls,
            last_error=" | ".join(errors[:_MAX_FEED_ERRORS]) or None,
            feed_states=feed_states,
            feed_item_cache=feed_item_cache,
        )

    def _normalized_feed_urls(self) -> tuple[str, ...]:
        deduped: OrderedDict[str, None] = OrderedDict()
        for raw_url in self.feed_urls:
            compact = _clean_url(raw_url)
            if not compact:
                continue
            try:
                canonical = _canonical_feed_url(compact)
            except ValueError:
                canonical = compact
            deduped[canonical] = None
        return tuple(deduped.keys())

    def _fetch_one_feed(
        self,
        feed_url: str,
        previous_snapshot: DisplayNewsTickerSnapshot | None,
        previous_state: DisplayNewsTickerFeedState | None,
        captured_at: datetime,
    ) -> _FetchedFeedResult:
        safe_url = _validate_public_feed_url(feed_url)
        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": _DEFAULT_ACCEPT,
            "Accept-Encoding": _DEFAULT_ACCEPT_ENCODING,
            "Cache-Control": "max-age=0",
        }
        if previous_state is not None and previous_state.etag:
            request_headers["If-None-Match"] = previous_state.etag
        if previous_state is not None and previous_state.last_modified:
            request_headers["If-Modified-Since"] = previous_state.last_modified

        try:
            downloaded = self._download_feed(safe_url, request_headers=request_headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                cached_items = self._items_for_feed(previous_snapshot, safe_url)
                state = DisplayNewsTickerFeedState(
                    url=safe_url,
                    resolved_url=_clean_url(exc.geturl()) or (previous_state.resolved_url if previous_state else safe_url),
                    etag=_clean_text(exc.headers.get("ETag") or "") or (previous_state.etag if previous_state else None),
                    last_modified=_clean_text(exc.headers.get("Last-Modified") or "") or (previous_state.last_modified if previous_state else None),
                    last_checked_at=captured_at.isoformat(),
                    last_success_at=previous_state.last_success_at if previous_state else captured_at.isoformat(),
                    consecutive_failures=0,
                    last_error=None,
                )
                return _FetchedFeedResult(feed_url=safe_url, items=cached_items, state=state, error=None)
            raise

        resolved_url = _validate_public_feed_url(downloaded.url)
        items, parse_error = self._parse_feed(
            downloaded.payload,
            feed_url=safe_url,
            resolved_url=resolved_url,
            response_headers=downloaded.headers,
        )
        ranked_feed_items = self._rank_items(items)[:_MAX_ITEMS_PER_FEED_CACHE]
        if not ranked_feed_items:
            ranked_feed_items = self._items_for_feed(previous_snapshot, safe_url)
        state = DisplayNewsTickerFeedState(
            url=safe_url,
            resolved_url=resolved_url,
            etag=_clean_text(downloaded.headers.get("ETag") or "") or (previous_state.etag if previous_state else None),
            last_modified=_clean_text(downloaded.headers.get("Last-Modified") or "") or (previous_state.last_modified if previous_state else None),
            last_checked_at=captured_at.isoformat(),
            last_success_at=captured_at.isoformat() if ranked_feed_items else (previous_state.last_success_at if previous_state else None),
            consecutive_failures=0 if ranked_feed_items else (previous_state.consecutive_failures + 1 if previous_state else 1),
            last_error=parse_error,
        )
        return _FetchedFeedResult(feed_url=safe_url, items=ranked_feed_items, state=state, error=parse_error)

    def _download_feed(
        self,
        feed_url: str,
        *,
        request_headers: Mapping[str, str],
    ) -> _DownloadedFeed:
        opener = urllib.request.build_opener(_SafeRedirectHandler(_validate_public_feed_url))
        request = urllib.request.Request(feed_url, headers=dict(request_headers), method="GET")
        with opener.open(request, timeout=max(0.5, float(self.timeout_s))) as response:
            raw_payload = response.read(_MAX_FEED_BYTES + 1)
            decoded_payload = self._decode_payload(raw_payload, response.headers.get("Content-Encoding") or "")
            _guard_xml_payload(decoded_payload)
            return _DownloadedFeed(
                url=response.geturl(),
                headers={key: value for key, value in response.headers.items()},
                payload=decoded_payload,
            )

    def _decode_payload(self, raw_payload: bytes, content_encoding: str) -> bytes:
        if len(raw_payload) > _MAX_FEED_BYTES:
            raise ValueError("feed_payload_too_large")
        encodings = [value.strip().lower() for value in content_encoding.split(",") if value.strip()]
        payload = raw_payload
        for encoding in encodings:
            if encoding in {"identity", ""}:
                continue
            if encoding in {"gzip", "x-gzip"}:
                with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
                    payload = handle.read(_MAX_FEED_BYTES + 1)
            elif encoding == "deflate":
                try:
                    payload = zlib.decompress(payload)
                except zlib.error:
                    payload = zlib.decompress(payload, -zlib.MAX_WBITS)
            else:
                raise ValueError("unsupported_content_encoding")
            if len(payload) > _MAX_FEED_BYTES:
                raise ValueError("feed_payload_too_large")
        return payload

    @overload
    def _parse_feed(
        self,
        payload: bytes,
        *,
        feed_url: str,
    ) -> tuple[DisplayNewsTickerItem, ...]: ...

    @overload
    def _parse_feed(
        self,
        payload: bytes,
        *,
        feed_url: str,
        resolved_url: str | None,
        response_headers: Mapping[str, str] | None,
    ) -> tuple[tuple[DisplayNewsTickerItem, ...], str | None]: ...

    def _parse_feed(
        self,
        payload: bytes,
        *,
        feed_url: str,
        resolved_url: str | None = None,
        response_headers: Mapping[str, str] | None = None,
    ) -> tuple[tuple[DisplayNewsTickerItem, ...], str | None] | tuple[DisplayNewsTickerItem, ...]:
        legacy_items_only = resolved_url is None and response_headers is None
        effective_resolved_url = _clean_url(resolved_url) or feed_url
        effective_headers = response_headers or {}
        if feedparser is not None and _version_tuple(str(getattr(feedparser, "__version__", "0"))) >= _MIN_FEEDPARSER_VERSION:
            items, parse_error = self._parse_with_feedparser(
                payload,
                feed_url=feed_url,
                resolved_url=effective_resolved_url,
                response_headers=effective_headers,
            )
            if items:
                result = (items, parse_error)
                return items if legacy_items_only else result
        result = self._parse_with_elementtree(payload, feed_url=feed_url, resolved_url=effective_resolved_url)
        return result[0] if legacy_items_only else result

    def _parse_with_feedparser(
        self,
        payload: bytes,
        *,
        feed_url: str,
        resolved_url: str,
        response_headers: Mapping[str, str],
    ) -> tuple[tuple[DisplayNewsTickerItem, ...], str | None]:
        parser_headers = dict(response_headers)
        parser_headers.setdefault("content-location", resolved_url)
        parsed = feedparser.parse(
            payload,
            response_headers=parser_headers,
            resolve_relative_uris=True,
            sanitize_html=False,
        )
        feed_info = getattr(parsed, "feed", {})
        source = _feed_source(resolved_url, _clean_text(str(feed_info.get("title") or "")))
        items: list[DisplayNewsTickerItem] = []
        for entry in getattr(parsed, "entries", []):
            title = _clean_text(str(entry.get("title") or ""))
            if not title:
                continue
            entry_source = entry.get("source")
            source_title = ""
            if isinstance(entry_source, Mapping):
                source_title = _clean_text(str(entry_source.get("title") or ""))
            items.append(
                DisplayNewsTickerItem(
                    title=title,
                    source=source_title or source,
                    link=self._feedparser_entry_link(entry, resolved_url=resolved_url),
                    published_at=(
                        _parse_struct_time(entry.get("published_parsed"))
                        or _parse_struct_time(entry.get("updated_parsed"))
                    ),
                    feed_url=feed_url,
                )
            )
        parse_error = None
        if getattr(parsed, "bozo", 0):
            parse_error = type(getattr(parsed, "bozo_exception", Exception())).__name__
        return tuple(items), parse_error

    def _feedparser_entry_link(self, entry: Mapping[str, object], *, resolved_url: str) -> str | None:
        scored_links: list[tuple[int, str]] = []
        raw_links = entry.get("links")
        if isinstance(raw_links, list):
            for raw_link in raw_links:
                if not isinstance(raw_link, Mapping):
                    continue
                href = _clean_url(str(raw_link.get("href") or ""))
                if not href:
                    continue
                rel = _clean_text(str(raw_link.get("rel") or "alternate")).casefold()
                mime = _clean_text(str(raw_link.get("type") or "")).casefold()
                score = 0
                if rel not in {"alternate", ""}:
                    score += 10
                if mime and "html" not in mime:
                    score += 1
                scored_links.append((score, urllib.parse.urljoin(resolved_url, href)))
        if scored_links:
            scored_links.sort(key=lambda pair: pair[0])
            return scored_links[0][1]
        direct_link = _clean_url(str(entry.get("link") or ""))
        if direct_link:
            return urllib.parse.urljoin(resolved_url, direct_link)
        return None

    def _parse_with_elementtree(
        self,
        payload: bytes,
        *,
        feed_url: str,
        resolved_url: str,
    ) -> tuple[tuple[DisplayNewsTickerItem, ...], str | None]:
        root = ET.fromstring(payload)
        root_name = _local_name(root.tag).lower()
        if root_name == "rss":
            return self._parse_rss_feed(root, feed_url=feed_url, resolved_url=resolved_url), None
        if root_name == "feed":
            return self._parse_atom_feed(root, feed_url=feed_url, resolved_url=resolved_url), None
        raise ValueError("unsupported_feed_format")

    def _parse_rss_feed(
        self,
        root: ET.Element,
        *,
        feed_url: str,
        resolved_url: str,
    ) -> tuple[DisplayNewsTickerItem, ...]:
        channel = next((child for child in root if _local_name(child.tag).lower() == "channel"), None)
        source = _feed_source(resolved_url, _child_text(channel, "title"))
        items: list[DisplayNewsTickerItem] = []
        for item in _child_elements(channel, "item"):
            title = _child_text(item, "title")
            if not title:
                continue
            raw_link = _child_text(item, "link") or _child_text(item, "guid")
            link = urllib.parse.urljoin(resolved_url, raw_link) if raw_link else None
            items.append(
                DisplayNewsTickerItem(
                    title=title,
                    source=source,
                    link=link,
                    published_at=_parse_timestamp(_child_text(item, "pubDate", "published", "updated")),
                    feed_url=feed_url,
                )
            )
        return tuple(items)

    def _parse_atom_feed(
        self,
        root: ET.Element,
        *,
        feed_url: str,
        resolved_url: str,
    ) -> tuple[DisplayNewsTickerItem, ...]:
        source = _feed_source(resolved_url, _child_text(root, "title"))
        items: list[DisplayNewsTickerItem] = []
        for entry in _child_elements(root, "entry"):
            title = _child_text(entry, "title")
            if not title:
                continue
            items.append(
                DisplayNewsTickerItem(
                    title=title,
                    source=source,
                    link=self._atom_entry_link(entry, resolved_url=resolved_url),
                    published_at=_parse_timestamp(_child_text(entry, "updated", "published")),
                    feed_url=feed_url,
                )
            )
        return tuple(items)

    def _atom_entry_link(self, entry: ET.Element, *, resolved_url: str) -> str | None:
        candidates: list[tuple[int, str]] = []
        for child in entry:
            if _local_name(child.tag).lower() != "link":
                continue
            href = _clean_url(child.attrib.get("href"))
            if not href:
                continue
            rel = _clean_text(child.attrib.get("rel")).casefold() or "alternate"
            mime = _clean_text(child.attrib.get("type")).casefold()
            score = 0
            if rel not in {"alternate", ""}:
                score += 10
            if mime and "html" not in mime:
                score += 1
            candidates.append((score, urllib.parse.urljoin(resolved_url, href)))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1]

    def _rank_items(self, items: Sequence[DisplayNewsTickerItem]) -> tuple[DisplayNewsTickerItem, ...]:
        ranked = sorted(
            items,
            key=lambda item: (
                _timestamp_sort_value(item.published_at),
                -len(item.title),
                item.source.casefold(),
                item.title.casefold(),
            ),
            reverse=True,
        )
        deduped: list[DisplayNewsTickerItem] = []
        seen: set[str] = set()
        for item in ranked:
            key = _dedupe_key(item.title)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        buckets: OrderedDict[str, deque[DisplayNewsTickerItem]] = OrderedDict()
        for item in deduped:
            source_key = item.source or "Feed"
            buckets.setdefault(source_key, deque()).append(item)

        interleaved: list[DisplayNewsTickerItem] = []
        while buckets:
            exhausted: list[str] = []
            for source_key, queue in list(buckets.items()):
                if not queue:
                    exhausted.append(source_key)
                    continue
                interleaved.append(queue.popleft())
                if not queue:
                    exhausted.append(source_key)
            for source_key in exhausted:
                buckets.pop(source_key, None)
        return tuple(interleaved)

    def _items_for_feed(
        self,
        snapshot: DisplayNewsTickerSnapshot | None,
        feed_url: str,
    ) -> tuple[DisplayNewsTickerItem, ...]:
        if snapshot is None:
            return ()
        if feed_url in snapshot.feed_item_cache:
            return snapshot.feed_item_cache[feed_url]
        return tuple(item for item in snapshot.items if item.feed_url == feed_url)


@dataclass(slots=True)
class DisplayWorldIntelligenceNewsTickerSource:
    resolver: DisplayWorldIntelligenceTickerFeedResolver
    timeout_s: float = 4.0
    max_items: int = 12
    user_agent: str = _DEFAULT_USER_AGENT
    max_workers: int = _MAX_CONCURRENT_FETCHES

    def fetch(
        self,
        *,
        now: datetime | None = None,
        previous_snapshot: DisplayNewsTickerSnapshot | None = None,
    ) -> DisplayNewsTickerSnapshot:
        feed_urls = tuple(
            url
            for url in (
                _clean_url(raw_url)
                for raw_url in self.resolver.resolve_feed_urls()
            )
            if url
        )
        if not feed_urls:
            return DisplayNewsTickerSnapshot(
                captured_at=_normalize_now(now).isoformat(),
                items=(),
                feed_urls=(),
                last_error=None,
                feed_states={},
                feed_item_cache={},
            )
        return DisplayNewsTickerFetcher(
            feed_urls=feed_urls,
            timeout_s=self.timeout_s,
            max_items=self.max_items,
            user_agent=self.user_agent,
            max_workers=self.max_workers,
        ).fetch(now=now, previous_snapshot=previous_snapshot)


@dataclass(slots=True)
class DisplayNewsTickerRuntime:
    enabled: bool
    store: DisplayNewsTickerStore
    fetcher: DisplayNewsTickerSource | None = None
    refresh_interval_s: float = 600.0
    rotation_interval_s: float = 12.0
    emit: Callable[[str], None] | None = None
    _snapshot: DisplayNewsTickerSnapshot | None = field(default=None, init=False, repr=False)
    _refresh_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _refresh_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _next_refresh_after_monotonic: float = field(default=0.0, init=False, repr=False)
    _refresh_failures: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "DisplayNewsTickerRuntime":
        return cls(
            enabled=bool(config.display_news_ticker_enabled),
            store=DisplayNewsTickerStore.from_config(config),
            fetcher=DisplayWorldIntelligenceNewsTickerSource(
                resolver=DisplayWorldIntelligenceTickerFeedResolver.from_config(config),
                timeout_s=config.display_news_ticker_timeout_s,
                max_items=config.display_news_ticker_max_items,
            )
            if config.display_news_ticker_enabled
            else None,
            refresh_interval_s=config.display_news_ticker_refresh_interval_s,
            rotation_interval_s=config.display_news_ticker_rotation_interval_s,
            emit=emit,
        )

    def current_text(self, *, now: datetime | None = None) -> str | None:
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
        if time.monotonic() < self._next_refresh_after_monotonic:
            return
        thread = self._refresh_thread
        if thread is not None and thread.is_alive():
            return
        snapshot = self._snapshot
        if snapshot is not None and snapshot.items and not self._is_snapshot_stale(snapshot, now=now):
            return
        with self._refresh_lock:
            if time.monotonic() < self._next_refresh_after_monotonic:
                return
            thread = self._refresh_thread
            if thread is not None and thread.is_alive():
                return
            self._next_refresh_after_monotonic = time.monotonic() + 5.0
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
            snapshot = self._fetch_snapshot(fetcher, previous_snapshot=previous)
        except Exception as exc:
            self._refresh_failures += 1
            self._next_refresh_after_monotonic = time.monotonic() + self._retry_delay_s()
            self._safe_emit(f"display_news_ticker_refresh_failed={type(exc).__name__}")
            if previous is None:
                self._snapshot = DisplayNewsTickerSnapshot(
                    captured_at=_utc_now().isoformat(),
                    items=(),
                    feed_urls=(),
                    last_error=type(exc).__name__,
                    feed_states={},
                    feed_item_cache={},
                )
            return

        if not snapshot.items and previous is not None and previous.items:
            snapshot = DisplayNewsTickerSnapshot(
                captured_at=snapshot.captured_at,
                items=previous.items,
                feed_urls=snapshot.feed_urls or previous.feed_urls,
                last_error=snapshot.last_error or previous.last_error,
                feed_states=snapshot.feed_states or previous.feed_states,
                feed_item_cache=snapshot.feed_item_cache or previous.feed_item_cache,
            )

        self._snapshot = snapshot
        if not snapshot.items and snapshot.last_error:
            self._refresh_failures += 1
            self._next_refresh_after_monotonic = time.monotonic() + self._retry_delay_s()
        else:
            self._refresh_failures = 0
            self._next_refresh_after_monotonic = time.monotonic() + max(30.0, float(self.refresh_interval_s))

        try:
            self.store.save(snapshot)
        except Exception as exc:
            self._safe_emit(f"display_news_ticker_cache_save_failed={type(exc).__name__}")

    def _fetch_snapshot(
        self,
        fetcher: DisplayNewsTickerSource,
        *,
        previous_snapshot: DisplayNewsTickerSnapshot | None,
    ) -> DisplayNewsTickerSnapshot:
        now = _utc_now()
        try:
            parameters = inspect.signature(fetcher.fetch).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "previous_snapshot" in parameters:
            return fetcher.fetch(now=now, previous_snapshot=previous_snapshot)
        return fetcher.fetch(now=now)

    def _retry_delay_s(self) -> float:
        exponent = max(0, self._refresh_failures - 1)
        return max(
            _MIN_REFRESH_RETRY_S,
            min(_MAX_REFRESH_RETRY_S, _MIN_REFRESH_RETRY_S * (2**exponent)),
        )

    def _safe_emit(self, line: str) -> None:
        if self.emit is None:
            return
        try:
            self.emit(line)
        except Exception:
            return
