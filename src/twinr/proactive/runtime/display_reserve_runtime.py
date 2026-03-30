"""Publish all right-lane reserve items through one shared runtime path.

# CHANGELOG: 2026-03-29
# BUG-1: Fixed reserved metadata key clobbering; user metadata could overwrite
#        runtime fields like reason/candidate_family and silently corrupt
#        exposure history analytics.
# BUG-2: Fixed naive-datetime handling; naive `now` values no longer get
#        interpreted as the Raspberry Pi system local timezone by astimezone().
# BUG-3: Fixed retry/duplicate publish behavior; repeated publishes of the same
#        visible item now dedupe through an idempotency path instead of writing
#        duplicate exposures during ambiguous retry scenarios.
# BUG-4: Fixed the default interprocess publish lock path; reserve-lane writes
#        no longer depend on one global `/tmp` lock that can become root-owned
#        and block Pi acceptance tests or mixed-user tooling.
# SEC-1: Fixed blind persistence of arbitrary metadata; metadata is now bounded,
#        JSON-safe, and sensitive keys are redacted before hitting the history
#        store.
# IMP-1: Added publisher-side synchronization for the shared reserve runtime so
#        concurrent producers serialize visible/history side-effects.
# IMP-2: Added optional OpenTelemetry tracing hooks plus stable publish IDs /
#        fingerprints for easier observability and replay debugging.
#
# BREAKING: A naive `now=` is now treated as UTC instead of "system local time".
#           This prevents silent timestamp skew, but callers that relied on the
#           old implicit-local behavior must pass an aware local datetime.
# BREAKING: Metadata keys that collide with reserved runtime fields are now
#           namespaced (for example `reason` -> `user_reason`) instead of
#           overwriting runtime metadata.
#
# The HDMI reserve area is a single surface even when the underlying items come
# from different producers such as the daily ambient planner or a real-time
# social/focus opener. This module centralizes the actual cue-store + exposure-
# history write path so those producers do not keep reimplementing slightly
# different reserve-side effects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from threading import RLock

try:  # pragma: no cover - only available on POSIX systems such as Raspberry Pi OS
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover - keep runtime fully optional
    _otel_trace = None

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseController,
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from .display_reserve_support import compact_text, utc_now


_LOGGER = logging.getLogger(__name__)
_TRACER = _otel_trace.get_tracer(__name__) if _otel_trace is not None else None

_DEFAULT_HOLD_SECONDS = 30.0
_MAX_BODY_LEN = 1200
_MAX_METADATA_KEY_LEN = 80
_MAX_METADATA_STRING_LEN = 512
_MAX_METADATA_ITEMS = 24
_MAX_METADATA_DEPTH = 3
_MAX_SEQUENCE_ITEMS = 16
_MAX_MATCH_ANCHORS = 8
_MAX_MATCH_ANCHOR_LEN = 160
_AUTO_IDEMPOTENCY_PREFIX = "auto:"
_RESERVED_METADATA_KEYS = frozenset(
    {
        "reason",
        "candidate_family",
        "publish_id",
        "request_fingerprint",
        "idempotency_key",
        "correlation_id",
        "runtime_path",
        "topic_key",
        "semantic_topic_key",
        "cue_source",
        "history_source",
    }
)
_SENSITIVE_METADATA_MARKERS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "session",
    "api_key",
    "apikey",
    "access_key",
    "refresh_token",
    "jwt",
    "bearer",
)


def _default_publish_lock_path(*, active_store_path: object | None = None) -> str:
    """Return a writable default lock path for reserve-runtime publishes."""

    if active_store_path is not None:
        resolved_store_path = os.path.abspath(os.fspath(active_store_path))
        store_dir = os.path.dirname(resolved_store_path)
        store_name = os.path.basename(resolved_store_path)
        if store_dir and store_name:
            return os.path.join(store_dir, f".{store_name}.reserve-runtime.lock")

    runtime_dir = str(os.getenv("XDG_RUNTIME_DIR", "") or "").strip()
    if runtime_dir and os.path.isabs(runtime_dir):
        return os.path.join(runtime_dir, "twinr-display-reserve-runtime.lock")

    get_uid = getattr(os, "getuid", None)
    uid = int(get_uid()) if callable(get_uid) else 0
    return os.path.join("/tmp", f"twinr-display-reserve-runtime-{uid}.lock")


@dataclass(frozen=True, slots=True)
class DisplayReserveRuntimeRequest:
    """Describe one concrete item to render on the right reserve lane."""

    topic_key: str
    title: str
    cue_source: str
    history_source: str
    action: str
    attention_state: str
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    hold_seconds: float
    reason: str
    semantic_topic_key: str = ""
    candidate_family: str = "general"
    match_anchors: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None
    idempotency_key: str = ""
    correlation_id: str = ""


@dataclass(frozen=True, slots=True)
class DisplayReserveRuntimePublishResult:
    """Summarize one shared reserve-lane publish."""

    cue: DisplayAmbientImpulseCue
    exposure_id: str | None = None


@dataclass(frozen=True, slots=True)
class _NormalizedDisplayReserveRuntimeRequest:
    topic_key: str
    title: str
    cue_source: str
    history_source: str
    action: str
    attention_state: str
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    hold_seconds: float
    reason: str
    semantic_topic_key: str
    candidate_family: str
    match_anchors: tuple[str, ...]
    metadata: dict[str, object]
    idempotency_key: str
    correlation_id: str
    fingerprint: str


@dataclass(slots=True)
class _PendingHistoryReplay:
    request: _NormalizedDisplayReserveRuntimeRequest
    cue: DisplayAmbientImpulseCue
    shown_at: datetime
    expires_at: datetime
    publish_id: str


@dataclass(slots=True)
class _CompletedPublishCacheEntry:
    result: DisplayReserveRuntimePublishResult
    expires_at: datetime
    fingerprint: str


@dataclass(slots=True)
class DisplayReserveRuntimePublisher:
    """Persist one reserve-lane item through the shared cue/history contract."""

    controller: DisplayAmbientImpulseController
    active_store: DisplayAmbientImpulseCueStore
    history_store: DisplayAmbientImpulseHistoryStore
    publish_lock_path: str = field(default_factory=_default_publish_lock_path)
    _publish_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _completed_cache: dict[str, _CompletedPublishCacheEntry] = field(default_factory=dict, init=False, repr=False)
    _pending_history: dict[str, _PendingHistoryReplay] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str,
    ) -> "DisplayReserveRuntimePublisher":
        """Build one shared reserve publisher from Twinr configuration."""

        controller = DisplayAmbientImpulseController.from_config(
            config,
            default_source=default_source,
        )
        publish_lock_path = (
            str(os.environ.get("TWINR_DISPLAY_RESERVE_RUNTIME_LOCK_PATH", "") or "").strip()
            or _default_publish_lock_path(active_store_path=controller.store.path)
        )
        return cls(
            controller=controller,
            active_store=controller.store,
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            publish_lock_path=publish_lock_path,
        )

    def publish(
        self,
        request: DisplayReserveRuntimeRequest,
        *,
        now: datetime | None = None,
    ) -> DisplayReserveRuntimePublishResult:
        """Show one reserve-lane item and append one exposure history record."""

        effective_now = self._coerce_input_now(now)
        normalized = self._normalize_request(request)
        idempotency_key = normalized.idempotency_key

        span_cm = (
            _TRACER.start_as_current_span("twinr.display.reserve.publish")
            if _TRACER is not None
            else nullcontext()
        )

        with span_cm as span:
            if span is not None:
                span.set_attribute("twinr.topic_key", normalized.topic_key)
                span.set_attribute("twinr.semantic_topic_key", normalized.semantic_topic_key)
                span.set_attribute("twinr.cue_source", normalized.cue_source)
                span.set_attribute("twinr.history_source", normalized.history_source)
                span.set_attribute("twinr.idempotency_key", idempotency_key)
                span.set_attribute("twinr.hold_seconds", normalized.hold_seconds)

            with self._exclusive_publish_window():
                self._prune_state(effective_now)

                cached = self._completed_cache.get(idempotency_key)
                if cached is not None:
                    self._ensure_matching_idempotency_key(
                        idempotency_key=idempotency_key,
                        known_fingerprint=cached.fingerprint,
                        request_fingerprint=normalized.fingerprint,
                    )
                    if cached.expires_at > effective_now:
                        if span is not None:
                            span.set_attribute("twinr.publish_path", "cached")
                        return cached.result

                pending = self._pending_history.get(idempotency_key)
                if pending is not None:
                    self._ensure_matching_idempotency_key(
                        idempotency_key=idempotency_key,
                        known_fingerprint=pending.request.fingerprint,
                        request_fingerprint=normalized.fingerprint,
                    )
                    if pending.expires_at > effective_now:
                        if span is not None:
                            span.set_attribute("twinr.publish_path", "history-replay")
                        return self._replay_pending_history(
                            idempotency_key=idempotency_key,
                            pending=pending,
                        )

                if span is not None:
                    span.set_attribute("twinr.publish_path", "fresh")

                cue = self._show_cue(normalized, effective_now)
                expires_at = self._coerce_cue_expiration(
                    cue.expires_at,
                    fallback_now=effective_now,
                    hold_seconds=normalized.hold_seconds,
                )
                publish_id = self._new_publish_id()

                self._pending_history[idempotency_key] = _PendingHistoryReplay(
                    request=normalized,
                    cue=cue,
                    shown_at=effective_now,
                    expires_at=expires_at,
                    publish_id=publish_id,
                )
                try:
                    result = self._append_history_for_cue(
                        request=normalized,
                        cue=cue,
                        shown_at=effective_now,
                        expires_at=expires_at,
                        publish_id=publish_id,
                    )
                except Exception as exc:
                    if span is not None:
                        span.record_exception(exc)
                    _LOGGER.warning(
                        "Reserve publish history append failed; pending replay retained "
                        "for retry. topic_key=%s idempotency_key=%s",
                        normalized.topic_key,
                        idempotency_key,
                        exc_info=True,
                    )
                    raise

                self._pending_history.pop(idempotency_key, None)
                self._completed_cache[idempotency_key] = _CompletedPublishCacheEntry(
                    result=result,
                    expires_at=expires_at,
                    fingerprint=normalized.fingerprint,
                )
                return result

    def show_visible_only(
        self,
        request: DisplayReserveRuntimeRequest,
        *,
        now: datetime | None = None,
    ) -> DisplayReserveRuntimePublishResult:
        """Show one reserve-lane item without appending a new exposure entry.

        This is used for passive idle-fill behavior after a daily plan is
        exhausted. The surface stays populated, but the same item is not
        counted as a second shown-card exposure.
        """

        effective_now = self._coerce_input_now(now)
        normalized = self._normalize_request(request)
        with self._exclusive_publish_window():
            cue = self._show_cue(normalized, effective_now)
        return DisplayReserveRuntimePublishResult(cue=cue, exposure_id=None)

    def _show_cue(
        self,
        request: _NormalizedDisplayReserveRuntimeRequest,
        shown_at: datetime,
    ) -> DisplayAmbientImpulseCue:
        return self.controller.show_impulse(
            topic_key=request.topic_key,
            semantic_topic_key=request.semantic_topic_key,
            eyebrow=request.eyebrow,
            headline=request.headline,
            body=request.body,
            symbol=request.symbol,
            accent=request.accent,
            action=request.action,
            attention_state=request.attention_state,
            hold_seconds=request.hold_seconds,
            source=request.cue_source,
            now=shown_at,
        )

    def _append_history_for_cue(
        self,
        *,
        request: _NormalizedDisplayReserveRuntimeRequest,
        cue: DisplayAmbientImpulseCue,
        shown_at: datetime,
        expires_at: datetime,
        publish_id: str,
    ) -> DisplayReserveRuntimePublishResult:
        metadata = self._build_history_metadata(request, publish_id=publish_id)
        exposure = self.history_store.append_exposure(
            source=request.history_source,
            topic_key=request.topic_key,
            semantic_topic_key=request.semantic_topic_key,
            title=request.title,
            headline=cue.headline,
            body=cue.body,
            action=request.action,
            attention_state=request.attention_state,
            shown_at=shown_at,
            expires_at=expires_at,
            match_anchors=request.match_anchors,
            metadata=metadata,
        )
        return DisplayReserveRuntimePublishResult(
            cue=cue,
            exposure_id=exposure.exposure_id,
        )

    def _replay_pending_history(
        self,
        *,
        idempotency_key: str,
        pending: _PendingHistoryReplay,
    ) -> DisplayReserveRuntimePublishResult:
        result = self._append_history_for_cue(
            request=pending.request,
            cue=pending.cue,
            shown_at=pending.shown_at,
            expires_at=pending.expires_at,
            publish_id=pending.publish_id,
        )
        self._pending_history.pop(idempotency_key, None)
        self._completed_cache[idempotency_key] = _CompletedPublishCacheEntry(
            result=result,
            expires_at=pending.expires_at,
            fingerprint=pending.request.fingerprint,
        )
        return result

    def _build_history_metadata(
        self,
        request: _NormalizedDisplayReserveRuntimeRequest,
        *,
        publish_id: str,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "reason": request.reason,
            "candidate_family": request.candidate_family,
            "publish_id": publish_id,
            "request_fingerprint": request.fingerprint,
            "runtime_path": "display.reserve.shared-runtime",
            "topic_key": request.topic_key,
            "semantic_topic_key": request.semantic_topic_key,
            "cue_source": request.cue_source,
            "history_source": request.history_source,
        }
        if request.idempotency_key:
            metadata["idempotency_key"] = request.idempotency_key
        if request.correlation_id:
            metadata["correlation_id"] = request.correlation_id

        for raw_key, raw_value in request.metadata.items():
            key = compact_text(str(raw_key), max_len=_MAX_METADATA_KEY_LEN)
            if not key:
                continue
            if key in _RESERVED_METADATA_KEYS:
                key = f"user_{key}"
            key = self._dedupe_metadata_key(key, metadata)
            metadata[key] = self._sanitize_metadata_value(
                key,
                raw_value,
                depth=0,
            )
        return metadata

    def _normalize_request(
        self,
        request: DisplayReserveRuntimeRequest,
    ) -> _NormalizedDisplayReserveRuntimeRequest:
        normalized = _NormalizedDisplayReserveRuntimeRequest(
            topic_key=self._clean_text(request.topic_key, max_len=160),
            title=self._clean_text(request.title, max_len=200),
            cue_source=self._clean_text(request.cue_source, max_len=120),
            history_source=self._clean_text(request.history_source, max_len=120),
            action=self._clean_text(request.action, max_len=120),
            attention_state=self._clean_text(request.attention_state, max_len=120),
            eyebrow=self._clean_text(request.eyebrow, max_len=120),
            headline=self._clean_text(request.headline, max_len=200),
            body=self._clean_text(request.body, max_len=_MAX_BODY_LEN),
            symbol=self._clean_text(request.symbol, max_len=32),
            accent=self._clean_text(request.accent, max_len=32),
            hold_seconds=self._normalize_hold_seconds(request.hold_seconds),
            reason=self._clean_text(request.reason, max_len=200),
            semantic_topic_key=self._clean_text(request.semantic_topic_key, max_len=160),
            candidate_family=self._clean_text(request.candidate_family, max_len=120) or "general",
            match_anchors=self._match_anchors(request.match_anchors),
            metadata=self._normalize_metadata(request.metadata),
            idempotency_key="",
            correlation_id=self._clean_text(request.correlation_id, max_len=160),
            fingerprint="",
        )
        fingerprint = self._fingerprint_request(normalized)
        explicit_idempotency_key = self._clean_text(request.idempotency_key, max_len=200)
        return replace(
            normalized,
            fingerprint=fingerprint,
            idempotency_key=explicit_idempotency_key or f"{_AUTO_IDEMPOTENCY_PREFIX}{fingerprint}",
        )

    def _normalize_metadata(
        self,
        metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if not isinstance(metadata, Mapping):
            return {}

        normalized: dict[str, object] = {}
        for raw_key, raw_value in metadata.items():
            key = compact_text(str(raw_key), max_len=_MAX_METADATA_KEY_LEN)
            if not key:
                continue
            if len(normalized) >= _MAX_METADATA_ITEMS:
                break
            normalized[key] = self._sanitize_metadata_value(key, raw_value, depth=0)
        return normalized

    def _sanitize_metadata_value(
        self,
        key: str,
        value: object,
        *,
        depth: int,
    ) -> object:
        if self._is_sensitive_metadata_key(key):
            return "[REDACTED]"
        if depth >= _MAX_METADATA_DEPTH:
            return self._stringify_metadata_value(value)

        if value is None or isinstance(value, (bool, int)):
            return value

        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return self._stringify_metadata_value(value)

        if isinstance(value, datetime):
            return self._coerce_history_datetime(value).isoformat()

        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<bytes:{len(bytes(value))}>"

        if isinstance(value, str):
            return self._clean_text(value, max_len=_MAX_METADATA_STRING_LEN)

        if isinstance(value, Mapping):
            nested: dict[str, object] = {}
            for idx, (raw_key, raw_child) in enumerate(value.items()):
                if idx >= _MAX_METADATA_ITEMS:
                    break
                nested_key = compact_text(str(raw_key), max_len=_MAX_METADATA_KEY_LEN)
                if not nested_key:
                    continue
                if nested_key in _RESERVED_METADATA_KEYS:
                    nested_key = f"user_{nested_key}"
                nested[nested_key] = self._sanitize_metadata_value(
                    nested_key,
                    raw_child,
                    depth=depth + 1,
                )
            return nested

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
            items: list[object] = []
            for idx, item in enumerate(value):
                if idx >= _MAX_SEQUENCE_ITEMS:
                    break
                items.append(self._sanitize_metadata_value(key, item, depth=depth + 1))
            return items

        return self._stringify_metadata_value(value)

    def _fingerprint_request(
        self,
        request: _NormalizedDisplayReserveRuntimeRequest,
    ) -> str:
        payload = {
            "topic_key": request.topic_key,
            "title": request.title,
            "cue_source": request.cue_source,
            "history_source": request.history_source,
            "action": request.action,
            "attention_state": request.attention_state,
            "eyebrow": request.eyebrow,
            "headline": request.headline,
            "body": request.body,
            "symbol": request.symbol,
            "accent": request.accent,
            "hold_seconds": request.hold_seconds,
            "reason": request.reason,
            "semantic_topic_key": request.semantic_topic_key,
            "candidate_family": request.candidate_family,
            "match_anchors": request.match_anchors,
            "metadata": request.metadata,
            "correlation_id": request.correlation_id,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    def _coerce_input_now(self, now: datetime | None) -> datetime:
        candidate = now or utc_now()
        if not isinstance(candidate, datetime):
            raise TypeError(f"now must be datetime | None, got {type(candidate)!r}")
        if self._is_naive_datetime(candidate):
            candidate = candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    def _coerce_cue_expiration(
        self,
        value: object,
        *,
        fallback_now: datetime,
        hold_seconds: float,
    ) -> datetime:
        if isinstance(value, (datetime, str)):
            try:
                return self._coerce_history_datetime(value)
            except (TypeError, ValueError):
                pass
        return fallback_now + timedelta(seconds=hold_seconds)

    def _coerce_history_datetime(
        self,
        value: datetime | str,
    ) -> datetime:
        if isinstance(value, str):
            candidate = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            candidate = value

        if self._is_naive_datetime(candidate):
            candidate = candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    def _normalize_hold_seconds(self, value: float) -> float:
        if isinstance(value, bool):
            return _DEFAULT_HOLD_SECONDS
        try:
            hold_seconds = float(value)
        except (TypeError, ValueError):
            return _DEFAULT_HOLD_SECONDS
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            return _DEFAULT_HOLD_SECONDS
        return hold_seconds

    @contextmanager
    def _exclusive_publish_window(self):
        with self._publish_lock:
            if not self.publish_lock_path or fcntl is None:
                yield
                return

            dirpath = os.path.dirname(self.publish_lock_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(self.publish_lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _prune_state(self, now: datetime) -> None:
        expired_completed = [
            key
            for key, entry in self._completed_cache.items()
            if entry.expires_at <= now
        ]
        for key in expired_completed:
            self._completed_cache.pop(key, None)

        expired_pending = [
            key
            for key, entry in self._pending_history.items()
            if entry.expires_at <= now
        ]
        for key in expired_pending:
            self._pending_history.pop(key, None)

    def _match_anchors(self, values: Sequence[str] | None) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes, bytearray, memoryview)):
            iterable = [values]
        elif isinstance(values, Sequence):
            iterable = values
        else:
            iterable = [values]

        anchors: list[str] = []
        for value in iterable:
            compact = compact_text(str(value), max_len=_MAX_MATCH_ANCHOR_LEN)
            if compact and compact not in anchors:
                anchors.append(compact)
            if len(anchors) >= _MAX_MATCH_ANCHORS:
                break
        return tuple(anchors)

    def _clean_text(self, value: object, *, max_len: int) -> str:
        if value is None:
            return ""
        return compact_text(str(value), max_len=max_len)

    def _dedupe_metadata_key(self, key: str, metadata: Mapping[str, object]) -> str:
        if key not in metadata:
            return key
        for idx in range(2, _MAX_METADATA_ITEMS + 2):
            candidate = compact_text(f"{key}_{idx}", max_len=_MAX_METADATA_KEY_LEN)
            if candidate and candidate not in metadata:
                return candidate
        return compact_text(f"{key}_{len(metadata)+1}", max_len=_MAX_METADATA_KEY_LEN) or "meta"

    def _stringify_metadata_value(self, value: object) -> str:
        return self._clean_text(repr(value), max_len=_MAX_METADATA_STRING_LEN)

    def _is_naive_datetime(self, value: datetime) -> bool:
        return value.tzinfo is None or value.tzinfo.utcoffset(value) is None

    def _is_sensitive_metadata_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in _SENSITIVE_METADATA_MARKERS)

    def _ensure_matching_idempotency_key(
        self,
        *,
        idempotency_key: str,
        known_fingerprint: str,
        request_fingerprint: str,
    ) -> None:
        if known_fingerprint == request_fingerprint:
            return
        raise ValueError(
            "DisplayReserveRuntimeRequest reused the same idempotency_key for a "
            "different payload: "
            f"{idempotency_key!r}"
        )

    def _new_publish_id(self) -> str:
        factory = getattr(uuid, "uuid7", uuid.uuid4)
        return str(factory())


__all__ = [
    "DisplayReserveRuntimePublishResult",
    "DisplayReserveRuntimePublisher",
    "DisplayReserveRuntimeRequest",
]
