"""Mirror long-term memory snapshots to the remote-primary backend.

This module provides ``LongTermRemoteStateStore``, the ChonkyDB-backed remote
snapshot adapter used when long-term memory runs in remote-primary mode. It
validates snapshot identifiers, bounds remote I/O, and exposes required-mode
failures through ``LongTermRemoteUnavailableError``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field  # AUDIT-FIX(#10): Field support is needed for stable exception/internal state handling.
from datetime import datetime, timezone
import hashlib
import json
import logging  # AUDIT-FIX(#5): Add diagnostics for remote-boundary failures without crashing callers.
import math  # AUDIT-FIX(#7): Clamp invalid numeric config values safely.
import os  # AUDIT-FIX(#2): Use no-follow local file opens to avoid symlink races during fallback reads.
from pathlib import Path
import stat  # AUDIT-FIX(#2): Require regular files for local snapshot fallback reads.
import threading  # AUDIT-FIX(#5): Protect lightweight circuit-breaker state across concurrent callers.
import time
from typing import Iterable, Iterator, Mapping
from urllib.parse import quote  # AUDIT-FIX(#6): Encode URI path segments safely.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig, ChonkyDBError, chonkydb_data_path
from twinr.memory.chonkydb.models import ChonkyDBRecordRequest
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteWriteContext,
    record_remote_write_diagnostic,
)


_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#5): Keep operational visibility for degraded-mode events.

_REMOTE_NAMESPACE_PREFIX = "twinr_longterm_v1"
_SNAPSHOT_SCHEMA = "twinr_remote_snapshot_v1"
_SNAPSHOT_POINTER_SCHEMA = "twinr_remote_snapshot_pointer_v1"
_SNAPSHOT_POINTER_VERSION = 1
_SNAPSHOT_POINTER_PREFIX = "__pointer__:"
_DOCUMENT_ID_HINTS_SCHEMA = "twinr_remote_snapshot_document_hints_v1"
_DOCUMENT_ID_HINTS_FILENAME = "remote_snapshot_document_hints.json"
_DEFAULT_REMOTE_READ_TIMEOUT_S = 10.0  # AUDIT-FIX(#7): Safe fallback defaults for malformed .env values.
_DEFAULT_REMOTE_WRITE_TIMEOUT_S = 15.0
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_S = 0.5
_DEFAULT_MAX_CONTENT_CHARS = 262_144
_DEFAULT_MAX_CONTENT_CHARS_CAP = 8_388_608  # 8 MiB cap for RPi-friendly snapshot fetches.
_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES = 8_388_608
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S = 15.0
_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S = 20.0
_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S = 30.0
_DEFAULT_STATUS_PROBE_TIMEOUT_S = 20.0
_DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S = 45.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S = 20.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S = 45.0
_DEFAULT_ASYNC_ATTESTATION_POLL_S = 0.05
_MAX_NAMESPACE_LENGTH = 255
_MAX_SNAPSHOT_KIND_LENGTH = 255
_MAX_DOCUMENT_HINTS_BYTES = 262_144


def _utcnow_iso() -> str:
    """Return the current UTC time encoded as ISO 8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


def remote_snapshot_document_hints_path(config: TwinrConfig) -> Path | None:
    """Return the local durable document-id hint file used by remote snapshots.

    The prompting layer uses this path as a cheap local invalidation signal for
    cached instruction bundles that depend on remote-managed prompt context.
    """

    try:
        return chonkydb_data_path(config) / _DOCUMENT_ID_HINTS_FILENAME
    except Exception:
        return None


def _strip_text(value: str | None) -> str:
    """Return trimmed string content for optional text input."""

    return str(value or "").strip()


def _mapping_dict(value: Mapping[str, object] | None) -> dict[str, object] | None:
    """Copy a mapping into a plain ``dict`` when one is present."""

    if value is None:
        return None
    return dict(value)


def _reject_non_finite_json_constant(value: str) -> object:  # AUDIT-FIX(#12): Reject NaN/Infinity so remote snapshots stay standards-compliant.
    """Reject non-finite JSON constants during strict snapshot parsing."""

    raise ValueError(f"Unsupported JSON constant {value!r}.")


def _safe_json_text(payload: Mapping[str, object]) -> str:
    """Serialize a mapping to strict JSON text for remote transport."""

    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,  # AUDIT-FIX(#12): Emit only strict JSON for remote interoperability.
    )


def _redact_secrets(text: str, *, secrets: Iterable[str]) -> str:
    """Replace configured secret values in log-bound text."""

    redacted = " ".join(str(text).split())
    for secret in secrets:
        cleaned = _strip_text(secret)
        if cleaned:
            redacted = redacted.replace(cleaned, "[redacted]")
    return redacted


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Coerce numeric config to an integer inside inclusive bounds."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Coerce numeric config to a finite float inside inclusive bounds."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_timeout_s(value: object, *, default: float) -> float:
    """Coerce timeout configuration to a safe bounded float."""

    return _coerce_float(value, default=default, minimum=0.1, maximum=300.0)


def _normalize_snapshot_document_id(value: object) -> str | None:
    """Normalize one remote document identifier used for pointer-based reads."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _snapshot_updated_at_sort_key(value: object) -> tuple[int, float]:
    """Sort snapshot candidates by updated_at while tolerating malformed timestamps."""

    normalized = _normalize_snapshot_document_id(value)
    if normalized is None:
        return (0, 0.0)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return (0, 0.0)
    aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return (1, aware.astimezone(timezone.utc).timestamp())


def _extract_store_document_id(result: Mapping[str, object] | None) -> str | None:
    """Extract a persisted document id from a ChonkyDB store response when present."""

    if not isinstance(result, Mapping):
        return None
    items = result.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for field_name in ("document_id", "payload_id", "chonky_id"):
                document_id = _normalize_snapshot_document_id(item.get(field_name))
                if document_id:
                    return document_id
    for field_name in ("document_id", "payload_id", "chonky_id"):
        document_id = _normalize_snapshot_document_id(result.get(field_name))
        if document_id:
            return document_id
    return None


def _store_result_failure_detail(result: Mapping[str, object] | None) -> str | None:
    """Summarize item-level ChonkyDB store failures from an otherwise 200 response."""

    if not isinstance(result, Mapping):
        return None
    failures: list[str] = []
    items = result.get("items")
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            if item.get("success") is False:
                error_type = _normalize_text(item.get("error_type"))
                error_text = _normalize_text(item.get("error"))
                if error_type and error_text:
                    failures.append(f"item[{index}] {error_type}: {error_text}")
                elif error_text:
                    failures.append(f"item[{index}] {error_text}")
                else:
                    failures.append(f"item[{index}] rejected")
    if failures:
        return "; ".join(failures)
    if result.get("success") is False:
        error_type = _normalize_text(result.get("error_type"))
        error_text = _normalize_text(result.get("error"))
        if error_type and error_text:
            return f"{error_type}: {error_text}"
        if error_text:
            return error_text
        return "request rejected"
    return None


def _safe_resolve_path(path: Path) -> Path:
    """Resolve a path defensively without propagating odd filesystem errors."""

    try:
        return path.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - defensive path hardening
        return Path(os.path.abspath(os.fspath(path)))


def _normalize_storage_token(value: str, *, field_name: str, max_length: int) -> str:
    """Validate namespace or snapshot tokens used in remote storage keys."""

    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters.")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def _encode_uri_path_segment(value: str) -> str:
    """Percent-encode one remote URI path segment."""

    return quote(value, safe="")


class LongTermRemoteUnavailableError(RuntimeError):
    """Signal that required remote long-term snapshot state is unavailable."""

    def __init__(self, message: str) -> None:  # AUDIT-FIX(#10): Keep normal exception semantics so traceback chaining works.
        self.message = str(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LongTermRemoteStatus:
    """Describe whether the remote snapshot backend is ready for use."""

    mode: str
    ready: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class LongTermRemoteFetchAttempt:
    """Capture one remote snapshot fetch attempt for readiness forensics."""

    source: str
    attempt: int
    status: str
    latency_ms: float
    status_code: int | None = None
    document_id: str | None = None
    error_type: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of one fetch attempt."""

        return {
            "source": self.source,
            "attempt": self.attempt,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "document_id": self.document_id,
            "error_type": self.error_type,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class LongTermRemoteSnapshotProbe:
    """Describe how one remote snapshot read was resolved."""

    snapshot_kind: str
    status: str
    latency_ms: float
    detail: str | None = None
    document_id: str | None = None
    pointer_document_id: str | None = None
    selected_source: str | None = None
    payload: dict[str, object] | None = None
    attempts: tuple[LongTermRemoteFetchAttempt, ...] = ()

    def to_dict(self, *, include_payload: bool = False) -> dict[str, object]:
        """Return a JSON-safe probe summary."""

        payload: dict[str, object] = {
            "snapshot_kind": self.snapshot_kind,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "document_id": self.document_id,
            "pointer_document_id": self.pointer_document_id,
            "selected_source": self.selected_source,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }
        if include_payload:
            payload["payload"] = self.payload
        return payload


@dataclass(frozen=True, slots=True)
class _RemoteSnapshotFetchResult:
    """Capture the outcome of one remote snapshot fetch attempt."""

    status: str
    payload: dict[str, object] | None = None
    detail: str | None = None
    document_id: str | None = None
    latency_ms: float = 0.0
    selected_source: str | None = None
    attempts: tuple[LongTermRemoteFetchAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class _RemoteSnapshotCandidate:
    """Bundle one parsed snapshot candidate with its source document id."""

    payload: Mapping[str, object]
    document_id: str | None = None


@dataclass(frozen=True, slots=True)
class _CachedSnapshotRead:
    """Hold one short-lived in-process remote snapshot read-through cache entry."""

    payload: dict[str, object]
    expires_at_monotonic: float
    document_id: str | None = None


@dataclass(slots=True)
class LongTermRemoteStateStore:
    """Load and save remote snapshot state for long-term memory.

    This adapter owns ChonkyDB URI construction, retry/backoff, a lightweight
    circuit breaker, and safe optional local fallback reads.
    """

    config: TwinrConfig
    read_client: ChonkyDBClient | None = None
    write_client: ChonkyDBClient | None = None
    namespace: str | None = None
    _state_lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)  # AUDIT-FIX(#5): Guard circuit state across concurrent callers.
    _circuit_open_until_monotonic: float = field(init=False, repr=False, default=0.0)
    _consecutive_failures: int = field(init=False, repr=False, default=0)
    _probe_cache_depth: int = field(init=False, repr=False, default=0)
    _probe_cache: dict[str, LongTermRemoteSnapshotProbe] = field(init=False, repr=False, default_factory=dict)
    _document_id_hints: dict[str, str] = field(init=False, repr=False, default_factory=dict)
    _snapshot_read_cache: dict[str, _CachedSnapshotRead] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the remote namespace once during construction."""

        namespace = self.namespace or _remote_namespace_for_config(self.config)
        self.namespace = _normalize_storage_token(  # AUDIT-FIX(#6): Normalize namespace once so URIs stay stable and safe.
            namespace,
            field_name="namespace",
            max_length=_MAX_NAMESPACE_LENGTH,
        )
        self._load_persisted_document_id_hints()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermRemoteStateStore":
        """Build a remote snapshot adapter from Twinr configuration."""

        namespace = _remote_namespace_for_config(config)
        if not (
            config.long_term_memory_enabled and config.long_term_memory_mode == "remote_primary"
        ):  # AUDIT-FIX(#11): Avoid creating remote clients when remote-primary mode is disabled.
            return cls(config=config, namespace=namespace)

        base_url = _strip_text(config.chonkydb_base_url)
        api_key = _strip_text(config.chonkydb_api_key)
        if not (base_url and api_key):
            return cls(config=config, namespace=namespace)

        try:  # AUDIT-FIX(#7): Fail closed on malformed client config instead of crashing during startup.
            read_client = ChonkyDBClient(
                ChonkyDBConnectionConfig(
                    base_url=base_url,
                    api_key=api_key,
                    api_key_header=config.chonkydb_api_key_header,
                    allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                    timeout_s=_coerce_timeout_s(
                        config.long_term_memory_remote_read_timeout_s,
                        default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
                    ),
                    max_response_bytes=config.chonkydb_max_response_bytes,
                )
            )
            write_client = ChonkyDBClient(
                ChonkyDBConnectionConfig(
                    base_url=base_url,
                    api_key=api_key,
                    api_key_header=config.chonkydb_api_key_header,
                    allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                    timeout_s=_coerce_timeout_s(
                        config.long_term_memory_remote_write_timeout_s,
                        default=_DEFAULT_REMOTE_WRITE_TIMEOUT_S,
                    ),
                    max_response_bytes=config.chonkydb_max_response_bytes,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on external client implementation
            _LOGGER.warning(
                "Failed to initialize ChonkyDB clients: %s",
                _redact_secrets(f"{type(exc).__name__}: {exc}", secrets=(api_key,)),
            )
            return cls(config=config, namespace=namespace)
        return cls(
            config=config,
            read_client=read_client,
            write_client=write_client,
            namespace=namespace,
        )

    @property
    def enabled(self) -> bool:
        """Return whether remote-primary long-term memory is active."""

        return self.config.long_term_memory_enabled and self.config.long_term_memory_mode == "remote_primary"

    @property
    def required(self) -> bool:
        """Return whether remote-primary failures must fail closed."""

        return self.enabled and self.config.long_term_memory_remote_required

    def status(self) -> LongTermRemoteStatus:
        """Probe whether the remote snapshot backend is ready for use.

        Returns:
            A status record containing the backend mode, readiness, and any
            operator-safe detail string.
        """

        if not self.enabled:
            return LongTermRemoteStatus(mode="disabled", ready=False)
        if self._circuit_is_open():  # AUDIT-FIX(#5): Short-circuit repeated failures so the device recovers faster under bad Wi-Fi.
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail="Remote long-term memory is temporarily cooling down after recent failures.",
            )
        if self.read_client is None or self.write_client is None:
            return LongTermRemoteStatus(mode="remote_primary", ready=False, detail="ChonkyDB is not configured.")
        try:
            instance = self._status_probe_client(self.read_client).instance()
        except Exception as exc:  # AUDIT-FIX(#5): Normalize all client-boundary failures instead of leaking unexpected exceptions.
            self._note_remote_failure()
            _LOGGER.warning("ChonkyDB health check failed: %s", self._safe_exception_text(exc))
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail=f"ChonkyDB health check failed ({type(exc).__name__}).",  # AUDIT-FIX(#8): Do not surface raw exception strings.
            )
        self._note_remote_success()
        if not bool(getattr(instance, "ready", False)):
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail="ChonkyDB instance responded but is not ready.",
            )
        return LongTermRemoteStatus(mode="remote_primary", ready=True)

    @contextmanager
    def cache_probe_reads(self) -> Iterator[None]:
        """Reuse successful snapshot probes within one bounded readiness scope."""

        with self._state_lock:
            self._probe_cache_depth += 1
            if self._probe_cache_depth == 1:
                self._probe_cache.clear()
        try:
            yield
        finally:
            with self._state_lock:
                if self._probe_cache_depth > 0:
                    self._probe_cache_depth -= 1
                if self._probe_cache_depth == 0:
                    self._probe_cache.clear()

    def load_snapshot(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool = True,
    ) -> dict[str, object] | None:
        """Load one snapshot from remote storage or a safe local fallback.

        Args:
            snapshot_kind: Logical snapshot name such as ``objects`` or
                ``midterm``.
            local_path: Optional local recovery snapshot path used in
                non-required mode when the remote backend is missing or flaky.
            prefer_cached_document_id: Reuse a previously remembered remote
                document id before resolving snapshot pointers again. This is
                enabled by default for ordinary snapshot reads because exact
                document ids are both faster and more deterministic right after
                local writes. Read-through reuse of remote results across
                ordinary reads is separately controlled by
                ``long_term_memory_remote_read_cache_ttl_s``.

        Returns:
            The loaded snapshot payload, or ``None`` when no usable payload is
            available.

        Raises:
            LongTermRemoteUnavailableError: If the remote backend is required
                and the snapshot cannot be read.
        """

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)  # AUDIT-FIX(#6): Validate snapshot IDs before they become remote keys.
        if not self.enabled:
            return None
        cached_payload = self._cached_snapshot_read(snapshot_kind=normalized_snapshot_kind)
        if cached_payload is not None:
            return cached_payload
        try:
            read_client = self._require_client(self.read_client, operation="read")
        except LongTermRemoteUnavailableError as exc:
            if self.required:
                raise
            local_payload = None
            if local_path is not None:
                local_payload = self._load_local_snapshot(
                    local_path,
                    snapshot_kind=normalized_snapshot_kind,
                )  # AUDIT-FIX(#4): Missing remote client config should still allow safe local fallback in optional mode.
            if local_payload is not None:
                return local_payload
            if self.required:
                raise
            _LOGGER.warning(
                "Remote long-term read client unavailable for %r: %s",
                normalized_snapshot_kind,
                self._safe_exception_text(exc),
            )
            return None

        probe = self._probe_snapshot_load_internal(
            snapshot_kind=normalized_snapshot_kind,
            local_path=local_path,
            prefer_cached_document_id=prefer_cached_document_id,
        )
        if probe.payload is not None:
            self._store_snapshot_read(
                snapshot_kind=normalized_snapshot_kind,
                payload=probe.payload,
                document_id=probe.document_id,
            )
            return probe.payload

        local_payload: dict[str, object] | None = None
        if local_path is not None and probe.status in {"not_found", "unavailable"}:
            local_payload = self._load_local_snapshot(
                local_path,
                snapshot_kind=normalized_snapshot_kind,
            )  # AUDIT-FIX(#1): Corrupt or unreadable local fallback data must not crash the caller.

        if probe.status == "unavailable":
            if self.required:
                raise LongTermRemoteUnavailableError(
                    probe.detail
                    or self._remote_failure_detail("read", normalized_snapshot_kind)
                )
            if local_payload is not None:
                return local_payload  # AUDIT-FIX(#4): Use local snapshot as graceful offline fallback when remote is flaky.
            return None

        if probe.status == "not_found":
            if self.required and not self.config.long_term_memory_migration_enabled:
                return None
            if local_payload is None:
                return None
            if self.config.long_term_memory_migration_enabled:
                try:
                    self.save_snapshot(snapshot_kind=normalized_snapshot_kind, payload=local_payload)
                except LongTermRemoteUnavailableError as exc:
                    if self.required:
                        raise
                    _LOGGER.warning(
                        "Failed to migrate local snapshot %r to remote store: %s",
                        normalized_snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return local_payload  # AUDIT-FIX(#4): Keep the usable local snapshot even when remote write-back fails.
            return local_payload

        return None

    def ensure_snapshot(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> bool:
        """Ensure one snapshot exists remotely, writing it if missing.

        Args:
            snapshot_kind: Logical snapshot name to check remotely.
            payload: Payload to persist if the snapshot is currently missing.

        Returns:
            ``True`` when the payload had to be written, otherwise ``False``.

        Raises:
            LongTermRemoteUnavailableError: If required remote state cannot be
                queried or written.
        """

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)  # AUDIT-FIX(#6): Keep snapshot identity validation consistent across operations.
        if not self.enabled:
            return False
        read_client = self._require_client(self.read_client, operation="read")
        probe = self.probe_snapshot_load(snapshot_kind=normalized_snapshot_kind)
        if probe.payload is not None:
            return False
        if probe.status == "unavailable":
            raise LongTermRemoteUnavailableError(
                probe.detail
                or self._remote_failure_detail("read", normalized_snapshot_kind)
            )
        self.save_snapshot(snapshot_kind=normalized_snapshot_kind, payload=payload)
        return True

    def save_snapshot(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> None:
        """Persist one snapshot payload to the remote backend.

        Args:
            snapshot_kind: Logical snapshot name to store remotely.
            payload: JSON-serializable snapshot body.

        Raises:
            LongTermRemoteUnavailableError: If the remote backend is enabled
                but unavailable for writes.
            ValueError: If ``payload`` is not JSON-safe.
        """

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)  # AUDIT-FIX(#6): Prevent malformed snapshot kinds from becoming URIs.
        if not self.enabled:
            return
        write_client = self._require_client(self.write_client, operation="write")
        read_client = self._require_client(self.read_client, operation="read")
        if self._circuit_is_open():  # AUDIT-FIX(#5): Fail fast while the remote circuit breaker is open.
            raise LongTermRemoteUnavailableError(
                "Remote long-term memory is temporarily cooling down after recent failures."
            )
        payload_dict = _mapping_dict(payload) or {}
        retry_attempts = self._retry_attempts()
        retry_backoff_s = self._retry_backoff_s()
        document_id: str | None = None
        attested_payload: dict[str, object] | None = None
        attested_source = "saved_snapshot"
        for attempt in range(retry_attempts):
            try:
                document_id = self._store_snapshot_record(
                    write_client,
                    snapshot_kind=normalized_snapshot_kind,
                    payload=payload_dict,
                    attempts=1,
                    backoff_s=0.0,
                )
            except LongTermRemoteUnavailableError:
                self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                if attempt + 1 >= retry_attempts:
                    raise
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s)
                continue
            break
        if not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            attested_probe = self._attest_saved_snapshot_readback(
                read_client,
                snapshot_kind=normalized_snapshot_kind,
                payload=payload_dict,
                document_id=document_id,
            )
            if attested_probe.payload is not None:
                document_id = attested_probe.document_id or document_id
                attested_payload = dict(attested_probe.payload)
                attested_source = attested_probe.selected_source or "saved_snapshot_attested"
            else:
                self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                if document_id is None:
                    raise LongTermRemoteUnavailableError(
                        attested_probe.detail
                        or (
                            f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                            "read back after write attestation."
                        )
                    )
                if retry_attempts <= 1:
                    raise LongTermRemoteUnavailableError(
                        attested_probe.detail
                        or (
                            f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                            "read back after write attestation."
                        )
                    )
                for retry_attempt in range(1, retry_attempts):
                    if retry_backoff_s > 0:
                        time.sleep(retry_backoff_s)
                    document_id = self._store_snapshot_record(
                        write_client,
                        snapshot_kind=normalized_snapshot_kind,
                        payload=payload_dict,
                        attempts=1,
                        backoff_s=0.0,
                    )
                    attested_probe = self._attest_saved_snapshot_readback(
                        read_client,
                        snapshot_kind=normalized_snapshot_kind,
                        payload=payload_dict,
                        document_id=document_id,
                    )
                    if attested_probe.payload is not None:
                        document_id = attested_probe.document_id or document_id
                        attested_payload = dict(attested_probe.payload)
                        attested_source = attested_probe.selected_source or "saved_snapshot_attested"
                        break
                    self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                    if retry_attempt + 1 >= retry_attempts:
                        raise LongTermRemoteUnavailableError(
                            attested_probe.detail
                            or (
                                f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                                "read back after write attestation."
                            )
                        )
        if document_id and not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            self._save_snapshot_pointer_with_attestation(
                write_client,
                read_client,
                snapshot_kind=normalized_snapshot_kind,
                document_id=document_id,
            )
            self._remember_snapshot_document_id(
                snapshot_kind=normalized_snapshot_kind,
                document_id=document_id,
                persist=True,
            )
        self._store_snapshot_read(
            snapshot_kind=normalized_snapshot_kind,
            payload=attested_payload or payload_dict,
            document_id=document_id,
        )
        self._clear_cached_probe(snapshot_kind=normalized_snapshot_kind)
        if not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            self._store_cached_probe(
                LongTermRemoteSnapshotProbe(
                    snapshot_kind=normalized_snapshot_kind,
                    status="found",
                    latency_ms=0.0,
                    document_id=document_id,
                    selected_source=attested_source,
                    payload=attested_payload or dict(payload_dict),
                )
            )

    def _require_client(self, client: ChonkyDBClient | None, *, operation: str) -> ChonkyDBClient:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    def _extract_snapshot_body(
        self,
        payload: Mapping[str, object],
        *,
        snapshot_kind: str,
    ) -> dict[str, object] | None:
        candidate = self._extract_snapshot_candidate(payload, snapshot_kind=snapshot_kind)
        if candidate is None:
            return None
        return dict(candidate.payload)

    def _extract_snapshot_candidate(
        self,
        payload: Mapping[str, object],
        *,
        snapshot_kind: str,
    ) -> _RemoteSnapshotCandidate | None:
        namespace = self.namespace or _REMOTE_NAMESPACE_PREFIX
        latest: tuple[tuple[int, float, int], _RemoteSnapshotCandidate] | None = None
        for ordinal, candidate in enumerate(self._iter_snapshot_candidates(payload)):
            schema = candidate.payload.get("schema")
            if schema is not None and schema != _SNAPSHOT_SCHEMA:  # AUDIT-FIX(#9): Ignore incompatible snapshot schemas instead of accepting stale/foreign records.
                continue
            if candidate.payload.get("namespace") != namespace:
                continue
            if candidate.payload.get("snapshot_kind") != snapshot_kind:
                continue
            body = candidate.payload.get("body")
            if isinstance(body, Mapping):
                match = _RemoteSnapshotCandidate(
                    payload=dict(body),
                    document_id=candidate.document_id,
                )
                sort_key = (*_snapshot_updated_at_sort_key(candidate.payload.get("updated_at")), ordinal)
                if latest is None or sort_key >= latest[0]:
                    latest = (sort_key, match)
        return None if latest is None else latest[1]

    def _iter_snapshot_candidates(self, payload: Mapping[str, object]) -> Iterable[_RemoteSnapshotCandidate]:
        top_level_document_id = self._candidate_document_id(payload)
        yield _RemoteSnapshotCandidate(payload=payload, document_id=top_level_document_id)

        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield _RemoteSnapshotCandidate(payload=direct, document_id=top_level_document_id)

        nested = payload.get("record")
        if isinstance(nested, Mapping):
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield _RemoteSnapshotCandidate(
                    payload=nested_payload,
                    document_id=self._candidate_document_id(nested) or top_level_document_id,
                )
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                parsed = self._parse_snapshot_content(nested_content)
                if parsed is not None:
                    yield _RemoteSnapshotCandidate(
                        payload=parsed,
                        document_id=self._candidate_document_id(nested) or top_level_document_id,
                    )

        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_snapshot_content(content)
            if parsed is not None:
                yield _RemoteSnapshotCandidate(payload=parsed, document_id=top_level_document_id)

        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                chunk_document_id = self._candidate_document_id(chunk) or top_level_document_id
                chunk_payload = chunk.get("payload")
                if isinstance(chunk_payload, Mapping):
                    yield _RemoteSnapshotCandidate(payload=chunk_payload, document_id=chunk_document_id)
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str):
                    parsed = self._parse_snapshot_content(chunk_content)
                    if parsed is not None:
                        yield _RemoteSnapshotCandidate(payload=parsed, document_id=chunk_document_id)

    def _parse_snapshot_content(self, content: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(
                content,
                parse_constant=_reject_non_finite_json_constant,  # AUDIT-FIX(#12): Keep remote snapshot parsing strict and deterministic.
            )
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None

    def _load_snapshot_via_uri(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        document_id: str | None = None,
        source: str,
        retry_not_found: bool = False,
    ) -> _RemoteSnapshotFetchResult:
        if self._circuit_is_open():  # AUDIT-FIX(#5): Skip repeat remote calls while the breaker is open.
            return _RemoteSnapshotFetchResult(
                status="unavailable",
                detail="Remote long-term memory is temporarily cooling down after recent failures.",
                selected_source=source,
            )

        last_error: Exception | None = None
        attempts = self._retry_attempts()
        backoff_s = self._retry_backoff_s()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        started = time.monotonic()
        effective_client = client
        if document_id is None:
            effective_client = self._origin_resolution_client(client)
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            try:
                payload = effective_client.fetch_full_document(
                    document_id=document_id,
                    origin_uri=None if document_id else self._snapshot_uri(snapshot_kind),
                    include_content=True,
                    max_content_chars=self._max_content_chars(local_path=local_path),
                )
            except Exception as exc:  # AUDIT-FIX(#5): Convert all client-boundary failures into stable fetch results.
                latency_ms = round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3)
                status_code = self._status_code_from_exception(exc)
                if isinstance(exc, ChonkyDBError) and exc.status_code == 404:
                    self._note_remote_success()
                    attempt_records.append(
                        LongTermRemoteFetchAttempt(
                            source=source,
                            attempt=attempt + 1,
                            status="not_found",
                            latency_ms=latency_ms,
                            status_code=status_code,
                            document_id=document_id,
                            error_type=type(exc).__name__,
                        )
                    )
                    if retry_not_found and attempt + 1 < attempts:
                        if backoff_s > 0:
                            time.sleep(backoff_s)
                        continue
                    return _RemoteSnapshotFetchResult(
                        status="not_found",
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        selected_source=source,
                        attempts=tuple(attempt_records),
                    )
                last_error = exc
                self._note_remote_failure()
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="error",
                        latency_ms=latency_ms,
                        status_code=status_code,
                        document_id=document_id,
                        error_type=type(exc).__name__,
                        detail=self._safe_exception_text(exc),
                    )
                )
                if attempt + 1 >= attempts:
                    _LOGGER.warning(
                        "Failed to read remote long-term snapshot %r: %s",
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return _RemoteSnapshotFetchResult(
                        status="unavailable",
                        detail=self._remote_failure_detail("read", snapshot_kind, exc=exc),  # AUDIT-FIX(#8): Avoid surfacing raw client exception text.
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        selected_source=source,
                        attempts=tuple(attempt_records),
                    )
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue

            if not isinstance(payload, Mapping):  # AUDIT-FIX(#9): Validate response shape before treating it as a mapping.
                self._note_remote_failure()
                _LOGGER.warning(
                    "Remote long-term snapshot %r returned payload type %s instead of Mapping.",
                    snapshot_kind,
                    type(payload).__name__,
                )
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="malformed",
                        latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                        document_id=document_id,
                        detail=f"payload_type={type(payload).__name__}",
                    )
                )
                if source != "pointer_document" and attempt + 1 < attempts:
                    if backoff_s > 0:
                        time.sleep(backoff_s)
                    continue
                return _RemoteSnapshotFetchResult(
                    status="unavailable",
                    detail=(
                        f"Remote long-term snapshot {snapshot_kind!r} returned malformed content "
                        "that Twinr could not parse."
                    ),
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    selected_source=source,
                    attempts=tuple(attempt_records),
                )

            self._note_remote_success()
            direct = self._extract_snapshot_candidate(payload, snapshot_kind=snapshot_kind)
            if direct is not None:
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="found",
                        latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                        document_id=direct.document_id or document_id,
                    )
                )
                return _RemoteSnapshotFetchResult(
                    status="found",
                    payload=dict(direct.payload),
                    document_id=direct.document_id,
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    selected_source=source,
                    attempts=tuple(attempt_records),
                )
            attempt_records.append(
                LongTermRemoteFetchAttempt(
                    source=source,
                    attempt=attempt + 1,
                    status="malformed",
                    latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                    document_id=document_id,
                    detail="snapshot_candidate_missing",
                )
            )
            if source != "pointer_document" and attempt + 1 < attempts:
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue
            return _RemoteSnapshotFetchResult(
                status="unavailable",
                detail=(
                    f"Remote long-term snapshot {snapshot_kind!r} returned malformed content "
                    "that Twinr could not parse."
                ),
                latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                selected_source=source,
                attempts=tuple(attempt_records),
            )

        return _RemoteSnapshotFetchResult(
            status="unavailable",
            detail=self._remote_failure_detail("read", snapshot_kind, exc=last_error),
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            selected_source=source,
            attempts=tuple(attempt_records),
        )

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool = False,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one remote snapshot read and preserve pointer/origin evidence.

        Args:
            snapshot_kind: Logical snapshot name to resolve remotely.
            local_path: Optional local recovery path used only for detail parity
                in the probe evidence.
            prefer_cached_document_id: When true, probe via an already learned
                exact ChonkyDB document id before falling back to the slower
                pointer/origin walk. Health checks use this to prove that the
                current remote snapshot is still readable without paying the
                full pointer-resolution cost on every warm probe.
        """

        return self._probe_snapshot_load_internal(
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            prefer_cached_document_id=prefer_cached_document_id,
        )

    def _probe_snapshot_load_internal(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one remote snapshot read, optionally reusing a learned document id."""

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)
        if not self.enabled:
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=normalized_snapshot_kind,
                status="disabled",
                latency_ms=0.0,
                detail="Remote-primary long-term memory is disabled.",
            )
        try:
            client = self._require_client(self.read_client, operation="read")
        except LongTermRemoteUnavailableError as exc:
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=normalized_snapshot_kind,
                status="unavailable",
                latency_ms=0.0,
                detail=str(exc),
            )
        cached_probe = self._cached_probe(snapshot_kind=normalized_snapshot_kind)
        if cached_probe is not None:
            return cached_probe
        if prefer_cached_document_id and not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            probe = self._load_snapshot_with_document_id_hint(
                client,
                snapshot_kind=normalized_snapshot_kind,
                local_path=local_path,
            )
        else:
            probe = self._load_snapshot_with_pointer_fallback(
                client,
                snapshot_kind=normalized_snapshot_kind,
                local_path=local_path,
            )
        self._store_cached_probe(probe)
        return probe

    def _load_snapshot_with_document_id_hint(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one snapshot by a remembered document id before pointer/origin resolution."""

        started = time.monotonic()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        hinted_document_id = self._cached_snapshot_document_id(snapshot_kind=snapshot_kind)
        if hinted_document_id:
            hinted_result = self._load_snapshot_via_uri(
                client,
                snapshot_kind=snapshot_kind,
                local_path=local_path,
                document_id=hinted_document_id,
                source="cached_document",
            )
            attempt_records.extend(hinted_result.attempts)
            if hinted_result.payload is not None:
                return LongTermRemoteSnapshotProbe(
                    snapshot_kind=snapshot_kind,
                    status=hinted_result.status,
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    detail=hinted_result.detail,
                    document_id=hinted_result.document_id or hinted_document_id,
                    selected_source=hinted_result.selected_source,
                    payload=hinted_result.payload,
                    attempts=tuple(attempt_records),
                )
            self._forget_snapshot_document_id(snapshot_kind=snapshot_kind)

        fallback = self._load_snapshot_with_pointer_fallback(
            client,
            snapshot_kind=snapshot_kind,
            local_path=local_path,
        )
        if (
            hinted_document_id
            and hinted_result.status == "unavailable"
            and fallback.payload is None
            and fallback.status == "not_found"
        ):
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=snapshot_kind,
                status="unavailable",
                latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                detail=hinted_result.detail,
                document_id=hinted_result.document_id or hinted_document_id,
                pointer_document_id=fallback.pointer_document_id,
                selected_source=hinted_result.selected_source,
                attempts=tuple(attempt_records) + fallback.attempts,
            )
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status=fallback.status,
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=fallback.detail,
            document_id=fallback.document_id,
            pointer_document_id=fallback.pointer_document_id,
            selected_source=fallback.selected_source,
            payload=fallback.payload,
            attempts=tuple(attempt_records) + fallback.attempts,
        )

    def _load_snapshot_with_pointer_fallback(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
    ) -> LongTermRemoteSnapshotProbe:
        started = time.monotonic()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        pointer_document_id: str | None = None
        pointer_result: _RemoteSnapshotFetchResult | None = None
        if not self._is_pointer_snapshot_kind(snapshot_kind):
            pointer_lookup_result = self._load_snapshot_via_uri(
                client,
                snapshot_kind=self._pointer_snapshot_kind(snapshot_kind),
                source="pointer_lookup",
            )
            attempt_records.extend(pointer_lookup_result.attempts)
            pointer_document_id = self._extract_pointer_document_id(
                snapshot_kind=snapshot_kind,
                pointer_payload=pointer_lookup_result.payload,
            )
            if pointer_document_id:
                pointer_result = self._load_snapshot_via_uri(
                    client,
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    document_id=pointer_document_id,
                    source="pointer_document",
                )
                attempt_records.extend(pointer_result.attempts)
                if pointer_result.payload is not None:
                    return LongTermRemoteSnapshotProbe(
                        snapshot_kind=snapshot_kind,
                        status=pointer_result.status,
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        detail=pointer_result.detail,
                        document_id=pointer_result.document_id,
                        pointer_document_id=pointer_document_id,
                        selected_source=pointer_result.selected_source,
                        payload=pointer_result.payload,
                        attempts=tuple(attempt_records),
                    )
                _LOGGER.warning(
                    "Remote snapshot pointer %r targeted unreadable document %r; reloading the latest origin snapshot.",
                    snapshot_kind,
                    pointer_document_id,
                )

        origin_result = self._load_snapshot_via_uri(
            client,
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            source="origin_uri",
        )
        attempt_records.extend(origin_result.attempts)
        if origin_result.payload is not None and origin_result.document_id and not self._is_pointer_snapshot_kind(snapshot_kind):
            write_client = self.write_client
            if write_client is not None:
                try:
                    self._save_snapshot_pointer(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        document_id=origin_result.document_id,
                    )
                except LongTermRemoteUnavailableError as exc:
                    _LOGGER.warning(
                        "Remote snapshot pointer %r repair failed after a successful origin read: %s",
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status=origin_result.status,
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=origin_result.detail,
            document_id=origin_result.document_id,
            pointer_document_id=pointer_document_id,
            selected_source=origin_result.selected_source,
            payload=origin_result.payload,
            attempts=tuple(attempt_records),
        )

    def _extract_pointer_document_id(
        self,
        *,
        snapshot_kind: str,
        pointer_payload: dict[str, object] | None,
    ) -> str | None:
        if pointer_payload is None:
            return None
        if pointer_payload.get("schema") != _SNAPSHOT_POINTER_SCHEMA:
            return None
        if pointer_payload.get("version") != _SNAPSHOT_POINTER_VERSION:
            return None
        if pointer_payload.get("snapshot_kind") != snapshot_kind:
            return None
        return _normalize_snapshot_document_id(pointer_payload.get("document_id"))

    def _snapshot_uri(self, snapshot_kind: str) -> str:
        namespace_segment = _encode_uri_path_segment(self.namespace or _REMOTE_NAMESPACE_PREFIX)
        snapshot_segment = _encode_uri_path_segment(
            self._normalize_snapshot_kind(snapshot_kind)
        )  # AUDIT-FIX(#6): Percent-encode each URI path segment to avoid ambiguous keys.
        return f"twinr://longterm/{namespace_segment}/{snapshot_segment}"

    def _retry_attempts(self) -> int:
        return _coerce_int(
            getattr(self.config, "long_term_memory_remote_retry_attempts", _DEFAULT_RETRY_ATTEMPTS),
            default=_DEFAULT_RETRY_ATTEMPTS,
            minimum=1,
            maximum=10,
        )  # AUDIT-FIX(#7): Clamp malformed or extreme retry configs.

    def _retry_backoff_s(self) -> float:
        return _coerce_float(
            getattr(self.config, "long_term_memory_remote_retry_backoff_s", _DEFAULT_RETRY_BACKOFF_S),
            default=_DEFAULT_RETRY_BACKOFF_S,
            minimum=0.0,
            maximum=30.0,
        )  # AUDIT-FIX(#7): Clamp invalid backoff config to sane bounds.

    def _store_snapshot_record(
        self,
        write_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        attempts: int | None = None,
        backoff_s: float | None = None,
    ) -> str | None:
        namespace = self.namespace or _REMOTE_NAMESPACE_PREFIX
        updated_at = _utcnow_iso()
        started = time.monotonic()
        snapshot_document = {
            "schema": _SNAPSHOT_SCHEMA,
            "namespace": namespace,
            "snapshot_kind": snapshot_kind,
            "updated_at": updated_at,
            "body": dict(payload),
        }
        try:  # AUDIT-FIX(#12): Reject non-JSON-safe payloads before handing them to the remote client.
            snapshot_content = _safe_json_text(snapshot_document)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Snapshot payload must be JSON-serializable and contain only finite JSON values."
            ) from exc

        record = ChonkyDBRecordRequest(
            payload=snapshot_document,
            metadata={
                "twinr_namespace": namespace,
                "twinr_snapshot_kind": snapshot_kind,
                "twinr_snapshot_updated_at": updated_at,
                "twinr_snapshot_schema": _SNAPSHOT_SCHEMA,
            },
            uri=self._snapshot_uri(snapshot_kind),
            content=snapshot_content,
            enable_chunking=False,
            include_insights_in_response=False,
            execution_mode="async",
        )
        last_error: Exception | None = None
        resolved_attempts = max(1, int(attempts or self._retry_attempts()))
        resolved_backoff_s = max(0.0, float(backoff_s if backoff_s is not None else self._retry_backoff_s()))
        for attempt in range(resolved_attempts):
            try:
                result = write_client.store_record(record)
                failure_detail = _store_result_failure_detail(result)
                if failure_detail:
                    raise ChonkyDBError(
                        f"ChonkyDB rejected remote snapshot write: {failure_detail}",
                        response_json=dict(result) if isinstance(result, Mapping) else None,
                    )
                self._note_remote_success()
                return _extract_store_document_id(result)
            except Exception as exc:  # AUDIT-FIX(#5): Catch all remote client failures, not only ChonkyDBError subclasses.
                last_error = exc
                self._note_remote_failure()
                if attempt + 1 >= resolved_attempts:
                    break
                if resolved_backoff_s > 0:
                    time.sleep(resolved_backoff_s)
        if last_error is not None:
            record_remote_write_diagnostic(
                remote_state=self,
                context=LongTermRemoteWriteContext(
                    snapshot_kind=snapshot_kind,
                    operation="store_snapshot_record",
                    uri_hint=record.uri,
                    attempt_count=resolved_attempts,
                    request_item_count=1,
                ),
                exc=last_error,
                started_monotonic=started,
                outcome="failed",
            )
            _LOGGER.warning(
                "Failed to write remote long-term snapshot %r: %s",
                snapshot_kind,
                self._safe_exception_text(last_error),
            )
        raise LongTermRemoteUnavailableError(
            self._remote_failure_detail("write", snapshot_kind, exc=last_error)  # AUDIT-FIX(#8): Keep outward-facing errors generic and secret-safe.
        ) from last_error

    def _attest_saved_snapshot_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        document_id: str | None,
    ) -> LongTermRemoteSnapshotProbe:
        """Verify one accepted snapshot write resolves to the expected payload.

        Async ChonkyDB writes may acknowledge before the corresponding
        ``origin_uri`` read exposes the new head. When the write response did not
        return a stable ``document_id``, poll readback instead of resubmitting the
        same snapshot under the same URI, because duplicate writes only amplify the
        append-only same-URI history that produced the stale read in the first
        place.
        """

        expected_payload = dict(payload)
        return self._attest_expected_snapshot_readback(
            read_client,
            snapshot_kind=snapshot_kind,
            expected_payload=expected_payload,
            document_id=document_id,
            source="saved_document" if document_id else "saved_origin_uri",
            mismatch_detail=(
                f"Remote long-term snapshot {snapshot_kind!r} readback did not match the just-written payload."
            ),
        )

    def _attest_saved_pointer_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
        pointer_document_id: str | None = None,
    ) -> LongTermRemoteSnapshotProbe:
        """Verify one accepted pointer write resolves to the expected payload."""

        pointer_snapshot_kind = self._pointer_snapshot_kind(snapshot_kind)
        expected_payload = {
            "schema": _SNAPSHOT_POINTER_SCHEMA,
            "version": _SNAPSHOT_POINTER_VERSION,
            "snapshot_kind": snapshot_kind,
            "document_id": document_id,
        }
        return self._attest_expected_snapshot_readback(
            read_client,
            snapshot_kind=pointer_snapshot_kind,
            expected_payload=expected_payload,
            document_id=pointer_document_id,
            source="saved_pointer_document" if pointer_document_id else "saved_pointer",
            mismatch_detail=(
                f"Remote long-term snapshot pointer {snapshot_kind!r} could not be "
                "read back after write attestation."
            ),
        )

    def _attest_expected_snapshot_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        expected_payload: Mapping[str, object],
        document_id: str | None,
        source: str,
        mismatch_detail: str,
    ) -> LongTermRemoteSnapshotProbe:
        """Poll one just-written snapshot until the expected payload is visible.

        Exact-document reads are deterministic and therefore checked once. When the
        write acknowledgement did not return a ``document_id``, Twinr must resolve
        by ``origin_uri`` and tolerate bounded propagation lag or a temporarily
        stale same-URI head before concluding the write is unreadable.
        """

        started = time.monotonic()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        resolved_attempts = 1 if document_id else self._retry_attempts()
        backoff_s = 0.0 if document_id else self._retry_backoff_s()
        last_result = _RemoteSnapshotFetchResult(status="not_found", selected_source=source)
        expected_payload_dict = dict(expected_payload)
        poll_interval_s = 0.0
        if document_id is None:
            poll_interval_s = max(self._retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
            visibility_timeout_s = self._async_attestation_visibility_timeout_s()
            resolved_attempts = max(
                resolved_attempts,
                int(math.ceil(visibility_timeout_s / poll_interval_s)),
            )
        for attempt in range(resolved_attempts):
            result = self._load_snapshot_via_uri(
                read_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
                source=source,
                retry_not_found=False,
            )
            attempt_records.extend(result.attempts)
            last_result = result
            if result.payload == expected_payload_dict:
                return LongTermRemoteSnapshotProbe(
                    snapshot_kind=snapshot_kind,
                    status="found",
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    detail=result.detail,
                    document_id=result.document_id or document_id,
                    selected_source=result.selected_source,
                    payload=dict(result.payload or expected_payload_dict),
                    attempts=tuple(attempt_records),
                )
            if document_id is not None or result.status == "unavailable":
                break
            if attempt + 1 >= resolved_attempts:
                break
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
        detail = last_result.detail or mismatch_detail
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status="unavailable",
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=detail,
            document_id=last_result.document_id or document_id,
            selected_source=last_result.selected_source,
            attempts=tuple(attempt_records),
        )

    def _save_snapshot_pointer_with_attestation(
        self,
        write_client: ChonkyDBClient,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
    ) -> None:
        """Persist one pointer snapshot and prove that fresh readers can see it."""

        retry_attempts = self._retry_attempts()
        retry_backoff_s = self._retry_backoff_s()
        for attempt in range(retry_attempts):
            pointer_document_id = self._save_snapshot_pointer(
                write_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
            )
            attested_probe = self._attest_saved_pointer_readback(
                read_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
                pointer_document_id=pointer_document_id,
            )
            if attested_probe.payload is not None:
                return
            if pointer_document_id is None or attempt + 1 >= retry_attempts:
                raise LongTermRemoteUnavailableError(
                    attested_probe.detail
                    or (
                        f"Remote long-term snapshot pointer {snapshot_kind!r} could not be "
                        "read back after write attestation."
                    )
                )
            if retry_backoff_s > 0:
                time.sleep(retry_backoff_s)

    def _save_snapshot_pointer(
        self,
        write_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
    ) -> str | None:
        """Persist one pointer snapshot and return the exact document id when known."""

        pointer_payload = {
            "schema": _SNAPSHOT_POINTER_SCHEMA,
            "version": _SNAPSHOT_POINTER_VERSION,
            "snapshot_kind": snapshot_kind,
            "document_id": document_id,
        }
        return self._store_snapshot_record(
            write_client,
            snapshot_kind=self._pointer_snapshot_kind(snapshot_kind),
            payload=pointer_payload,
        )

    def _candidate_document_id(self, payload: Mapping[str, object]) -> str | None:
        for field_name in ("document_id", "payload_id", "chonky_id"):
            document_id = _normalize_snapshot_document_id(payload.get(field_name))
            if document_id:
                return document_id
        return None

    def _pointer_snapshot_kind(self, snapshot_kind: str) -> str:
        return f"{_SNAPSHOT_POINTER_PREFIX}{snapshot_kind}"

    def _is_pointer_snapshot_kind(self, snapshot_kind: str) -> bool:
        return snapshot_kind.startswith(_SNAPSHOT_POINTER_PREFIX)

    def _max_content_chars(self, *, local_path: Path | None) -> int:
        cap = self._max_content_chars_cap()
        configured = _coerce_int(
            getattr(self.config, "long_term_memory_remote_max_content_chars", _DEFAULT_MAX_CONTENT_CHARS),
            default=_DEFAULT_MAX_CONTENT_CHARS,
            minimum=1,
            maximum=cap,
        )
        if local_path is None:
            return configured
        candidate_path = self._validated_local_snapshot_path(local_path, warn=False)
        if candidate_path is None:
            return configured
        try:
            path_stat = candidate_path.lstat()
        except OSError:
            return configured
        if not stat.S_ISREG(path_stat.st_mode):  # AUDIT-FIX(#2): Ignore non-regular paths when sizing remote fetches.
            return configured
        requested = max(configured, int(path_stat.st_size) + 131_072)
        return min(requested, cap)  # AUDIT-FIX(#3): Cap remote fetch size so oversized local files cannot pressure RPi memory.

    def _max_content_chars_cap(self) -> int:
        return _coerce_int(
            getattr(
                self.config,
                "long_term_memory_remote_max_content_chars_cap",
                _DEFAULT_MAX_CONTENT_CHARS_CAP,
            ),
            default=_DEFAULT_MAX_CONTENT_CHARS_CAP,
            minimum=_DEFAULT_MAX_CONTENT_CHARS,
            maximum=64 * 1024 * 1024,
        )

    def _local_snapshot_max_bytes(self) -> int:
        return _coerce_int(
            getattr(
                self.config,
                "long_term_memory_remote_local_snapshot_max_bytes",
                _DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES,
            ),
            default=_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES,
            minimum=1,
            maximum=64 * 1024 * 1024,
        )

    def _circuit_breaker_cooldown_s(self) -> float:
        return _coerce_float(
            getattr(
                self.config,
                "long_term_memory_remote_circuit_breaker_cooldown_s",
                _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
            ),
            default=_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
            minimum=0.0,
            maximum=300.0,
        )

    def _circuit_is_open(self) -> bool:
        with self._state_lock:
            return time.monotonic() < self._circuit_open_until_monotonic

    def _note_remote_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until_monotonic = 0.0

    def _note_remote_failure(self) -> None:
        cooldown_s = self._circuit_breaker_cooldown_s()
        if cooldown_s <= 0:
            return
        with self._state_lock:
            self._consecutive_failures += 1
            multiplier = min(self._consecutive_failures, 4)
            self._circuit_open_until_monotonic = max(
                self._circuit_open_until_monotonic,
                time.monotonic() + (cooldown_s * multiplier),
            )

    def _remote_failure_detail(
        self,
        operation: str,
        snapshot_kind: str,
        *,
        exc: Exception | None = None,
    ) -> str:
        if exc is None:
            return f"Failed to {operation} remote long-term snapshot {snapshot_kind!r}."
        return f"Failed to {operation} remote long-term snapshot {snapshot_kind!r} ({type(exc).__name__})."

    @staticmethod
    def _status_code_from_exception(exc: Exception) -> int | None:
        """Extract an HTTP-like status code from ChonkyDB exceptions when present."""

        if isinstance(exc, ChonkyDBError):
            return exc.status_code
        return None

    def _safe_exception_text(self, exc: BaseException) -> str:
        return _redact_secrets(
            f"{type(exc).__name__}: {exc}",
            secrets=(self.config.chonkydb_api_key,),
        )

    def _normalize_snapshot_kind(self, snapshot_kind: str) -> str:
        return _normalize_storage_token(
            snapshot_kind,
            field_name="snapshot_kind",
            max_length=_MAX_SNAPSHOT_KIND_LENGTH,
        )

    def _clear_probe_cache(self) -> None:
        with self._state_lock:
            self._probe_cache.clear()

    def _clear_cached_probe(self, *, snapshot_kind: str) -> None:
        with self._state_lock:
            self._probe_cache.pop(snapshot_kind, None)

    def _remote_read_cache_ttl_s(self) -> float:
        try:
            ttl_s = float(getattr(self.config, "long_term_memory_remote_read_cache_ttl_s", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return ttl_s if math.isfinite(ttl_s) and ttl_s > 0.0 else 0.0

    def _cached_snapshot_read(self, *, snapshot_kind: str) -> dict[str, object] | None:
        if self._remote_read_cache_ttl_s() <= 0.0:
            return None
        now = time.monotonic()
        with self._state_lock:
            cached = self._snapshot_read_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._snapshot_read_cache.pop(snapshot_kind, None)
                return None
            payload = dict(cached.payload)
            document_id = cached.document_id
        if document_id:
            self._remember_snapshot_document_id(snapshot_kind=snapshot_kind, document_id=document_id)
        return payload

    def _store_snapshot_read(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        document_id: str | None,
    ) -> None:
        ttl_s = self._remote_read_cache_ttl_s()
        normalized_document_id = _normalize_snapshot_document_id(document_id)
        if ttl_s > 0.0 and normalized_document_id is not None:
            self._remember_snapshot_document_id(
                snapshot_kind=snapshot_kind,
                document_id=normalized_document_id,
            )
        if ttl_s <= 0.0:
            return
        payload_dict = _mapping_dict(payload)
        if payload_dict is None:
            return
        entry = _CachedSnapshotRead(
            payload=payload_dict,
            expires_at_monotonic=time.monotonic() + ttl_s,
            document_id=normalized_document_id,
        )
        with self._state_lock:
            self._snapshot_read_cache[snapshot_kind] = entry

    def _clear_snapshot_read(self, *, snapshot_kind: str) -> None:
        with self._state_lock:
            self._snapshot_read_cache.pop(snapshot_kind, None)

    def _cached_snapshot_document_id(self, *, snapshot_kind: str) -> str | None:
        with self._state_lock:
            return self._document_id_hints.get(snapshot_kind)

    def _remember_snapshot_document_id(
        self,
        *,
        snapshot_kind: str,
        document_id: str,
        persist: bool = False,
    ) -> None:
        normalized_document_id = _normalize_snapshot_document_id(document_id)
        if normalized_document_id is None:
            return
        snapshot_hints: dict[str, str] | None = None
        with self._state_lock:
            self._document_id_hints[snapshot_kind] = normalized_document_id
            if persist:
                snapshot_hints = self._persistent_document_id_hints_locked()
        if snapshot_hints is not None:
            self._persist_document_id_hints(snapshot_hints=snapshot_hints)

    def _forget_snapshot_document_id(self, *, snapshot_kind: str) -> None:
        snapshot_hints: dict[str, str] | None = None
        with self._state_lock:
            removed = self._document_id_hints.pop(snapshot_kind, None)
            self._snapshot_read_cache.pop(snapshot_kind, None)
            if removed is not None:
                snapshot_hints = self._persistent_document_id_hints_locked()
        if snapshot_hints is not None:
            self._persist_document_id_hints(snapshot_hints=snapshot_hints)

    def _cached_probe(self, *, snapshot_kind: str) -> LongTermRemoteSnapshotProbe | None:
        with self._state_lock:
            cached = self._probe_cache.get(snapshot_kind)
        if cached is None:
            return None
        payload = dict(cached.payload) if isinstance(cached.payload, dict) else cached.payload
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=cached.snapshot_kind,
            status=cached.status,
            latency_ms=0.0,
            detail=cached.detail,
            document_id=cached.document_id,
            pointer_document_id=cached.pointer_document_id,
            selected_source=cached.selected_source,
            payload=payload,
            attempts=cached.attempts,
        )

    def _origin_resolution_client(self, client: ChonkyDBClient) -> ChonkyDBClient:
        """Return one client tuned for cold `origin_uri` resolution."""

        target_timeout_s = self._origin_resolution_bootstrap_timeout_s()
        if float(client.config.timeout_s) >= target_timeout_s:
            return client
        return client.clone_with_timeout(target_timeout_s)

    def _async_attestation_visibility_timeout_s(self) -> float:
        """Return the bounded visibility window for async same-URI readback.

        Accepted async writes may need a few extra polling rounds before the new
        `origin_uri` head becomes visible under shared server load. Twinr should
        wait within one bounded visibility budget instead of re-submitting the
        same write and growing same-URI history.
        """

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        write_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_write_timeout_s", _DEFAULT_REMOTE_WRITE_TIMEOUT_S),
            default=_DEFAULT_REMOTE_WRITE_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
            self._status_probe_timeout_s(),
            read_timeout_s,
            min(write_timeout_s * 2.0, _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S),
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
            minimum=_DEFAULT_ASYNC_ATTESTATION_POLL_S,
            maximum=_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
        )

    def _status_probe_client(self, client: ChonkyDBClient) -> ChonkyDBClient:
        """Return one client tuned for fail-closed remote health probes.

        Health and readiness probes guard Twinr startup and fresh-reader
        attestation. They are not on the hot retrieval path and can therefore
        tolerate a bounded longer timeout than ordinary snapshot reads, which
        avoids classifying a loaded-but-live backend as down under shared
        server load.
        """

        target_timeout_s = self._status_probe_timeout_s()
        if float(client.config.timeout_s) >= target_timeout_s:
            return client
        return client.clone_with_timeout(target_timeout_s)

    def _status_probe_timeout_s(self) -> float:
        """Return the bounded timeout used for remote health and readiness probes."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        chonky_timeout_s = _coerce_timeout_s(
            getattr(self.config, "chonkydb_timeout_s", _DEFAULT_STATUS_PROBE_TIMEOUT_S),
            default=_DEFAULT_STATUS_PROBE_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_STATUS_PROBE_TIMEOUT_S,
            read_timeout_s * 2.0,
            chonky_timeout_s,
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_STATUS_PROBE_TIMEOUT_S,
            minimum=read_timeout_s,
            maximum=_DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S,
        )

    def _origin_resolution_bootstrap_timeout_s(self) -> float:
        """Return the bounded timeout used for cold current-head resolution."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S,
            read_timeout_s * 3.0,
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S,
            minimum=read_timeout_s,
            maximum=_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S,
        )

    def _document_id_hints_path(self) -> Path | None:
        """Return the local durable hint path when the storage root is valid."""

        try:
            base_dir = chonkydb_data_path(self.config)
        except Exception:
            return None
        return base_dir / _DOCUMENT_ID_HINTS_FILENAME

    def _load_persisted_document_id_hints(self) -> None:
        """Load durable write-learned snapshot document ids from local state."""

        hints_path = self._document_id_hints_path()
        if hints_path is None:
            return
        try:
            path_stat = hints_path.lstat()
        except OSError:
            return
        if not stat.S_ISREG(path_stat.st_mode):
            return
        if path_stat.st_size > _MAX_DOCUMENT_HINTS_BYTES:
            _LOGGER.warning(
                "Ignoring remote snapshot document-id hints at %s because the file exceeds %d bytes.",
                hints_path,
                _MAX_DOCUMENT_HINTS_BYTES,
            )
            return
        try:
            payload = json.loads(
                hints_path.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return
        if payload.get("schema") != _DOCUMENT_ID_HINTS_SCHEMA:
            return
        if _normalize_text(payload.get("namespace")) != self.namespace:
            return
        raw_hints = payload.get("document_ids")
        if not isinstance(raw_hints, Mapping):
            return
        loaded_hints: dict[str, str] = {}
        for raw_kind, raw_document_id in raw_hints.items():
            try:
                normalized_kind = self._normalize_snapshot_kind(str(raw_kind))
            except ValueError:
                continue
            if self._is_pointer_snapshot_kind(normalized_kind):
                continue
            normalized_document_id = _normalize_snapshot_document_id(raw_document_id)
            if normalized_document_id is None:
                continue
            loaded_hints[normalized_kind] = normalized_document_id
        if not loaded_hints:
            return
        with self._state_lock:
            self._document_id_hints.update(loaded_hints)

    def _persistent_document_id_hints_locked(self) -> dict[str, str]:
        """Return the persistable document-id hints while the state lock is held."""

        return {
            snapshot_kind: document_id
            for snapshot_kind, document_id in self._document_id_hints.items()
            if not self._is_pointer_snapshot_kind(snapshot_kind)
        }

    def _persist_document_id_hints(self, *, snapshot_hints: Mapping[str, str]) -> None:
        """Atomically persist the current write-learned snapshot document ids."""

        hints_path = self._document_id_hints_path()
        if hints_path is None:
            return
        try:
            hints_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _LOGGER.warning(
                "Failed to create remote snapshot hint directory %s: %s",
                hints_path.parent,
                self._safe_exception_text(exc),
            )
            return
        payload = {
            "schema": _DOCUMENT_ID_HINTS_SCHEMA,
            "namespace": self.namespace,
            "updated_at": _utcnow_iso(),
            "document_ids": dict(sorted(snapshot_hints.items())),
        }
        try:
            json_text = _safe_json_text(payload)
        except (TypeError, ValueError):
            return
        temp_path = hints_path.with_name(
            f"{hints_path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(json_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, hints_path)
        except OSError as exc:
            _LOGGER.warning(
                "Failed to persist remote snapshot document-id hints at %s: %s",
                hints_path,
                self._safe_exception_text(exc),
            )
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _store_cached_probe(self, probe: LongTermRemoteSnapshotProbe) -> None:
        if probe.snapshot_kind.startswith(_SNAPSHOT_POINTER_PREFIX):
            return
        with self._state_lock:
            if self._probe_cache_depth <= 0:
                return
            payload = dict(probe.payload) if isinstance(probe.payload, dict) else probe.payload
            self._probe_cache[probe.snapshot_kind] = LongTermRemoteSnapshotProbe(
                snapshot_kind=probe.snapshot_kind,
                status=probe.status,
                latency_ms=probe.latency_ms,
                detail=probe.detail,
                document_id=probe.document_id,
                pointer_document_id=probe.pointer_document_id,
                selected_source=probe.selected_source,
                payload=payload,
                attempts=probe.attempts,
            )

    def _validated_local_snapshot_path(self, local_path: Path, *, warn: bool = True) -> Path | None:
        project_root = _safe_resolve_path(Path(self.config.project_root))
        candidate = local_path if local_path.is_absolute() else (project_root / local_path)
        candidate_absolute = Path(os.path.abspath(os.fspath(candidate)))
        resolved_target = _safe_resolve_path(candidate_absolute)

        configured_memory_path = Path(self.config.long_term_memory_path)
        configured_candidate = (
            configured_memory_path
            if configured_memory_path.is_absolute()
            else (project_root / configured_memory_path)
        )
        configured_absolute = Path(os.path.abspath(os.fspath(configured_candidate)))
        resolved_configured_target = _safe_resolve_path(configured_absolute)

        if configured_absolute.exists() and configured_absolute.is_dir():
            allowed = resolved_target.is_relative_to(resolved_configured_target)
        else:
            allowed = resolved_target == resolved_configured_target

        if not allowed:
            if warn:
                _LOGGER.warning(
                    "Skipped local snapshot fallback %s because the caller supplied a path outside the configured Twinr memory root %s. This is a probe/operator path issue, not corrupted memory data.",
                    candidate_absolute,
                    configured_absolute,
                )
            return None  # AUDIT-FIX(#2): Prevent traversal/symlink escapes outside the configured Twinr memory path.
        return candidate_absolute

    def _load_local_snapshot(self, local_path: Path, *, snapshot_kind: str) -> dict[str, object] | None:
        candidate_path = self._validated_local_snapshot_path(local_path)
        if candidate_path is None:
            return None

        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(candidate_path, open_flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            _LOGGER.warning(
                "Failed to open local snapshot fallback %s for %r: %s",
                candidate_path,
                snapshot_kind,
                self._safe_exception_text(exc),
            )
            return None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):  # AUDIT-FIX(#2): Only regular files are valid snapshot fallbacks.
                _LOGGER.warning(
                    "Rejected local snapshot fallback %s for %r because it is not a regular file.",
                    candidate_path,
                    snapshot_kind,
                )
                return None
            if file_stat.st_size > self._local_snapshot_max_bytes():  # AUDIT-FIX(#3): Refuse oversized fallback files on constrained RPi hardware.
                _LOGGER.warning(
                    "Rejected local snapshot fallback %s for %r because it exceeds %s bytes.",
                    candidate_path,
                    snapshot_kind,
                    self._local_snapshot_max_bytes(),
                )
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                try:
                    payload = json.load(
                        handle,
                        parse_constant=_reject_non_finite_json_constant,  # AUDIT-FIX(#12): Reject non-standard JSON values in local fallback files too.
                    )
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
                    _LOGGER.warning(
                        "Failed to parse local snapshot fallback %s for %r: %s",
                        candidate_path,
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return None
        finally:
            if fd >= 0:
                os.close(fd)

        if not isinstance(payload, Mapping):
            _LOGGER.warning(
                "Rejected local snapshot fallback %s for %r because the JSON root is %s, not an object.",
                candidate_path,
                snapshot_kind,
                type(payload).__name__,
            )
            return None
        return dict(payload)

def _remote_namespace_for_config(config: TwinrConfig) -> str:
    """Derive the stable remote namespace for one Twinr configuration."""

    override = _normalize_text(config.long_term_memory_remote_namespace)
    if override:
        return _normalize_storage_token(
            override,
            field_name="namespace",
            max_length=_MAX_NAMESPACE_LENGTH,
        )  # AUDIT-FIX(#6): Validate configured namespace overrides before they become URI components.
    root = _safe_resolve_path(Path(config.project_root))  # AUDIT-FIX(#7): Avoid crashing on odd paths or symlink loops during namespace derivation.
    memory_path = Path(config.long_term_memory_path)
    resolved_memory_path = memory_path if memory_path.is_absolute() else (root / memory_path)
    resolved_memory_path = _safe_resolve_path(resolved_memory_path)
    digest = hashlib.sha1(str(resolved_memory_path).encode("utf-8")).hexdigest()[:12]
    stem = _normalize_text(root.name) or "twinr"
    return _normalize_storage_token(
        f"{_REMOTE_NAMESPACE_PREFIX}:{stem}:{digest}",
        field_name="namespace",
        max_length=_MAX_NAMESPACE_LENGTH,
    )


__all__ = [
    "LongTermRemoteStateStore",
    "LongTermRemoteStatus",
    "LongTermRemoteUnavailableError",
]
