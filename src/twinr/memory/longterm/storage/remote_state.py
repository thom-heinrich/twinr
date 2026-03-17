"""Mirror long-term memory snapshots to the remote-primary backend.

This module provides ``LongTermRemoteStateStore``, the ChonkyDB-backed remote
snapshot adapter used when long-term memory runs in remote-primary mode. It
validates snapshot identifiers, bounds remote I/O, and exposes required-mode
failures through ``LongTermRemoteUnavailableError``.
"""

from __future__ import annotations

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
from typing import Iterable, Mapping
from urllib.parse import quote  # AUDIT-FIX(#6): Encode URI path segments safely.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig, ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBRecordRequest


_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#5): Keep operational visibility for degraded-mode events.

_REMOTE_NAMESPACE_PREFIX = "twinr_longterm_v1"
_SNAPSHOT_SCHEMA = "twinr_remote_snapshot_v1"
_SNAPSHOT_POINTER_SCHEMA = "twinr_remote_snapshot_pointer_v1"
_SNAPSHOT_POINTER_VERSION = 1
_SNAPSHOT_POINTER_PREFIX = "__pointer__:"
_DEFAULT_REMOTE_READ_TIMEOUT_S = 10.0  # AUDIT-FIX(#7): Safe fallback defaults for malformed .env values.
_DEFAULT_REMOTE_WRITE_TIMEOUT_S = 15.0
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_S = 0.5
_DEFAULT_MAX_CONTENT_CHARS = 262_144
_DEFAULT_MAX_CONTENT_CHARS_CAP = 8_388_608  # 8 MiB cap for RPi-friendly snapshot fetches.
_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES = 8_388_608
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S = 15.0
_MAX_NAMESPACE_LENGTH = 255
_MAX_SNAPSHOT_KIND_LENGTH = 255


def _utcnow_iso() -> str:
    """Return the current UTC time encoded as ISO 8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


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

    def __post_init__(self) -> None:
        """Normalize the remote namespace once during construction."""

        namespace = self.namespace or _remote_namespace_for_config(self.config)
        self.namespace = _normalize_storage_token(  # AUDIT-FIX(#6): Normalize namespace once so URIs stay stable and safe.
            namespace,
            field_name="namespace",
            max_length=_MAX_NAMESPACE_LENGTH,
        )

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
            instance = self.read_client.instance()
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

    def load_snapshot(self, *, snapshot_kind: str, local_path: Path | None = None) -> dict[str, object] | None:
        """Load one snapshot from remote storage or a safe local fallback.

        Args:
            snapshot_kind: Logical snapshot name such as ``objects`` or
                ``midterm``.
            local_path: Optional local recovery snapshot path used in
                non-required mode when the remote backend is missing or flaky.

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

        probe = self.probe_snapshot_load(snapshot_kind=normalized_snapshot_kind, local_path=local_path)
        if probe.payload is not None:
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
        if self._circuit_is_open():  # AUDIT-FIX(#5): Fail fast while the remote circuit breaker is open.
            raise LongTermRemoteUnavailableError(
                "Remote long-term memory is temporarily cooling down after recent failures."
            )

        document_id = self._store_snapshot_record(
            write_client,
            snapshot_kind=normalized_snapshot_kind,
            payload=_mapping_dict(payload) or {},
        )
        if document_id and not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            self._save_snapshot_pointer(
                write_client,
                snapshot_kind=normalized_snapshot_kind,
                document_id=document_id,
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
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            try:
                payload = client.fetch_full_document(
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
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one remote snapshot read and preserve pointer/origin evidence."""

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
        return self._load_snapshot_with_pointer_fallback(
            client,
            snapshot_kind=normalized_snapshot_kind,
            local_path=local_path,
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
    ) -> str | None:
        namespace = self.namespace or _REMOTE_NAMESPACE_PREFIX
        updated_at = _utcnow_iso()
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
        )
        last_error: Exception | None = None
        attempts = self._retry_attempts()
        backoff_s = self._retry_backoff_s()
        for attempt in range(attempts):
            try:
                result = write_client.store_record(record)
                self._note_remote_success()
                return _extract_store_document_id(result)
            except Exception as exc:  # AUDIT-FIX(#5): Catch all remote client failures, not only ChonkyDBError subclasses.
                last_error = exc
                self._note_remote_failure()
                if attempt + 1 >= attempts:
                    break
                if backoff_s > 0:
                    time.sleep(backoff_s)
        if last_error is not None:
            _LOGGER.warning(
                "Failed to write remote long-term snapshot %r: %s",
                snapshot_kind,
                self._safe_exception_text(last_error),
            )
        raise LongTermRemoteUnavailableError(
            self._remote_failure_detail("write", snapshot_kind, exc=last_error)  # AUDIT-FIX(#8): Keep outward-facing errors generic and secret-safe.
        ) from last_error

    def _save_snapshot_pointer(
        self,
        write_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
    ) -> None:
        pointer_payload = {
            "schema": _SNAPSHOT_POINTER_SCHEMA,
            "version": _SNAPSHOT_POINTER_VERSION,
            "snapshot_kind": snapshot_kind,
            "document_id": document_id,
        }
        self._store_snapshot_record(
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
        candidate_path = self._validated_local_snapshot_path(local_path)
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

    def _validated_local_snapshot_path(self, local_path: Path) -> Path | None:
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
            _LOGGER.warning(
                "Rejected local snapshot path %s because it escapes the configured Twinr memory path.",
                candidate_absolute,
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
