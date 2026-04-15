"""Shared constants, helper functions, and public value types for remote state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import chonkydb_data_path

_LOGGER = logging.getLogger("twinr.memory.longterm.storage.remote_state")

_REMOTE_NAMESPACE_PREFIX = "twinr_longterm_v1"
_SNAPSHOT_SCHEMA = "twinr_remote_snapshot_v1"
_SNAPSHOT_POINTER_SCHEMA = "twinr_remote_snapshot_pointer_v1"
_SNAPSHOT_POINTER_VERSION = 1
_SNAPSHOT_POINTER_PREFIX = "__pointer__:"
_DOCUMENT_ID_HINTS_SCHEMA = "twinr_remote_snapshot_document_hints_v1"
_DOCUMENT_ID_HINTS_FILENAME = "remote_snapshot_document_hints.json"
_DEFAULT_REMOTE_READ_TIMEOUT_S = 10.0
_DEFAULT_REMOTE_WRITE_TIMEOUT_S = 15.0
_DEFAULT_REMOTE_FLUSH_TIMEOUT_S = 60.0
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_S = 0.5
_DEFAULT_MAX_CONTENT_CHARS = 262_144
_DEFAULT_MAX_CONTENT_CHARS_CAP = 8_388_608
_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES = 8_388_608
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S = 15.0
_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S = 20.0
_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S = 30.0
_DEFAULT_STATUS_PROBE_TIMEOUT_S = 20.0
_DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S = 45.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S = 20.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S = 45.0
_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S = 20.0
_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S = 180.0
_DEFAULT_ASYNC_ATTESTATION_POLL_S = 0.05
_MAX_NAMESPACE_LENGTH = 255
_MAX_SNAPSHOT_KIND_LENGTH = 255
_MAX_DOCUMENT_HINTS_BYTES = 262_144
_EXPLICIT_REMOTE_TRANSIENT_DETAILS = frozenset(
    {
        "service warmup in progress",
        "upstream unavailable or restarting",
        "warmup_pending",
        "query_surface_unhealthy: warmup_pending",
    }
)


def _utcnow_iso() -> str:
    """Return the current UTC time encoded as ISO 8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


def is_explicit_remote_transient_detail(detail: str | None) -> bool:
    """Return whether one backend detail explicitly reports restart/warmup transit."""

    normalized = _normalize_text(detail).lower()
    return normalized in _EXPLICIT_REMOTE_TRANSIENT_DETAILS


def remote_snapshot_document_hints_path(config: TwinrConfig) -> Path | None:
    """Return the local durable document-id hint file used by remote snapshots."""

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


def _reject_non_finite_json_constant(value: str) -> object:
    """Reject non-finite JSON constants during strict snapshot parsing."""

    raise ValueError(f"Unsupported JSON constant {value!r}.")


def _safe_json_text(payload: Mapping[str, object]) -> str:
    """Serialize a mapping to strict JSON text for remote transport."""

    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
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

    if not isinstance(value, (bool, int, float, str)):
        return default
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

    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return default
    else:
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
            document_id = _candidate_document_id(item)
            if document_id:
                return document_id
    return _candidate_document_id(result)


def _extract_store_job_id(result: Mapping[str, object] | None) -> str | None:
    """Extract one async ChonkyDB job identifier from a store response when present."""

    if not isinstance(result, Mapping):
        return None
    raw_job_id = result.get("job_id")
    normalized = _strip_text(raw_job_id if isinstance(raw_job_id, str) else None)
    return normalized or None


def _candidate_document_id(payload: Mapping[str, object]) -> str | None:
    """Extract one document id from common ChonkyDB response field names."""

    for field_name in ("document_id", "payload_id", "chonky_id"):
        document_id = _normalize_snapshot_document_id(payload.get(field_name))
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
    detail = _normalize_text(str(result.get("detail") or ""))
    if detail:
        return detail
    error = _normalize_text(str(result.get("error") or ""))
    if error:
        return error
    return None


def _safe_resolve_path(path: Path) -> Path:
    """Resolve one path without requiring the target to already exist."""

    try:
        return path.expanduser().resolve(strict=False)
    except RuntimeError:
        return Path(path.expanduser())


def _normalize_storage_token(value: str, *, field_name: str, max_length: int) -> str:
    """Normalize and validate namespace/snapshot identifiers."""

    normalized = _strip_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if any(ch.isspace() for ch in normalized):
        raise ValueError(f"{field_name} must not contain whitespace.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters.")
    return normalized


def _encode_uri_path_segment(value: str) -> str:
    """Encode one URI path segment without leaking reserved separators."""

    return quote(value, safe="")


class LongTermRemoteUnavailableError(RuntimeError):
    """Signal that required remote long-term snapshot state is unavailable."""

    def __init__(self, message: str) -> None:
        self.message = str(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class LongTermRemoteReadFailedError(LongTermRemoteUnavailableError):
    """Signal that one required remote read failed despite a configured backend."""

    def __init__(self, message: str, *, details: Mapping[str, object] | None = None) -> None:
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LongTermRemoteStatus:
    """Describe whether the remote snapshot backend is ready for use.

    `ready=False` can still allow one deeper operational readiness proof when
    the shallow backend status route responded but withheld its startup-ready
    bit. That state is surfaced explicitly via `operational_probe_allowed`
    instead of overloading `ready=True`.
    """

    mode: str
    ready: bool
    detail: str | None = None
    operational_probe_allowed: bool = False


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


def _remote_namespace_for_config(config: TwinrConfig) -> str:
    """Derive the stable remote namespace for one Twinr configuration."""

    override = _normalize_text(config.long_term_memory_remote_namespace)
    if override:
        return _normalize_storage_token(
            override,
            field_name="namespace",
            max_length=_MAX_NAMESPACE_LENGTH,
        )
    root = _safe_resolve_path(Path(config.project_root))
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
    "LongTermRemoteFetchAttempt",
    "LongTermRemoteReadFailedError",
    "LongTermRemoteSnapshotProbe",
    "LongTermRemoteStatus",
    "LongTermRemoteUnavailableError",
    "_CachedSnapshotRead",
    "_DEFAULT_ASYNC_ATTESTATION_POLL_S",
    "_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S",
    "_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S",
    "_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S",
    "_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S",
    "_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S",
    "_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES",
    "_DEFAULT_MAX_CONTENT_CHARS",
    "_DEFAULT_MAX_CONTENT_CHARS_CAP",
    "_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S",
    "_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S",
    "_DEFAULT_REMOTE_FLUSH_TIMEOUT_S",
    "_DEFAULT_REMOTE_READ_TIMEOUT_S",
    "_DEFAULT_REMOTE_WRITE_TIMEOUT_S",
    "_DEFAULT_RETRY_ATTEMPTS",
    "_DEFAULT_RETRY_BACKOFF_S",
    "_DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S",
    "_DEFAULT_STATUS_PROBE_TIMEOUT_S",
    "_DOCUMENT_ID_HINTS_FILENAME",
    "_DOCUMENT_ID_HINTS_SCHEMA",
    "_LOGGER",
    "_MAX_DOCUMENT_HINTS_BYTES",
    "_MAX_NAMESPACE_LENGTH",
    "_MAX_SNAPSHOT_KIND_LENGTH",
    "_REMOTE_NAMESPACE_PREFIX",
    "_RemoteSnapshotCandidate",
    "_RemoteSnapshotFetchResult",
    "_SNAPSHOT_POINTER_PREFIX",
    "_SNAPSHOT_POINTER_SCHEMA",
    "_SNAPSHOT_POINTER_VERSION",
    "_SNAPSHOT_SCHEMA",
    "_candidate_document_id",
    "_coerce_float",
    "_coerce_int",
    "_coerce_timeout_s",
    "_encode_uri_path_segment",
    "_extract_store_document_id",
    "_extract_store_job_id",
    "_mapping_dict",
    "_normalize_snapshot_document_id",
    "_normalize_storage_token",
    "_normalize_text",
    "_redact_secrets",
    "_reject_non_finite_json_constant",
    "_remote_namespace_for_config",
    "_safe_json_text",
    "_safe_resolve_path",
    "_snapshot_updated_at_sort_key",
    "_store_result_failure_detail",
    "_strip_text",
    "_utcnow_iso",
    "is_explicit_remote_transient_detail",
    "remote_snapshot_document_hints_path",
]
