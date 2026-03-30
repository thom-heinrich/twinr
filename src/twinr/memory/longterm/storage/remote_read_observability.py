"""Persist histogram and alert artifacts for remote long-term request operations.

The historical module name stays stable for import compatibility, but the
artifact contract now covers both read and write traffic. This keeps the
request-path classification logic in one focused module so catalog/state
adapters can emit concise operational telemetry without embedding histogram
bookkeeping or alert policy in their main I/O paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import threading
import tempfile
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from twinr.memory.longterm.storage.remote_read_diagnostics import (
        LongTermRemoteReadContext,
        LongTermRemoteWriteContext,
    )


_LOG = logging.getLogger(__name__)
_PROJECT_ROOT_OVERRIDE_ENV = "TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"
_HISTOGRAM_SCHEMA = "twinr_longterm_remote_request_histograms_v2"
_ALERT_TIMEOUT_CLASS = "timeout"
_SLOW_ALERT_LATENCY_MS = 2_000.0
_ARTIFACT_FILE_MODE = 0o644
_LOCK_FILE_MODE = 0o666
_HISTOGRAM_BUCKETS_MS: tuple[tuple[float, str], ...] = (
    (50.0, "lt_50_ms"),
    (100.0, "50_100_ms"),
    (250.0, "100_250_ms"),
    (500.0, "250_500_ms"),
    (1_000.0, "500_1000_ms"),
    (2_000.0, "1000_2000_ms"),
    (5_000.0, "2000_5000_ms"),
)
_SUPPORTED_OPERATIONS = frozenset(
    {
        "snapshot_load",
        "load_catalog_current_head",
        "fetch_item_document",
        "fetch_catalog_segment",
        "retrieve_search",
        "retrieve_batch",
        "topk_search",
        "topk_batch",
        "fast_topic_topk_search",
        "graph_neighbors_query",
        "graph_path_query",
        "store_snapshot_record",
        "store_records_bulk",
        "graph_store_many",
    }
)
_STATE_LOCK = threading.Lock()


def _log_ops_event_append_failure(
    *,
    logger: logging.Logger,
    event_kind: str,
    request_kind: str,
    exc: BaseException,
) -> None:
    """Keep optional ops-event append failures quiet for foreign-owned Pi stores."""

    if isinstance(exc, PermissionError):
        logger.debug(
            "Skipping remote long-term %s %s event because the ops event store is not writable in this runtime user context: %s",
            request_kind,
            event_kind,
            exc,
        )
        return
    logger.warning(
        "Failed to append remote long-term %s %s event.",
        request_kind,
        event_kind,
        exc_info=True,
    )


def record_remote_request_observation(
    *,
    remote_state: object | None,
    context: "LongTermRemoteReadContext | LongTermRemoteWriteContext",
    latency_ms: float,
    outcome: str,
    classification: str,
    request_kind: str,
) -> None:
    """Update persisted histograms and emit explicit ops alerts when needed."""

    operation = str(getattr(context, "operation", "") or "").strip()
    if operation not in _SUPPORTED_OPERATIONS:
        return
    store = _ops_event_store(remote_state)
    project_root = _project_root(remote_state)
    if store is None or project_root is None:
        return

    normalized_request_kind = _normalize_request_kind(request_kind)
    normalized_outcome = str(outcome or "ok").strip().lower() or "ok"
    normalized_classification = str(classification or "ok").strip().lower() or "ok"
    bounded_latency_ms = max(0.0, float(latency_ms))
    access_classification = resolve_remote_access_classification(
        request_kind=normalized_request_kind,
        context=context,
    )
    data = {
        "request_kind": normalized_request_kind,
        "snapshot_kind": _normalize_text(getattr(context, "snapshot_kind", None)),
        "operation": operation,
        "request_method": _normalize_text(getattr(context, "request_method", None)),
        "request_path": _normalize_text(getattr(context, "request_path", None)),
        "request_payload_kind": _normalize_text(getattr(context, "request_payload_kind", None)),
        "access_classification": access_classification,
        "outcome": normalized_outcome,
        "classification": normalized_classification,
        "latency_ms": round(bounded_latency_ms, 3),
        "latency_bucket": _latency_bucket(bounded_latency_ms),
        "allowed_doc_count": _normalize_int(getattr(context, "allowed_doc_count", None)),
        "result_limit": _normalize_int(getattr(context, "result_limit", None)),
        "batch_size": _normalize_int(getattr(context, "batch_size", None)),
        "catalog_entry_count": _normalize_int(getattr(context, "catalog_entry_count", None)),
    }

    histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
    try:
        with _STATE_LOCK:
            with _histogram_file_lock(histogram_path):
                payload = _load_histogram_payload(histogram_path)
                _update_histogram_payload(payload, data)
                _write_json_atomic(histogram_path, payload)
    except Exception:
        _LOG.warning("Failed to persist remote-read histogram artifact.", exc_info=True)
        return

    if (
        normalized_request_kind == "read"
        or (normalized_request_kind == "write" and normalized_outcome == "ok")
    ) and (normalized_classification == _ALERT_TIMEOUT_CLASS or bounded_latency_ms >= _SLOW_ALERT_LATENCY_MS):
        alert_level = "error" if normalized_outcome == "failed" else "warning"
        alert_kind = "timeout" if normalized_classification == _ALERT_TIMEOUT_CLASS else "slow_read"
        alert_data = dict(data)
        alert_data["alert_kind"] = alert_kind
        alert_data["histogram_path"] = str(histogram_path)
        message = (
            f"Remote long-term {normalized_request_kind} {data['snapshot_kind'] or 'unknown'} {operation} "
            f"hit {alert_kind} at {data['latency_ms']} ms."
        )
        try:
            store.append(
                event=f"longterm_remote_{normalized_request_kind}_alert",
                level=alert_level,
                message=message,
                data=alert_data,
            )
        except Exception as exc:
            _log_ops_event_append_failure(
                logger=_LOG,
                event_kind="alert",
                request_kind=normalized_request_kind,
                exc=exc,
            )


def record_remote_read_observation(
    *,
    remote_state: object | None,
    context: "LongTermRemoteReadContext",
    latency_ms: float,
    outcome: str,
    classification: str,
) -> None:
    """Compatibility wrapper for read-path observations."""

    record_remote_request_observation(
        remote_state=remote_state,
        context=context,
        latency_ms=latency_ms,
        outcome=outcome,
        classification=classification,
        request_kind="read",
    )


def record_remote_write_observation(
    *,
    remote_state: object | None,
    context: "LongTermRemoteWriteContext",
    latency_ms: float,
    outcome: str,
    classification: str,
) -> None:
    """Emit one successful or degraded write-path observation."""

    record_remote_request_observation(
        remote_state=remote_state,
        context=context,
        latency_ms=latency_ms,
        outcome=outcome,
        classification=classification,
        request_kind="write",
    )


def _project_root(remote_state: object | None) -> Path | None:
    config = getattr(remote_state, "config", None)
    project_root = os.environ.get(_PROJECT_ROOT_OVERRIDE_ENV) or getattr(config, "project_root", None)
    if not project_root:
        return None
    return Path(str(project_root)).expanduser().resolve(strict=False)


def _ops_event_store(remote_state: object | None) -> object | None:
    project_root = _project_root(remote_state)
    if project_root is None:
        return None
    try:
        from twinr.ops.events import TwinrOpsEventStore

        return TwinrOpsEventStore.from_project_root(project_root)
    except Exception:
        _LOG.warning("Failed to initialize remote-read alert store.", exc_info=True)
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refresh_fd_mode_best_effort(fd: int, mode: int) -> None:
    """Best-effort chmod for shared lock files that may already be foreign-owned."""

    try:
        current_mode = os.fstat(fd).st_mode & 0o777
    except OSError:
        return
    if current_mode == mode:
        return
    try:
        os.fchmod(fd, mode)
    except PermissionError:
        return


def _normalize_text(value: object | None) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _normalize_request_kind(value: object | None) -> str:
    normalized = " ".join(str(value or "").split()).strip().lower()
    return normalized if normalized in {"read", "write"} else "read"


def _normalize_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latency_bucket(latency_ms: float) -> str:
    for upper_bound, bucket_name in _HISTOGRAM_BUCKETS_MS:
        if latency_ms < upper_bound:
            return bucket_name
    return "gte_5000_ms"


def resolve_remote_access_classification(
    *,
    request_kind: str,
    context: "LongTermRemoteReadContext | LongTermRemoteWriteContext",
) -> str:
    """Normalize one request into the coarse live-path class Twinr cares about."""

    explicit = _normalize_text(getattr(context, "access_classification", None))
    if explicit:
        return explicit
    normalized_request_kind = _normalize_request_kind(request_kind)
    operation = _normalize_text(getattr(context, "operation", None)) or ""
    payload_kind = _normalize_text(getattr(context, "request_payload_kind", None)) or ""
    if normalized_request_kind == "write":
        if operation == "graph_store_many":
            return "graph_topology_write"
        if operation == "store_snapshot_record":
            return "legacy_snapshot_write"
        uri_hint = _normalize_text(getattr(context, "uri_hint", None)) or ""
        if uri_hint.endswith("/catalog/current"):
            return "catalog_current_head_write"
        return "record_bulk_write"
    if payload_kind == "catalog_current_head_document" or operation == "load_catalog_current_head":
        return "catalog_current_head"
    if payload_kind == "topk_scope_query" or operation in {"topk_search", "fast_topic_topk_search"}:
        return "topk_scope_query"
    if payload_kind in {
        "topk_allowed_doc_batch",
        "retrieve_allowed_doc_batch",
        "topk_allowed_doc_query",
        "retrieve_allowed_doc_query",
        "graph_neighbors_query",
        "graph_path_query",
    } or operation in {"topk_batch", "retrieve_batch", "retrieve_search", "graph_neighbors_query", "graph_path_query"}:
        return "retrieve_batch" if operation in {"topk_batch", "retrieve_batch", "retrieve_search"} else "topk_scope_query"
    if payload_kind in {"full_document_lookup", "catalog_segment_document"} or operation in {
        "fetch_item_document",
        "fetch_catalog_segment",
    }:
        return "documents_full"
    if operation == "snapshot_load" or payload_kind in {
        "document_id_cached_head",
        "document_id_pointer_head",
        "origin_uri_pointer_lookup",
        "origin_uri_snapshot_head",
        "snapshot_lookup",
    }:
        return "legacy_snapshot_compat"
    return "unknown"


def _load_histogram_payload(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    except Exception:
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    if not isinstance(loaded, Mapping):
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    operations = loaded.get("operations")
    return {
        "schema": _HISTOGRAM_SCHEMA,
        "version": 1,
        "updated_at": _utc_now_iso(),
        "operations": dict(operations) if isinstance(operations, Mapping) else {},
    }


def _update_histogram_payload(payload: dict[str, object], data: Mapping[str, object]) -> None:
    operations = payload.setdefault("operations", {})
    assert isinstance(operations, dict)
    request_kind = _normalize_request_kind(data.get("request_kind"))
    operation = str(data.get("operation") or "unknown")
    snapshot_kind = str(data.get("snapshot_kind") or "unknown")
    key = f"{snapshot_kind}:{operation}"
    request_method = _normalize_text(data.get("request_method"))
    request_path = _normalize_text(data.get("request_path"))
    request_payload_kind = _normalize_text(data.get("request_payload_kind"))
    access_classification = _normalize_text(data.get("access_classification"))
    request_endpoint = None
    if request_method and request_path:
        request_endpoint = f"{request_method} {request_path}"
    entry = operations.get(key)
    if not isinstance(entry, dict):
        entry = {
            "request_kind": request_kind,
            "snapshot_kind": snapshot_kind,
            "operation": operation,
            "total_count": 0,
            "outcome_counts": {},
            "classification_counts": {},
            "request_kind_counts": {},
            "access_classification_counts": {},
            "latency_buckets_ms": {},
            "request_endpoint_counts": {},
            "request_payload_kind_counts": {},
            "last_latency_ms": 0.0,
            "last_updated_at": _utc_now_iso(),
        }
        operations[key] = entry
    entry["total_count"] = int(entry.get("total_count", 0)) + 1
    outcome_counts = dict(entry.get("outcome_counts") or {})
    classification_counts = dict(entry.get("classification_counts") or {})
    request_kind_counts = dict(entry.get("request_kind_counts") or {})
    access_classification_counts = dict(entry.get("access_classification_counts") or {})
    latency_buckets = dict(entry.get("latency_buckets_ms") or {})
    request_endpoint_counts = dict(entry.get("request_endpoint_counts") or {})
    request_payload_kind_counts = dict(entry.get("request_payload_kind_counts") or {})
    outcome = str(data.get("outcome") or "ok")
    classification = str(data.get("classification") or "ok")
    latency_bucket = str(data.get("latency_bucket") or _latency_bucket(float(data.get("latency_ms") or 0.0)))
    outcome_counts[outcome] = int(outcome_counts.get(outcome, 0)) + 1
    classification_counts[classification] = int(classification_counts.get(classification, 0)) + 1
    request_kind_counts[request_kind] = int(request_kind_counts.get(request_kind, 0)) + 1
    if access_classification is not None:
        access_classification_counts[access_classification] = int(
            access_classification_counts.get(access_classification, 0)
        ) + 1
    latency_buckets[latency_bucket] = int(latency_buckets.get(latency_bucket, 0)) + 1
    if request_endpoint is not None:
        request_endpoint_counts[request_endpoint] = int(request_endpoint_counts.get(request_endpoint, 0)) + 1
    if request_payload_kind is not None:
        request_payload_kind_counts[request_payload_kind] = int(request_payload_kind_counts.get(request_payload_kind, 0)) + 1
    entry["outcome_counts"] = outcome_counts
    entry["classification_counts"] = classification_counts
    entry["request_kind_counts"] = request_kind_counts
    entry["access_classification_counts"] = access_classification_counts
    entry["latency_buckets_ms"] = latency_buckets
    entry["request_endpoint_counts"] = request_endpoint_counts
    entry["request_payload_kind_counts"] = request_payload_kind_counts
    entry["last_latency_ms"] = float(data.get("latency_ms") or 0.0)
    entry["last_request_kind"] = request_kind
    entry["last_access_classification"] = access_classification
    entry["last_outcome"] = outcome
    entry["last_classification"] = classification
    entry["last_request_method"] = request_method
    entry["last_request_path"] = request_path
    entry["last_request_endpoint"] = request_endpoint
    entry["last_request_payload_kind"] = request_payload_kind
    entry["last_updated_at"] = _utc_now_iso()
    payload["updated_at"] = entry["last_updated_at"]


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fchmod(handle.fileno(), _ARTIFACT_FILE_MODE)
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        os.chmod(path, _ARTIFACT_FILE_MODE)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def _histogram_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    lock_handle = None
    try:
        lock_fd = os.open(lock_path, lock_flags, _LOCK_FILE_MODE)
        _refresh_fd_mode_best_effort(lock_fd, _LOCK_FILE_MODE)
        lock_handle = os.fdopen(lock_fd, "a+", encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()

__all__ = [
    "record_remote_read_observation",
    "record_remote_request_observation",
    "record_remote_write_observation",
    "resolve_remote_access_classification",
]
