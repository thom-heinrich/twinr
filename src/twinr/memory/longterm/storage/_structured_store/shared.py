"""Shared constants and helper functions for structured long-term storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import time

from twinr.agent.workflows.forensics import workflow_event
from twinr.text_utils import retrieval_terms


_OBJECT_STORE_SCHEMA = "twinr_memory_object_store"
_OBJECT_STORE_VERSION = 1
_OBJECT_STORE_MANIFEST_SCHEMA = "twinr_memory_object_store_manifest"
_OBJECT_STORE_SHARD_SCHEMA = "twinr_memory_object_store_shard"
_CONFLICT_STORE_SCHEMA = "twinr_memory_conflict_store"
_CONFLICT_STORE_VERSION = 1
_ARCHIVE_STORE_SCHEMA = "twinr_memory_archive_store"
_ARCHIVE_STORE_VERSION = 1
_ARCHIVE_STORE_MANIFEST_SCHEMA = "twinr_memory_archive_store_manifest"
_ARCHIVE_STORE_SHARD_SCHEMA = "twinr_memory_archive_store_shard"
_SNAPSHOT_WRITTEN_AT_KEY = "written_at"
_MIN_AWARE_DATETIME = datetime.min.replace(tzinfo=timezone.utc)
_CROSS_SERVICE_READ_MODE = 0o644
_OBJECT_STATE_QUERY_TERMS = frozenset(
    {
        "active",
        "aktuell",
        "available",
        "bestaetigt",
        "candidate",
        "confirmed",
        "confirmed_by_user",
        "current",
        "discarded",
        "expired",
        "former",
        "frueher",
        "gespeichert",
        "invalid",
        "outdated",
        "pending",
        "previous",
        "stored",
        "superseded",
        "unbestaetigt",
        "unclear",
        "uncertain",
        "unklar",
        "unconfirmed",
        "user_confirmed",
        "vorher",
    }
)
_NON_SEMANTIC_ATTRIBUTE_KEYS = frozenset(
    {
        "support_count",
        "event_names",
    }
)

_LOG = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


def _retrieval_trace_details(
    query_text: str | None,
    *,
    episodic_limit: int | None = None,
    durable_limit: int | None = None,
    candidate_limit: int | None = None,
    payload_count: int | None = None,
    entry_count: int | None = None,
) -> dict[str, object]:
    """Summarize retrieval inputs for trace-safe latency diagnostics."""

    clean_query = _normalize_text(query_text)
    details: dict[str, object] = {
        "query_chars": len(clean_query),
        "query_terms": len(tuple(term for term in retrieval_terms(clean_query) if isinstance(term, str))),
    }
    if episodic_limit is not None:
        details["episodic_limit"] = max(0, int(episodic_limit))
    if durable_limit is not None:
        details["durable_limit"] = max(0, int(durable_limit))
    if candidate_limit is not None:
        details["candidate_limit"] = max(0, int(candidate_limit))
    if payload_count is not None:
        details["payload_count"] = max(0, int(payload_count))
    if entry_count is not None:
        details["entry_count"] = max(0, int(entry_count))
    return details


def _run_timed_workflow_step(
    *,
    name: str,
    kind: str,
    details: dict[str, object],
    operation: Callable[[], object],
) -> object:
    """Emit bounded timing events for one retrieval step without span rethrow bugs."""

    workflow_event(kind="span_start", msg=name, details={"kind": kind, **details})
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        workflow_event(
            kind="exception",
            msg=f"{name}_exception",
            level="ERROR",
            details={
                "span": name,
                "kind": kind,
                "exception": {"type": type(exc).__name__},
            },
            kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        )
        raise
    workflow_event(
        kind="span_end",
        msg=name,
        details={"kind": kind, **details},
        kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
    )
    return result


def _utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""

    return datetime.now(timezone.utc)


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce a value to a positive integer or fall back to ``default``."""

    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _coerce_aware_utc(value: object) -> datetime:
    """Normalize a datetime-like value into an aware UTC timestamp."""

    if not isinstance(value, datetime):
        return _MIN_AWARE_DATETIME
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_snapshot_written_at(payload: Mapping[str, object]) -> datetime:
    """Parse the stored snapshot write time or return the minimum sentinel."""

    raw_value = payload.get(_SNAPSHOT_WRITTEN_AT_KEY)
    if not isinstance(raw_value, str) or not raw_value:
        return _MIN_AWARE_DATETIME
    try:
        return _coerce_aware_utc(datetime.fromisoformat(raw_value))
    except ValueError:
        return _MIN_AWARE_DATETIME


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry to disk after an atomic file replacement."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object atomically and durably within ``path.parent``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fchmod(handle.fileno(), _CROSS_SERVICE_READ_MODE)
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
        os.chmod(path, _CROSS_SERVICE_READ_MODE)
        _fsync_directory(path.parent)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
