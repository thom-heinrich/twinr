"""Capture structured diagnostics for remote long-term memory I/O failures.

This module keeps ChonkyDB failure classification out of the catalog and
snapshot adapters so operators can distinguish timeouts, DNS/transport
failures, backend HTTP errors, and client-contract payload issues from the
shared ops event stream without bloating the storage orchestration path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import TYPE_CHECKING
from urllib.error import URLError

from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation

if TYPE_CHECKING:
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_LOG = logging.getLogger(__name__)
_MAX_ERROR_TEXT_CHARS = 240
_CLIENT_CONTRACT_ERRORS = (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError)
_PROJECT_ROOT_OVERRIDE_ENV = "TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"


@dataclass(frozen=True, slots=True)
class LongTermRemoteReadContext:
    """Describe one remote ChonkyDB read that may need operator diagnostics."""

    snapshot_kind: str
    operation: str
    item_id: str | None = None
    document_id_hint: str | None = None
    uri_hint: str | None = None
    query_text: str | None = None
    catalog_entry_count: int | None = None
    allowed_doc_count: int | None = None
    result_limit: int | None = None
    segment_index: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class LongTermRemoteWriteContext:
    """Describe one remote ChonkyDB write that may need operator diagnostics."""

    snapshot_kind: str
    operation: str
    document_id_hint: str | None = None
    uri_hint: str | None = None
    attempt_count: int | None = None
    request_item_count: int | None = None
    request_correlation_id: str | None = None
    batch_index: int | None = None
    batch_count: int | None = None
    request_bytes: int | None = None


def extract_remote_write_context(exc: BaseException | None) -> dict[str, object] | None:
    """Return normalized bulk-write correlation metadata attached to one exception."""

    if exc is None:
        return None
    raw_context = getattr(exc, "remote_write_context", None)
    if not isinstance(raw_context, Mapping):
        return None
    context = {
        "snapshot_kind": _normalize_text(raw_context.get("snapshot_kind")),
        "operation": _normalize_text(raw_context.get("operation")),
        "request_correlation_id": _normalize_text(raw_context.get("request_correlation_id")),
        "batch_index": _normalize_int(raw_context.get("batch_index")),
        "batch_count": _normalize_int(raw_context.get("batch_count")),
        "request_item_count": _normalize_int(raw_context.get("request_item_count")),
        "request_bytes": _normalize_int(raw_context.get("request_bytes")),
    }
    normalized = {key: value for key, value in context.items() if value is not None}
    return normalized or None


def record_remote_read_diagnostic(
    *,
    remote_state: LongTermRemoteStateStore | object | None,
    context: LongTermRemoteReadContext,
    exc: BaseException,
    started_monotonic: float,
    outcome: str = "failed",
) -> None:
    """Append one sanitized remote-read ops event when a read degrades or fails."""

    _record_remote_request_diagnostic(
        remote_state=remote_state,
        context=context,
        exc=exc,
        started_monotonic=started_monotonic,
        outcome=outcome,
        request_kind="read",
    )


def record_remote_write_diagnostic(
    *,
    remote_state: LongTermRemoteStateStore | object | None,
    context: LongTermRemoteWriteContext,
    exc: BaseException,
    started_monotonic: float,
    outcome: str = "failed",
) -> None:
    """Append one sanitized remote-write ops event when a write fails."""

    _record_remote_request_diagnostic(
        remote_state=remote_state,
        context=context,
        exc=exc,
        started_monotonic=started_monotonic,
        outcome=outcome,
        request_kind="write",
    )


def _classify_remote_request_exception(exc: BaseException) -> str:
    chain = tuple(_exception_chain(exc))
    if _is_timeout_chain(chain):
        return "timeout"
    if _is_dns_resolution_chain(chain):
        return "dns_resolution_error"
    for item in chain:
        if not isinstance(item, ChonkyDBError):
            continue
        if item.status_code is not None:
            return "backend_http_error" if int(item.status_code) >= 500 else "client_contract_error"
        message = " ".join(str(item).lower().split())
        if item.response_json is not None or "invalid payload" in message or "non-json" in message:
            return "client_contract_error"
    if any(isinstance(item, _CLIENT_CONTRACT_ERRORS) for item in chain):
        return "client_contract_error"
    if any(isinstance(item, (URLError, OSError)) for item in chain):
        return "transport_error"
    return "unexpected_error"


def _classify_remote_read_exception(exc: BaseException) -> str:
    return _classify_remote_request_exception(exc)


def _ops_event_store(remote_state: LongTermRemoteStateStore | object | None) -> object | None:
    config = getattr(remote_state, "config", None)
    project_root = os.environ.get(_PROJECT_ROOT_OVERRIDE_ENV) or getattr(config, "project_root", None)
    if not project_root:
        return None
    try:
        from twinr.ops.events import TwinrOpsEventStore

        return TwinrOpsEventStore.from_project_root(Path(project_root))
    except Exception:
        _LOG.warning("Failed to initialize remote long-term read diagnostics store.", exc_info=True)
        return None


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_timeout_chain(chain: tuple[BaseException, ...]) -> bool:
    if any(isinstance(item, (TimeoutError, socket.timeout)) for item in chain):
        return True
    for item in chain:
        message = " ".join(str(item).lower().split())
        if "timed out" in message or "timeout" in message:
            return True
    return False


def _is_dns_resolution_chain(chain: tuple[BaseException, ...]) -> bool:
    if any(isinstance(item, socket.gaierror) for item in chain):
        return True
    dns_markers = (
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided",
        "getaddrinfo",
    )
    for item in chain:
        message = " ".join(str(item).lower().split())
        if any(marker in message for marker in dns_markers):
            return True
    return False


def _root_cause(exc: BaseException) -> BaseException:
    chain = _exception_chain(exc)
    return chain[-1] if chain else exc


def _status_code_from_exception(exc: BaseException) -> int | None:
    for item in _exception_chain(exc):
        if isinstance(item, ChonkyDBError) and item.status_code is not None:
            try:
                return int(item.status_code)
            except (TypeError, ValueError):
                return None
    return None


def _response_json_keys(exc: BaseException) -> list[str]:
    for item in _exception_chain(exc):
        if isinstance(item, ChonkyDBError) and isinstance(item.response_json, dict):
            return [str(key) for key in sorted(item.response_json)[:12]]
    return []


def _response_json_value(exc: BaseException, key: str) -> str | None:
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return None
    for item in _exception_chain(exc):
        if not isinstance(item, ChonkyDBError) or not isinstance(item.response_json, dict):
            continue
        value = item.response_json.get(normalized_key)
        text = _normalize_text(value)
        if text:
            return text
    return None


def _response_text_excerpt(exc: BaseException) -> str | None:
    for item in _exception_chain(exc):
        if not isinstance(item, ChonkyDBError):
            continue
        text = _normalize_text(item.response_text)
        if text:
            return text
    return None


def _query_sha256(query_text: str | None) -> str | None:
    normalized = " ".join(str(query_text or "").split()).strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _query_term_count(query_text: str | None) -> int:
    normalized = " ".join(str(query_text or "").split()).strip()
    if not normalized:
        return 0
    return len(normalized.split(" "))


def _normalize_text(value: object | None) -> str | None:
    text = _compact_text(str(value or ""), limit=160).strip()
    return text or None


def _normalize_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_text(value: object | None, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    normalized_limit = max(int(limit), 0)
    if len(text) <= normalized_limit:
        return text
    if normalized_limit <= 3:
        return text[:normalized_limit]
    return text[: normalized_limit - 3].rstrip() + "..."


def _record_remote_request_diagnostic(
    *,
    remote_state: LongTermRemoteStateStore | object | None,
    context: LongTermRemoteReadContext | LongTermRemoteWriteContext,
    exc: BaseException,
    started_monotonic: float,
    outcome: str,
    request_kind: str,
) -> None:
    store = _ops_event_store(remote_state)
    if store is None:
        return

    normalized_request_kind = str(request_kind or "read").strip().lower() or "read"
    normalized_outcome = str(outcome or "failed").strip().lower() or "failed"
    classification = _classify_remote_request_exception(exc)
    root_cause = _root_cause(exc)
    elapsed_ms = max(0.0, (time.monotonic() - float(started_monotonic)) * 1000.0)
    data: dict[str, object] = {
        "request_kind": normalized_request_kind,
        "snapshot_kind": _normalize_text(getattr(context, "snapshot_kind", None)),
        "operation": _normalize_text(getattr(context, "operation", None)),
        "outcome": normalized_outcome,
        "classification": classification,
        "latency_ms": round(elapsed_ms, 3),
        "namespace": _normalize_text(getattr(remote_state, "namespace", None)),
        "item_id": _normalize_text(getattr(context, "item_id", None)),
        "document_id_hint": _normalize_text(getattr(context, "document_id_hint", None)),
        "uri_hint": _compact_text(_normalize_text(getattr(context, "uri_hint", None)), limit=160),
        "segment_index": _normalize_int(getattr(context, "segment_index", None)),
        "catalog_entry_count": _normalize_int(getattr(context, "catalog_entry_count", None)),
        "allowed_doc_count": _normalize_int(getattr(context, "allowed_doc_count", None)),
        "result_limit": _normalize_int(getattr(context, "result_limit", None)),
        "batch_size": _normalize_int(getattr(context, "batch_size", None)),
        "attempt_count": _normalize_int(getattr(context, "attempt_count", None)),
        "request_item_count": _normalize_int(getattr(context, "request_item_count", None)),
        "request_correlation_id": _normalize_text(getattr(context, "request_correlation_id", None)),
        "batch_index": _normalize_int(getattr(context, "batch_index", None)),
        "batch_count": _normalize_int(getattr(context, "batch_count", None)),
        "request_bytes": _normalize_int(getattr(context, "request_bytes", None)),
        "query_sha256": _query_sha256(getattr(context, "query_text", None)),
        "query_term_count": _query_term_count(getattr(context, "query_text", None)),
        "query_chars": len(str(getattr(context, "query_text", None) or "")),
        "read_timeout_s": _normalize_float(
            getattr(getattr(remote_state, "config", None), "long_term_memory_remote_read_timeout_s", None)
        )
        if normalized_request_kind == "read"
        else None,
        "write_timeout_s": _normalize_float(
            getattr(getattr(remote_state, "config", None), "long_term_memory_remote_write_timeout_s", None)
        )
        if normalized_request_kind == "write"
        else None,
        "error_type": type(exc).__name__,
        "error_message": _compact_text(str(exc), limit=_MAX_ERROR_TEXT_CHARS),
        "root_cause_type": type(root_cause).__name__,
        "root_cause_message": _compact_text(str(root_cause), limit=_MAX_ERROR_TEXT_CHARS),
        "exception_chain": [type(item).__name__ for item in _exception_chain(exc)],
    }
    if isinstance(root_cause, OSError):
        data["root_cause_errno"] = _normalize_int(getattr(root_cause, "errno", None))
    status_code = _status_code_from_exception(exc)
    if status_code is not None:
        data["status_code"] = status_code
    response_keys = _response_json_keys(exc)
    if response_keys:
        data["response_json_keys"] = response_keys
    response_detail = _response_json_value(exc, "detail")
    if response_detail is None:
        response_detail = _response_text_excerpt(exc)
    if response_detail is not None:
        data["response_detail"] = _compact_text(response_detail, limit=_MAX_ERROR_TEXT_CHARS)
    response_error = _response_json_value(exc, "error")
    if response_error is not None:
        data["response_error"] = _compact_text(response_error, limit=_MAX_ERROR_TEXT_CHARS)
    response_error_type = _response_json_value(exc, "error_type")
    if response_error_type is not None:
        data["response_error_type"] = _compact_text(response_error_type, limit=80)

    event_name = (
        f"longterm_remote_{normalized_request_kind}_failed"
        if normalized_outcome == "failed"
        else f"longterm_remote_{normalized_request_kind}_degraded"
    )
    action = "failed" if normalized_outcome == "failed" else "degraded to fallback"
    message = (
        f"Remote long-term {data['snapshot_kind'] or 'unknown'} {data['operation'] or normalized_request_kind} "
        f"{action} ({classification})."
    )
    request_correlation_id = data.get("request_correlation_id")
    batch_index = data.get("batch_index")
    batch_count = data.get("batch_count")
    if isinstance(request_correlation_id, str) and request_correlation_id:
        message = f"{message[:-1]} request_id={request_correlation_id}."
    if isinstance(batch_index, int) and isinstance(batch_count, int) and batch_count > 0:
        message = f"{message[:-1]} batch={batch_index}/{batch_count}."
    try:
        store.append(
            event=event_name,
            level="warning",
            message=message,
            data=data,
        )
    except Exception:
        _LOG.warning("Failed to append remote long-term %s diagnostic event.", normalized_request_kind, exc_info=True)
    if normalized_request_kind == "read":
        record_remote_read_observation(
            remote_state=remote_state,
            context=context,
            latency_ms=elapsed_ms,
            outcome=normalized_outcome,
            classification=classification,
        )
