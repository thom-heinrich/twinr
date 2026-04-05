# CHANGELOG: 2026-03-27
# BUG-1: Root-trace binding no longer leaves workflow_event()/workflow_decision() outside spans with per-event random trace IDs; bind_workflow_forensics() now materializes one stable root trace ID.
# BUG-2: The writer no longer flushes both files on every record; it now batches writes and forces flush/fsync on critical events, reducing Raspberry-Pi SD-card churn and write-path latency.
# BUG-3: TWINR_WORKFLOW_TRACE_SCOPE and TWINR_WORKFLOW_TRACE_SAMPLE_RATE are now actually implemented; they were previously defined but ignored.
# BUG-4: Critical lifecycle records (run_end / exception) now bypass the normal max-event truncation path so the trace does not silently lose its most important terminal evidence.
# SEC-1: Privacy redaction now also applies to msg/body-like payloads and adds practical PII/token heuristics, reducing accidental storage of user utterances, phone numbers, emails, and credentials.
# SEC-2: Existing trace directories are chmod-hardened when possible, and atomic sidecar writes now fsync both file and directory for power-loss-safe replacement on Linux-class deployments.
# IMP-1: Records now carry OpenTelemetry-aligned resource/instrumentation metadata, severity numbers, W3C traceparent helpers, and subprocess env-carrier helpers.
# IMP-2: Context propagation now includes helpers for child threads and process env carriers, matching 2026 trace propagation practice for edge/IoT workflows.
# IMP-3: Trace retention is now enforced locally to prevent long-lived Raspberry-Pi deployments from accumulating unbounded run packs.
# BREAKING: Automatic retention cleanup now prunes old run packs by default. Set TWINR_WORKFLOW_TRACE_RETENTION_MAX_RUNS=0, TWINR_WORKFLOW_TRACE_RETENTION_MAX_TOTAL_BYTES=0, and TWINR_WORKFLOW_TRACE_RETENTION_MAX_AGE_DAYS=0 to disable.
# BREAKING: from_env() now defaults TWINR_WORKFLOW_TRACE_MODE to "balanced" instead of the old implicit "forensic" behavior, because per-record flushing was the main Pi-side bottleneck.

"""Forensic-grade workflow tracing for Twinr hardware loops.

This module provides a bounded, structured run pack for difficult live-runtime
bugs. It is intentionally workflow-local so button/session/audio issues can be
traced without scattering JSON-writing logic across the orchestration code.

2026 upgrade notes
------------------
* Keeps the original drop-in API intact.
* Adds OpenTelemetry-aligned metadata and W3C trace-context helpers.
* Adds context propagation helpers for helper threads and child processes.
* Adds batched file writes plus retention controls suited to Raspberry Pi 4.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
from types import TracebackType
from typing import Any, Iterator, TextIO, TypeVar
import atexit
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid


_MODULE_VERSION = "2026.03.27"
_SCHEMA_VERSION = "twinr.workflow.forensics/2026-03-27"
_INSTRUMENTATION_SCOPE_NAME = "twinr.workflow_forensics"
_TRACE_FILE_KINDS: frozenset[str] = frozenset({"span_start", "span_end", "exception", "decision"})
_CRITICAL_RECORD_KINDS: frozenset[str] = frozenset({"run_start", "run_end", "exception"})
_DEFAULT_TRACE_FLAGS = "01"

_TRACE_ENV_ENABLED = "TWINR_WORKFLOW_TRACE_ENABLED"
_TRACE_ENV_MODE = "TWINR_WORKFLOW_TRACE_MODE"
_TRACE_ENV_DIR = "TWINR_WORKFLOW_TRACE_DIR"
_TRACE_ENV_SCOPE = "TWINR_WORKFLOW_TRACE_SCOPE"
_TRACE_ENV_SAMPLE_RATE = "TWINR_WORKFLOW_TRACE_SAMPLE_RATE"
_TRACE_ENV_ALLOW_RAW_TEXT = "TWINR_WORKFLOW_TRACE_ALLOW_RAW_TEXT"
_TRACE_ENV_QUEUE_MAXSIZE = "TWINR_WORKFLOW_TRACE_QUEUE_MAXSIZE"
_TRACE_ENV_MAX_EVENTS = "TWINR_WORKFLOW_TRACE_MAX_EVENTS"
_TRACE_ENV_MAX_MSG_CARDINALITY = "TWINR_WORKFLOW_TRACE_MAX_MSG_CARDINALITY"
_TRACE_ENV_MAX_SPAN_HISTORY = "TWINR_WORKFLOW_TRACE_MAX_SPAN_HISTORY"
_TRACE_ENV_FLUSH_INTERVAL_MS = "TWINR_WORKFLOW_TRACE_FLUSH_INTERVAL_MS"
_TRACE_ENV_FLUSH_EVERY_EVENTS = "TWINR_WORKFLOW_TRACE_FLUSH_EVERY_EVENTS"
_TRACE_ENV_FSYNC_INTERVAL_MS = "TWINR_WORKFLOW_TRACE_FSYNC_INTERVAL_MS"
_TRACE_ENV_RETENTION_MAX_RUNS = "TWINR_WORKFLOW_TRACE_RETENTION_MAX_RUNS"
_TRACE_ENV_RETENTION_MAX_TOTAL_BYTES = "TWINR_WORKFLOW_TRACE_RETENTION_MAX_TOTAL_BYTES"
_TRACE_ENV_RETENTION_MAX_AGE_DAYS = "TWINR_WORKFLOW_TRACE_RETENTION_MAX_AGE_DAYS"
_QUEUE_SENTINEL = object()
_SUMMARY_FLUSH_INTERVAL = 25
_MAX_DETAIL_TEXT = 240
_MAX_STACK_LINES = 12
_DEFAULT_QUEUE_MAXSIZE = 2048
_DEFAULT_MAX_EVENTS = 5000
_DEFAULT_MAX_MSG_CARDINALITY = 1024
_DEFAULT_MAX_SPAN_HISTORY = 256
_DEFAULT_ALLOW_RAW_TEXT = False
_DEFAULT_FLUSH_INTERVAL_MS = 250
_DEFAULT_FLUSH_EVERY_EVENTS = 32
_DEFAULT_FSYNC_INTERVAL_MS = 2000
_DEFAULT_RETENTION_MAX_RUNS = 32
_DEFAULT_RETENTION_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_DEFAULT_RETENTION_MAX_AGE_DAYS = 14
_COUNT_OVERFLOW_BUCKET = "[other]"

_SECRET_KEYWORDS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "code",
    "cookie",
    "credential",
    "dob",
    "email",
    "lat",
    "lon",
    "otp",
    "passphrase",
    "password",
    "patient",
    "phone",
    "pin",
    "secret",
    "session",
    "ssn",
    "token",
)
_RAW_TEXT_HINT_KEYWORDS: tuple[str, ...] = (
    "body",
    "content",
    "input",
    "message",
    "output",
    "prompt",
    "question",
    "response",
    "text",
    "transcript",
    "utterance",
)
_RAW_TEXT_SAFE_KEYS: tuple[str, ...] = (
    "cwd",
    "event_kind",
    "file",
    "func",
    "host",
    "hostname",
    "kind",
    "line",
    "mode",
    "module",
    "platform",
    "python",
    "scope",
    "service",
    "stack",
    "status",
    "trace_flags",
    "type",
    "traceparent",
)
_ARG_SECRET_PREFIXES: tuple[str, ...] = (
    "--api-key",
    "--apikey",
    "--authorization",
    "--bearer",
    "--cookie",
    "--password",
    "--secret",
    "--token",
)
_ENV_WHITELIST: tuple[str, ...] = (
    "TWINR_WORKFLOW_TRACE_ENABLED",
    "TWINR_WORKFLOW_TRACE_MODE",
    "TWINR_WORKFLOW_TRACE_SCOPE",
    "TWINR_WORKFLOW_TRACE_SAMPLE_RATE",
    "TWINR_WORKFLOW_TRACE_ALLOW_RAW_TEXT",
    "TWINR_WORKFLOW_TRACE_QUEUE_MAXSIZE",
    "TWINR_WORKFLOW_TRACE_MAX_EVENTS",
    "TWINR_WORKFLOW_TRACE_MAX_MSG_CARDINALITY",
    "TWINR_WORKFLOW_TRACE_MAX_SPAN_HISTORY",
    "TWINR_WORKFLOW_TRACE_FLUSH_INTERVAL_MS",
    "TWINR_WORKFLOW_TRACE_FLUSH_EVERY_EVENTS",
    "TWINR_WORKFLOW_TRACE_FSYNC_INTERVAL_MS",
    "TWINR_WORKFLOW_TRACE_RETENTION_MAX_RUNS",
    "TWINR_WORKFLOW_TRACE_RETENTION_MAX_TOTAL_BYTES",
    "TWINR_WORKFLOW_TRACE_RETENTION_MAX_AGE_DAYS",
    "TWINR_LLM_PROVIDER",
    "TWINR_STT_PROVIDER",
    "TWINR_TTS_PROVIDER",
    "TWINR_LONG_TERM_MEMORY_MODE",
    "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED",
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d .\-()]{6,}\d)")
_HEX_TOKEN_RE = re.compile(r"\b[a-f0-9]{24,}\b", re.IGNORECASE)
_B64ISH_RE = re.compile(r"\b[A-Za-z0-9_\-/+=]{32,}\b")
_SAFE_EVENT_NAME_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,96}$")
_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-(?P<span_id>[0-9a-f]{16})-(?P<trace_flags>[0-9a-f]{2})$"
)

_LOGGER = logging.getLogger(__name__)

_ACTIVE_TRACER: ContextVar["WorkflowForensics | None"] = ContextVar(
    "twinr_workflow_active_tracer",
    default=None,
)
_ACTIVE_TRACE_ID: ContextVar[str | None] = ContextVar(
    "twinr_workflow_active_trace_id",
    default=None,
)
_ACTIVE_SPAN_ID: ContextVar[str | None] = ContextVar(
    "twinr_workflow_active_span_id",
    default=None,
)
_ACTIVE_PARENT_SPAN_HINT: ContextVar[str | None] = ContextVar(
    "twinr_workflow_active_parent_span_hint",
    default=None,
)
_ACTIVE_TRACE_FLAGS: ContextVar[str] = ContextVar(
    "twinr_workflow_active_trace_flags",
    default=_DEFAULT_TRACE_FLAGS,
)
_ACTIVE_TRACE_STATE: ContextVar[str | None] = ContextVar(
    "twinr_workflow_active_tracestate",
    default=None,
)

T = TypeVar("T")


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple dotenv-style ``KEY=VALUE`` pairs for trace configuration."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        return {}
    values: dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _normalize_trace_base_dir(raw_value: str, *, project_root: Path) -> Path:
    """Normalize one configured workflow-trace directory for the active checkout.

    Twinr commonly syncs the leading repo `.env` onto the Pi acceptance checkout.
    Repo-owned trace directories therefore need to survive a root move from
    `/home/thh/twinr` to `/twinr`. When a configured absolute path still points
    at another Twinr checkout's `state/forensics/...` subtree, rebase that tail
    onto the active `project_root` instead of writing into the foreign checkout.
    """

    project_root_resolved = project_root.expanduser().resolve(strict=False)
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        return candidate
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.is_relative_to(project_root_resolved):
        return resolved_candidate
    parts = resolved_candidate.parts
    for index in range(len(parts) - 1):
        if parts[index] != "state" or parts[index + 1] != "forensics":
            continue
        return (project_root_resolved / Path(*parts[index:])).resolve(strict=False)
    return resolved_candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_id() -> str:
    while True:
        trace_id = uuid.uuid4().hex
        if trace_id != "0" * 32:
            return trace_id


def _span_id() -> str:
    while True:
        span = uuid.uuid4().hex[:16]
        if span != "0" * 16:
            return span


def _token() -> str:
    return uuid.uuid4().hex


def _safe_str(value: object, *, default: str = "") -> str:
    try:
        return str(value if value is not None else default)
    except Exception:
        fallback_type = type(value).__name__ if value is not None else "unknown"
        return f"<unstringifiable:{fallback_type}>"


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unreprable:{type(value).__name__}>"


def _bounded_text(value: object, *, default: str = "") -> str:
    text = _safe_str(value, default=default).strip()
    if not text:
        return default
    if len(text) > _MAX_DETAIL_TEXT:
        return f"{text[:_MAX_DETAIL_TEXT - 3]}..."
    return text


def _redacted_text_summary(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"[redacted_text len={len(text)} sha256={digest}]"


def _looks_sensitive_text(text: str) -> bool:
    return bool(
        _EMAIL_RE.search(text)
        or _PHONE_RE.search(text)
        or _HEX_TOKEN_RE.search(text)
        or _B64ISH_RE.search(text)
    )


def _should_redact_text(text: str, *, allow_raw_text: bool, key_hint: str | None) -> bool:
    if allow_raw_text:
        return False
    lowered_key = (key_hint or "").casefold()
    if lowered_key == "msg" and _SAFE_EVENT_NAME_RE.fullmatch(text):
        return False
    if lowered_key in _RAW_TEXT_SAFE_KEYS:
        return False
    if any(secret in lowered_key for secret in _SECRET_KEYWORDS):
        return True
    if any(marker in lowered_key for marker in _RAW_TEXT_HINT_KEYWORDS):
        return True
    if _looks_sensitive_text(text):
        return True
    if "\n" in text or "\r" in text or "\t" in text:
        return True
    if text and any(char.isspace() for char in text):
        return True
    if len(text) > 64 and not _SAFE_EVENT_NAME_RE.fullmatch(text):
        return True
    return False


def _sanitize_message_text(text: object, *, allow_raw_text: bool) -> str:
    bounded = _bounded_text(text, default="unknown")
    if _should_redact_text(bounded, allow_raw_text=allow_raw_text, key_hint="msg"):
        return _redacted_text_summary(bounded)
    return bounded


def _sanitize_details(
    value: object,
    *,
    allow_raw_text: bool = True,
    key_hint: str | None = None,
) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = _bounded_text(value)
        if _should_redact_text(text, allow_raw_text=allow_raw_text, key_hint=key_hint):
            return _redacted_text_summary(text)
        return text
    if isinstance(value, Path):
        return _sanitize_details(str(value), allow_raw_text=allow_raw_text, key_hint=key_hint)
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, item in list(value.items())[:64]:
            key_text = _bounded_text(key, default="_")
            lowered = key_text.casefold()
            if any(secret in lowered for secret in _SECRET_KEYWORDS):
                cleaned[key_text] = "[redacted]"
                continue
            cleaned[key_text] = _sanitize_details(
                item,
                allow_raw_text=allow_raw_text,
                key_hint=key_text,
            )
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _sanitize_details(item, allow_raw_text=allow_raw_text, key_hint=key_hint)
            for item in list(value)[:32]
        ]
    return _sanitize_details(_safe_repr(value), allow_raw_text=allow_raw_text, key_hint=key_hint)


def _normalize_details_payload(details: object | None) -> dict[str, object]:
    if details is None:
        return {}
    if isinstance(details, dict):
        payload = dict(details)
        exception_payload = payload.get("exception")
        if isinstance(exception_payload, str):
            normalized = exception_payload.strip()
            if normalized:
                payload["exception"] = {
                    "type": normalized.split(":", 1)[0].strip() or "unknown",
                    "message": normalized,
                }
        return payload
    if isinstance(details, Mapping):
        return {str(key): value for key, value in details.items()}
    return {"value": details}


def _exception_summary_from_parts(
    exc_type: object,
    exc_value: object,
    exc_traceback: TracebackType | None,
    *,
    allow_raw_text: bool,
) -> dict[str, object]:
    try:
        if isinstance(exc_value, BaseException):
            formatted_stack = traceback.format_exception(exc_value)[-_MAX_STACK_LINES:]
        else:
            formatted_stack = [_bounded_text(exc_value)]
    except Exception:
        formatted_stack = [_bounded_text(exc_value)]
    return {
        "type": _bounded_text(getattr(exc_type, "__name__", _safe_str(exc_type)), default="unknown"),
        "message": _sanitize_details(
            _bounded_text(exc_value),
            allow_raw_text=allow_raw_text,
            key_hint="message",
        ),
        "stack": _sanitize_details(
            formatted_stack,
            allow_raw_text=allow_raw_text,
            key_hint="stack",
        ),
    }


def _exception_summary(exc: BaseException, *, allow_raw_text: bool) -> dict[str, object]:
    return _exception_summary_from_parts(
        type(exc),
        exc,
        exc.__traceback__,
        allow_raw_text=allow_raw_text,
    )


def _exception_type_from_record(record: dict[str, object]) -> str:
    details = record.get("details")
    if not isinstance(details, dict):
        return "unknown"
    exception_payload = details.get("exception")
    if isinstance(exception_payload, dict):
        exception_type = exception_payload.get("type")
        if isinstance(exception_type, str):
            normalized = exception_type.strip()
            if normalized:
                return normalized
        return "unknown"
    if isinstance(exception_payload, str):
        normalized = exception_payload.strip()
        if normalized:
            return normalized.split(":", 1)[0].strip() or "unknown"
    return "unknown"


def _task_id() -> int | None:
    try:
        import asyncio

        task = asyncio.current_task()
    except Exception:
        return None
    if task is None:
        return None
    return id(task)


def _bool_from_env(raw: str | None, *, default: bool = False) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _int_from_env(raw: str | None, *, default: int, minimum: int) -> int:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        parsed = int(normalized)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else minimum


def _int_or_disabled_from_env(raw: str | None, *, default: int, minimum: int = 0) -> int:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        parsed = int(normalized)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return 0
    return parsed if parsed >= minimum else minimum


def _float_from_env(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        parsed = float(normalized)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return min(max(parsed, minimum), maximum)


def _normalize_base_dir(base_dir: Path) -> Path:
    expanded = base_dir.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (Path.cwd() / expanded).resolve(strict=False)


def _sanitize_argv(argv: list[str], *, allow_raw_text: bool) -> list[object]:
    sanitized: list[object] = []
    for raw_arg in argv[:64]:
        arg = _bounded_text(raw_arg)
        lowered = arg.casefold()
        if any(secret in lowered for secret in _SECRET_KEYWORDS) or any(
            lowered.startswith(prefix) for prefix in _ARG_SECRET_PREFIXES
        ):
            sanitized.append("[redacted]")
            continue
        sanitized.append(_sanitize_details(arg, allow_raw_text=allow_raw_text, key_hint="argv"))
    return sanitized


def _parse_scope(scope_raw: str | None) -> tuple[str, ...]:
    if scope_raw is None:
        return ()
    tokens = tuple(token.strip() for token in scope_raw.split(",") if token.strip())
    return tokens


def _service_matches_scope(service: str, scope: Iterable[str]) -> bool:
    scope_tokens = tuple(scope)
    if not scope_tokens:
        return True
    service_name = service.strip()
    for token in scope_tokens:
        lowered = token.casefold()
        if lowered in {"*", "all", "global"}:
            return True
        if fnmatchcase(service_name, token):
            return True
    return False


def _mode_write_profile(mode: str) -> tuple[int, int, int]:
    normalized = mode.strip().lower()
    if normalized == "forensic":
        return 100, 1, 500
    if normalized in {"throughput", "bulk"}:
        return 1000, 128, 5000
    return (
        _DEFAULT_FLUSH_INTERVAL_MS,
        _DEFAULT_FLUSH_EVERY_EVENTS,
        _DEFAULT_FSYNC_INTERVAL_MS,
    )


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    if not path.is_dir():
        raise NotADirectoryError(_bounded_text(path))


def _file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += _file_size_bytes(child)
    except OSError:
        return total
    return total


def _traceparent_from_ids(trace_id: str, span_id: str, *, trace_flags: str = _DEFAULT_TRACE_FLAGS) -> str:
    return f"00-{trace_id}-{span_id}-{trace_flags}"


def _parse_traceparent(traceparent: str | None) -> tuple[str, str, str] | None:
    if not traceparent:
        return None
    normalized = traceparent.strip().lower()
    match = _TRACEPARENT_RE.fullmatch(normalized)
    if match is None:
        return None
    trace_id = match.group("trace_id")
    span_id = match.group("span_id")
    trace_flags = match.group("trace_flags")
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id, trace_flags


def capture_thread_snapshot(
    thread: Thread | None,
    *,
    max_frames: int = 8,
) -> dict[str, object]:
    """Return one bounded stack/location snapshot for a Python thread."""

    if thread is None:
        return {"present": False}
    snapshot: dict[str, object] = {
        "present": True,
        "name": _bounded_text(getattr(thread, "name", "")),
        "ident": getattr(thread, "ident", None),
        "native_id": getattr(thread, "native_id", None),
        "daemon": bool(getattr(thread, "daemon", False)),
    }
    try:
        snapshot["alive"] = bool(thread.is_alive())
    except Exception:
        snapshot["alive"] = None
    ident = snapshot.get("ident")
    if not isinstance(ident, int):
        snapshot["stack_present"] = False
        snapshot["stack"] = []
        snapshot["top_frame"] = None
        return snapshot
    try:
        frame = sys._current_frames().get(ident)
    except Exception:
        frame = None
    if frame is None:
        snapshot["stack_present"] = False
        snapshot["stack"] = []
        snapshot["top_frame"] = None
        return snapshot
    extracted = traceback.extract_stack(frame, limit=max(1, int(max_frames)))
    bounded_stack: list[dict[str, object]] = []
    for entry in extracted[-max(1, int(max_frames)) :]:
        line_no = entry.lineno if isinstance(entry.lineno, int) else 0
        bounded_stack.append(
            {
                "file": _bounded_text(entry.filename),
                "line": line_no,
                "func": _bounded_text(entry.name),
            }
        )
    snapshot["stack_present"] = bool(bounded_stack)
    snapshot["stack"] = bounded_stack
    snapshot["top_frame"] = bounded_stack[-1] if bounded_stack else None
    return snapshot


@dataclass(frozen=True, slots=True)
class _SpanState:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    started_mono_ns: int
    name: str
    kind: str
    details: dict[str, object]


class WorkflowForensics:
    """Write a structured trace run pack for one workflow process."""

    def __init__(
        self,
        *,
        project_root: Path,
        service: str,
        enabled: bool,
        mode: str,
        base_dir: Path | None = None,
        allow_raw_text: bool = _DEFAULT_ALLOW_RAW_TEXT,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        max_events: int = _DEFAULT_MAX_EVENTS,
        max_msg_cardinality: int = _DEFAULT_MAX_MSG_CARDINALITY,
        max_span_history: int = _DEFAULT_MAX_SPAN_HISTORY,
        scope: tuple[str, ...] = (),
        sample_rate: float = 1.0,
        flush_interval_ms: int = _DEFAULT_FLUSH_INTERVAL_MS,
        flush_every_events: int = _DEFAULT_FLUSH_EVERY_EVENTS,
        fsync_interval_ms: int = _DEFAULT_FSYNC_INTERVAL_MS,
        retention_max_runs: int = _DEFAULT_RETENTION_MAX_RUNS,
        retention_max_total_bytes: int = _DEFAULT_RETENTION_MAX_TOTAL_BYTES,
        retention_max_age_days: int = _DEFAULT_RETENTION_MAX_AGE_DAYS,
    ) -> None:
        self.project_root = project_root
        self.service = service
        self.mode = _bounded_text(mode, default="balanced").lower() or "balanced"
        self.base_dir = _normalize_base_dir(base_dir) if base_dir is not None else None
        self.allow_raw_text = bool(allow_raw_text)
        self.scope = tuple(scope)
        self.sample_rate = min(max(float(sample_rate), 0.0), 1.0)
        self.run_id = _token()
        self._queue_maxsize = max(64, int(queue_maxsize))
        self._max_events = max(128, int(max_events))
        self._max_msg_cardinality = max(64, int(max_msg_cardinality))
        self._max_span_history = max(64, int(max_span_history))
        self._flush_interval_s = max(0.05, int(flush_interval_ms) / 1000.0)
        self._flush_every_events = max(1, int(flush_every_events))
        self._fsync_interval_s = max(0.25, int(fsync_interval_ms) / 1000.0)
        self._retention_max_runs = max(0, int(retention_max_runs))
        self._retention_max_total_bytes = max(0, int(retention_max_total_bytes))
        self._retention_max_age_days = max(0, int(retention_max_age_days))
        self._sampled_out = False
        if enabled and 0.0 < self.sample_rate < 1.0 and random.random() > self.sample_rate:
            enabled = False
            self._sampled_out = True
        self.enabled = bool(enabled)
        self._event_queue: Queue[dict[str, object] | object] = Queue(maxsize=self._queue_maxsize)
        self._writer_stop = Event()
        self._lifecycle_lock = Lock()
        self._stats_lock = Lock()
        self._closed = False
        self._accepting_events = False
        self._counts_by_kind: dict[str, int] = {}
        self._counts_by_msg: dict[str, int] = {}
        self._span_durations_ms: list[dict[str, object]] = []
        self._exception_counts: dict[str, int] = {}
        self._event_count = 0
        self._slowest_ms = 0.0
        self._last_event_ts: str | None = None
        self._queue_dropped = 0
        self._events_dropped_for_limit = 0
        self._trace_truncated = False
        self._writer_failed = False
        self._writer_error: str | None = None
        self._writer: Thread | None = None
        self._run_dir: Path | None = None
        self._base_dir_effective: Path | None = None
        self._jsonl_path: Path | None = None
        self._trace_path: Path | None = None
        self._metrics_path: Path | None = None
        self._summary_path: Path | None = None
        self._repro_dir: Path | None = None
        self._previous_sys_excepthook: object | None = None
        self._previous_threading_excepthook: object | None = None
        self._previous_unraisablehook: object | None = None
        self._installed_sys_excepthook: object | None = None
        self._installed_threading_excepthook: object | None = None
        self._installed_unraisablehook: object | None = None
        self._resource_attrs = self._build_resource_attributes()
        self._instrumentation_scope = {
            "name": _INSTRUMENTATION_SCOPE_NAME,
            "version": _MODULE_VERSION,
        }

        if self.enabled:
            try:
                self._initialize_enabled_state()
            except Exception as exc:
                self.enabled = False
                self._accepting_events = False
                self._writer_stop.set()
                self._restore_exception_hooks()
                self._safe_stderr(f"[twinr-workflow-trace] disabled after setup failure: {_bounded_text(exc)}")
        atexit.register(self.close)

    @classmethod
    def from_env(cls, *, project_root: Path, service: str) -> "WorkflowForensics":
        file_values: dict[str, str] = {}
        for candidate in (Path.cwd() / ".env", project_root / ".env"):
            file_values.update(_read_dotenv(candidate))
        enabled_raw = os.environ.get(_TRACE_ENV_ENABLED, file_values.get(_TRACE_ENV_ENABLED, "0"))
        mode = os.environ.get(_TRACE_ENV_MODE, file_values.get(_TRACE_ENV_MODE, "balanced")).strip().lower() or "balanced"
        base_dir_raw = os.environ.get(_TRACE_ENV_DIR, file_values.get(_TRACE_ENV_DIR, "")).strip()
        scope_raw = os.environ.get(_TRACE_ENV_SCOPE, file_values.get(_TRACE_ENV_SCOPE, "")).strip()
        sample_rate_raw = os.environ.get(
            _TRACE_ENV_SAMPLE_RATE,
            file_values.get(_TRACE_ENV_SAMPLE_RATE, "1.0"),
        )
        allow_raw_text_raw = os.environ.get(
            _TRACE_ENV_ALLOW_RAW_TEXT,
            file_values.get(_TRACE_ENV_ALLOW_RAW_TEXT, "0"),
        )
        queue_maxsize_raw = os.environ.get(
            _TRACE_ENV_QUEUE_MAXSIZE,
            file_values.get(_TRACE_ENV_QUEUE_MAXSIZE, str(_DEFAULT_QUEUE_MAXSIZE)),
        )
        max_events_raw = os.environ.get(
            _TRACE_ENV_MAX_EVENTS,
            file_values.get(_TRACE_ENV_MAX_EVENTS, str(_DEFAULT_MAX_EVENTS)),
        )
        max_msg_cardinality_raw = os.environ.get(
            _TRACE_ENV_MAX_MSG_CARDINALITY,
            file_values.get(_TRACE_ENV_MAX_MSG_CARDINALITY, str(_DEFAULT_MAX_MSG_CARDINALITY)),
        )
        max_span_history_raw = os.environ.get(
            _TRACE_ENV_MAX_SPAN_HISTORY,
            file_values.get(_TRACE_ENV_MAX_SPAN_HISTORY, str(_DEFAULT_MAX_SPAN_HISTORY)),
        )
        mode_flush_interval_ms, mode_flush_every_events, mode_fsync_interval_ms = _mode_write_profile(mode)
        flush_interval_raw = os.environ.get(
            _TRACE_ENV_FLUSH_INTERVAL_MS,
            file_values.get(_TRACE_ENV_FLUSH_INTERVAL_MS, str(mode_flush_interval_ms)),
        )
        flush_every_events_raw = os.environ.get(
            _TRACE_ENV_FLUSH_EVERY_EVENTS,
            file_values.get(_TRACE_ENV_FLUSH_EVERY_EVENTS, str(mode_flush_every_events)),
        )
        fsync_interval_raw = os.environ.get(
            _TRACE_ENV_FSYNC_INTERVAL_MS,
            file_values.get(_TRACE_ENV_FSYNC_INTERVAL_MS, str(mode_fsync_interval_ms)),
        )
        retention_max_runs_raw = os.environ.get(
            _TRACE_ENV_RETENTION_MAX_RUNS,
            file_values.get(_TRACE_ENV_RETENTION_MAX_RUNS, str(_DEFAULT_RETENTION_MAX_RUNS)),
        )
        retention_max_total_bytes_raw = os.environ.get(
            _TRACE_ENV_RETENTION_MAX_TOTAL_BYTES,
            file_values.get(_TRACE_ENV_RETENTION_MAX_TOTAL_BYTES, str(_DEFAULT_RETENTION_MAX_TOTAL_BYTES)),
        )
        retention_max_age_days_raw = os.environ.get(
            _TRACE_ENV_RETENTION_MAX_AGE_DAYS,
            file_values.get(_TRACE_ENV_RETENTION_MAX_AGE_DAYS, str(_DEFAULT_RETENTION_MAX_AGE_DAYS)),
        )
        enabled = _bool_from_env(enabled_raw, default=False)
        scope = _parse_scope(scope_raw)
        if enabled and not _service_matches_scope(service, scope):
            enabled = False
        return cls(
            project_root=project_root,
            service=service,
            enabled=enabled,
            mode=mode,
            base_dir=_normalize_trace_base_dir(base_dir_raw, project_root=project_root) if base_dir_raw else None,
            allow_raw_text=_bool_from_env(allow_raw_text_raw, default=_DEFAULT_ALLOW_RAW_TEXT),
            queue_maxsize=_int_from_env(queue_maxsize_raw, default=_DEFAULT_QUEUE_MAXSIZE, minimum=64),
            max_events=_int_from_env(max_events_raw, default=_DEFAULT_MAX_EVENTS, minimum=128),
            max_msg_cardinality=_int_from_env(
                max_msg_cardinality_raw,
                default=_DEFAULT_MAX_MSG_CARDINALITY,
                minimum=64,
            ),
            max_span_history=_int_from_env(
                max_span_history_raw,
                default=_DEFAULT_MAX_SPAN_HISTORY,
                minimum=64,
            ),
            scope=scope,
            sample_rate=_float_from_env(sample_rate_raw, default=1.0, minimum=0.0, maximum=1.0),
            flush_interval_ms=_int_from_env(
                flush_interval_raw,
                default=_DEFAULT_FLUSH_INTERVAL_MS,
                minimum=50,
            ),
            flush_every_events=_int_from_env(
                flush_every_events_raw,
                default=_DEFAULT_FLUSH_EVERY_EVENTS,
                minimum=1,
            ),
            fsync_interval_ms=_int_from_env(
                fsync_interval_raw,
                default=_DEFAULT_FSYNC_INTERVAL_MS,
                minimum=250,
            ),
            retention_max_runs=_int_or_disabled_from_env(
                retention_max_runs_raw,
                default=_DEFAULT_RETENTION_MAX_RUNS,
                minimum=1,
            ),
            retention_max_total_bytes=_int_or_disabled_from_env(
                retention_max_total_bytes_raw,
                default=_DEFAULT_RETENTION_MAX_TOTAL_BYTES,
                minimum=1024,
            ),
            retention_max_age_days=_int_or_disabled_from_env(
                retention_max_age_days_raw,
                default=_DEFAULT_RETENTION_MAX_AGE_DAYS,
                minimum=1,
            ),
        )

    def event(
        self,
        *,
        kind: str,
        msg: str,
        details: dict[str, object] | None = None,
        reason: dict[str, object] | None = None,
        kpi: dict[str, object] | None = None,
        level: str = "INFO",
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        parent_event_id: str | None = None,
        loc_skip: int = 2,
    ) -> None:
        if not self.enabled:
            return
        with self._stats_lock:
            if self._closed or not self._accepting_events or self._writer_failed:
                return
            if self._trace_truncated and _bounded_text(kind, default="event") not in _CRITICAL_RECORD_KINDS:
                self._events_dropped_for_limit += 1
                return
        try:
            record = self._base_record(
                kind=kind,
                msg=msg,
                level=level,
                details=details,
                reason=reason,
                kpi=kpi,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                parent_event_id=parent_event_id,
                loc_skip=loc_skip + 1,
            )
        except Exception as exc:
            self._safe_stderr(f"[twinr-workflow-trace] record build failed: {_bounded_text(exc)}")
            return
        self._enqueue_record(record, critical=_bounded_text(kind) in _CRITICAL_RECORD_KINDS)

    def can_accept_events(self) -> bool:
        if not self.enabled:
            return False
        with self._stats_lock:
            return not (
                self._closed
                or not self._accepting_events
                or self._writer_failed
                or self._trace_truncated
            )

    def remaining_event_budget(self) -> int:
        """Return the bounded number of non-critical events this run can still accept."""

        if not self.enabled:
            return 0
        with self._stats_lock:
            if self._closed or self._writer_failed:
                return 0
            return max(0, self._max_events - self._event_count)

    def decision(
        self,
        *,
        msg: str,
        question: str,
        selected: dict[str, object],
        options: list[dict[str, object]],
        context: dict[str, object] | None = None,
        confidence: object | None = None,
        guardrails: list[str] | None = None,
        kpi_impact_estimate: dict[str, object] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        self.event(
            kind="decision",
            msg=msg,
            details={"question": question, "context": context or {}},
            reason={
                "selected": selected,
                "options": options,
                "confidence": confidence,
                "guardrails": guardrails or [],
                "kpi_impact_estimate": kpi_impact_estimate or {},
            },
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            loc_skip=2,
        )

    @contextmanager
    def span(
        self,
        *,
        name: str,
        kind: str = "span",
        details: dict[str, object] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[_SpanState]:
        state = _SpanState(
            trace_id=trace_id or _trace_id(),
            span_id=_span_id(),
            parent_span_id=parent_span_id,
            started_mono_ns=time.monotonic_ns(),
            name=name,
            kind=kind,
            details=details or {},
        )
        if not self.enabled:
            yield state
            return
        self.event(
            kind="span_start",
            msg=name,
            details={"kind": kind, **(details or {})},
            trace_id=state.trace_id,
            span_id=state.span_id,
            parent_span_id=state.parent_span_id,
            loc_skip=2,
        )
        try:
            yield state
        except Exception as exc:
            duration_ms = (time.monotonic_ns() - state.started_mono_ns) / 1_000_000.0
            self.event(
                kind="exception",
                msg=f"{name}_exception",
                level="ERROR",
                details={
                    "span": name,
                    "kind": kind,
                    "exception": _exception_summary(exc, allow_raw_text=self.allow_raw_text),
                },
                kpi={"duration_ms": round(duration_ms, 3)},
                trace_id=state.trace_id,
                span_id=state.span_id,
                parent_span_id=state.parent_span_id,
                loc_skip=2,
            )
            raise
        else:
            duration_ms = (time.monotonic_ns() - state.started_mono_ns) / 1_000_000.0
            self.event(
                kind="span_end",
                msg=name,
                details={"kind": kind, **(details or {})},
                kpi={"duration_ms": round(duration_ms, 3)},
                trace_id=state.trace_id,
                span_id=state.span_id,
                parent_span_id=state.parent_span_id,
                loc_skip=2,
            )

    def close(self) -> None:
        if not self.enabled:
            return
        writer: Thread | None
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting_events = False
            writer = self._writer
        try:
            record = self._base_record(
                kind="run_end",
                msg="workflow_trace_stopped",
                level="INFO",
                details={},
                reason=None,
                kpi=None,
                trace_id=None,
                span_id=None,
                parent_span_id=None,
                parent_event_id=None,
                loc_skip=2,
            )
            self._enqueue_record(record, critical=True)
        except Exception:
            self._safe_stderr("[twinr-workflow-trace] failed to build or enqueue final run_end record")
        self._enqueue_sentinel()
        if writer is not None:
            deadline = time.monotonic() + 3.0
            while writer.is_alive() and time.monotonic() < deadline:
                writer.join(timeout=0.25)
        self._writer_stop.set()
        self._restore_exception_hooks()
        self._safe_flush_sidecars(force=True)

    def _build_resource_attributes(self) -> dict[str, object]:
        return {
            "service.name": self.service,
            "service.instance.id": f"{socket.gethostname()}-{os.getpid()}-{self.run_id[:8]}",
            "host.name": socket.gethostname(),
            "process.pid": os.getpid(),
            "process.runtime.name": platform.python_implementation(),
            "process.runtime.version": platform.python_version(),
            "process.executable.name": Path(sys.executable).name,
            "telemetry.sdk.name": _INSTRUMENTATION_SCOPE_NAME,
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": _MODULE_VERSION,
            "twinr.trace.mode": self.mode,
        }

    def _initialize_enabled_state(self) -> None:
        self._run_dir = self._build_run_dir()
        self._jsonl_path = self._run_dir / "run.jsonl"
        self._trace_path = self._run_dir / "run.trace"
        self._metrics_path = self._run_dir / "run.metrics.json"
        self._summary_path = self._run_dir / "run.summary.json"
        self._repro_dir = self._run_dir / "run.repro"
        _ensure_private_dir(self._repro_dir)
        self._install_exception_hooks()
        self._write_repro_pack()
        self._writer = Thread(target=self._writer_main, daemon=True, name="twinr-workflow-trace")
        self._accepting_events = True
        self._writer.start()
        self.event(
            kind="run_start",
            msg="workflow_trace_started",
            details={
                "mode": self.mode,
                "cwd": str(Path.cwd()),
                "pid": os.getpid(),
                "service": self.service,
                "scope": list(self.scope),
                "sample_rate": self.sample_rate,
            },
        )

    def _enqueue_record(self, record: dict[str, object], *, critical: bool = False) -> None:
        try:
            if critical:
                self._event_queue.put(record, timeout=0.5)
            else:
                self._event_queue.put_nowait(record)
        except Full:
            with self._stats_lock:
                self._queue_dropped += 1

    def _enqueue_sentinel(self) -> None:
        deadline = time.monotonic() + 2.0
        while not self._writer_stop.is_set():
            try:
                self._event_queue.put(_QUEUE_SENTINEL, timeout=0.25)
                return
            except Full:
                if time.monotonic() >= deadline:
                    self._safe_stderr("[twinr-workflow-trace] shutdown forced before queue fully drained")
                    self._writer_stop.set()
                    return

    def _build_run_dir(self) -> Path:
        base_dir = self.base_dir if self.base_dir is not None else self.project_root / "state" / "forensics" / "workflow"
        base_dir = _normalize_base_dir(base_dir)
        _ensure_private_dir(base_dir)
        self._base_dir_effective = base_dir
        for _ in range(4):
            run_dir = base_dir / self.run_id
            try:
                run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
                try:
                    os.chmod(run_dir, 0o700)
                except OSError:
                    pass
                break
            except FileExistsError:
                self.run_id = _token()
        else:
            raise FileExistsError("unable to allocate unique workflow trace directory")
        latest_path = base_dir / "LATEST"
        self._atomic_write_text(latest_path, f"{self.run_id}\n")
        self._enforce_retention(base_dir, keep_run_id=self.run_id)
        return run_dir

    def _base_record(
        self,
        *,
        kind: str,
        msg: str,
        level: str,
        details: dict[str, object] | None,
        reason: dict[str, object] | None,
        kpi: dict[str, object] | None,
        trace_id: str | None,
        span_id: str | None,
        parent_span_id: str | None,
        parent_event_id: str | None,
        loc_skip: int,
    ) -> dict[str, object]:
        try:
            frame = sys._getframe(loc_skip)
        except ValueError:
            frame = None
        except Exception:
            frame = None
        try:
            module = frame.f_globals.get("__name__", __name__) if frame is not None else __name__
            file_name = frame.f_code.co_filename if frame is not None else __file__
            func_name = frame.f_code.co_name if frame is not None else "_unknown"
            line_no = frame.f_lineno if frame is not None else 0
        finally:
            del frame
        normalized_details = _normalize_details_payload(details)
        normalized_reason = _normalize_details_payload(reason)
        normalized_kpi = _normalize_details_payload(kpi)
        normalized_trace_id = trace_id or current_workflow_trace_id() or _trace_id()
        normalized_span_id = span_id
        normalized_parent_span_id = parent_span_id
        if normalized_span_id is None:
            active_span = current_workflow_span_id()
            if active_span:
                normalized_span_id = active_span
        if normalized_parent_span_id is None and normalized_span_id is None:
            normalized_parent_span_id = current_workflow_parent_span_id()
        trace_flags = _ACTIVE_TRACE_FLAGS.get()
        severity_text = _bounded_text(level, default="INFO").upper()
        traceparent = (
            _traceparent_from_ids(normalized_trace_id, normalized_span_id, trace_flags=trace_flags)
            if normalized_span_id
            else None
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "ts_wall_utc": _utc_now(),
            "ts_mono_ns": time.monotonic_ns(),
            "observed_time_unix_nano": time.time_ns(),
            "level": severity_text,
            "severity_text": severity_text,
            "severity_number": self._severity_number(severity_text),
            "run_id": self.run_id,
            "event_id": _token(),
            "parent_event_id": parent_event_id,
            "trace_id": normalized_trace_id,
            "span_id": normalized_span_id,
            "parent_span_id": normalized_parent_span_id,
            "trace_flags": trace_flags,
            "traceparent": traceparent,
            "proc_id": os.getpid(),
            "thread_id": current_thread().ident,
            "thread_name": _bounded_text(current_thread().name, default="unknown"),
            "task_id": _task_id(),
            "host": socket.gethostname(),
            "service": self.service,
            "version": platform.python_version(),
            "resource": dict(self._resource_attrs),
            "instrumentation_scope": dict(self._instrumentation_scope),
            "loc": {
                "module": module,
                "func": func_name,
                "file": file_name,
                "line": line_no,
            },
            "kind": _bounded_text(kind, default="event"),
            "msg": _sanitize_message_text(msg, allow_raw_text=self.allow_raw_text),
            "body": _sanitize_message_text(msg, allow_raw_text=self.allow_raw_text),
            "details": _sanitize_details(normalized_details, allow_raw_text=self.allow_raw_text),
            "kpi": _sanitize_details(normalized_kpi or {}, allow_raw_text=True),
            "reason": _sanitize_details(normalized_reason or {}, allow_raw_text=self.allow_raw_text),
        }

    def _writer_main(self) -> None:
        jsonl_path = self._jsonl_path
        trace_path = self._trace_path
        if jsonl_path is None or trace_path is None:
            self._writer_stop.set()
            return
        pending_run_lines: list[str] = []
        pending_trace_lines: list[str] = []
        pending_count = 0
        last_flush_at = time.monotonic()
        last_fsync_at = last_flush_at

        def flush_pending(*, force_fsync: bool = False) -> None:
            nonlocal pending_count, last_flush_at, last_fsync_at
            if not pending_run_lines and not pending_trace_lines:
                return
            try:
                if pending_run_lines:
                    run_handle.writelines(pending_run_lines)
                    pending_run_lines.clear()
                if pending_trace_lines:
                    trace_handle.writelines(pending_trace_lines)
                    pending_trace_lines.clear()
                run_handle.flush()
                trace_handle.flush()
                now = time.monotonic()
                if force_fsync or (now - last_fsync_at) >= self._fsync_interval_s:
                    os.fsync(run_handle.fileno())
                    os.fsync(trace_handle.fileno())
                    last_fsync_at = now
                last_flush_at = now
                pending_count = 0
            except OSError as exc:
                with self._stats_lock:
                    self._writer_failed = True
                    self._writer_error = _bounded_text(exc)
                    self._accepting_events = False
                self._safe_stderr(f"[twinr-workflow-trace] writer stopped after I/O failure: {_bounded_text(exc)}")
                self._writer_stop.set()
                raise

        try:
            with self._open_append_text(jsonl_path) as run_handle, self._open_append_text(trace_path) as trace_handle:
                while not self._writer_stop.is_set():
                    timeout = max(0.05, self._flush_interval_s - (time.monotonic() - last_flush_at))
                    try:
                        item = self._event_queue.get(timeout=timeout)
                    except Empty:
                        if pending_count > 0:
                            flush_pending(force_fsync=False)
                        continue
                    if item is _QUEUE_SENTINEL:
                        self._writer_stop.set()
                        break
                    if not isinstance(item, dict):
                        continue
                    kind = _bounded_text(item.get("kind"), default="event")
                    with self._stats_lock:
                        if self._event_count >= self._max_events and kind not in _CRITICAL_RECORD_KINDS:
                            self._trace_truncated = True
                            self._events_dropped_for_limit += 1
                            continue
                    try:
                        encoded = self._json_dumps(item)
                    except Exception as exc:
                        with self._stats_lock:
                            self._queue_dropped += 1
                        self._safe_stderr(f"[twinr-workflow-trace] dropped unserializable record: {_bounded_text(exc)}")
                        continue
                    pending_run_lines.append(encoded + "\n")
                    if kind in _TRACE_FILE_KINDS:
                        pending_trace_lines.append(encoded + "\n")
                    pending_count += 1
                    self._update_stats_from_record(item)
                    should_force_flush = kind in _CRITICAL_RECORD_KINDS
                    should_flush = (
                        should_force_flush
                        or pending_count >= self._flush_every_events
                        or (time.monotonic() - last_flush_at) >= self._flush_interval_s
                    )
                    if should_flush:
                        flush_pending(force_fsync=should_force_flush)
                    if self._should_flush_sidecars():
                        self._safe_flush_sidecars(force=False)
                if pending_count > 0:
                    flush_pending(force_fsync=True)
        except Exception as exc:
            if not self._writer_stop.is_set():
                with self._stats_lock:
                    self._writer_failed = True
                    self._writer_error = _bounded_text(exc)
                    self._accepting_events = False
                self._safe_stderr(f"[twinr-workflow-trace] writer crashed: {_bounded_text(exc)}")
        finally:
            self._writer_stop.set()
            self._safe_flush_sidecars(force=True)

    def _update_stats_from_record(self, record: dict[str, object]) -> None:
        kind = _bounded_text(record.get("kind"), default="unknown")
        msg = _bounded_text(record.get("msg"), default="unknown")
        duration_ms = self._extract_duration_ms(record)
        with self._stats_lock:
            self._event_count += 1
            wall_ts = record.get("ts_wall_utc")
            self._last_event_ts = wall_ts if isinstance(wall_ts, str) else None
            self._counts_by_kind[kind] = self._counts_by_kind.get(kind, 0) + 1
            if msg in self._counts_by_msg:
                self._counts_by_msg[msg] += 1
            elif len(self._counts_by_msg) < self._max_msg_cardinality:
                self._counts_by_msg[msg] = 1
            else:
                self._counts_by_msg[_COUNT_OVERFLOW_BUCKET] = self._counts_by_msg.get(_COUNT_OVERFLOW_BUCKET, 0) + 1
            if kind == "exception":
                exc_type = _exception_type_from_record(record)
                self._exception_counts[exc_type] = self._exception_counts.get(exc_type, 0) + 1
            if duration_ms > self._slowest_ms:
                self._slowest_ms = duration_ms
            if kind == "span_end":
                self._record_span_duration_locked(name=msg, duration_ms=duration_ms, status="ok")
            elif kind == "exception":
                details = record.get("details")
                if isinstance(details, dict) and "span" in details:
                    self._record_span_duration_locked(
                        name=_bounded_text(details.get("span"), default=msg),
                        duration_ms=duration_ms,
                        status="error",
                    )

    def _record_span_duration_locked(self, *, name: str, duration_ms: float, status: str) -> None:
        self._span_durations_ms.append(
            {
                "name": _bounded_text(name, default="unknown"),
                "duration_ms": round(duration_ms, 3),
                "status": _bounded_text(status, default="unknown"),
            }
        )
        if len(self._span_durations_ms) > self._max_span_history:
            del self._span_durations_ms[:-self._max_span_history]

    def _extract_duration_ms(self, record: dict[str, object]) -> float:
        raw_kpi = record.get("kpi")
        if not isinstance(raw_kpi, dict):
            return 0.0
        raw_duration = raw_kpi.get("duration_ms")
        try:
            duration_ms = float(raw_duration or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(duration_ms):
            return 0.0
        return duration_ms

    def _summary_duration_ms(self, item: Mapping[str, object]) -> float:
        raw_duration = item.get("duration_ms")
        if raw_duration is None:
            return 0.0
        if isinstance(raw_duration, bool):
            return 1.0 if raw_duration else 0.0
        if not isinstance(raw_duration, (int, float, str, bytes, bytearray)):
            return 0.0
        try:
            duration_ms = float(raw_duration)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(duration_ms):
            return 0.0
        return duration_ms

    def _should_flush_sidecars(self) -> bool:
        with self._stats_lock:
            return self._event_count == 1 or self._event_count % _SUMMARY_FLUSH_INTERVAL == 0

    def _snapshot_sidecars(self) -> tuple[dict[str, object], dict[str, object]]:
        with self._stats_lock:
            counts_by_kind = dict(self._counts_by_kind)
            counts_by_msg = dict(self._counts_by_msg)
            exception_counts = dict(self._exception_counts)
            span_durations_ms = list(self._span_durations_ms[-64:])
            event_count = self._event_count
            slowest_ms = self._slowest_ms
            last_event_ts = self._last_event_ts
            queue_dropped = self._queue_dropped
            events_dropped_for_limit = self._events_dropped_for_limit
            trace_truncated = self._trace_truncated
            writer_failed = self._writer_failed
            writer_error = self._writer_error
        metrics = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "service": self.service,
            "resource": dict(self._resource_attrs),
            "event_count": event_count,
            "counts_by_kind": counts_by_kind,
            "counts_by_msg": counts_by_msg,
            "exception_counts": exception_counts,
            "slowest_duration_ms": round(slowest_ms, 3),
            "span_durations_ms": span_durations_ms,
            "queue_dropped": queue_dropped,
            "events_dropped_for_limit": events_dropped_for_limit,
            "trace_truncated": trace_truncated,
            "writer_failed": writer_failed,
            "writer_error": writer_error,
            "sample_rate": self.sample_rate,
            "scope": list(self.scope),
        }
        summary = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "service": self.service,
            "mode": self.mode,
            "event_count": event_count,
            "last_event_ts": last_event_ts,
            "top_kinds": sorted(counts_by_kind.items(), key=lambda item: item[1], reverse=True)[:16],
            "top_msgs": sorted(counts_by_msg.items(), key=lambda item: item[1], reverse=True)[:16],
            "exception_counts": exception_counts,
            "slowest_spans": sorted(
                span_durations_ms,
                key=self._summary_duration_ms,
                reverse=True,
            )[:16],
            "queue_dropped": queue_dropped,
            "events_dropped_for_limit": events_dropped_for_limit,
            "trace_truncated": trace_truncated,
            "writer_failed": writer_failed,
            "writer_error": writer_error,
            "sample_rate": self.sample_rate,
            "scope": list(self.scope),
        }
        return metrics, summary

    def _safe_flush_sidecars(self, *, force: bool) -> None:
        if not self.enabled:
            return
        metrics_path = self._metrics_path
        summary_path = self._summary_path
        if metrics_path is None or summary_path is None:
            return
        if not force and not self._should_flush_sidecars():
            return
        metrics, summary = self._snapshot_sidecars()
        try:
            self._atomic_write_text(metrics_path, self._json_dumps(metrics, pretty=True))
            self._atomic_write_text(summary_path, self._json_dumps(summary, pretty=True))
        except OSError as exc:
            self._safe_stderr(f"[twinr-workflow-trace] sidecar flush failed: {_bounded_text(exc)}")

    def _install_exception_hooks(self) -> None:
        if not self.enabled:
            return

        previous_excepthook = sys.excepthook
        previous_threading_excepthook = getattr(threading, "excepthook", None)
        previous_unraisablehook = getattr(sys, "unraisablehook", None)

        def excepthook(exc_type, exc_value, exc_traceback):
            self.event(
                kind="exception",
                msg="sys_excepthook",
                level="ERROR",
                details={
                    "exception": _exception_summary_from_parts(
                        exc_type,
                        exc_value,
                        exc_traceback,
                        allow_raw_text=self.allow_raw_text,
                    )
                },
                loc_skip=2,
            )
            if callable(previous_excepthook):
                previous_excepthook(exc_type, exc_value, exc_traceback)

        def thread_excepthook(args):
            self.event(
                kind="exception",
                msg="threading_excepthook",
                level="ERROR",
                details={
                    "thread": {
                        "name": _bounded_text(getattr(args.thread, "name", "unknown")),
                        "ident": getattr(args.thread, "ident", None),
                    },
                    "exception": _exception_summary_from_parts(
                        args.exc_type,
                        args.exc_value,
                        args.exc_traceback,
                        allow_raw_text=self.allow_raw_text,
                    ),
                },
                loc_skip=2,
            )
            if callable(previous_threading_excepthook):
                previous_threading_excepthook(args)

        def unraisablehook(unraisable):
            self.event(
                kind="exception",
                msg="sys_unraisablehook",
                level="ERROR",
                details={
                    "err_msg": _sanitize_details(
                        getattr(unraisable, "err_msg", None),
                        allow_raw_text=self.allow_raw_text,
                        key_hint="message",
                    ),
                    "object_type": _bounded_text(type(getattr(unraisable, "object", None)).__name__),
                    "exception": _exception_summary_from_parts(
                        getattr(unraisable, "exc_type", None),
                        getattr(unraisable, "exc_value", None),
                        getattr(unraisable, "exc_traceback", None),
                        allow_raw_text=self.allow_raw_text,
                    ),
                },
                loc_skip=2,
            )
            if callable(previous_unraisablehook):
                previous_unraisablehook(unraisable)

        self._previous_sys_excepthook = previous_excepthook
        self._previous_threading_excepthook = previous_threading_excepthook
        self._previous_unraisablehook = previous_unraisablehook
        self._installed_sys_excepthook = excepthook
        self._installed_threading_excepthook = thread_excepthook
        self._installed_unraisablehook = unraisablehook
        sys.excepthook = excepthook
        if previous_threading_excepthook is not None:
            threading.excepthook = thread_excepthook
        if previous_unraisablehook is not None:
            sys.unraisablehook = unraisablehook

    def _restore_exception_hooks(self) -> None:
        if self._installed_sys_excepthook is not None and sys.excepthook is self._installed_sys_excepthook:
            previous = self._previous_sys_excepthook
            if callable(previous):
                sys.excepthook = previous
        if (
            self._installed_threading_excepthook is not None
            and getattr(threading, "excepthook", None) is self._installed_threading_excepthook
        ):
            previous_threading = self._previous_threading_excepthook
            if callable(previous_threading):
                threading.excepthook = previous_threading
        if self._installed_unraisablehook is not None and getattr(sys, "unraisablehook", None) is self._installed_unraisablehook:
            previous_unraisable = self._previous_unraisablehook
            if callable(previous_unraisable):
                sys.unraisablehook = previous_unraisable

    def _write_repro_pack(self) -> None:
        repro_dir = self._repro_dir
        if repro_dir is None:
            return
        repro = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "service": self.service,
            "cwd": str(Path.cwd()),
            "argv": _sanitize_argv(sys.argv, allow_raw_text=self.allow_raw_text),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mode": self.mode,
            "scope": list(self.scope),
            "sample_rate": self.sample_rate,
            "resource": dict(self._resource_attrs),
        }
        env_snapshot = {
            key: _sanitize_details(os.environ.get(key, ""), allow_raw_text=self.allow_raw_text, key_hint=key)
            for key in _ENV_WHITELIST
            if key in os.environ
        }
        _ensure_private_dir(repro_dir)
        self._atomic_write_text(repro_dir / "runtime.json", self._json_dumps(repro, pretty=True))
        self._atomic_write_text(repro_dir / "env.json", self._json_dumps(env_snapshot, pretty=True))

    def _open_append_text(self, path: Path) -> TextIO:
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8", buffering=1)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        directory = path.parent
        _ensure_private_dir(directory)
        temp_path: str | None = None
        dir_fd: int | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(directory), delete=False) as temp_handle:
                temp_path = temp_handle.name
                try:
                    os.chmod(temp_path, 0o600)
                except OSError:
                    pass
                temp_handle.write(content)
                temp_handle.flush()
                os.fsync(temp_handle.fileno())
            os.replace(temp_path, path)
            if hasattr(os, "O_DIRECTORY"):
                try:
                    dir_fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
                    os.fsync(dir_fd)
                finally:
                    if dir_fd is not None:
                        os.close(dir_fd)
        except Exception:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def _json_dumps(self, payload: object, *, pretty: bool = False) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            allow_nan=False,
        )

    def _severity_number(self, severity_text: str) -> int:
        return {
            "DEBUG": 5,
            "INFO": 9,
            "WARNING": 13,
            "ERROR": 17,
            "CRITICAL": 21,
        }.get(severity_text.upper(), 9)

    def _safe_stderr(self, message: str) -> None:
        try:
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            try:
                fallback_stderr = sys.__stderr__
                if fallback_stderr is None:
                    return
                fallback_stderr.write(f"{message}\n")
                fallback_stderr.flush()
            except Exception:
                _LOGGER.warning("Workflow forensics failed to write to stderr fallback.", exc_info=True)

    def _enforce_retention(self, base_dir: Path, *, keep_run_id: str) -> None:
        if self._retention_max_runs == 0 and self._retention_max_total_bytes == 0 and self._retention_max_age_days == 0:
            return
        try:
            candidates: list[tuple[float, int, Path]] = []
            for child in base_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name == keep_run_id:
                    continue
                try:
                    stat = child.stat()
                except OSError:
                    continue
                candidates.append((stat.st_mtime, _dir_size_bytes(child), child))
            now = time.time()
            if self._retention_max_age_days > 0:
                max_age_seconds = float(timedelta(days=self._retention_max_age_days).total_seconds())
                for mtime, _, path in list(candidates):
                    if (now - mtime) > max_age_seconds:
                        self._delete_run_dir(path)
                candidates = [entry for entry in candidates if entry[2].exists()]
            candidates.sort(key=lambda item: item[0], reverse=True)
            keep_old_runs = max(0, self._retention_max_runs - 1) if self._retention_max_runs > 0 else 0
            if self._retention_max_runs > 0 and len(candidates) > keep_old_runs:
                for _, _, path in candidates[keep_old_runs:]:
                    self._delete_run_dir(path)
                candidates = [entry for entry in candidates[:keep_old_runs] if entry[2].exists()]
            if self._retention_max_total_bytes > 0:
                current_size = _dir_size_bytes(base_dir / keep_run_id)
                total_bytes = current_size + sum(size for _, size, path in candidates if path.exists())
                for _, size, path in reversed(candidates):
                    if total_bytes <= self._retention_max_total_bytes:
                        break
                    if path.exists():
                        self._delete_run_dir(path)
                        total_bytes -= size
        except Exception as exc:
            self._safe_stderr(f"[twinr-workflow-trace] retention cleanup skipped: {_bounded_text(exc)}")

    def _delete_run_dir(self, path: Path) -> None:
        try:
            for child in sorted(path.rglob("*"), reverse=True):
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                except OSError:
                    continue
            path.rmdir()
        except OSError:
            return


# ---- Active context helpers -------------------------------------------------


def current_workflow_forensics() -> WorkflowForensics | None:
    tracer = _ACTIVE_TRACER.get()
    if isinstance(tracer, WorkflowForensics) and tracer.enabled:
        return tracer
    return None


def current_workflow_trace_id() -> str | None:
    return _ACTIVE_TRACE_ID.get()


def current_workflow_span_id() -> str | None:
    return _ACTIVE_SPAN_ID.get()


def current_workflow_parent_span_id() -> str | None:
    active_span = _ACTIVE_SPAN_ID.get()
    if active_span is not None:
        return active_span
    return _ACTIVE_PARENT_SPAN_HINT.get()


def current_workflow_traceparent() -> str | None:
    trace_id = _ACTIVE_TRACE_ID.get()
    span_id = _ACTIVE_SPAN_ID.get()
    if not trace_id or not span_id:
        return None
    return _traceparent_from_ids(trace_id, span_id, trace_flags=_ACTIVE_TRACE_FLAGS.get())


def workflow_context_to_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an env mapping with TRACEPARENT/TRACESTATE injected for child processes."""

    target = dict(os.environ if env is None else env)
    traceparent = current_workflow_traceparent()
    tracestate = _ACTIVE_TRACE_STATE.get()
    if traceparent:
        target["TRACEPARENT"] = traceparent
    if tracestate:
        target["TRACESTATE"] = tracestate
    return target


def workflow_context_from_env(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """Extract TRACEPARENT/TRACESTATE from one env mapping."""

    source = os.environ if env is None else env
    traceparent = source.get("TRACEPARENT") or source.get("traceparent")
    tracestate = source.get("TRACESTATE") or source.get("tracestate")
    return traceparent, tracestate


@contextmanager
def bind_workflow_forensics(
    tracer: WorkflowForensics | None,
    *,
    trace_id: str | None = None,
    traceparent: str | None = None,
    tracestate: str | None = None,
) -> Iterator[str | None]:
    """Bind one tracer and trace context to the current execution context."""

    active_tracer = tracer if isinstance(tracer, WorkflowForensics) and tracer.enabled else None
    parsed = _parse_traceparent(traceparent)
    bound_trace_id = trace_id or _ACTIVE_TRACE_ID.get()
    bound_span_id = _ACTIVE_SPAN_ID.get()
    bound_trace_flags = _ACTIVE_TRACE_FLAGS.get()
    bound_tracestate = tracestate or _ACTIVE_TRACE_STATE.get()
    if parsed is not None:
        bound_trace_id, bound_span_id, bound_trace_flags = parsed
    if active_tracer is not None and bound_trace_id is None:
        bound_trace_id = _trace_id()
    tracer_token = _ACTIVE_TRACER.set(active_tracer)
    trace_token = _ACTIVE_TRACE_ID.set(bound_trace_id)
    span_token = _ACTIVE_SPAN_ID.set(bound_span_id)
    parent_hint_token = _ACTIVE_PARENT_SPAN_HINT.set(bound_span_id)
    trace_flags_token = _ACTIVE_TRACE_FLAGS.set(bound_trace_flags)
    tracestate_token = _ACTIVE_TRACE_STATE.set(bound_tracestate)
    try:
        yield _ACTIVE_TRACE_ID.get()
    finally:
        _ACTIVE_TRACE_STATE.reset(tracestate_token)
        _ACTIVE_TRACE_FLAGS.reset(trace_flags_token)
        _ACTIVE_PARENT_SPAN_HINT.reset(parent_hint_token)
        _ACTIVE_SPAN_ID.reset(span_token)
        _ACTIVE_TRACE_ID.reset(trace_token)
        _ACTIVE_TRACER.reset(tracer_token)


@contextmanager
def bind_workflow_traceparent(
    tracer: WorkflowForensics | None,
    *,
    traceparent: str,
    tracestate: str | None = None,
) -> Iterator[str | None]:
    """Convenience wrapper to bind inbound W3C trace context."""

    with bind_workflow_forensics(tracer, traceparent=traceparent, tracestate=tracestate) as bound:
        yield bound


def workflow_event(
    *,
    kind: str,
    msg: str,
    details: dict[str, object] | None = None,
    reason: dict[str, object] | None = None,
    kpi: dict[str, object] | None = None,
    level: str = "INFO",
) -> None:
    """Emit one structured event on the currently bound workflow trace."""

    tracer = current_workflow_forensics()
    if tracer is None:
        return
    tracer.event(
        kind=kind,
        msg=msg,
        details=details,
        reason=reason,
        kpi=kpi,
        level=level,
        trace_id=current_workflow_trace_id(),
        span_id=current_workflow_span_id(),
        parent_span_id=None,
        loc_skip=2,
    )


def workflow_decision(
    *,
    msg: str,
    question: str,
    selected: dict[str, object],
    options: list[dict[str, object]],
    context: dict[str, object] | None = None,
    confidence: object | None = None,
    guardrails: list[str] | None = None,
    kpi_impact_estimate: dict[str, object] | None = None,
) -> None:
    """Emit one decision record on the currently bound workflow trace."""

    tracer = current_workflow_forensics()
    if tracer is None:
        return
    tracer.decision(
        msg=msg,
        question=question,
        selected=selected,
        options=options,
        context=context,
        confidence=confidence,
        guardrails=guardrails,
        kpi_impact_estimate=kpi_impact_estimate,
        trace_id=current_workflow_trace_id(),
        span_id=current_workflow_span_id(),
        parent_span_id=None,
    )


@contextmanager
def workflow_span(
    *,
    name: str,
    kind: str = "span",
    details: dict[str, object] | None = None,
) -> Iterator[_SpanState]:
    """Create one child span on the currently bound workflow trace context."""

    tracer = current_workflow_forensics()
    trace_id = current_workflow_trace_id() or _trace_id()
    parent_span_id = current_workflow_span_id() or _ACTIVE_PARENT_SPAN_HINT.get()
    if tracer is None:
        state = _SpanState(
            trace_id=trace_id,
            span_id=_span_id(),
            parent_span_id=parent_span_id,
            started_mono_ns=time.monotonic_ns(),
            name=name,
            kind=kind,
            details=details or {},
        )
        trace_token = _ACTIVE_TRACE_ID.set(state.trace_id)
        span_token = _ACTIVE_SPAN_ID.set(state.span_id)
        parent_hint_token = _ACTIVE_PARENT_SPAN_HINT.set(state.span_id)
        try:
            yield state
        finally:
            _ACTIVE_PARENT_SPAN_HINT.reset(parent_hint_token)
            _ACTIVE_SPAN_ID.reset(span_token)
            _ACTIVE_TRACE_ID.reset(trace_token)
        return
    with tracer.span(
        name=name,
        kind=kind,
        details=details,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    ) as state:
        trace_token = _ACTIVE_TRACE_ID.set(state.trace_id)
        span_token = _ACTIVE_SPAN_ID.set(state.span_id)
        parent_hint_token = _ACTIVE_PARENT_SPAN_HINT.set(state.span_id)
        try:
            yield state
        finally:
            _ACTIVE_PARENT_SPAN_HINT.reset(parent_hint_token)
            _ACTIVE_SPAN_ID.reset(span_token)
            _ACTIVE_TRACE_ID.reset(trace_token)


def workflow_context() -> Context:
    """Return a copy of the current contextvars context for later propagation."""

    return copy_context()


def workflow_callable(
    target: Callable[..., T],
    *,
    context: Context | None = None,
) -> Callable[..., T]:
    """Bind one callable to the current workflow context for thread/process helpers."""

    ctx = context or copy_context()

    def _runner(*args: Any, **kwargs: Any) -> T:
        return ctx.run(target, *args, **kwargs)

    return _runner


def start_workflow_thread(
    *,
    target: Callable[..., Any],
    name: str | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    daemon: bool | None = None,
    context: Context | None = None,
    start: bool = True,
) -> Thread:
    """Start a thread that inherits the current workflow tracing context."""

    bound_target = workflow_callable(target, context=context)
    thread = Thread(target=bound_target, name=name, args=args, kwargs=kwargs or {}, daemon=daemon)
    if start:
        thread.start()
    return thread
