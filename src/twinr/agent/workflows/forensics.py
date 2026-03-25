
"""Forensic-grade workflow tracing for Twinr hardware loops.

This module provides a bounded, structured run pack for difficult live-runtime
bugs. It is intentionally workflow-local so button/session/audio issues can be
traced without scattering JSON-writing logic across the orchestration code.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue   # AUDIT-FIX(#1): Begrenzte Queue braucht explizite Full-Behandlung.
from threading import Event, Lock, Thread, current_thread   # AUDIT-FIX(#7): Koordinierter Shutdown und Snapshots brauchen Locks.
from types import TracebackType
from typing import Iterator, TextIO
import atexit
import hashlib   # AUDIT-FIX(#3): Redaktions-Summaries nutzen stabile Hashes ohne Rohtext zu speichern.
import inspect
import json
import logging
import math   # AUDIT-FIX(#11): Nicht-endliche Floats werden vor JSON-Serialisierung normalisiert.
import os
import platform
import socket
import sys
import tempfile   # AUDIT-FIX(#6): Sidecars werden per Temp-File plus atomarem Replace geschrieben.
import threading   # AUDIT-FIX(#5): Ungefangene Exceptions aus Thread.run werden erfasst.
import time
import traceback
import uuid


_TRACE_ENV_ENABLED = "TWINR_WORKFLOW_TRACE_ENABLED"
_TRACE_ENV_MODE = "TWINR_WORKFLOW_TRACE_MODE"
_TRACE_ENV_DIR = "TWINR_WORKFLOW_TRACE_DIR"
_TRACE_ENV_SCOPE = "TWINR_WORKFLOW_TRACE_SCOPE"
_TRACE_ENV_SAMPLE_RATE = "TWINR_WORKFLOW_TRACE_SAMPLE_RATE"
_TRACE_ENV_ALLOW_RAW_TEXT = "TWINR_WORKFLOW_TRACE_ALLOW_RAW_TEXT"
_TRACE_ENV_QUEUE_MAXSIZE = "TWINR_WORKFLOW_TRACE_QUEUE_MAXSIZE"  # AUDIT-FIX(#1): Queue-Größe muss konfigurierbar sein.
_TRACE_ENV_MAX_EVENTS = "TWINR_WORKFLOW_TRACE_MAX_EVENTS"  # AUDIT-FIX(#2): Event-Caps pro Run verhindern unendliches Plattenwachstum.
_TRACE_ENV_MAX_MSG_CARDINALITY = "TWINR_WORKFLOW_TRACE_MAX_MSG_CARDINALITY"  # AUDIT-FIX(#2): Cardinality-Caps begrenzen Summary-Strukturen.
_TRACE_ENV_MAX_SPAN_HISTORY = "TWINR_WORKFLOW_TRACE_MAX_SPAN_HISTORY"  # AUDIT-FIX(#2): Span-Historie wird explizit begrenzt.
_QUEUE_SENTINEL = object()
_SUMMARY_FLUSH_INTERVAL = 25  # AUDIT-FIX(#2): Vermeidet SD-Card-Churn durch Sidecar-Rewrite bei jedem Event.
_MAX_DETAIL_TEXT = 240
_MAX_STACK_LINES = 12
_DEFAULT_QUEUE_MAXSIZE = 2048  # AUDIT-FIX(#1): Verhindert unendliches RAM-Wachstum bei Log-Stürmen.
_DEFAULT_MAX_EVENTS = 5000  # AUDIT-FIX(#2): Begrenzt Trace-Größe pro Run für langlebige Services.
_DEFAULT_MAX_MSG_CARDINALITY = 1024  # AUDIT-FIX(#2): Begrenzt Anzahl unterschiedlicher Message-Counter.
_DEFAULT_MAX_SPAN_HISTORY = 256  # AUDIT-FIX(#2): Begrenzt im Speicher gehaltene Span-Historie.
_DEFAULT_ALLOW_RAW_TEXT = False  # AUDIT-FIX(#3): Datenschutzsicherer Default für Traces im Seniorenhaushalt.
_COUNT_OVERFLOW_BUCKET = "[other]"  # AUDIT-FIX(#2): Überschüssige Cardinality wird aggregiert statt endlos zu wachsen.
_SECRET_KEYWORDS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "authorization",
    "cookie",
    "session",
    "bearer",
)
_RAW_TEXT_HINT_KEYWORDS: tuple[str, ...] = (
    "text",
    "transcript",
    "utterance",
    "prompt",
    "response",
    "content",
    "input",
    "output",
    "question",
    "message",
)
_RAW_TEXT_SAFE_KEYS: tuple[str, ...] = (
    "cwd",
    "mode",
    "service",
    "host",
    "hostname",
    "module",
    "func",
    "file",
    "line",
    "kind",
    "type",
    "stack",
    "status",
    "platform",
    "python",
)
_ARG_SECRET_PREFIXES: tuple[str, ...] = (
    "--api-key",
    "--apikey",
    "--token",
    "--password",
    "--secret",
    "--authorization",
    "--cookie",
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
    "TWINR_LLM_PROVIDER",
    "TWINR_STT_PROVIDER",
    "TWINR_TTS_PROVIDER",
    "TWINR_LONG_TERM_MEMORY_MODE",
    "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED",
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


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple dotenv-style ``KEY=VALUE`` pairs for trace configuration."""

    try:  # AUDIT-FIX(#10): Unlesbare .env darf Bootstrap nicht crashen lassen.
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token() -> str:
    return uuid.uuid4().hex


def _safe_str(value: object, *, default: str = "") -> str:
    try:  # AUDIT-FIX(#11): Beliebige __str__-Implementierungen dürfen Tracing nicht crashen.
        return str(value if value is not None else default)
    except Exception:
        fallback_type = type(value).__name__ if value is not None else "unknown"
        return f"<unstringifiable:{fallback_type}>"


def _safe_repr(value: object) -> str:
    try:  # AUDIT-FIX(#11): Beliebige __repr__-Implementierungen dürfen Tracing nicht crashen.
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


def _should_redact_text(text: str, *, allow_raw_text: bool, key_hint: str | None) -> bool:
    if allow_raw_text:
        return False
    lowered_key = (key_hint or "").casefold()
    if lowered_key in _RAW_TEXT_SAFE_KEYS:
        return False
    if any(marker in lowered_key for marker in _RAW_TEXT_HINT_KEYWORDS):
        return True
    return len(text) > 32 or any(char.isspace() for char in text)


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
        return value if math.isfinite(value) else None  # AUDIT-FIX(#11): Hält ausgegebenes JSON standardkonform.
    if isinstance(value, str):
        text = _bounded_text(value)
        if _should_redact_text(text, allow_raw_text=allow_raw_text, key_hint=key_hint):
            return _redacted_text_summary(text)
        return text
    if isinstance(value, Path):
        return _sanitize_details(str(value), allow_raw_text=allow_raw_text, key_hint=key_hint)
    if isinstance(value, dict):
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


def _exception_summary_from_parts(
    exc_type: object,
    exc_value: object,
    exc_traceback: TracebackType | None,
    *,
    allow_raw_text: bool,
) -> dict[str, object]:
    try:
        formatted_stack = traceback.format_exception(exc_type, exc_value, exc_traceback)[-1 * _MAX_STACK_LINES :]
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
    """Extract one stable exception type from a trace record.

    The live workflow occasionally emits a legacy string payload in
    ``details.exception``. The forensics writer must stay crash-safe because it
    is responsible for preserving evidence while the runtime is still serving
    real button presses.
    """

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


def _normalize_legacy_exception_details(details: dict[str, object] | None) -> dict[str, object]:
    """Promote legacy string exceptions into structured type/message payloads."""

    if not isinstance(details, dict):
        return dict(details or {})
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


def _normalize_base_dir(base_dir: Path) -> Path:
    expanded = base_dir.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (Path.cwd() / expanded).resolve(strict=False)  # AUDIT-FIX(#6): Konfigurierbare Pfade werden vor Nutzung normalisiert.


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


def capture_thread_snapshot(
    thread: Thread | None,
    *,
    max_frames: int = 8,
) -> dict[str, object]:
    """Return one bounded stack/location snapshot for a Python thread.

    The snapshot is intentionally shallow and metadata-first so it can be
    written into live workflow runpacks without dumping locals or other
    unbounded payloads.
    """

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
        bounded_stack.append(
            {
                "file": _bounded_text(entry.filename),
                "line": int(entry.lineno),
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
    ) -> None:
        self.project_root = project_root
        self.service = service
        self.enabled = bool(enabled)
        self.mode = mode
        self.base_dir = _normalize_base_dir(base_dir) if base_dir is not None else None  # AUDIT-FIX(#6): Basisverzeichnis wird gehärtet normalisiert.
        self.allow_raw_text = bool(allow_raw_text)  # AUDIT-FIX(#3): Privacy-Modus muss zur Laufzeit konfigurierbar sein.
        self.run_id = _token()
        self._queue_maxsize = max(64, int(queue_maxsize))  # AUDIT-FIX(#1): Erzwingt sinnvolle Untergrenze.
        self._max_events = max(128, int(max_events))  # AUDIT-FIX(#2): Erzwingt sinnvolle Untergrenze.
        self._max_msg_cardinality = max(64, int(max_msg_cardinality))  # AUDIT-FIX(#2)
        self._max_span_history = max(64, int(max_span_history))  # AUDIT-FIX(#2)
        self._event_queue: Queue[dict[str, object] | object] = Queue(maxsize=self._queue_maxsize)
        self._writer_stop = Event()
        self._lifecycle_lock = Lock()  # AUDIT-FIX(#7): Close/Event-Races müssen serialisiert werden.
        self._stats_lock = Lock()  # AUDIT-FIX(#8): Snapshot-Erzeugung und Counter müssen threadsicher sein.
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

        if self.enabled:
            try:  # AUDIT-FIX(#4): Tracing muss fail-open auf "disabled" gehen und nie den Primärservice crashen.
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
        mode = os.environ.get(_TRACE_ENV_MODE, file_values.get(_TRACE_ENV_MODE, "forensic")).strip().lower() or "forensic"
        base_dir_raw = os.environ.get(_TRACE_ENV_DIR, file_values.get(_TRACE_ENV_DIR, "")).strip()
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
        return cls(
            project_root=project_root,
            service=service,
            enabled=_bool_from_env(enabled_raw, default=False),
            mode=mode,
            base_dir=Path(base_dir_raw) if base_dir_raw else None,
            allow_raw_text=_bool_from_env(allow_raw_text_raw, default=_DEFAULT_ALLOW_RAW_TEXT),
            queue_maxsize=_int_from_env(
                queue_maxsize_raw,
                default=_DEFAULT_QUEUE_MAXSIZE,
                minimum=64,
            ),
            max_events=_int_from_env(
                max_events_raw,
                default=_DEFAULT_MAX_EVENTS,
                minimum=128,
            ),
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
            if self._trace_truncated:
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
        except Exception as exc:  # AUDIT-FIX(#11): Record-Build darf auf Caller-Pfaden nie fail-closed enden.
            self._safe_stderr(f"[twinr-workflow-trace] record build failed: {_bounded_text(exc)}")
            return
        self._enqueue_record(record)

    def can_accept_events(self) -> bool:
        """Return whether the tracer can still accept non-critical events."""

        if not self.enabled:
            return False
        with self._stats_lock:
            return not (
                self._closed
                or not self._accepting_events
                or self._writer_failed
                or self._trace_truncated
            )

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
            trace_id=trace_id or _token(),
            span_id=_token()[:16],
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
            self._accepting_events = False  # AUDIT-FIX(#7): Späte Events werden ab Shutdown-Beginn abgewiesen.
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
            deadline = time.monotonic() + 3.0  # AUDIT-FIX(#7): Begrenzt drainen, bevor aufgegeben wird.
            while writer.is_alive() and time.monotonic() < deadline:
                writer.join(timeout=0.25)
        self._writer_stop.set()
        self._restore_exception_hooks()
        self._safe_flush_sidecars(force=True)

    def _initialize_enabled_state(self) -> None:
        self._run_dir = self._build_run_dir()  # AUDIT-FIX(#4): On-Disk-Artefakte nur bei aktivem Tracing erzeugen.
        self._jsonl_path = self._run_dir / "run.jsonl"
        self._trace_path = self._run_dir / "run.trace"
        self._metrics_path = self._run_dir / "run.metrics.json"
        self._summary_path = self._run_dir / "run.summary.json"
        self._repro_dir = self._run_dir / "run.repro"
        self._repro_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
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
            },
        )

    def _enqueue_record(self, record: dict[str, object], *, critical: bool = False) -> None:
        try:
            if critical:
                self._event_queue.put(record, timeout=0.5)
            else:
                self._event_queue.put_nowait(record)
        except Full:  # AUDIT-FIX(#1): Drop-Counter halten Überlast fest ohne Workflow-Threads zu blockieren.
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
        base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)  # AUDIT-FIX(#6): Nutzt private-by-default-Berechtigungen.
        if not base_dir.is_dir():
            raise NotADirectoryError(_bounded_text(base_dir))
        for _ in range(4):
            run_dir = base_dir / self.run_id
            try:
                run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
                break
            except FileExistsError:
                self.run_id = _token()
        else:
            raise FileExistsError("unable to allocate unique workflow trace directory")
        latest_path = base_dir / "LATEST"
        self._atomic_write_text(latest_path, f"{self.run_id}\n")
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
        frame = inspect.currentframe()
        try:
            for _ in range(loc_skip):
                if frame is not None:
                    frame = frame.f_back
            module = frame.f_globals.get("__name__", __name__) if frame is not None else __name__
            file_name = frame.f_code.co_filename if frame is not None else __file__
            func_name = frame.f_code.co_name if frame is not None else "_unknown"
            line_no = frame.f_lineno if frame is not None else 0
        finally:
            del frame  # AUDIT-FIX(#12): Bricht Referenzzyklen im Hot-Path der Callsite-Inspektion.
        return {
            "ts_wall_utc": _utc_now(),
            "ts_mono_ns": time.monotonic_ns(),
            "level": _bounded_text(level, default="INFO"),
            "run_id": self.run_id,
            "event_id": _token(),
            "parent_event_id": parent_event_id,
            "trace_id": trace_id or _token(),
            "span_id": span_id,
            "parent_span_id": parent_span_id,  # AUDIT-FIX(#8): Erhält Parent-Span-Verkettung für Trace-Rekonstruktion.
            "proc_id": os.getpid(),
            "thread_id": current_thread().ident,
            "task_id": _task_id(),
            "host": socket.gethostname(),
            "service": self.service,
            "version": platform.python_version(),
            "loc": {
                "module": module,
                "func": func_name,
                "file": file_name,
                "line": line_no,
            },
            "kind": _bounded_text(kind, default="event"),
            "msg": _bounded_text(msg, default="unknown"),
            "details": _sanitize_details(
                _normalize_legacy_exception_details(details),
                allow_raw_text=self.allow_raw_text,
            ),
            "kpi": _sanitize_details(kpi or {}, allow_raw_text=True),
            "reason": _sanitize_details(reason or {}, allow_raw_text=self.allow_raw_text),
        }

    def _writer_main(self) -> None:
        jsonl_path = self._jsonl_path
        trace_path = self._trace_path
        if jsonl_path is None or trace_path is None:
            self._writer_stop.set()
            return
        try:
            with self._open_append_text(jsonl_path) as run_handle, self._open_append_text(trace_path) as trace_handle:
                while not self._writer_stop.is_set():
                    try:
                        item = self._event_queue.get(timeout=0.25)
                    except Empty:
                        continue
                    if item is _QUEUE_SENTINEL:
                        self._writer_stop.set()
                        break
                    if not isinstance(item, dict):
                        continue
                    with self._stats_lock:
                        if self._event_count >= self._max_events:
                            self._trace_truncated = True  # AUDIT-FIX(#2): Begrenzt Run-Größe statt die Disk zu füllen.
                            self._events_dropped_for_limit += 1
                            continue
                    try:
                        encoded = self._json_dumps(item)
                    except Exception as exc:
                        with self._stats_lock:
                            self._queue_dropped += 1
                        self._safe_stderr(f"[twinr-workflow-trace] dropped unserializable record: {_bounded_text(exc)}")
                        continue
                    try:
                        run_handle.write(encoded + "\n")
                        if item.get("kind") in {"span_start", "span_end", "exception"}:
                            trace_handle.write(encoded + "\n")
                        run_handle.flush()
                        trace_handle.flush()
                    except OSError as exc:  # AUDIT-FIX(#9): Stoppt sauber bei persistenten Storage-Fehlern statt still zu sterben.
                        with self._stats_lock:
                            self._writer_failed = True
                            self._writer_error = _bounded_text(exc)
                            self._accepting_events = False
                        self._safe_stderr(f"[twinr-workflow-trace] writer stopped after I/O failure: {_bounded_text(exc)}")
                        self._writer_stop.set()
                        break
                    self._update_stats_from_record(item)
                    if self._should_flush_sidecars():
                        self._safe_flush_sidecars(force=False)
        except Exception as exc:  # AUDIT-FIX(#9): Unerwartete Writer-Crashes werden eingegrenzt und deterministisch sichtbar gemacht.
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
            del self._span_durations_ms[:-self._max_span_history]  # AUDIT-FIX(#2): Hält nur begrenzte Span-Historie.

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
            "run_id": self.run_id,
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
        }
        summary = {
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
                key=lambda item: float(item.get("duration_ms", 0.0)),
                reverse=True,
            )[:16],
            "queue_dropped": queue_dropped,
            "events_dropped_for_limit": events_dropped_for_limit,
            "trace_truncated": trace_truncated,
            "writer_failed": writer_failed,
            "writer_error": writer_error,
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
        try:  # AUDIT-FIX(#6): Sidecars werden atomar ersetzt, um zerrissenes JSON nach Crash/Power-Loss zu vermeiden.
            self._atomic_write_text(
                metrics_path,
                self._json_dumps(metrics, pretty=True),
            )
            self._atomic_write_text(
                summary_path,
                self._json_dumps(summary, pretty=True),
            )
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
                details={"exception": _exception_summary_from_parts(
                    exc_type,
                    exc_value,
                    exc_traceback,
                    allow_raw_text=self.allow_raw_text,
                )},
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
            threading.excepthook = thread_excepthook  # AUDIT-FIX(#5): Deckt ungefangene Exceptions in Helper-Threads ab.
        if previous_unraisablehook is not None:
            sys.unraisablehook = unraisablehook  # AUDIT-FIX(#5): Erfasst unraisable Exceptions aus Finalizern/GC.

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
            "run_id": self.run_id,
            "service": self.service,
            "cwd": str(Path.cwd()),
            "argv": _sanitize_argv(sys.argv, allow_raw_text=self.allow_raw_text),  # AUDIT-FIX(#3): Dumpt Startup-Argumente standardmäßig nicht im Klartext.
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        env_snapshot = {
            key: _sanitize_details(os.environ.get(key, ""), allow_raw_text=self.allow_raw_text, key_hint=key)
            for key in _ENV_WHITELIST
            if key in os.environ
        }
        repro_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._atomic_write_text(
            repro_dir / "runtime.json",
            self._json_dumps(repro, pretty=True),
        )
        self._atomic_write_text(
            repro_dir / "env.json",
            self._json_dumps(env_snapshot, pretty=True),
        )

    def _open_append_text(self, path: Path) -> TextIO:
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW  # AUDIT-FIX(#6): Folgt keinen Symlinks, falls die Plattform das unterstützt.
        fd = os.open(path, flags, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8", buffering=1)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        directory = path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(directory),
                delete=False,
            ) as temp_handle:
                temp_path = temp_handle.name
                try:
                    os.chmod(temp_path, 0o600)
                except OSError:
                    pass
                temp_handle.write(content)
                temp_handle.flush()
            os.replace(temp_path, path)  # AUDIT-FIX(#6): Atomarer Replace verhindert TOCTOU-/Partial-Write-Probleme.
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

    def _safe_stderr(self, message: str) -> None:
        try:
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            try:
                sys.__stderr__.write(f"{message}\n")
                sys.__stderr__.flush()
            except Exception:
                _LOGGER.warning("Workflow forensics failed to write to stderr fallback.", exc_info=True)


def current_workflow_forensics() -> WorkflowForensics | None:
    """Return the tracer currently bound to this execution context."""

    tracer = _ACTIVE_TRACER.get()
    if isinstance(tracer, WorkflowForensics) and tracer.enabled:
        return tracer
    return None


def current_workflow_trace_id() -> str | None:
    """Return the trace identifier currently bound to this execution context."""

    return _ACTIVE_TRACE_ID.get()


def current_workflow_span_id() -> str | None:
    """Return the span identifier currently bound to this execution context."""

    return _ACTIVE_SPAN_ID.get()


@contextmanager
def bind_workflow_forensics(
    tracer: WorkflowForensics | None,
    *,
    trace_id: str | None = None,
) -> Iterator[str | None]:
    """Bind one tracer and optional trace id to the current execution context."""

    active_tracer = tracer if isinstance(tracer, WorkflowForensics) and tracer.enabled else None
    tracer_token = _ACTIVE_TRACER.set(active_tracer)
    trace_token = _ACTIVE_TRACE_ID.set(trace_id or _ACTIVE_TRACE_ID.get())
    span_token = _ACTIVE_SPAN_ID.set(None)
    try:
        yield _ACTIVE_TRACE_ID.get()
    finally:
        _ACTIVE_SPAN_ID.reset(span_token)
        _ACTIVE_TRACE_ID.reset(trace_token)
        _ACTIVE_TRACER.reset(tracer_token)


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
    trace_id = current_workflow_trace_id() or _token()
    parent_span_id = current_workflow_span_id()
    if tracer is None:
        state = _SpanState(
            trace_id=trace_id,
            span_id=_token()[:16],
            parent_span_id=parent_span_id,
            started_mono_ns=time.monotonic_ns(),
            name=name,
            kind=kind,
            details=details or {},
        )
        yield state
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
        try:
            yield state
        finally:
            _ACTIVE_SPAN_ID.reset(span_token)
            _ACTIVE_TRACE_ID.reset(trace_token)
