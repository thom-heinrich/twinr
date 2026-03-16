"""Forensic-grade workflow tracing for Twinr hardware loops.

This module provides a bounded, structured run pack for difficult live-runtime
bugs. It is intentionally workflow-local so button/session/audio issues can be
traced without scattering JSON-writing logic across the orchestration code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread, current_thread
from types import TracebackType
from typing import Callable, Iterator
import atexit
import inspect
import json
import os
import platform
import socket
import sys
import time
import traceback
import uuid


_TRACE_ENV_ENABLED = "TWINR_WORKFLOW_TRACE_ENABLED"
_TRACE_ENV_MODE = "TWINR_WORKFLOW_TRACE_MODE"
_TRACE_ENV_DIR = "TWINR_WORKFLOW_TRACE_DIR"
_TRACE_ENV_SCOPE = "TWINR_WORKFLOW_TRACE_SCOPE"
_TRACE_ENV_SAMPLE_RATE = "TWINR_WORKFLOW_TRACE_SAMPLE_RATE"
_TRACE_ENV_ALLOW_RAW_TEXT = "TWINR_WORKFLOW_TRACE_ALLOW_RAW_TEXT"
_QUEUE_SENTINEL = object()
_SUMMARY_FLUSH_INTERVAL = 1
_MAX_DETAIL_TEXT = 240
_MAX_STACK_LINES = 12
_ENV_WHITELIST: tuple[str, ...] = (
    "TWINR_WORKFLOW_TRACE_ENABLED",
    "TWINR_WORKFLOW_TRACE_MODE",
    "TWINR_WORKFLOW_TRACE_SCOPE",
    "TWINR_WORKFLOW_TRACE_SAMPLE_RATE",
    "TWINR_WORKFLOW_TRACE_ALLOW_RAW_TEXT",
    "TWINR_LLM_PROVIDER",
    "TWINR_STT_PROVIDER",
    "TWINR_TTS_PROVIDER",
    "TWINR_LONG_TERM_MEMORY_MODE",
    "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED",
)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple dotenv-style ``KEY=VALUE`` pairs for trace configuration."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
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


def _bounded_text(value: object, *, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        return default
    if len(text) > _MAX_DETAIL_TEXT:
        return f"{text[:_MAX_DETAIL_TEXT - 3]}..."
    return text


def _sanitize_details(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, Path):
        return _bounded_text(value)
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in list(value.items())[:64]:
            lowered = str(key).casefold()
            if any(secret in lowered for secret in ("api_key", "token", "password", "secret", "authorization", "cookie")):
                cleaned[str(key)] = "[redacted]"
                continue
            cleaned[str(key)] = _sanitize_details(item)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_details(item) for item in list(value)[:32]]
    return _bounded_text(repr(value))


def _exception_summary(exc: BaseException) -> dict[str, object]:
    return {
        "type": type(exc).__name__,
        "message": _bounded_text(exc),
        "stack": traceback.format_exception(type(exc), exc, exc.__traceback__)[-1 * _MAX_STACK_LINES :],
    }


def _task_id() -> int | None:
    try:
        import asyncio

        task = asyncio.current_task()
    except Exception:
        return None
    if task is None:
        return None
    return id(task)


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
    ) -> None:
        self.project_root = project_root
        self.service = service
        self.enabled = bool(enabled)
        self.mode = mode
        self.base_dir = base_dir
        self.run_id = _token()
        self._event_queue: Queue[dict[str, object] | object] = Queue()
        self._writer_stop = Event()
        self._closed = False
        self._counts_by_kind: dict[str, int] = {}
        self._counts_by_msg: dict[str, int] = {}
        self._span_durations_ms: list[dict[str, object]] = []
        self._exception_counts: dict[str, int] = {}
        self._event_count = 0
        self._slowest_ms = 0.0
        self._last_event_ts = None
        self._writer: Thread | None = None
        self._run_dir = self._build_run_dir()
        self._jsonl_path = self._run_dir / "run.jsonl"
        self._trace_path = self._run_dir / "run.trace"
        self._metrics_path = self._run_dir / "run.metrics.json"
        self._summary_path = self._run_dir / "run.summary.json"
        self._repro_dir = self._run_dir / "run.repro"
        self._repro_dir.mkdir(parents=True, exist_ok=True)
        self._install_exception_hooks()
        self._write_repro_pack()
        if self.enabled:
            self._writer = Thread(target=self._writer_main, daemon=True, name="twinr-workflow-trace")
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
        atexit.register(self.close)

    @classmethod
    def from_env(cls, *, project_root: Path, service: str) -> "WorkflowForensics":
        file_values: dict[str, str] = {}
        for candidate in (Path.cwd() / ".env", project_root / ".env"):
            file_values.update(_read_dotenv(candidate))
        enabled_raw = os.environ.get(_TRACE_ENV_ENABLED, file_values.get(_TRACE_ENV_ENABLED, "0"))
        mode = os.environ.get(_TRACE_ENV_MODE, file_values.get(_TRACE_ENV_MODE, "forensic")).strip().lower() or "forensic"
        base_dir_raw = os.environ.get(_TRACE_ENV_DIR, file_values.get(_TRACE_ENV_DIR, "")).strip()
        return cls(
            project_root=project_root,
            service=service,
            enabled=enabled_raw.strip().lower() in {"1", "true", "yes", "on"},
            mode=mode,
            base_dir=Path(base_dir_raw) if base_dir_raw else None,
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
        parent_event_id: str | None = None,
        loc_skip: int = 2,
    ) -> None:
        if not self.enabled:
            return
        record = self._base_record(
            kind=kind,
            msg=msg,
            level=level,
            details=details,
            reason=reason,
            kpi=kpi,
            trace_id=trace_id,
            span_id=span_id,
            parent_event_id=parent_event_id,
            loc_skip=loc_skip + 1,
        )
        self._event_queue.put(record)

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
    ) -> None:
        self.event(
            kind="decision",
            msg=msg,
            details={"question": _bounded_text(question), "context": _sanitize_details(context or {})},
            reason={
                "selected": _sanitize_details(selected),
                "options": _sanitize_details(options),
                "confidence": _sanitize_details(confidence),
                "guardrails": _sanitize_details(guardrails or []),
                "kpi_impact_estimate": _sanitize_details(kpi_impact_estimate or {}),
            },
            trace_id=trace_id,
            span_id=span_id,
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
        if not self.enabled:
            yield _SpanState(
                trace_id=trace_id or _token(),
                span_id=_token()[:16],
                parent_span_id=parent_span_id,
                started_mono_ns=time.monotonic_ns(),
                name=name,
                kind=kind,
                details=details or {},
            )
            return
        state = _SpanState(
            trace_id=trace_id or _token(),
            span_id=_token()[:16],
            parent_span_id=parent_span_id,
            started_mono_ns=time.monotonic_ns(),
            name=name,
            kind=kind,
            details=details or {},
        )
        self.event(
            kind="span_start",
            msg=name,
            details={"kind": kind, **(details or {})},
            trace_id=state.trace_id,
            span_id=state.span_id,
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
                details={"span": name, "exception": _exception_summary(exc)},
                kpi={"duration_ms": round(duration_ms, 3)},
                trace_id=state.trace_id,
                span_id=state.span_id,
                loc_skip=2,
            )
            self._span_durations_ms.append(
                {"name": name, "duration_ms": round(duration_ms, 3), "status": "error"}
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
                loc_skip=2,
            )
            self._span_durations_ms.append(
                {"name": name, "duration_ms": round(duration_ms, 3), "status": "ok"}
            )

    def close(self) -> None:
        if self._closed or not self.enabled:
            return
        self._closed = True
        try:
            self.event(kind="run_end", msg="workflow_trace_stopped", details={}, loc_skip=2)
        except Exception:
            pass
        self._event_queue.put(_QUEUE_SENTINEL)
        writer = self._writer
        if writer is not None:
            writer.join(timeout=1.0)
        self._flush_sidecars()

    def _build_run_dir(self) -> Path:
        if self.base_dir is not None:
            base_dir = self.base_dir
        else:
            base_dir = self.project_root / "state" / "forensics" / "workflow"
        run_dir = base_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        latest_path = base_dir / "LATEST"
        try:
            latest_path.write_text(f"{self.run_id}\n", encoding="utf-8")
        except Exception:
            pass
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
        parent_event_id: str | None,
        loc_skip: int,
    ) -> dict[str, object]:
        frame = inspect.currentframe()
        for _ in range(loc_skip):
            if frame is not None:
                frame = frame.f_back
        module = frame.f_globals.get("__name__", __name__) if frame is not None else __name__
        file_name = frame.f_code.co_filename if frame is not None else __file__
        func_name = frame.f_code.co_name if frame is not None else "_unknown"
        line_no = frame.f_lineno if frame is not None else 0
        return {
            "ts_wall_utc": _utc_now(),
            "ts_mono_ns": time.monotonic_ns(),
            "level": level,
            "run_id": self.run_id,
            "event_id": _token(),
            "parent_event_id": parent_event_id,
            "trace_id": trace_id or _token(),
            "span_id": span_id,
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
            "kind": kind,
            "msg": _bounded_text(msg),
            "details": _sanitize_details(details or {}),
            "kpi": _sanitize_details(kpi or {}),
            "reason": _sanitize_details(reason or {}),
        }

    def _writer_main(self) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as run_handle, self._trace_path.open(
            "a", encoding="utf-8"
        ) as trace_handle:
            while not self._writer_stop.is_set():
                try:
                    item = self._event_queue.get(timeout=0.25)
                except Empty:
                    continue
                if item is _QUEUE_SENTINEL:
                    self._writer_stop.set()
                    break
                record = item
                self._event_count += 1
                self._last_event_ts = record["ts_wall_utc"]
                kind = str(record["kind"])
                msg = str(record["msg"])
                self._counts_by_kind[kind] = self._counts_by_kind.get(kind, 0) + 1
                self._counts_by_msg[msg] = self._counts_by_msg.get(msg, 0) + 1
                if kind == "exception":
                    exc_type = str(((record.get("details") or {}).get("exception") or {}).get("type", "unknown"))
                    self._exception_counts[exc_type] = self._exception_counts.get(exc_type, 0) + 1
                encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
                run_handle.write(encoded + "\n")
                if kind in {"span_start", "span_end"}:
                    trace_handle.write(encoded + "\n")
                run_handle.flush()
                trace_handle.flush()
                duration_ms = float(((record.get("kpi") or {}).get("duration_ms") or 0.0) or 0.0)
                if duration_ms > self._slowest_ms:
                    self._slowest_ms = duration_ms
                if self._event_count % _SUMMARY_FLUSH_INTERVAL == 0:
                    self._flush_sidecars()

    def _flush_sidecars(self) -> None:
        metrics = {
            "run_id": self.run_id,
            "event_count": self._event_count,
            "counts_by_kind": self._counts_by_kind,
            "counts_by_msg": self._counts_by_msg,
            "exception_counts": self._exception_counts,
            "slowest_duration_ms": round(self._slowest_ms, 3),
            "span_durations_ms": self._span_durations_ms[-64:],
        }
        summary = {
            "run_id": self.run_id,
            "service": self.service,
            "mode": self.mode,
            "event_count": self._event_count,
            "last_event_ts": self._last_event_ts,
            "top_kinds": sorted(self._counts_by_kind.items(), key=lambda item: item[1], reverse=True)[:16],
            "top_msgs": sorted(self._counts_by_msg.items(), key=lambda item: item[1], reverse=True)[:16],
            "exception_counts": self._exception_counts,
            "slowest_spans": sorted(
                self._span_durations_ms,
                key=lambda item: float(item.get("duration_ms", 0.0)),
                reverse=True,
            )[:16],
        }
        self._metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self._summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _install_exception_hooks(self) -> None:
        if not self.enabled:
            return

        previous_excepthook = sys.excepthook

        def excepthook(exc_type, exc_value, exc_traceback):
            self.event(
                kind="exception",
                msg="sys_excepthook",
                level="ERROR",
                details={
                    "exception": {
                        "type": getattr(exc_type, "__name__", str(exc_type)),
                        "message": _bounded_text(exc_value),
                        "stack": traceback.format_exception(exc_type, exc_value, exc_traceback)[-1 * _MAX_STACK_LINES :],
                    }
                },
                loc_skip=2,
            )
            if callable(previous_excepthook):
                previous_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = excepthook

    def _write_repro_pack(self) -> None:
        repro = {
            "run_id": self.run_id,
            "service": self.service,
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        env_snapshot = {key: _bounded_text(os.environ.get(key, "")) for key in _ENV_WHITELIST if key in os.environ}
        self._repro_dir.mkdir(parents=True, exist_ok=True)
        (self._repro_dir / "runtime.json").write_text(
            json.dumps(repro, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._repro_dir / "env.json").write_text(
            json.dumps(env_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
