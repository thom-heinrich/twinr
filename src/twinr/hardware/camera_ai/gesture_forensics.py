# CHANGELOG: 2026-03-28
# BUG-1: Fixed incorrect memory KPI reporting: rss_mb now reports current RSS on Linux instead of ru_maxrss peak RSS.
# BUG-2: Fixed deep-exec semantics: Python 3.12+ now records real control-flow edges (BRANCH/JUMP) instead of every line transition.
# BUG-3: Fixed sampled-out telemetry spam: skipped deep-exec refreshes are now rate-limited to avoid saturating the tracer queue.
# BUG-4: Fixed scope matching: monitored files are now matched by exact file / directory boundaries rather than naive string prefixes.
# SEC-1: Fixed raw exception leakage: exception messages are redacted when raw-text capture is disabled.
# IMP-1: Upgraded deep-exec to sys.monitoring on Python 3.12+ with per-code-object activation and automatic fallback to tracing when unavailable.
# IMP-2: Improved fallback coverage and artifact hygiene: current-thread tracing fallback now attempts all-thread coverage and emits project-relative paths.

"""Scoped run-pack forensics for Twinr's live Pi gesture path.

This module binds the existing workflow forensics tracer to the dedicated
gesture acknowledgement lane and adds an opt-in deep-exec mode that records
control-flow edges only for a small set of gesture modules.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, MethodType
from typing import Any, Iterator
import hashlib
import os
import random
import resource
import sys
import threading
import time
import uuid

from twinr.agent.workflows.forensics import (
    WorkflowForensics,
    bind_workflow_forensics,
    workflow_event,
    workflow_span,
)


_ENV_ENABLED = "TWINR_GESTURE_FORENSICS_ENABLED"
_ENV_MODE = "TWINR_GESTURE_FORENSICS_MODE"
_ENV_DIR = "TWINR_GESTURE_FORENSICS_DIR"
_ENV_ALLOW_RAW_TEXT = "TWINR_GESTURE_FORENSICS_ALLOW_RAW_TEXT"
_ENV_SAMPLE_RATE = "TWINR_GESTURE_FORENSICS_SAMPLE_RATE"
_ENV_SCOPE = "TWINR_GESTURE_FORENSICS_SCOPE"
_ENV_MAX_BRANCH_EVENTS = "TWINR_GESTURE_FORENSICS_MAX_BRANCH_EVENTS"
_ENV_QUEUE_MAXSIZE = "TWINR_GESTURE_FORENSICS_QUEUE_MAXSIZE"
_ENV_MAX_EVENTS = "TWINR_GESTURE_FORENSICS_MAX_EVENTS"
_ENV_MAX_SPAN_HISTORY = "TWINR_GESTURE_FORENSICS_MAX_SPAN_HISTORY"
_ENV_DEEP_EXEC_MAX_REFRESHES = "TWINR_GESTURE_FORENSICS_DEEP_EXEC_MAX_REFRESHES"
_ENV_UNSAFE_CONTINUOUS_DEEP_EXEC = "TWINR_GESTURE_FORENSICS_UNSAFE_CONTINUOUS_DEEP_EXEC"

_DEFAULT_SCOPE = (
    "src/twinr/proactive/runtime/service.py",
    "src/twinr/proactive/runtime/gesture_ack_lane.py",
    "src/twinr/proactive/runtime/gesture_wakeup_lane.py",
    "src/twinr/hardware/camera_ai/adapter.py",
    "src/twinr/hardware/camera_ai/live_gesture_pipeline.py",
    "src/twinr/hardware/hand_landmarks.py",
)
_DEFAULT_SAMPLE_RATE = 1.0
_DEFAULT_MAX_BRANCH_EVENTS = 768
_DEFAULT_QUEUE_MAXSIZE = 4096
_DEFAULT_MAX_EVENTS = 20000
_DEFAULT_MAX_SPAN_HISTORY = 512
_DEFAULT_DEEP_EXEC_MAX_REFRESHES = 1

_MONITORING_TOOL_CANDIDATES = (4, 3, 5)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Return simple ``KEY=VALUE`` entries from one dotenv-style file."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
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


def _bool_from_env(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _float_from_env(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None:
        return default
    try:
        parsed = float(raw.strip())
    except (AttributeError, TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return max(minimum, min(maximum, parsed))


def _int_from_env(raw: str | None, *, default: int, minimum: int, maximum: int) -> int:
    if raw is None:
        return default
    try:
        parsed = int(raw.strip())
    except (AttributeError, TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _safe_resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _resolve_scope_prefixes(project_root: Path, raw_scope: str | None) -> tuple[Path, ...]:
    tokens = [item.strip() for item in str(raw_scope or "").split(",") if item.strip()]
    if not tokens:
        tokens = list(_DEFAULT_SCOPE)
    prefixes: list[Path] = []
    for token in tokens:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)
        prefixes.append(candidate)
    return tuple(prefixes)


def _display_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(project_root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def _matches_scope(file_path: Path, scope_prefixes: tuple[Path, ...]) -> bool:
    resolved = file_path.resolve(strict=False)
    for prefix in scope_prefixes:
        candidate = prefix.resolve(strict=False)
        if candidate.suffix:
            if resolved == candidate:
                return True
            continue
        try:
            resolved.relative_to(candidate)
            return True
        except ValueError:
            continue
    return False


def _current_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        try:
            statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            resident_pages = int(statm[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return round((resident_pages * page_size) / (1024.0 * 1024.0), 3)
        except (FileNotFoundError, OSError, IndexError, ValueError):
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_raw = float(getattr(usage, "ru_maxrss", 0.0) or 0.0)
    if sys.platform == "darwin":
        rss_mb = rss_raw / (1024.0 * 1024.0)
    else:
        rss_mb = rss_raw / 1024.0
    return round(max(0.0, rss_mb), 3)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_raw = float(getattr(usage, "ru_maxrss", 0.0) or 0.0)
    if sys.platform == "darwin":
        rss_mb = rss_raw / (1024.0 * 1024.0)
    else:
        rss_mb = rss_raw / 1024.0
    return round(max(0.0, rss_mb), 3)


def _resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_user_s": float(getattr(usage, "ru_utime", 0.0) or 0.0),
        "cpu_system_s": float(getattr(usage, "ru_stime", 0.0) or 0.0),
        "rss_mb": _current_rss_mb(),
        "rss_peak_mb": _peak_rss_mb(),
    }


def _resource_delta(*, started_mono_ns: int, started: dict[str, float]) -> dict[str, object]:
    # BREAKING: rss_mb now reports current RSS. Peak RSS moved to rss_peak_mb,
    # and rss_delta_mb was added for regression triage.
    finished = _resource_snapshot()
    return {
        "duration_ms": round(max(0.0, (time.monotonic_ns() - started_mono_ns) / 1_000_000.0), 3),
        "cpu_user_ms": round(max(0.0, (finished["cpu_user_s"] - started["cpu_user_s"]) * 1000.0), 3),
        "cpu_system_ms": round(max(0.0, (finished["cpu_system_s"] - started["cpu_system_s"]) * 1000.0), 3),
        "rss_mb": finished["rss_mb"],
        "rss_delta_mb": round(finished["rss_mb"] - started["rss_mb"], 3),
        "rss_peak_mb": finished["rss_peak_mb"],
    }


def _redact_exception_text(exc: BaseException, *, allow_raw_text: bool) -> str:
    if allow_raw_text:
        return str(exc or type(exc).__name__)
    raw = str(exc or "")
    if not raw:
        return type(exc).__name__
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"<redacted:{type(exc).__name__}:sha256={digest}:len={len(raw)}>"


def _should_emit_log_counter(count: int) -> bool:
    return count > 0 and (count & (count - 1)) == 0


def _iter_nested_code_objects(code: CodeType, seen_codes: set[int]) -> Iterator[CodeType]:
    code_id = id(code)
    if code_id in seen_codes:
        return
    seen_codes.add(code_id)
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from _iter_nested_code_objects(const, seen_codes)


def _iter_code_objects_from_object(obj: object, seen_objects: set[int], seen_codes: set[int]) -> Iterator[CodeType]:
    obj_id = id(obj)
    if obj_id in seen_objects:
        return
    seen_objects.add(obj_id)

    if isinstance(obj, CodeType):
        yield from _iter_nested_code_objects(obj, seen_codes)
        return
    if isinstance(obj, (FunctionType, MethodType)):
        code = getattr(obj, "__code__", None)
        if isinstance(code, CodeType):
            yield from _iter_nested_code_objects(code, seen_codes)
        return
    if isinstance(obj, (staticmethod, classmethod)):
        func = getattr(obj, "__func__", None)
        if func is not None:
            yield from _iter_code_objects_from_object(func, seen_objects, seen_codes)
        return
    if isinstance(obj, property):
        for attr in (obj.fget, obj.fset, obj.fdel):
            if attr is not None:
                yield from _iter_code_objects_from_object(attr, seen_objects, seen_codes)
        return
    if isinstance(obj, type):
        for value in obj.__dict__.values():
            yield from _iter_code_objects_from_object(value, seen_objects, seen_codes)
        return


def _collect_scoped_code_objects(scope_prefixes: tuple[Path, ...]) -> tuple[CodeType, ...]:
    seen_objects: set[int] = set()
    seen_codes: set[int] = set()
    collected: list[CodeType] = []
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            resolved_module_file = _safe_resolve_path(module_file)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if not _matches_scope(resolved_module_file, scope_prefixes):
            continue
        module_dict = getattr(module, "__dict__", None)
        if not isinstance(module_dict, dict):
            continue
        for value in module_dict.values():
            collected.extend(_iter_code_objects_from_object(value, seen_objects, seen_codes))
    return tuple(collected)


def _code_offset_line_map(code: CodeType) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}
    try:
        iterator = code.co_lines()
    except AttributeError:
        return mapping
    for start, end, line in iterator:
        for offset in range(int(start), int(end)):
            mapping[offset] = int(line) if line is not None else None
    return mapping


@dataclass(frozen=True, slots=True)
class GestureForensicsConfig:
    """Store bounded runtime settings for gesture-path tracing."""

    enabled: bool = False
    mode: str = "forensic"
    base_dir: Path | None = None
    allow_raw_text: bool = False
    sample_rate: float = _DEFAULT_SAMPLE_RATE
    scope_prefixes: tuple[Path, ...] = ()
    max_branch_events: int = _DEFAULT_MAX_BRANCH_EVENTS
    queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE
    max_events: int = _DEFAULT_MAX_EVENTS
    max_span_history: int = _DEFAULT_MAX_SPAN_HISTORY
    deep_exec_max_refreshes: int = _DEFAULT_DEEP_EXEC_MAX_REFRESHES
    unsafe_continuous_deep_exec: bool = False


class _DeepExecMonitor:
    """Trace scoped control-flow edges for the configured gesture modules."""

    def __init__(
        self,
        *,
        project_root: Path,
        scope_prefixes: tuple[Path, ...],
        max_branch_events: int,
    ) -> None:
        self._project_root = project_root.resolve(strict=False)
        self._scope_prefixes = tuple(path.resolve(strict=False) for path in scope_prefixes)
        self._scope_display_prefixes = tuple(
            _display_path(path, project_root=self._project_root) for path in self._scope_prefixes
        )
        self._max_branch_events = max(1, int(max_branch_events))
        self._branch_events = 0
        self._dropped_branch_events = 0
        self._reentrant = False
        self._active = False
        self._backend = "inactive"
        self._monitoring = getattr(sys, "monitoring", None)
        self._monitoring_tool_id: int | None = None
        self._monitoring_code_objects: tuple[CodeType, ...] = ()
        self._monitoring_callbacks_registered = False
        self._line_map_cache: dict[int, dict[int, int | None]] = {}
        self._previous_trace: Any = None
        self._previous_threading_trace: Any = None
        self._last_lines: dict[int, int] = {}

    def __enter__(self) -> "_DeepExecMonitor":
        if self._start_monitoring_backend():
            return self
        self._start_settrace_backend()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        had_backend = self._backend != "inactive"
        backend = self._backend
        active = self._active
        try:
            if self._backend == "sys.monitoring":
                self._stop_monitoring_backend()
            elif self._backend.startswith("settrace"):
                self._stop_settrace_backend()
        finally:
            if had_backend:
                workflow_event(
                    kind="metric",
                    msg="gesture_deep_exec_stopped",
                    details={
                        "backend": backend,
                        "branch_events": self._branch_events,
                        "dropped_branch_events": self._dropped_branch_events,
                        "active": active,
                    },
                )
            self._last_lines.clear()
            self._line_map_cache.clear()
            self._active = False
            self._backend = "inactive"

    def _start_monitoring_backend(self) -> bool:
        monitoring = self._monitoring
        if monitoring is None:
            return False
        tool_id = self._acquire_monitoring_tool_id(monitoring)
        if tool_id is None:
            workflow_event(
                kind="warning",
                msg="gesture_deep_exec_monitoring_unavailable",
                details={"reason": "no_free_tool_id"},
            )
            return False
        self._monitoring_tool_id = tool_id

        code_objects = _collect_scoped_code_objects(self._scope_prefixes)
        if not code_objects:
            workflow_event(
                kind="warning",
                msg="gesture_deep_exec_monitoring_unavailable",
                details={"reason": "no_scoped_code_objects"},
            )
            self._stop_monitoring_backend()
            return False
        self._monitoring_code_objects = code_objects

        try:
            self._register_monitoring_callbacks(monitoring, tool_id)
            event_set = self._monitoring_event_set(monitoring)
            for code in code_objects:
                monitoring.set_local_events(tool_id, code, event_set)
        except Exception:
            self._stop_monitoring_backend()
            raise

        # BREAKING: On Python 3.12+, deep-exec now records real BRANCH/JUMP
        # control-flow edges via sys.monitoring instead of every line transition.
        self._backend = "sys.monitoring"
        self._active = True
        workflow_event(
            kind="metric",
            msg="gesture_deep_exec_started",
            details={
                "backend": self._backend,
                "scope_prefixes": self._scope_display_prefixes,
                "max_branch_events": self._max_branch_events,
                "traced_code_objects": len(code_objects),
            },
        )
        return True

    @staticmethod
    def _acquire_monitoring_tool_id(monitoring: Any) -> int | None:
        get_tool = getattr(monitoring, "get_tool", None)
        for tool_id in _MONITORING_TOOL_CANDIDATES:
            try:
                if callable(get_tool) and get_tool(tool_id) is not None:
                    continue
                monitoring.use_tool_id(tool_id, "twinr_gesture_forensics")
                return int(tool_id)
            except Exception:
                continue
        return None

    def _register_monitoring_callbacks(self, monitoring: Any, tool_id: int) -> None:
        events = monitoring.events
        if hasattr(events, "BRANCH_LEFT") and hasattr(events, "BRANCH_RIGHT"):
            monitoring.register_callback(tool_id, events.BRANCH_LEFT, self._on_branch_left)
            monitoring.register_callback(tool_id, events.BRANCH_RIGHT, self._on_branch_right)
        elif hasattr(events, "BRANCH"):
            monitoring.register_callback(tool_id, events.BRANCH, self._on_branch)
        monitoring.register_callback(tool_id, events.JUMP, self._on_jump)
        self._monitoring_callbacks_registered = True

    @staticmethod
    def _monitoring_event_set(monitoring: Any) -> int:
        events = monitoring.events
        event_set = int(events.JUMP)
        if hasattr(events, "BRANCH_LEFT") and hasattr(events, "BRANCH_RIGHT"):
            event_set |= int(events.BRANCH_LEFT) | int(events.BRANCH_RIGHT)
        elif hasattr(events, "BRANCH"):
            event_set |= int(events.BRANCH)
        return event_set

    def _stop_monitoring_backend(self) -> None:
        monitoring = self._monitoring
        tool_id = self._monitoring_tool_id
        if monitoring is None or tool_id is None:
            self._monitoring_tool_id = None
            self._monitoring_code_objects = ()
            self._monitoring_callbacks_registered = False
            return
        try:
            if self._monitoring_callbacks_registered:
                events = monitoring.events
                if hasattr(events, "BRANCH_LEFT") and hasattr(events, "BRANCH_RIGHT"):
                    monitoring.register_callback(tool_id, events.BRANCH_LEFT, None)
                    monitoring.register_callback(tool_id, events.BRANCH_RIGHT, None)
                elif hasattr(events, "BRANCH"):
                    monitoring.register_callback(tool_id, events.BRANCH, None)
                monitoring.register_callback(tool_id, events.JUMP, None)
            clear_tool_id = getattr(monitoring, "clear_tool_id", None)
            if callable(clear_tool_id):
                clear_tool_id(tool_id)
            free_tool_id = getattr(monitoring, "free_tool_id", None)
            if callable(free_tool_id):
                free_tool_id(tool_id)
        finally:
            self._monitoring_tool_id = None
            self._monitoring_code_objects = ()
            self._monitoring_callbacks_registered = False
            self._active = False

    def _on_branch_left(self, code: CodeType, instruction_offset: int, destination_offset: int) -> object | None:
        return self._on_control_flow_event(
            code=code,
            instruction_offset=instruction_offset,
            destination_offset=destination_offset,
            event_kind="branch_left",
        )

    def _on_branch_right(self, code: CodeType, instruction_offset: int, destination_offset: int) -> object | None:
        return self._on_control_flow_event(
            code=code,
            instruction_offset=instruction_offset,
            destination_offset=destination_offset,
            event_kind="branch_right",
        )

    def _on_branch(self, code: CodeType, instruction_offset: int, destination_offset: int) -> object | None:
        return self._on_control_flow_event(
            code=code,
            instruction_offset=instruction_offset,
            destination_offset=destination_offset,
            event_kind="branch",
        )

    def _on_jump(self, code: CodeType, instruction_offset: int, destination_offset: int) -> object | None:
        return self._on_control_flow_event(
            code=code,
            instruction_offset=instruction_offset,
            destination_offset=destination_offset,
            event_kind="jump",
        )

    def _on_control_flow_event(
        self,
        *,
        code: CodeType,
        instruction_offset: int,
        destination_offset: int,
        event_kind: str,
    ) -> object | None:
        if self._reentrant:
            return None
        file_path = _safe_resolve_path(getattr(code, "co_filename", ""))
        if not _matches_scope(file_path, self._scope_prefixes):
            return None
        line_from = self._line_for_offset(code, int(instruction_offset))
        line_to = self._line_for_offset(code, int(destination_offset))
        if line_from is None or line_to is None or line_from == line_to:
            return None
        if self._branch_events >= self._max_branch_events:
            self._dropped_branch_events += 1
            if self._dropped_branch_events == 1:
                self._emit_with_guard(
                    kind="warning",
                    msg="gesture_deep_exec_branch_cap_reached",
                    details={"backend": self._backend, "max_branch_events": self._max_branch_events},
                )
            return getattr(self._monitoring, "DISABLE", None)
        self._branch_events += 1
        # BREAKING: file fields are now project-relative when possible so
        # artifacts are portable across devices and leak less filesystem detail.
        self._emit_branch(
            file_name=_display_path(file_path, project_root=self._project_root),
            func_name=str(getattr(code, "co_qualname", getattr(code, "co_name", "_unknown")) or "_unknown"),
            previous_line=line_from,
            current_line=line_to,
            event_kind=event_kind,
        )
        return None

    def _line_for_offset(self, code: CodeType, offset: int) -> int | None:
        code_id = id(code)
        mapping = self._line_map_cache.get(code_id)
        if mapping is None:
            mapping = _code_offset_line_map(code)
            self._line_map_cache[code_id] = mapping
        line = mapping.get(int(offset))
        return int(line) if line is not None else None

    def _start_settrace_backend(self) -> bool:
        current_trace = sys.gettrace()
        threading_trace = threading.gettrace() if hasattr(threading, "gettrace") else None
        if current_trace is not None or threading_trace is not None:
            workflow_event(
                kind="warning",
                msg="gesture_deep_exec_skipped_existing_trace",
                details={
                    "existing_trace_type": type(current_trace).__name__ if current_trace is not None else None,
                    "existing_thread_trace_type": (
                        type(threading_trace).__name__ if threading_trace is not None else None
                    ),
                },
            )
            return False

        self._previous_trace = current_trace
        self._previous_threading_trace = threading_trace
        if hasattr(threading, "settrace_all_threads"):
            threading.settrace_all_threads(self._trace)
            self._backend = "settrace_all_threads"
        else:
            threading.settrace(self._trace)
            sys.settrace(self._trace)
            self._backend = "settrace"
        self._active = True
        workflow_event(
            kind="metric",
            msg="gesture_deep_exec_started",
            details={
                "backend": self._backend,
                "scope_prefixes": self._scope_display_prefixes,
                "max_branch_events": self._max_branch_events,
                "traced_code_objects": 0,
            },
        )
        return True

    def _stop_settrace_backend(self) -> None:
        try:
            if self._backend == "settrace_all_threads" and hasattr(threading, "settrace_all_threads"):
                threading.settrace_all_threads(self._previous_threading_trace)
            else:
                threading.settrace(self._previous_threading_trace)
                sys.settrace(self._previous_trace)
        finally:
            self._active = False
            self._last_lines.clear()

    def _trace(self, frame, event, arg):  # pragma: no cover - covered indirectly through file artifacts.
        if self._reentrant:
            return None
        code = getattr(frame, "f_code", None)
        if code is None:
            return None
        file_path = _safe_resolve_path(getattr(code, "co_filename", ""))
        if not _matches_scope(file_path, self._scope_prefixes):
            return None
        if event == "call":
            self._last_lines[id(frame)] = int(getattr(frame, "f_lineno", 0) or 0)
            return self._trace
        if event == "line":
            current_line = int(getattr(frame, "f_lineno", 0) or 0)
            previous_line = self._last_lines.get(id(frame))
            self._last_lines[id(frame)] = current_line
            if previous_line is not None and previous_line != current_line:
                if self._branch_events >= self._max_branch_events:
                    self._dropped_branch_events += 1
                    if self._dropped_branch_events == 1:
                        self._emit_with_guard(
                            kind="warning",
                            msg="gesture_deep_exec_branch_cap_reached",
                            details={"backend": self._backend, "max_branch_events": self._max_branch_events},
                        )
                    return None
                self._branch_events += 1
                self._emit_branch(
                    file_name=_display_path(file_path, project_root=self._project_root),
                    func_name=str(getattr(code, "co_name", "_unknown") or "_unknown"),
                    previous_line=previous_line,
                    current_line=current_line,
                    event_kind="line_fallback",
                )
            return self._trace
        if event in {"return", "exception"}:
            self._last_lines.pop(id(frame), None)
            return self._trace
        return self._trace

    def _emit_branch(
        self,
        *,
        file_name: str,
        func_name: str,
        previous_line: int,
        current_line: int,
        event_kind: str,
    ) -> None:
        self._emit_with_guard(
            kind="branch",
            msg="gesture_deep_exec_edge",
            details={
                "backend": self._backend,
                "event_kind": event_kind,
                "file": file_name,
                "func": func_name,
                "line_from": previous_line,
                "line_to": current_line,
                "edge": f"{previous_line}->{current_line}",
            },
            reason={
                "selected": {
                    "id": f"{previous_line}->{current_line}",
                    "justification": "Captured one scoped gesture-path control-flow edge.",
                    "expected_outcome": "Preserve an exact edge for replay-grade debugging.",
                },
                "options": [
                    {
                        "id": f"{previous_line}->{current_line}",
                        "summary": "Observed control-flow edge",
                    }
                ],
                "confidence": "forensic",
                "guardrails": ["gesture_deep_exec_scoped", "branch_cap_enforced"],
                "kpi_impact_estimate": {"overhead": "high"},
            },
        )

    def _emit_with_guard(
        self,
        *,
        kind: str,
        msg: str,
        details: dict[str, object],
        reason: dict[str, object] | None = None,
        level: str = "INFO",
    ) -> None:
        self._reentrant = True
        try:
            workflow_event(
                kind=kind,
                msg=msg,
                details=details,
                reason=reason,
                level=level,
            )
        finally:
            self._reentrant = False


class GestureForensics:
    """Own one scoped tracer for the live gesture acknowledgement path."""

    def __init__(
        self,
        *,
        project_root: Path,
        service: str,
        config: GestureForensicsConfig,
    ) -> None:
        self.project_root = project_root.resolve(strict=False)
        self.service = service
        self.config = config
        self.base_dir = (
            config.base_dir.resolve(strict=False)
            if config.base_dir is not None
            else (self.project_root / "state" / "forensics" / "gesture").resolve(strict=False)
        )
        self.tracer = WorkflowForensics(
            project_root=self.project_root,
            service=service,
            enabled=config.enabled,
            mode=config.mode,
            base_dir=self.base_dir,
            allow_raw_text=config.allow_raw_text,
            queue_maxsize=config.queue_maxsize,
            max_events=config.max_events,
            max_span_history=config.max_span_history,
        )
        self._deep_exec_refreshes_started = 0
        self._deep_exec_guardrail_notice_emitted = False
        self._deep_exec_tracer_saturated_notice_emitted = False
        self._deep_exec_sampled_out_count = 0

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path,
        service: str,
    ) -> "GestureForensics":
        """Build one gesture-path tracer from env or project ``.env`` files."""

        project_root = project_root.expanduser().resolve(strict=False)
        file_values: dict[str, str] = {}
        for candidate in (Path.cwd() / ".env", project_root / ".env"):
            file_values.update(_read_dotenv(candidate))
        enabled_raw = os.environ.get(_ENV_ENABLED, file_values.get(_ENV_ENABLED, "0"))
        mode = str(os.environ.get(_ENV_MODE, file_values.get(_ENV_MODE, "forensic")) or "forensic").strip().lower()
        if mode not in {"forensic", "prod", "deep-exec"}:
            mode = "forensic"
        base_dir_raw = str(os.environ.get(_ENV_DIR, file_values.get(_ENV_DIR, "")) or "").strip()
        scope_raw = os.environ.get(_ENV_SCOPE, file_values.get(_ENV_SCOPE, ""))
        return cls(
            project_root=project_root,
            service=service,
            config=GestureForensicsConfig(
                enabled=_bool_from_env(enabled_raw, default=False),
                mode=mode,
                base_dir=(Path(base_dir_raw).expanduser() if base_dir_raw else None),
                allow_raw_text=_bool_from_env(
                    os.environ.get(_ENV_ALLOW_RAW_TEXT, file_values.get(_ENV_ALLOW_RAW_TEXT, "0")),
                    default=False,
                ),
                sample_rate=_float_from_env(
                    os.environ.get(_ENV_SAMPLE_RATE, file_values.get(_ENV_SAMPLE_RATE, str(_DEFAULT_SAMPLE_RATE))),
                    default=_DEFAULT_SAMPLE_RATE,
                    minimum=0.0,
                    maximum=1.0,
                ),
                scope_prefixes=_resolve_scope_prefixes(project_root, scope_raw),
                max_branch_events=_int_from_env(
                    os.environ.get(
                        _ENV_MAX_BRANCH_EVENTS,
                        file_values.get(_ENV_MAX_BRANCH_EVENTS, str(_DEFAULT_MAX_BRANCH_EVENTS)),
                    ),
                    default=_DEFAULT_MAX_BRANCH_EVENTS,
                    minimum=32,
                    maximum=5000,
                ),
                queue_maxsize=_int_from_env(
                    os.environ.get(
                        _ENV_QUEUE_MAXSIZE,
                        file_values.get(_ENV_QUEUE_MAXSIZE, str(_DEFAULT_QUEUE_MAXSIZE)),
                    ),
                    default=_DEFAULT_QUEUE_MAXSIZE,
                    minimum=128,
                    maximum=20000,
                ),
                max_events=_int_from_env(
                    os.environ.get(
                        _ENV_MAX_EVENTS,
                        file_values.get(_ENV_MAX_EVENTS, str(_DEFAULT_MAX_EVENTS)),
                    ),
                    default=_DEFAULT_MAX_EVENTS,
                    minimum=256,
                    maximum=500000,
                ),
                max_span_history=_int_from_env(
                    os.environ.get(
                        _ENV_MAX_SPAN_HISTORY,
                        file_values.get(_ENV_MAX_SPAN_HISTORY, str(_DEFAULT_MAX_SPAN_HISTORY)),
                    ),
                    default=_DEFAULT_MAX_SPAN_HISTORY,
                    minimum=64,
                    maximum=5000,
                ),
                deep_exec_max_refreshes=_int_from_env(
                    os.environ.get(
                        _ENV_DEEP_EXEC_MAX_REFRESHES,
                        file_values.get(_ENV_DEEP_EXEC_MAX_REFRESHES, str(_DEFAULT_DEEP_EXEC_MAX_REFRESHES)),
                    ),
                    default=_DEFAULT_DEEP_EXEC_MAX_REFRESHES,
                    minimum=1,
                    maximum=32,
                ),
                unsafe_continuous_deep_exec=_bool_from_env(
                    os.environ.get(
                        _ENV_UNSAFE_CONTINUOUS_DEEP_EXEC,
                        file_values.get(_ENV_UNSAFE_CONTINUOUS_DEEP_EXEC, "0"),
                    ),
                    default=False,
                ),
            ),
        )

    @property
    def enabled(self) -> bool:
        """Return whether gesture-path tracing is active."""

        return bool(self.tracer.enabled)

    @property
    def run_id(self) -> str:
        """Return the current run identifier."""

        return self.tracer.run_id

    @property
    def run_dir(self) -> Path:
        """Return the on-disk run directory."""

        return self.base_dir / self.run_id

    def close(self) -> None:
        """Flush and close the underlying tracer."""

        self.tracer.close()

    @contextmanager
    def bind_refresh(
        self,
        *,
        observed_at: float,
        runtime_status_value: object,
        vision_mode: str | None,
        refresh_interval_s: float | None,
    ) -> Iterator[str | None]:
        """Bind one end-to-end gesture refresh to a trace context."""

        if not self.enabled:
            yield None
            return
        trace_id = uuid.uuid4().hex
        started_mono_ns = time.monotonic_ns()
        started_resources = _resource_snapshot()
        refresh_details: dict[str, object] = {
            "observed_at": round(float(observed_at), 6),
            "runtime_status": str(runtime_status_value or "").strip().lower() or None,
            "vision_mode": vision_mode,
            "refresh_interval_s": refresh_interval_s,
            "forensics_mode": self.config.mode,
            "run_id": self.run_id,
        }
        with bind_workflow_forensics(self.tracer, trace_id=trace_id):
            workflow_event(
                kind="turn_start",
                msg="gesture_refresh_started",
                details=refresh_details,
            )
            with workflow_span(
                name="gesture_refresh_turn",
                kind="turn",
                details=refresh_details,
            ):
                with self._deep_exec_scope():
                    try:
                        yield trace_id
                    except Exception as exc:
                        workflow_event(
                            kind="exception",
                            msg="gesture_refresh_failed",
                            level="ERROR",
                            details={
                                **refresh_details,
                                "error_type": type(exc).__name__,
                                "error": _redact_exception_text(exc, allow_raw_text=self.config.allow_raw_text),
                            },
                            kpi=_resource_delta(started_mono_ns=started_mono_ns, started=started_resources),
                        )
                        raise
                    else:
                        workflow_event(
                            kind="turn_end",
                            msg="gesture_refresh_completed",
                            details=refresh_details,
                            kpi=_resource_delta(started_mono_ns=started_mono_ns, started=started_resources),
                        )

    def _deep_exec_scope(self):
        if self.config.mode != "deep-exec":
            return nullcontext()
        if not self.tracer.can_accept_events():
            if not self._deep_exec_tracer_saturated_notice_emitted:
                self._deep_exec_tracer_saturated_notice_emitted = True
                workflow_event(
                    kind="warning",
                    msg="gesture_deep_exec_skipped_tracer_saturated",
                    details={"run_id": self.run_id},
                )
            return nullcontext()
        if self.config.sample_rate <= 0.0:
            self._record_sampled_out("disabled")
            return nullcontext()
        if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
            self._record_sampled_out("bernoulli")
            return nullcontext()
        if not self.config.unsafe_continuous_deep_exec:
            if self._deep_exec_refreshes_started >= self.config.deep_exec_max_refreshes:
                if not self._deep_exec_guardrail_notice_emitted:
                    self._deep_exec_guardrail_notice_emitted = True
                    workflow_event(
                        kind="warning",
                        msg="gesture_deep_exec_guardrail_skipped",
                        details={
                            "deep_exec_max_refreshes": self.config.deep_exec_max_refreshes,
                            "deep_exec_refreshes_started": self._deep_exec_refreshes_started,
                        },
                    )
                return nullcontext()
            self._deep_exec_refreshes_started += 1
        return _DeepExecMonitor(
            project_root=self.project_root,
            scope_prefixes=self.config.scope_prefixes,
            max_branch_events=self.config.max_branch_events,
        )

    def _record_sampled_out(self, reason: str) -> None:
        self._deep_exec_sampled_out_count += 1
        if _should_emit_log_counter(self._deep_exec_sampled_out_count):
            workflow_event(
                kind="metric",
                msg="gesture_deep_exec_sampled_out",
                details={
                    "sample_rate": self.config.sample_rate,
                    "reason": reason,
                    "count": self._deep_exec_sampled_out_count,
                },
            )


__all__ = ["GestureForensics", "GestureForensicsConfig"]