"""Scoped run-pack forensics for Twinr's live Pi gesture path.

This module binds the existing workflow forensics tracer to the dedicated
gesture acknowledgement lane and adds an opt-in deep-exec mode that records
line-to-line branch edges only for a small set of gesture modules.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import os
import random
import resource
import sys
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


def _resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_raw = float(getattr(usage, "ru_maxrss", 0.0) or 0.0)
    if sys.platform == "darwin":
        rss_mb = rss_raw / (1024.0 * 1024.0)
    else:
        rss_mb = rss_raw / 1024.0
    return {
        "cpu_user_s": float(getattr(usage, "ru_utime", 0.0) or 0.0),
        "cpu_system_s": float(getattr(usage, "ru_stime", 0.0) or 0.0),
        "rss_mb": round(max(0.0, rss_mb), 3),
    }


def _resource_delta(*, started_mono_ns: int, started: dict[str, float]) -> dict[str, object]:
    finished = _resource_snapshot()
    return {
        "duration_ms": round(max(0.0, (time.monotonic_ns() - started_mono_ns) / 1_000_000.0), 3),
        "cpu_user_ms": round(max(0.0, (finished["cpu_user_s"] - started["cpu_user_s"]) * 1000.0), 3),
        "cpu_system_ms": round(max(0.0, (finished["cpu_system_s"] - started["cpu_system_s"]) * 1000.0), 3),
        "rss_mb": finished["rss_mb"],
    }


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
    """Trace line-to-line edges only for the configured gesture modules."""

    def __init__(
        self,
        *,
        scope_prefixes: tuple[Path, ...],
        max_branch_events: int,
    ) -> None:
        self._scope_prefixes = tuple(str(path) for path in scope_prefixes)
        self._max_branch_events = max(1, int(max_branch_events))
        self._branch_events = 0
        self._dropped_branch_events = 0
        self._previous_trace: Any = None
        self._last_lines: dict[int, int] = {}
        self._reentrant = False
        self._active = False

    def __enter__(self) -> "_DeepExecMonitor":
        self._previous_trace = sys.gettrace()
        if self._previous_trace is not None:
            workflow_event(
                kind="warning",
                msg="gesture_deep_exec_skipped_existing_trace",
                details={"existing_trace_type": type(self._previous_trace).__name__},
            )
            return self
        sys.settrace(self._trace)
        self._active = True
        workflow_event(
            kind="metric",
            msg="gesture_deep_exec_started",
            details={
                "scope_prefixes": self._scope_prefixes,
                "max_branch_events": self._max_branch_events,
            },
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            sys.settrace(self._previous_trace)
        workflow_event(
            kind="metric",
            msg="gesture_deep_exec_stopped",
            details={
                "branch_events": self._branch_events,
                "dropped_branch_events": self._dropped_branch_events,
                "active": self._active,
            },
        )
        self._last_lines.clear()
        self._active = False

    def _trace(self, frame, event, arg):  # pragma: no cover - covered indirectly through file artifacts.
        if self._reentrant:
            return None
        code = getattr(frame, "f_code", None)
        if code is None:
            return None
        file_name = str(Path(code.co_filename).resolve(strict=False))
        if not self._matches_scope(file_name):
            return None
        if event == "call":
            self._last_lines[id(frame)] = int(getattr(frame, "f_lineno", 0) or 0)
            return self._trace
        if event == "line":
            current_line = int(getattr(frame, "f_lineno", 0) or 0)
            previous_line = self._last_lines.get(id(frame))
            self._last_lines[id(frame)] = current_line
            if previous_line is not None and previous_line != current_line:
                self._emit_branch(
                    file_name=file_name,
                    func_name=str(getattr(code, "co_name", "_unknown") or "_unknown"),
                    previous_line=previous_line,
                    current_line=current_line,
                )
            return self._trace
        if event == "return":
            self._last_lines.pop(id(frame), None)
            return self._trace
        return self._trace

    def _matches_scope(self, file_name: str) -> bool:
        return any(file_name.startswith(prefix) for prefix in self._scope_prefixes)

    def _emit_branch(
        self,
        *,
        file_name: str,
        func_name: str,
        previous_line: int,
        current_line: int,
    ) -> None:
        if self._branch_events >= self._max_branch_events:
            self._dropped_branch_events += 1
            if self._dropped_branch_events == 1:
                self._emit_with_guard(
                    kind="warning",
                    msg="gesture_deep_exec_branch_cap_reached",
                    details={"max_branch_events": self._max_branch_events},
                )
            return
        self._branch_events += 1
        self._emit_with_guard(
            kind="branch",
            msg="gesture_deep_exec_edge",
            details={
                "file": file_name,
                "func": func_name,
                "line_from": previous_line,
                "line_to": current_line,
                "edge": f"{previous_line}->{current_line}",
            },
            reason={
                "selected": {
                    "id": f"{previous_line}->{current_line}",
                    "justification": "Captured one scoped deep-exec line transition in the gesture path.",
                    "expected_outcome": "Preserve an exact line-to-line edge for replay-grade debugging.",
                },
                "options": [
                    {
                        "id": f"{previous_line}->{current_line}",
                        "summary": "Observed line transition",
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
                                "error": str(exc or type(exc).__name__),
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
            workflow_event(
                kind="metric",
                msg="gesture_deep_exec_sampled_out",
                details={"sample_rate": self.config.sample_rate},
            )
            return nullcontext()
        if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
            workflow_event(
                kind="metric",
                msg="gesture_deep_exec_sampled_out",
                details={"sample_rate": self.config.sample_rate},
            )
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
            scope_prefixes=self.config.scope_prefixes,
            max_branch_events=self.config.max_branch_events,
        )


__all__ = ["GestureForensics", "GestureForensicsConfig"]
