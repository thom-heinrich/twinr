# CHANGELOG: 2026-03-30
# BUG-1: Fix race on shutdown where close() could tear down the shared LongTermMemoryService while a probe thread was still using it.
# BUG-2: Fix watchdog blindness during hung probes by detecting probe time-budget overruns, persisting synthetic fail snapshots, and exposing stall state to systemd/watchdog consumers.
# BUG-3: Fix crash-on-observability-failure paths so snapshot/event-store write errors (for example on full/read-only SD cards) no longer kill the watchdog loop.
# BUG-4: Add a first-success startup timeout budget so cold-start remote probes do not false-fail before the first healthy sample lands.
# BUG-5: Persist monotonic freshness and PID attestation fields in watchdog snapshots so supervisor startup gating survives Pi wall-clock jumps without false restart blocks.
# SEC-1: Redact secret-bearing and content-bearing fields from probe payloads / remote_write_context before persisting to logs or ops artifacts.
# IMP-1: Add optional systemd sd_notify integration (READY/STATUS/WATCHDOG/STOPPING) for Raspberry Pi deployments managed by systemd.
# IMP-2: Add adaptive exponential backoff with jitter after failures/degradation to avoid synchronized probe storms against the remote backend.
# IMP-3: Emit sub-second/1s-class heartbeats while probes are inflight, cache host/process metadata, and harden all internal telemetry emission paths.

"""Continuously probe required remote ChonkyDB readiness.

This module runs a dedicated watchdog loop outside the GPIO and conversation
runtime. It reuses ``LongTermMemoryService.ensure_remote_ready()`` as the
canonical fail-closed readiness check, writes one rolling JSON snapshot under
Twinr's ops store, emits transition events into the local ops event log, and
prints one JSON line per sample for journald/systemd.

2026 upgrade notes:
- Shutdown is now cooperative and race-safe with active probe threads.
- Hung probes are explicitly surfaced as watchdog failures instead of leaving
  the artifact in a stale "last good" state forever.
- Diagnostics are recursively sanitized before they are persisted or logged.
- systemd ``sd_notify`` integration is supported without a hard dependency on
  ``python-systemd``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Protocol
import inspect
import json
import os
import random
import socket
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_read_diagnostics import extract_remote_write_context
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.events import TwinrOpsEventStore, compact_text
from twinr.ops.remote_memory_watchdog_state import (
    DEFAULT_HISTORY_LIMIT as _DEFAULT_HISTORY_LIMIT,
    DEFAULT_WATCHDOG_INTERVAL_S as _DEFAULT_WATCHDOG_INTERVAL_S,
    PERSISTED_RECENT_SAMPLE_LIMIT as _PERSISTED_RECENT_SAMPLE_LIMIT,
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
    RemoteMemoryWatchdogStore,
    SNAPSHOT_SCHEMA_VERSION as _SNAPSHOT_SCHEMA_VERSION,
    build_remote_memory_watchdog_bootstrap_snapshot,
    build_starting_sample as _build_starting_sample,
    coerce_history_limit as _coerce_history_limit,
    coerce_interval_s as _coerce_interval_s,
    compact_history_sample as _compact_history_sample,
    current_boot_id as _current_boot_id,
    monotonic_seconds_to_ns as _monotonic_seconds_to_ns,
    read_proc_create_time_s as _read_proc_create_time_s,
    read_proc_stat_start_ticks as _read_proc_stat_start_ticks,
    utc_now_iso as _utc_now_iso,
)


_DEFAULT_DEEP_PROBE_IDLE_FLOOR_S = 5.0
_DEFAULT_PROBE_TIMEOUT_FLOOR_S = 15.0
_DEFAULT_STARTUP_PROBE_TIMEOUT_FLOOR_S = 45.0
_DEFAULT_PROBE_TIMEOUT_CAP_S = 300.0
_DEFAULT_FAILURE_BACKOFF_CAP_S = 120.0
_DEFAULT_FAILURE_BACKOFF_BASE_MULTIPLIER = 2.0
_DEFAULT_FAILURE_BACKOFF_JITTER_RATIO = 0.35
_COLD_BOOTSTRAP_STORE_STEP_COUNT = 4
_COLD_WARM_CHECK_COUNT = 8
_RECENT_SUCCESS_TIMEOUT_SAMPLE_LIMIT = 8
_RECENT_SUCCESS_TIMEOUT_HEADROOM_RATIO = 0.5
_RECENT_SUCCESS_TIMEOUT_MIN_HEADROOM_S = 2.0
_RECENT_SUCCESS_TIMEOUT_MAX_HEADROOM_S = 10.0
_DEFAULT_HEARTBEAT_WHILE_PROBE_INFLIGHT_S = 1.0
_DEFAULT_CLOSE_JOIN_TIMEOUT_S = 2.0
_MAX_DETAIL_LENGTH = 240
_MAX_DIAGNOSTIC_DEPTH = 4
_MAX_DIAGNOSTIC_ITEMS = 24
_MAX_DIAGNOSTIC_STRING_LENGTH = 256

_REDACTED = "[REDACTED]"
_TRUNCATED = "[TRUNCATED]"

_SECRET_FIELD_FRAGMENTS = (
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "client_secret",
    "cookie",
    "credentials",
    "jwt",
    "passwd",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session",
    "set_cookie",
    "signature",
    "token",
)
_CONTENT_FIELD_FRAGMENTS = (
    "archive_query",
    "body",
    "content",
    "conversation",
    "document",
    "memory",
    "message",
    "messages",
    "payload_text",
    "prompt",
    "query_text",
    "request_body",
    "response_body",
    "text",
    "transcript",
)


def _copy_object_dict(value: object) -> dict[str, object] | None:
    """Copy one generic dict payload into a plain object dict."""

    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _coerce_positive_float(value: object, *, default: float, minimum: float | None = None) -> float:
    """Coerce one config value to a positive float."""

    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = default
    if minimum is not None:
        resolved = max(minimum, resolved)
    if resolved <= 0.0:
        resolved = default
        if minimum is not None:
            resolved = max(minimum, resolved)
    return resolved


def _default_startup_probe_timeout_s(config: TwinrConfig, *, probe_timeout_s: float) -> float:
    """Estimate the bounded first-probe budget for a fresh required-remote reader."""

    read_timeout_s = _coerce_positive_float(
        getattr(config, "long_term_memory_remote_read_timeout_s", 8.0),
        default=8.0,
        minimum=1.0,
    )
    chonky_timeout_s = _coerce_positive_float(
        getattr(config, "chonkydb_timeout_s", 20.0),
        default=20.0,
        minimum=read_timeout_s,
    )
    status_probe_timeout_s = max(
        _DEFAULT_PROBE_TIMEOUT_FLOOR_S,
        read_timeout_s * 2.0,
        chonky_timeout_s,
    )
    origin_resolution_timeout_s = max(
        chonky_timeout_s,
        read_timeout_s * 3.0,
    )
    estimated_timeout_s = (
        status_probe_timeout_s
        + (_COLD_BOOTSTRAP_STORE_STEP_COUNT * origin_resolution_timeout_s)
        + (_COLD_WARM_CHECK_COUNT * read_timeout_s)
    )
    return max(
        probe_timeout_s,
        _DEFAULT_STARTUP_PROBE_TIMEOUT_FLOOR_S,
        min(_DEFAULT_PROBE_TIMEOUT_CAP_S, estimated_timeout_s),
    )


def _looks_sensitive_key(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return False
    return any(fragment in normalized for fragment in _SECRET_FIELD_FRAGMENTS)


def _looks_content_key(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return False
    return any(fragment in normalized for fragment in _CONTENT_FIELD_FRAGMENTS)


def _normalize_jsonish_value(
    value: object,
    *,
    key_hint: str | None = None,
    depth: int = 0,
) -> object:
    """Recursively sanitize diagnostics for persistence/logging."""

    if key_hint and (_looks_sensitive_key(key_hint) or _looks_content_key(key_hint)):
        return _REDACTED
    if depth >= _MAX_DIAGNOSTIC_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        text = compact_text(value.strip(), limit=_MAX_DIAGNOSTIC_STRING_LENGTH)
        return text or None
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for index, (raw_key, raw_item) in enumerate(value.items()):
            if index >= _MAX_DIAGNOSTIC_ITEMS:
                sanitized["__truncated__"] = True
                break
            key = compact_text(str(raw_key).strip(), limit=96) or f"key_{index}"
            sanitized[key] = _normalize_jsonish_value(raw_item, key_hint=key, depth=depth + 1)
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        items = list(value)
        sanitized_items = [
            _normalize_jsonish_value(item, depth=depth + 1)
            for item in items[:_MAX_DIAGNOSTIC_ITEMS]
        ]
        if len(items) > _MAX_DIAGNOSTIC_ITEMS:
            sanitized_items.append(_TRUNCATED)
        return sanitized_items
    text = compact_text(repr(value), limit=_MAX_DIAGNOSTIC_STRING_LENGTH)
    return text or f"<{type(value).__name__}>"


def _normalize_probe_payload(value: object) -> dict[str, object] | None:
    """Keep only sanitized dict-like probe payloads for watchdog artifacts."""

    if not isinstance(value, dict):
        return None
    normalized = _normalize_jsonish_value(value, depth=0)
    return normalized if isinstance(normalized, dict) else None


def _normalize_remote_write_context(value: object) -> dict[str, object] | None:
    """Sanitize remote write context before it reaches logs or persistent state."""

    if not isinstance(value, dict):
        return None
    normalized = _normalize_jsonish_value(value, depth=0)
    return normalized if isinstance(normalized, dict) else None


class _RemoteMemoryService(Protocol):
    """Protocol used by the watchdog so tests can provide fakes."""

    def ensure_remote_ready(self):
        """Raise if required remote memory is unavailable."""

    def probe_remote_ready(self):
        """Return structured remote-memory readiness evidence when available."""

    def remote_required(self) -> bool:
        """Report whether this runtime must fail closed on remote loss."""

    def remote_status(self):  # pragma: no cover - protocol surface only
        """Return the effective remote status object."""

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Shut down any owned background workers."""


class _SystemdNotifier:
    """Minimal pure-Python sd_notify helper with optional watchdog support."""

    def __init__(self) -> None:
        self._notify_socket = os.environ.get("NOTIFY_SOCKET") or ""
        self._watchdog_interval_s = self._detect_watchdog_interval_s()

    @property
    def enabled(self) -> bool:
        return bool(self._notify_socket)

    @property
    def watchdog_interval_s(self) -> float | None:
        return self._watchdog_interval_s

    def ready(self, *, status: str | None = None) -> bool:
        payload = ["READY=1"]
        if status:
            payload.append(f"STATUS={status}")
        return self._send("\n".join(payload))

    def stopping(self, *, status: str | None = None) -> bool:
        payload = ["STOPPING=1"]
        if status:
            payload.append(f"STATUS={status}")
        return self._send("\n".join(payload))

    def status(self, status: str) -> bool:
        if not status:
            return False
        return self._send(f"STATUS={status}")

    def watchdog_ping(self, *, status: str | None = None) -> bool:
        payload = ["WATCHDOG=1"]
        if status:
            payload.append(f"STATUS={status}")
        return self._send("\n".join(payload))

    def _detect_watchdog_interval_s(self) -> float | None:
        raw_value = os.environ.get("WATCHDOG_USEC")
        if not raw_value:
            return None
        try:
            watchdog_usec = int(raw_value)
        except (TypeError, ValueError):
            return None
        if watchdog_usec <= 0:
            return None
        return max(0.5, watchdog_usec / 2_000_000.0)

    def _send(self, state: str) -> bool:
        if not self.enabled or not state:
            return False
        address = self._notify_socket
        if not address:
            return False
        if address[0] == "@":
            address = f"\0{address[1:]}"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.connect(address)
                sock.sendall(state.encode("utf-8", errors="replace"))
            return True
        except OSError:
            return False


class RemoteMemoryWatchdog:
    """Continuously verify remote-primary long-term memory readiness."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        service_factory=None,
        store: RemoteMemoryWatchdogStore | None = None,
        event_store: TwinrOpsEventStore | None = None,
        emit=None,
        monotonic=None,
        sleep=None,
        rand=None,
    ) -> None:
        self.config = config
        self.interval_s = _coerce_interval_s(
            getattr(config, "long_term_memory_remote_watchdog_interval_s", _DEFAULT_WATCHDOG_INTERVAL_S),
            default=_DEFAULT_WATCHDOG_INTERVAL_S,
        )
        self.history_limit = _coerce_history_limit(
            getattr(config, "long_term_memory_remote_watchdog_history_limit", _DEFAULT_HISTORY_LIMIT),
            default=_DEFAULT_HISTORY_LIMIT,
        )
        self.probe_timeout_s = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_probe_timeout_s",
                max(_DEFAULT_PROBE_TIMEOUT_FLOOR_S, min(_DEFAULT_PROBE_TIMEOUT_CAP_S, self.interval_s * 4.0)),
            ),
            default=max(_DEFAULT_PROBE_TIMEOUT_FLOOR_S, min(_DEFAULT_PROBE_TIMEOUT_CAP_S, self.interval_s * 4.0)),
            minimum=1.0,
        )
        self.startup_probe_timeout_s = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_startup_probe_timeout_s",
                _default_startup_probe_timeout_s(config, probe_timeout_s=self.probe_timeout_s),
            ),
            default=_default_startup_probe_timeout_s(config, probe_timeout_s=self.probe_timeout_s),
            minimum=self.probe_timeout_s,
        )
        self.failure_backoff_cap_s = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_failure_backoff_cap_s",
                _DEFAULT_FAILURE_BACKOFF_CAP_S,
            ),
            default=_DEFAULT_FAILURE_BACKOFF_CAP_S,
            minimum=self.interval_s,
        )
        self.failure_backoff_base_multiplier = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_failure_backoff_base_multiplier",
                _DEFAULT_FAILURE_BACKOFF_BASE_MULTIPLIER,
            ),
            default=_DEFAULT_FAILURE_BACKOFF_BASE_MULTIPLIER,
            minimum=1.0,
        )
        self.failure_backoff_jitter_ratio = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_failure_backoff_jitter_ratio",
                _DEFAULT_FAILURE_BACKOFF_JITTER_RATIO,
            ),
            default=_DEFAULT_FAILURE_BACKOFF_JITTER_RATIO,
            minimum=0.0,
        )
        self.heartbeat_while_probe_inflight_s = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_heartbeat_while_probe_inflight_s",
                _DEFAULT_HEARTBEAT_WHILE_PROBE_INFLIGHT_S,
            ),
            default=_DEFAULT_HEARTBEAT_WHILE_PROBE_INFLIGHT_S,
            minimum=0.1,
        )
        self.close_join_timeout_s = _coerce_positive_float(
            getattr(
                config,
                "long_term_memory_remote_watchdog_close_join_timeout_s",
                _DEFAULT_CLOSE_JOIN_TIMEOUT_S,
            ),
            default=_DEFAULT_CLOSE_JOIN_TIMEOUT_S,
            minimum=0.1,
        )

        self.service_factory = service_factory or (lambda: LongTermMemoryService.from_config(config))
        self.store = store or RemoteMemoryWatchdogStore.from_config(config)
        self.event_store = event_store or TwinrOpsEventStore.from_config(config)
        self.emit = emit or self._default_emit
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._rand = rand or random.Random()

        self._service: _RemoteMemoryService | None = None
        self._probe_lock = Lock()
        self._store_lock = Lock()
        self._run_lock = Lock()
        self._stop_event = Event()
        self._systemd = _SystemdNotifier()

        self._started_at = _utc_now_iso()
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._boot_id = _current_boot_id()
        self._pid_starttime_ticks = _read_proc_stat_start_ticks(self._pid)
        self._pid_create_time_s = _read_proc_create_time_s(self._pid)

        self._sample_count = 0
        self._failure_count = 0
        self._consecutive_ok = 0
        self._consecutive_fail = 0
        self._last_ok_at: str | None = None
        self._last_failure_at: str | None = None
        self._last_status: str | None = None
        self._last_detail: str | None = None
        self._recent_samples: deque[RemoteMemoryWatchdogSample] = deque(maxlen=self.history_limit)

        self._probe_thread: Thread | None = None
        self._probe_started_at: str | None = None
        self._probe_started_monotonic: float | None = None
        self._probe_timeout_reported = False
        self._next_probe_not_before_monotonic = self._monotonic()
        self._last_systemd_watchdog_ping_monotonic: float | None = None
        self._ready_notified = False
        self._has_success_since_start = False

        self._restore_recent_history_from_store()

    def _persisted_recent_samples(self) -> tuple[RemoteMemoryWatchdogSample, ...]:
        """Return a bounded, compact recent history for persisted snapshots."""

        if not self._recent_samples:
            return ()
        window = max(1, min(self.history_limit, _PERSISTED_RECENT_SAMPLE_LIMIT))
        return tuple(_compact_history_sample(sample) for sample in tuple(self._recent_samples)[-window:])

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteMemoryWatchdog":
        """Build a watchdog from Twinr config."""

        return cls(config=config)

    @property
    def artifact_path(self) -> Path:
        """Return the rolling JSON artifact path."""

        return self.store.path

    def close(self) -> None:
        """Shut down the owned long-term memory service best-effort."""

        self._stop_event.set()
        self._notify_systemd_stopping()
        probe_thread = self._probe_thread
        if probe_thread is not None and probe_thread is not current_thread():
            try:
                probe_thread.join(timeout=self.close_join_timeout_s)
            except Exception as exc:
                self._emit_internal_error(
                    "remote_memory_watchdog_join_error",
                    exc,
                    extra={"phase": "close"},
                )
        if probe_thread is not None and probe_thread.is_alive():
            # Do not shut down the shared service while the probe thread is still
            # using it. The process may be about to exit (daemon thread), and
            # racing a shutdown here can corrupt diagnostics or create false
            # negatives.
            self._safe_emit_payload(
                {
                    "event": "remote_memory_watchdog_close_deferred",
                    "artifact_path": str(self.artifact_path),
                    "probe_inflight": True,
                    "probe_started_at": self._probe_started_at,
                    "probe_timeout_s": round(self._effective_probe_timeout_s(), 1),
                }
            )
            return
        self._drop_service_instance()

    def probe_once(self) -> RemoteMemoryWatchdogSnapshot:
        """Run one remote-memory readiness probe and persist the rolling state."""

        with self._probe_lock:
            started = self._monotonic()
            status = "fail"
            ready = False
            mode = "unknown"
            required = False
            detail: str | None = None
            probe_payload: dict[str, object] | None = None

            try:
                service = self._service_instance()
                required = bool(service.remote_required())
                remote_status = service.remote_status()
                mode = str(getattr(remote_status, "mode", "unknown") or "unknown")
                status_detail = self._normalize_detail(getattr(remote_status, "detail", None))
                if not required:
                    status = "disabled"
                    ready = False
                    detail = status_detail or "Remote-primary long-term memory is not required."
                elif not bool(getattr(remote_status, "ready", False)):
                    # Keep the same service instance alive so the remote-state
                    # circuit breaker can cool down after transient backend
                    # outages instead of being reset on every watchdog tick.
                    status = "fail"
                    ready = False
                    detail = status_detail or "Required remote long-term memory is unavailable."
                else:
                    probe_remote_ready = getattr(service, "probe_remote_ready", None)
                    if callable(probe_remote_ready):
                        bootstrap_probe = self._should_run_bootstrap_probe()
                        probe_result = self._call_probe_remote_ready(
                            probe_remote_ready=probe_remote_ready,
                            bootstrap=bootstrap_probe,
                            include_archive=True,
                        )
                        probe_payload = self._probe_result_to_payload(probe_result)
                        raw_ready = bool(getattr(probe_result, "ready", False))
                        archive_safe = self._probe_result_archive_safe(
                            probe_result=probe_result,
                            include_archive=True,
                        )
                        ready = bool(raw_ready and archive_safe)
                        if raw_ready and not archive_safe:
                            status = "degraded"
                            detail = (
                                self._normalize_detail(getattr(probe_result, "detail", None))
                                or "Remote readiness probe stayed current-only and is not archive-safe."
                            )
                        else:
                            status = "ok" if ready else "fail"
                            detail = self._normalize_detail(getattr(probe_result, "detail", None)) or status_detail
                        mode = str(
                            getattr(
                                getattr(probe_result, "remote_status", None),
                                "mode",
                                mode,
                            )
                            or mode
                        )
                    else:
                        service.ensure_remote_ready()
                        status = "ok"
                        ready = True
                        detail = status_detail
            except LongTermRemoteUnavailableError as exc:
                status = "fail"
                ready = False
                detail = self._normalize_detail(str(exc))
                probe_payload = self._merge_remote_write_context(
                    probe_payload,
                    extract_remote_write_context(exc),
                )
            except Exception as exc:
                status = "fail"
                ready = False
                detail = self._normalize_detail(f"{type(exc).__name__}: {exc}")
                self._drop_service_instance()

            latency_ms = round(max(0.0, (self._monotonic() - started) * 1000.0), 1)
            captured_at = _utc_now_iso()
            sample = self._build_sample(
                captured_at=captured_at,
                status=status,
                ready=ready,
                mode=mode,
                required=required,
                latency_ms=latency_ms,
                detail=detail,
                probe=probe_payload,
            )
            snapshot = self._build_snapshot(sample)
            self._safe_store_save(snapshot)
            self._emit_transition_event(sample)
            self._safe_emit_payload(
                {
                    "event": "remote_memory_watchdog_sample",
                    **sample.to_dict(),
                    "artifact_path": str(self.artifact_path),
                }
            )
            if sample.status == "ok":
                self._has_success_since_start = True
            self._probe_timeout_reported = False
            self._update_next_probe_deadline_after_sample(sample)
            return snapshot

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the watchdog loop until stopped or the optional duration expires."""

        with self._run_lock:
            resolved_duration = None
            if duration_s is not None:
                resolved_duration = max(0.0, float(duration_s))
            deadline = None if resolved_duration is None else self._monotonic() + resolved_duration

            self._safe_event_append(
                event="remote_memory_watchdog_started",
                message="Remote memory watchdog started.",
                data={
                    "artifact_path": str(self.artifact_path),
                    "interval_s": self.interval_s,
                    "history_limit": self.history_limit,
                    "probe_timeout_s": self.probe_timeout_s,
                },
            )
            self._notify_systemd_ready()

            if resolved_duration == 0.0:
                try:
                    self.probe_once()
                    return 0
                except KeyboardInterrupt:
                    return 0
                finally:
                    self._safe_event_append(
                        event="remote_memory_watchdog_stopped",
                        message="Remote memory watchdog stopped.",
                        data={
                            "artifact_path": str(self.artifact_path),
                            "sample_count": self._sample_count,
                            "failure_count": self._failure_count,
                        },
                    )
                    self.close()

            try:
                while not self._stop_event.is_set():
                    now_monotonic = self._monotonic()
                    self._reap_probe_thread(now_monotonic=now_monotonic)
                    if self._probe_thread is None and now_monotonic >= self._next_probe_not_before_monotonic:
                        self._start_probe_thread(now_monotonic=now_monotonic)
                    probe_inflight = bool(self._probe_thread is not None and self._probe_thread.is_alive())
                    self._emit_heartbeat(
                        probe_started_at=self._probe_started_at,
                        probe_started_monotonic=self._probe_started_monotonic,
                        probe_inflight=probe_inflight,
                    )
                    self._mark_stalled_probe_if_needed(now_monotonic=now_monotonic)
                    self._maybe_notify_systemd_watchdog(now_monotonic=now_monotonic)
                    if deadline is not None and now_monotonic >= deadline:
                        return 0
                    self._sleep_until_next_tick(
                        now_monotonic=now_monotonic,
                        deadline=deadline,
                        probe_inflight=probe_inflight,
                    )
                return 0
            except KeyboardInterrupt:
                return 0
            finally:
                self._safe_event_append(
                    event="remote_memory_watchdog_stopped",
                    message="Remote memory watchdog stopped.",
                    data={
                        "artifact_path": str(self.artifact_path),
                        "sample_count": self._sample_count,
                        "failure_count": self._failure_count,
                    },
                )
                self.close()

    @staticmethod
    def _default_emit(line: str) -> None:
        """Print one watchdog line for journald/systemd capture."""

        print(line, flush=True)

    def _deep_probe_idle_gap_s(self, *, last_sample: RemoteMemoryWatchdogSample | None) -> float:
        """Return the minimum quiet window between completed deep probes."""

        raw_keepalive = getattr(
            self.config,
            "long_term_memory_remote_keepalive_interval_s",
            _DEFAULT_DEEP_PROBE_IDLE_FLOOR_S,
        )
        keepalive_s = _coerce_positive_float(
            raw_keepalive,
            default=_DEFAULT_DEEP_PROBE_IDLE_FLOOR_S,
            minimum=_DEFAULT_DEEP_PROBE_IDLE_FLOOR_S,
        )
        latency_s = 0.0
        if last_sample is not None:
            latency_s = max(0.0, float(last_sample.latency_ms or 0.0) / 1000.0)
        return max(self.interval_s, keepalive_s, latency_s)

    def _recent_success_latency_s(self) -> float | None:
        """Return the slowest recent successful deep-probe latency in seconds."""

        latencies: list[float] = []
        for sample in reversed(self._recent_samples):
            if sample.status != "ok":
                continue
            latency_s = max(0.0, float(sample.latency_ms or 0.0) / 1000.0)
            if latency_s <= 0.0:
                continue
            latencies.append(latency_s)
            if len(latencies) >= _RECENT_SUCCESS_TIMEOUT_SAMPLE_LIMIT:
                break
        if not latencies:
            return None
        return max(latencies)

    def _effective_probe_timeout_s(self) -> float:
        """Return the watchdog timeout budget for the current in-flight probe.

        A hard 15s cutoff caused false fail-closed flips on the Pi both during
        cold boot and during slow-but-healthy archive-inclusive checks. Keep a
        larger startup-only budget until the first successful probe lands, then
        pad the steady-state timeout from observed successful probe latencies
        while still keeping the timeout bounded.
        """

        timeout_s = self.probe_timeout_s
        if not self._has_success_since_start:
            timeout_s = max(timeout_s, self.startup_probe_timeout_s)
        recent_success_latency_s = self._recent_success_latency_s()
        if recent_success_latency_s is None:
            return timeout_s
        headroom_s = max(
            _RECENT_SUCCESS_TIMEOUT_MIN_HEADROOM_S,
            min(
                _RECENT_SUCCESS_TIMEOUT_MAX_HEADROOM_S,
                recent_success_latency_s * _RECENT_SUCCESS_TIMEOUT_HEADROOM_RATIO,
            ),
        )
        return min(
            max(_DEFAULT_PROBE_TIMEOUT_CAP_S, self.startup_probe_timeout_s),
            max(timeout_s, recent_success_latency_s + headroom_s),
        )

    def _restore_recent_history_from_store(self) -> None:
        """Restore bounded counters/history from the persisted rolling snapshot."""

        try:
            snapshot = self.store.load()
        except Exception:
            return
        if snapshot is None:
            return
        restored_samples = tuple(snapshot.recent_samples[-self.history_limit :])
        if not restored_samples:
            current_sample = snapshot.current
            if current_sample.seq > 0 or current_sample.status not in {"starting", "unknown"}:
                restored_samples = (_compact_history_sample(current_sample),)
        self._recent_samples.extend(restored_samples)
        self._sample_count = max(len(restored_samples), int(snapshot.sample_count or 0))
        self._failure_count = max(0, int(snapshot.failure_count or 0))
        self._last_ok_at = snapshot.last_ok_at
        self._last_failure_at = snapshot.last_failure_at
        if not restored_samples:
            return
        last_sample = restored_samples[-1]
        self._consecutive_ok = max(0, int(last_sample.consecutive_ok))
        self._consecutive_fail = max(0, int(last_sample.consecutive_fail))
        self._last_status = last_sample.status
        self._last_detail = last_sample.detail

    def _failure_backoff_s(self, *, last_sample: RemoteMemoryWatchdogSample | None) -> float:
        """Return a jittered capped backoff after failed/degraded probes."""

        if last_sample is None:
            return self.interval_s
        if last_sample.status == "ok":
            return self._deep_probe_idle_gap_s(last_sample=last_sample)
        failure_streak = max(1, int(last_sample.consecutive_fail or 1))
        base_delay = self.interval_s * (self.failure_backoff_base_multiplier ** max(0, failure_streak - 1))
        capped_delay = min(self.failure_backoff_cap_s, max(self.interval_s, base_delay))
        jitter = capped_delay * min(1.0, max(0.0, self.failure_backoff_jitter_ratio))
        low = max(self.interval_s, capped_delay - jitter)
        high = capped_delay + jitter
        if high <= low:
            return capped_delay
        return float(self._rand.uniform(low, high))

    def _update_next_probe_deadline_after_sample(self, sample: RemoteMemoryWatchdogSample) -> None:
        """Schedule the next deep probe using health-aware backoff."""

        quiet_window_s = (
            self._deep_probe_idle_gap_s(last_sample=sample)
            if sample.status == "ok"
            else self._failure_backoff_s(last_sample=sample)
        )
        self._next_probe_not_before_monotonic = max(
            self._next_probe_not_before_monotonic,
            self._monotonic() + quiet_window_s,
        )

    def _service_instance(self) -> _RemoteMemoryService:
        service = self._service
        if service is None:
            service = self.service_factory()
            self._service = service
        return service

    def _drop_service_instance(self) -> None:
        """Discard one potentially poisoned service instance after a probe failure."""

        service = self._service
        self._service = None
        if service is None:
            return
        try:
            service.shutdown(timeout_s=2.0)
        except Exception:
            return

    def _should_run_bootstrap_probe(self) -> bool:
        """Return whether the next readiness probe must perform full bootstrap."""

        last_sample = self._recent_samples[-1] if self._recent_samples else None
        return last_sample is None or last_sample.status != "ok"

    @staticmethod
    def _call_probe_remote_ready(*, probe_remote_ready, bootstrap: bool, include_archive: bool):
        """Call probe helpers compatibly across older and newer service surfaces."""

        try:
            signature = inspect.signature(probe_remote_ready)
        except (TypeError, ValueError):
            return probe_remote_ready()
        parameters = signature.parameters.values()
        supports_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        kwargs: dict[str, object] = {}
        if supports_kwargs or "bootstrap" in signature.parameters:
            kwargs["bootstrap"] = bootstrap
        if supports_kwargs or "include_archive" in signature.parameters:
            kwargs["include_archive"] = include_archive
        return probe_remote_ready(**kwargs)

    @staticmethod
    def _probe_result_archive_safe(*, probe_result, include_archive: bool) -> bool:
        """Return whether one structured readiness result proves archive safety."""

        if probe_result is None:
            return False
        warm_result = getattr(probe_result, "warm_result", None)
        archive_safe = getattr(warm_result, "archive_safe", None)
        if archive_safe is not None:
            return bool(archive_safe)
        probe_payload = None
        try:
            probe_payload = getattr(probe_result, "to_dict", lambda: None)()
        except Exception:
            probe_payload = None
        if isinstance(probe_payload, dict):
            warm_payload = probe_payload.get("warm_result")
            if isinstance(warm_payload, dict) and "archive_safe" in warm_payload:
                return bool(warm_payload.get("archive_safe"))
        return bool(getattr(probe_result, "ready", False) and include_archive)

    @staticmethod
    def _normalize_detail(value: object) -> str | None:
        text = compact_text(str(value or "").strip(), limit=_MAX_DETAIL_LENGTH)
        return text or None

    def _probe_result_to_payload(self, probe_result) -> dict[str, object] | None:
        """Safely serialize a structured probe result to one sanitized dict payload."""

        if probe_result is None:
            return None
        try:
            raw_payload = getattr(probe_result, "to_dict", lambda: None)()
        except Exception as exc:
            self._emit_internal_error(
                "remote_memory_watchdog_probe_payload_error",
                exc,
                extra={"phase": "probe_result_to_payload"},
            )
            return None
        return _normalize_probe_payload(raw_payload)

    @staticmethod
    def _merge_remote_write_context(
        probe: dict[str, object] | None,
        remote_write_context: dict[str, object] | None,
    ) -> dict[str, object] | None:
        sanitized_context = _normalize_remote_write_context(remote_write_context)
        if not sanitized_context:
            return probe
        merged = dict(probe or {})
        merged["remote_write_context"] = sanitized_context
        return merged

    def _start_probe_thread(self, *, now_monotonic: float) -> None:
        """Start one deep probe thread if none is active."""

        if self._probe_thread is not None and self._probe_thread.is_alive():
            return
        self._probe_started_at = _utc_now_iso()
        self._probe_started_monotonic = now_monotonic
        self._probe_timeout_reported = False
        self._probe_thread = Thread(
            target=self._probe_worker_main,
            name="twinr-remote-memory-watchdog-probe",
            daemon=True,
        )
        self._probe_thread.start()

    def _reap_probe_thread(self, *, now_monotonic: float) -> None:
        """Join and clear one completed background probe thread."""

        probe_thread = self._probe_thread
        if probe_thread is None or probe_thread.is_alive():
            return
        try:
            probe_thread.join(timeout=0.0)
        except Exception as exc:
            self._emit_internal_error(
                "remote_memory_watchdog_join_error",
                exc,
                extra={"phase": "reap"},
            )
        self._probe_thread = None
        self._probe_started_at = None
        self._probe_started_monotonic = None
        self._probe_timeout_reported = False
        if self._next_probe_not_before_monotonic < now_monotonic:
            self._next_probe_not_before_monotonic = now_monotonic

    def _probe_worker_main(self) -> None:
        """Run one deep remote probe in the background."""

        try:
            self.probe_once()
        except Exception as exc:
            self._emit_internal_error(
                "remote_memory_watchdog_worker_error",
                exc,
                extra={"artifact_path": str(self.artifact_path)},
            )

    def _mark_stalled_probe_if_needed(self, *, now_monotonic: float) -> None:
        """Persist a synthetic fail sample if the active probe exceeds its budget."""

        if self._probe_timeout_reported:
            return
        if self._probe_thread is None or not self._probe_thread.is_alive():
            return
        if self._probe_started_monotonic is None:
            return
        probe_age_s = max(0.0, now_monotonic - self._probe_started_monotonic)
        probe_timeout_s = self._effective_probe_timeout_s()
        if probe_age_s < probe_timeout_s:
            return

        captured_at = _utc_now_iso()
        last_sample = self._recent_samples[-1] if self._recent_samples else None
        mode = last_sample.mode if last_sample is not None else "unknown"
        required = last_sample.required if last_sample is not None else True
        detail = self._normalize_detail(
            f"Remote readiness probe exceeded {round(probe_timeout_s, 1)}s and is assumed stuck."
        )
        sample = self._build_sample(
            captured_at=captured_at,
            status="fail",
            ready=False,
            mode=mode,
            required=required,
            latency_ms=round(probe_age_s * 1000.0, 1),
            detail=detail,
            probe={
                "watchdog_timeout": {
                    "probe_started_at": self._probe_started_at,
                    "probe_age_s": round(probe_age_s, 1),
                    "probe_timeout_s": round(probe_timeout_s, 1),
                }
            },
        )
        snapshot = self._build_snapshot(sample)
        self._safe_store_save(snapshot)
        self._emit_transition_event(sample)
        self._safe_emit_payload(
            {
                "event": "remote_memory_watchdog_probe_timeout",
                **sample.to_dict(),
                "artifact_path": str(self.artifact_path),
                "probe_started_at": self._probe_started_at,
                "probe_age_s": round(probe_age_s, 1),
                "probe_timeout_s": round(probe_timeout_s, 1),
            }
        )
        self._probe_timeout_reported = True
        self._update_next_probe_deadline_after_sample(sample)

    def _emit_heartbeat(
        self,
        *,
        probe_started_at: str | None,
        probe_started_monotonic: float | None,
        probe_inflight: bool,
    ) -> None:
        """Emit one heartbeat line while the deep probe runs."""

        captured_at = _utc_now_iso()
        now_monotonic = self._monotonic()
        heartbeat_monotonic_ns = self._current_monotonic_ns(now_monotonic=now_monotonic)
        probe_age_s: float | None = None
        probe_timeout_s: float | None = None
        if probe_inflight and probe_started_monotonic is not None:
            probe_age_s = round(max(0.0, now_monotonic - probe_started_monotonic), 1)
            probe_timeout_s = round(self._effective_probe_timeout_s(), 1)
        self._persist_heartbeat_snapshot(
            heartbeat_at=captured_at,
            heartbeat_monotonic_ns=heartbeat_monotonic_ns,
            probe_inflight=probe_inflight,
            probe_started_at=probe_started_at,
            probe_started_monotonic_ns=_monotonic_seconds_to_ns(probe_started_monotonic),
            probe_age_s=probe_age_s,
        )
        last_sample = self._recent_samples[-1] if self._recent_samples else None
        payload: dict[str, object] = {
            "event": "remote_memory_watchdog_heartbeat",
            "artifact_path": str(self.artifact_path),
            "captured_at": captured_at,
            "probe_inflight": probe_inflight,
            "probe_started_at": probe_started_at,
            "sample_count": self._sample_count,
            "failure_count": self._failure_count,
        }
        if probe_age_s is not None:
            payload["probe_age_s"] = probe_age_s
            payload["probe_timeout_s"] = probe_timeout_s
            payload["probe_timed_out"] = bool(self._probe_timeout_reported)
        if last_sample is None:
            payload["status"] = "starting"
            payload["ready"] = False
        else:
            payload.update(
                {
                    "status": last_sample.status,
                    "ready": last_sample.ready,
                    "mode": last_sample.mode,
                    "required": last_sample.required,
                    "last_sample_at": last_sample.captured_at,
                    "last_latency_ms": last_sample.latency_ms,
                    "detail": last_sample.detail,
                }
            )
        self._safe_emit_payload(payload)

    def _build_sample(
        self,
        *,
        captured_at: str,
        status: str,
        ready: bool,
        mode: str,
        required: bool,
        latency_ms: float,
        detail: str | None,
        probe: dict[str, object] | None = None,
    ) -> RemoteMemoryWatchdogSample:
        self._sample_count += 1
        if status == "ok":
            self._consecutive_ok += 1
            self._consecutive_fail = 0
            self._last_ok_at = captured_at
        elif status == "fail":
            self._consecutive_fail += 1
            self._consecutive_ok = 0
            self._failure_count += 1
            self._last_failure_at = captured_at
        else:
            self._consecutive_ok = 0
            self._consecutive_fail = 0
        sample = RemoteMemoryWatchdogSample(
            seq=self._sample_count,
            captured_at=captured_at,
            status=status,
            ready=ready,
            mode=mode,
            required=required,
            latency_ms=latency_ms,
            consecutive_ok=self._consecutive_ok,
            consecutive_fail=self._consecutive_fail,
            captured_monotonic_ns=self._current_monotonic_ns(),
            detail=detail,
            probe=probe,
        )
        self._recent_samples.append(sample)
        return sample

    def _build_snapshot(self, current: RemoteMemoryWatchdogSample) -> RemoteMemoryWatchdogSnapshot:
        return RemoteMemoryWatchdogSnapshot(
            schema_version=_SNAPSHOT_SCHEMA_VERSION,
            started_at=self._started_at,
            updated_at=current.captured_at,
            hostname=self._hostname,
            pid=self._pid,
            interval_s=self.interval_s,
            history_limit=self.history_limit,
            sample_count=self._sample_count,
            failure_count=self._failure_count,
            last_ok_at=self._last_ok_at,
            last_failure_at=self._last_failure_at,
            artifact_path=str(self.artifact_path),
            current=current,
            recent_samples=self._persisted_recent_samples(),
            updated_monotonic_ns=current.captured_monotonic_ns,
            heartbeat_at=current.captured_at,
            heartbeat_monotonic_ns=current.captured_monotonic_ns,
            boot_id=self._boot_id,
            pid_starttime_ticks=self._pid_starttime_ticks,
            pid_create_time_s=self._pid_create_time_s,
            probe_inflight=False,
            probe_started_at=None,
            probe_started_monotonic_ns=None,
            probe_age_s=None,
        )

    def _persist_heartbeat_snapshot(
        self,
        *,
        heartbeat_at: str,
        heartbeat_monotonic_ns: int | None,
        probe_inflight: bool,
        probe_started_at: str | None,
        probe_started_monotonic_ns: int | None,
        probe_age_s: float | None,
    ) -> None:
        """Persist liveness metadata even while a deep probe is still running."""

        last_sample = self._recent_samples[-1] if self._recent_samples else None
        current = last_sample or self._starting_sample(
            captured_at=heartbeat_at,
            captured_monotonic_ns=heartbeat_monotonic_ns,
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=_SNAPSHOT_SCHEMA_VERSION,
            started_at=self._started_at,
            updated_at=current.captured_at,
            hostname=self._hostname,
            pid=self._pid,
            interval_s=self.interval_s,
            history_limit=self.history_limit,
            sample_count=self._sample_count,
            failure_count=self._failure_count,
            last_ok_at=self._last_ok_at,
            last_failure_at=self._last_failure_at,
            artifact_path=str(self.artifact_path),
            current=current,
            recent_samples=self._persisted_recent_samples(),
            updated_monotonic_ns=current.captured_monotonic_ns,
            heartbeat_at=heartbeat_at,
            heartbeat_monotonic_ns=heartbeat_monotonic_ns,
            boot_id=self._boot_id,
            pid_starttime_ticks=self._pid_starttime_ticks,
            pid_create_time_s=self._pid_create_time_s,
            probe_inflight=probe_inflight,
            probe_started_at=probe_started_at,
            probe_started_monotonic_ns=probe_started_monotonic_ns,
            probe_age_s=probe_age_s,
        )
        self._safe_store_save(snapshot)

    def _starting_sample(
        self,
        *,
        captured_at: str,
        captured_monotonic_ns: int | None = None,
    ) -> RemoteMemoryWatchdogSample:
        """Synthesize the startup state before the first real probe sample exists."""

        return _build_starting_sample(
            self.config,
            captured_at=captured_at,
            captured_monotonic_ns=captured_monotonic_ns,
        )

    def _current_monotonic_ns(self, *, now_monotonic: float | None = None) -> int | None:
        """Return the injected monotonic clock as integer nanoseconds."""

        resolved_now = self._monotonic() if now_monotonic is None else now_monotonic
        return _monotonic_seconds_to_ns(resolved_now)

    def _emit_transition_event(self, sample: RemoteMemoryWatchdogSample) -> None:
        previous_status = self._last_status
        previous_detail = self._last_detail
        self._last_status = sample.status
        self._last_detail = sample.detail
        remote_write_context = (
            _copy_object_dict(sample.probe.get("remote_write_context"))
            if isinstance(sample.probe, dict)
            else None
        )
        sanitized_remote_write_context = _normalize_remote_write_context(remote_write_context)
        if previous_status == sample.status and previous_detail == sample.detail:
            self._maybe_notify_systemd_status(sample=sample)
            return
        if previous_status is None:
            message = f"Remote memory watchdog observed initial state {sample.status}."
        else:
            message = f"Remote memory watchdog changed from {previous_status} to {sample.status}."
        level = "error" if sample.status == "fail" else "info"
        self._safe_event_append(
            event="remote_memory_watchdog_status_changed",
            message=message,
            level=level,
            data={
                "artifact_path": str(self.artifact_path),
                "captured_at": sample.captured_at,
                "previous_status": previous_status,
                "status": sample.status,
                "ready": sample.ready,
                "mode": sample.mode,
                "required": sample.required,
                "latency_ms": sample.latency_ms,
                "detail": sample.detail,
                "consecutive_ok": sample.consecutive_ok,
                "consecutive_fail": sample.consecutive_fail,
                "remote_write_context": sanitized_remote_write_context,
            },
        )
        self._maybe_notify_systemd_status(sample=sample)

    def _safe_store_save(self, snapshot: RemoteMemoryWatchdogSnapshot) -> None:
        """Persist one snapshot without letting store failures kill the watchdog."""

        try:
            with self._store_lock:
                self.store.save(snapshot)
        except Exception as exc:
            self._emit_internal_error(
                "remote_memory_watchdog_store_error",
                exc,
                extra={
                    "artifact_path": str(self.artifact_path),
                    "sample_count": self._sample_count,
                },
            )

    def _safe_event_append(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, object] | None = None,
    ) -> None:
        """Append one ops event best-effort."""

        safe_data = _normalize_jsonish_value(data or {}, depth=0)
        try:
            self.event_store.append(
                event=event,
                message=message,
                level=level,
                data=safe_data if isinstance(safe_data, dict) else {},
            )
        except Exception as exc:
            self._emit_internal_error(
                "remote_memory_watchdog_event_store_error",
                exc,
                extra={
                    "event_name": event,
                    "level": level,
                    "artifact_path": str(self.artifact_path),
                },
            )

    def _safe_emit_payload(self, payload: dict[str, object]) -> None:
        """Serialize and emit one NDJSON payload best-effort."""

        safe_payload = _normalize_jsonish_value(payload, depth=0)
        if not isinstance(safe_payload, dict):
            safe_payload = {"event": "remote_memory_watchdog_emit_error", "detail": "payload normalization failed"}
        line: str
        try:
            line = json.dumps(safe_payload, ensure_ascii=True, sort_keys=True)
        except Exception:
            line = json.dumps(
                {
                    "event": "remote_memory_watchdog_emit_error",
                    "detail": "payload serialization failed",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        try:
            self.emit(line)
        except Exception:
            try:
                self._default_emit(line)
            except Exception:
                return

    def _emit_internal_error(
        self,
        event: str,
        exc: Exception,
        *,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Emit one internal watchdog error without raising."""

        payload = {
            "event": event,
            "error": self._normalize_detail(f"{type(exc).__name__}: {exc}"),
        }
        if extra:
            payload.update(extra)
        self._safe_emit_payload(payload)

    def _sleep_until_next_tick(
        self,
        *,
        now_monotonic: float,
        deadline: float | None,
        probe_inflight: bool,
    ) -> None:
        """Sleep or wait until the next scheduler tick."""

        sleep_s = self.interval_s
        if probe_inflight:
            sleep_s = min(sleep_s, self.heartbeat_while_probe_inflight_s)
        systemd_watchdog_interval_s = self._systemd.watchdog_interval_s
        if systemd_watchdog_interval_s is not None:
            sleep_s = min(sleep_s, systemd_watchdog_interval_s)
        if deadline is not None:
            sleep_s = min(sleep_s, max(0.0, deadline - now_monotonic))
        if sleep_s <= 0.0:
            return
        if self._sleep is time.sleep:
            self._stop_event.wait(timeout=sleep_s)
            return
        self._sleep(sleep_s)

    def _notify_systemd_ready(self) -> None:
        """Notify systemd that the watchdog service is ready."""

        if self._ready_notified:
            return
        status = self._systemd_status_text()
        self._systemd.ready(status=status)
        self._ready_notified = True

    def _notify_systemd_stopping(self) -> None:
        """Notify systemd that the watchdog service is stopping."""

        self._systemd.stopping(
            status=compact_text(
                f"Remote memory watchdog stopping; samples={self._sample_count} failures={self._failure_count}.",
                limit=160,
            )
            or "Remote memory watchdog stopping."
        )

    def _maybe_notify_systemd_status(self, *, sample: RemoteMemoryWatchdogSample) -> None:
        """Push one compact status string to systemd when state changes."""

        status_text = compact_text(
            (
                f"remote-memory={sample.status} ready={str(sample.ready).lower()} "
                f"mode={sample.mode} latency_ms={sample.latency_ms} detail={sample.detail or '-'}"
            ),
            limit=160,
        )
        if status_text:
            self._systemd.status(status_text)

    def _maybe_notify_systemd_watchdog(self, *, now_monotonic: float) -> None:
        """Send WATCHDOG=1 keep-alives while the service is healthy enough to supervise."""

        watchdog_interval_s = self._systemd.watchdog_interval_s
        if watchdog_interval_s is None:
            return
        if self._probe_timeout_reported:
            # Intentionally stop pinging systemd once the probe is considered
            # stuck so the unit can be restarted automatically when configured
            # with WatchdogSec=.
            return
        last_ping = self._last_systemd_watchdog_ping_monotonic
        if last_ping is not None and (now_monotonic - last_ping) < watchdog_interval_s:
            return
        self._systemd.watchdog_ping(status=self._systemd_status_text())
        self._last_systemd_watchdog_ping_monotonic = now_monotonic

    def _systemd_status_text(self) -> str:
        """Build one compact human-readable status summary for sd_notify."""

        last_sample = self._recent_samples[-1] if self._recent_samples else None
        if last_sample is None:
            return "Remote memory watchdog starting."
        return (
            compact_text(
                (
                    f"remote-memory={last_sample.status} ready={str(last_sample.ready).lower()} "
                    f"mode={last_sample.mode} samples={self._sample_count} failures={self._failure_count} "
                    f"detail={last_sample.detail or '-'}"
                ),
                limit=160,
            )
            or "Remote memory watchdog running."
        )


__all__ = [
    "RemoteMemoryWatchdog",
    "RemoteMemoryWatchdogSample",
    "RemoteMemoryWatchdogSnapshot",
    "RemoteMemoryWatchdogStore",
    "build_remote_memory_watchdog_bootstrap_snapshot",
]
