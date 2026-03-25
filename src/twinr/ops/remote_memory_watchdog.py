"""Continuously probe required remote ChonkyDB readiness.

This module runs a dedicated watchdog loop outside the GPIO and conversation
runtime. It reuses ``LongTermMemoryService.ensure_remote_ready()`` as the
canonical fail-closed readiness check, writes one rolling JSON snapshot under
Twinr's ops store, emits transition events into the local ops event log, and
prints one JSON line per sample for journald/systemd.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Protocol
import inspect
import json
import os
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
    utc_now_iso as _utc_now_iso,
)


_DEFAULT_DEEP_PROBE_IDLE_FLOOR_S = 5.0


def _copy_object_dict(value: object) -> dict[str, object] | None:
    """Copy one generic dict payload into a plain object dict."""

    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


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
        self.service_factory = service_factory or (lambda: LongTermMemoryService.from_config(config))
        self.store = store or RemoteMemoryWatchdogStore.from_config(config)
        self.event_store = event_store or TwinrOpsEventStore.from_config(config)
        self.emit = emit or self._default_emit
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._service: _RemoteMemoryService | None = None
        self._probe_lock = Lock()
        self._store_lock = Lock()
        self._started_at = _utc_now_iso()
        self._sample_count = 0
        self._failure_count = 0
        self._consecutive_ok = 0
        self._consecutive_fail = 0
        self._last_ok_at: str | None = None
        self._last_failure_at: str | None = None
        self._last_status: str | None = None
        self._last_detail: str | None = None
        self._recent_samples: deque[RemoteMemoryWatchdogSample] = deque(maxlen=self.history_limit)

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
                            include_archive=bootstrap_probe,
                        )
                        probe_payload = self._normalize_probe_payload(
                            getattr(probe_result, "to_dict", lambda: None)()
                            if probe_result is not None
                            else None
                        )
                        ready = bool(getattr(probe_result, "ready", False))
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
            with self._store_lock:
                self.store.save(snapshot)
            self._emit_transition_event(sample)
            self.emit(
                json.dumps(
                    {
                        "event": "remote_memory_watchdog_sample",
                        **sample.to_dict(),
                        "artifact_path": str(self.artifact_path),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            return snapshot

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the watchdog loop until stopped or the optional duration expires."""

        resolved_duration = None
        if duration_s is not None:
            resolved_duration = max(0.0, float(duration_s))
        deadline = None if resolved_duration is None else self._monotonic() + resolved_duration
        self.event_store.append(
            event="remote_memory_watchdog_started",
            message="Remote memory watchdog started.",
            data={
                "artifact_path": str(self.artifact_path),
                "interval_s": self.interval_s,
                "history_limit": self.history_limit,
            },
        )
        if resolved_duration == 0.0:
            try:
                self.probe_once()
                return 0
            except KeyboardInterrupt:
                return 0
            finally:
                self.event_store.append(
                    event="remote_memory_watchdog_stopped",
                    message="Remote memory watchdog stopped.",
                    data={
                        "artifact_path": str(self.artifact_path),
                        "sample_count": self._sample_count,
                        "failure_count": self._failure_count,
                    },
                )
                self.close()

        probe_thread: Thread | None = None
        probe_started_at: str | None = None
        probe_started_monotonic: float | None = None
        probe_was_inflight = False
        next_probe_not_before_monotonic = self._monotonic()
        try:
            while True:
                now_monotonic = self._monotonic()
                probe_inflight = bool(probe_thread is not None and probe_thread.is_alive())
                if probe_was_inflight and not probe_inflight:
                    last_sample = self._recent_samples[-1] if self._recent_samples else None
                    next_probe_not_before_monotonic = max(
                        next_probe_not_before_monotonic,
                        now_monotonic + self._deep_probe_idle_gap_s(last_sample=last_sample),
                    )
                probe_was_inflight = probe_inflight
                if not probe_inflight and now_monotonic >= next_probe_not_before_monotonic:
                    probe_started_at = _utc_now_iso()
                    probe_started_monotonic = now_monotonic
                    probe_thread = Thread(
                        target=self._probe_worker_main,
                        name="twinr-remote-memory-watchdog-probe",
                        daemon=True,
                    )
                    probe_thread.start()
                    probe_inflight = True
                    probe_was_inflight = True
                self._emit_heartbeat(
                    probe_started_at=probe_started_at,
                    probe_started_monotonic=probe_started_monotonic,
                    probe_inflight=probe_inflight,
                )
                if deadline is not None and now_monotonic >= deadline:
                    return 0
                sleep_s = self.interval_s
                if deadline is not None:
                    sleep_s = min(sleep_s, max(0.0, deadline - now_monotonic))
                if sleep_s > 0.0:
                    self._sleep(sleep_s)
        except KeyboardInterrupt:
            return 0
        finally:
            self.event_store.append(
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
        try:
            keepalive_s = float(raw_keepalive)
        except (TypeError, ValueError):
            keepalive_s = _DEFAULT_DEEP_PROBE_IDLE_FLOOR_S
        if keepalive_s <= 0.0:
            keepalive_s = _DEFAULT_DEEP_PROBE_IDLE_FLOOR_S
        latency_s = 0.0
        if last_sample is not None:
            latency_s = max(0.0, float(last_sample.latency_ms or 0.0) / 1000.0)
        return max(self.interval_s, keepalive_s, latency_s)

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
    def _normalize_detail(value: object) -> str | None:
        text = compact_text(str(value or "").strip(), limit=240)
        return text or None

    @staticmethod
    def _normalize_probe_payload(value: object) -> dict[str, object] | None:
        """Keep only dict-like probe payloads for watchdog artifacts."""

        if not isinstance(value, dict):
            return None
        return dict(value)

    @staticmethod
    def _merge_remote_write_context(
        probe: dict[str, object] | None,
        remote_write_context: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if not remote_write_context:
            return probe
        merged = dict(probe or {})
        merged["remote_write_context"] = dict(remote_write_context)
        return merged

    def _probe_worker_main(self) -> None:
        """Run one deep remote probe in the background."""

        try:
            self.probe_once()
        except Exception as exc:
            self.emit(
                json.dumps(
                    {
                        "event": "remote_memory_watchdog_worker_error",
                        "error": self._normalize_detail(f"{type(exc).__name__}: {exc}"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )

    def _emit_heartbeat(
        self,
        *,
        probe_started_at: str | None,
        probe_started_monotonic: float | None,
        probe_inflight: bool,
    ) -> None:
        """Emit one per-second heartbeat line while the deep probe runs."""

        captured_at = _utc_now_iso()
        probe_age_s: float | None = None
        if probe_inflight and probe_started_monotonic is not None:
            probe_age_s = round(max(0.0, self._monotonic() - probe_started_monotonic), 1)
        self._persist_heartbeat_snapshot(
            heartbeat_at=captured_at,
            probe_inflight=probe_inflight,
            probe_started_at=probe_started_at,
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
        self.emit(json.dumps(payload, ensure_ascii=True, sort_keys=True))

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
            hostname=socket.gethostname(),
            pid=os.getpid(),
            interval_s=self.interval_s,
            history_limit=self.history_limit,
            sample_count=self._sample_count,
            failure_count=self._failure_count,
            last_ok_at=self._last_ok_at,
            last_failure_at=self._last_failure_at,
            artifact_path=str(self.artifact_path),
            current=current,
            recent_samples=self._persisted_recent_samples(),
            heartbeat_at=current.captured_at,
            probe_inflight=False,
            probe_started_at=None,
            probe_age_s=None,
        )

    def _persist_heartbeat_snapshot(
        self,
        *,
        heartbeat_at: str,
        probe_inflight: bool,
        probe_started_at: str | None,
        probe_age_s: float | None,
    ) -> None:
        """Persist liveness metadata even while a deep probe is still running."""

        last_sample = self._recent_samples[-1] if self._recent_samples else None
        current = last_sample or self._starting_sample(captured_at=heartbeat_at)
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=_SNAPSHOT_SCHEMA_VERSION,
            started_at=self._started_at,
            updated_at=current.captured_at,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            interval_s=self.interval_s,
            history_limit=self.history_limit,
            sample_count=self._sample_count,
            failure_count=self._failure_count,
            last_ok_at=self._last_ok_at,
            last_failure_at=self._last_failure_at,
            artifact_path=str(self.artifact_path),
            current=current,
            recent_samples=self._persisted_recent_samples(),
            heartbeat_at=heartbeat_at,
            probe_inflight=probe_inflight,
            probe_started_at=probe_started_at,
            probe_age_s=probe_age_s,
        )
        with self._store_lock:
            self.store.save(snapshot)

    def _starting_sample(self, *, captured_at: str) -> RemoteMemoryWatchdogSample:
        """Synthesize the startup state before the first real probe sample exists."""

        return _build_starting_sample(self.config, captured_at=captured_at)

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
        if previous_status == sample.status and previous_detail == sample.detail:
            return
        if previous_status is None:
            message = f"Remote memory watchdog observed initial state {sample.status}."
        else:
            message = (
                f"Remote memory watchdog changed from {previous_status} to {sample.status}."
            )
        level = "error" if sample.status == "fail" else "info"
        self.event_store.append(
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
                "remote_write_context": remote_write_context,
            },
        )


__all__ = [
    "RemoteMemoryWatchdog",
    "RemoteMemoryWatchdogSample",
    "RemoteMemoryWatchdogSnapshot",
    "RemoteMemoryWatchdogStore",
    "build_remote_memory_watchdog_bootstrap_snapshot",
]
