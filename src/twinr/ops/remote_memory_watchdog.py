"""Continuously probe required remote ChonkyDB readiness.

This module runs a dedicated watchdog loop outside the GPIO and conversation
runtime. It reuses ``LongTermMemoryService.ensure_remote_ready()`` as the
canonical fail-closed readiness check, writes one rolling JSON snapshot under
Twinr's ops store, emits transition events into the local ops event log, and
prints one JSON line per sample for journald/systemd.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Protocol
import json
import os
import socket
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.events import TwinrOpsEventStore, compact_text
from twinr.ops.paths import resolve_ops_paths_for_config


_DEFAULT_WATCHDOG_INTERVAL_S = 1.0
_DEFAULT_HISTORY_LIMIT = 3600
_SNAPSHOT_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_interval_s(value: object, *, default: float) -> float:
    """Normalize the watchdog interval to a finite positive float."""

    try:
        interval_s = float(value)
    except (TypeError, ValueError):
        return default
    if interval_s <= 0.0:
        return default
    return interval_s


def _coerce_history_limit(value: object, *, default: int) -> int:
    """Normalize the rolling sample-history length."""

    try:
        history_limit = int(value)
    except (TypeError, ValueError):
        return default
    if history_limit <= 0:
        return default
    return history_limit


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)


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


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSample:
    """Capture one remote-memory probe result."""

    seq: int
    captured_at: str
    status: str
    ready: bool
    mode: str
    required: bool
    latency_ms: float
    consecutive_ok: int
    consecutive_fail: int
    detail: str | None = None
    probe: dict[str, object] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSample":
        """Hydrate one sample from persisted JSON data."""

        return cls(
            seq=int(payload.get("seq", 0) or 0),
            captured_at=str(payload.get("captured_at", "") or ""),
            status=str(payload.get("status", "unknown") or "unknown"),
            ready=bool(payload.get("ready", False)),
            mode=str(payload.get("mode", "unknown") or "unknown"),
            required=bool(payload.get("required", False)),
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
            consecutive_ok=int(payload.get("consecutive_ok", 0) or 0),
            consecutive_fail=int(payload.get("consecutive_fail", 0) or 0),
            detail=cls._coerce_optional_text(payload.get("detail")),
            probe=dict(payload.get("probe")) if isinstance(payload.get("probe"), dict) else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        text = compact_text(str(value or "").strip(), limit=240)
        return text or None


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSnapshot:
    """Persisted rolling watchdog state."""

    schema_version: int
    started_at: str
    updated_at: str
    hostname: str
    pid: int
    interval_s: float
    history_limit: int
    sample_count: int
    failure_count: int
    last_ok_at: str | None
    last_failure_at: str | None
    artifact_path: str
    current: RemoteMemoryWatchdogSample
    recent_samples: tuple[RemoteMemoryWatchdogSample, ...]
    heartbeat_at: str | None = None
    probe_inflight: bool = False
    probe_started_at: str | None = None
    probe_age_s: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSnapshot":
        """Hydrate one persisted snapshot from JSON data."""

        current_payload = payload.get("current")
        if not isinstance(current_payload, dict):
            raise ValueError("Remote memory watchdog snapshot is missing `current`.")
        raw_recent_samples = payload.get("recent_samples")
        if raw_recent_samples is None:
            raw_recent_samples = ()
        if not isinstance(raw_recent_samples, list):
            raise ValueError("Remote memory watchdog snapshot has malformed `recent_samples`.")
        return cls(
            schema_version=int(payload.get("schema_version", 0) or 0),
            started_at=str(payload.get("started_at", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
            hostname=str(payload.get("hostname", "") or ""),
            pid=int(payload.get("pid", 0) or 0),
            interval_s=float(payload.get("interval_s", 0.0) or 0.0),
            history_limit=int(payload.get("history_limit", 0) or 0),
            sample_count=int(payload.get("sample_count", 0) or 0),
            failure_count=int(payload.get("failure_count", 0) or 0),
            last_ok_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_ok_at")),
            last_failure_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_failure_at")),
            artifact_path=str(payload.get("artifact_path", "") or ""),
            current=RemoteMemoryWatchdogSample.from_dict(current_payload),
            recent_samples=tuple(
                RemoteMemoryWatchdogSample.from_dict(item)
                for item in raw_recent_samples
                if isinstance(item, dict)
            ),
            heartbeat_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("heartbeat_at")),
            probe_inflight=bool(payload.get("probe_inflight", False)),
            probe_started_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("probe_started_at")),
            probe_age_s=cls._coerce_optional_float(payload.get("probe_age_s")),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["current"] = self.current.to_dict()
        payload["recent_samples"] = [sample.to_dict() for sample in self.recent_samples]
        return payload

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not parsed >= 0.0:
            return None
        return parsed


class RemoteMemoryWatchdogStore:
    """Persist the current rolling watchdog snapshot to disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteMemoryWatchdogStore":
        """Build the canonical watchdog-state store for one Twinr config."""

        ops_paths = resolve_ops_paths_for_config(config)
        return cls(ops_paths.ops_store_root / "remote_memory_watchdog.json")

    def save(self, snapshot: RemoteMemoryWatchdogSnapshot) -> None:
        """Persist the latest rolling snapshot atomically."""

        _atomic_write_json(self.path, snapshot.to_dict())

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        """Load the persisted rolling snapshot when it exists."""

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is not valid JSON: {self.path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is unreadable: {self.path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Remote memory watchdog snapshot has an invalid top-level payload: {self.path}")
        try:
            return RemoteMemoryWatchdogSnapshot.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is malformed: {self.path}") from exc


def build_remote_memory_watchdog_bootstrap_snapshot(
    config: TwinrConfig,
    *,
    pid: int,
    artifact_path: str | Path,
    started_at: str | None = None,
    captured_at: str | None = None,
) -> RemoteMemoryWatchdogSnapshot:
    """Build the supervisor-seeded startup snapshot for a fresh watchdog child."""

    resolved_started_at = str(started_at or _utc_now_iso())
    resolved_captured_at = str(captured_at or resolved_started_at)
    required = bool(
        config.long_term_memory_enabled
        and config.long_term_memory_mode == "remote_primary"
        and config.long_term_memory_remote_required
    )
    current = RemoteMemoryWatchdogSample(
        seq=0,
        captured_at=resolved_captured_at,
        status="starting",
        ready=False,
        mode="remote_primary" if required else "disabled",
        required=required,
        latency_ms=0.0,
        consecutive_ok=0,
        consecutive_fail=0,
        detail="Remote memory watchdog is starting.",
        probe=None,
    )
    interval_s = _coerce_interval_s(
        getattr(config, "long_term_memory_remote_watchdog_interval_s", _DEFAULT_WATCHDOG_INTERVAL_S),
        default=_DEFAULT_WATCHDOG_INTERVAL_S,
    )
    history_limit = _coerce_history_limit(
        getattr(config, "long_term_memory_remote_watchdog_history_limit", _DEFAULT_HISTORY_LIMIT),
        default=_DEFAULT_HISTORY_LIMIT,
    )
    return RemoteMemoryWatchdogSnapshot(
        schema_version=_SNAPSHOT_SCHEMA_VERSION,
        started_at=resolved_started_at,
        updated_at=resolved_captured_at,
        hostname=socket.gethostname(),
        pid=int(pid),
        interval_s=interval_s,
        history_limit=history_limit,
        sample_count=0,
        failure_count=0,
        last_ok_at=None,
        last_failure_at=None,
        artifact_path=str(artifact_path),
        current=current,
        recent_samples=(),
        heartbeat_at=resolved_captured_at,
        probe_inflight=True,
        probe_started_at=resolved_captured_at,
        probe_age_s=0.0,
    )


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
            captured_at = _utc_now_iso()
            started = self._monotonic()
            status = "fail"
            ready = False
            mode = "unknown"
            required = False
            detail: str | None = None
            probe_payload: dict[str, object] | None = None
            deep_probe_attempted = False

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
                    deep_probe_attempted = True
                    probe_remote_ready = getattr(service, "probe_remote_ready", None)
                    if callable(probe_remote_ready):
                        probe_result = probe_remote_ready()
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
                if deep_probe_attempted:
                    self._drop_service_instance()
            except Exception as exc:
                status = "fail"
                ready = False
                detail = self._normalize_detail(f"{type(exc).__name__}: {exc}")
                self._drop_service_instance()

            latency_ms = round(max(0.0, (self._monotonic() - started) * 1000.0), 1)
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
        try:
            while True:
                if probe_thread is None or not probe_thread.is_alive():
                    probe_started_at = _utc_now_iso()
                    probe_started_monotonic = self._monotonic()
                    probe_thread = Thread(
                        target=self._probe_worker_main,
                        name="twinr-remote-memory-watchdog-probe",
                        daemon=True,
                    )
                    probe_thread.start()
                self._emit_heartbeat(
                    probe_started_at=probe_started_at,
                    probe_started_monotonic=probe_started_monotonic,
                    probe_inflight=probe_thread.is_alive(),
                )
                if deadline is not None and self._monotonic() >= deadline:
                    return 0
                sleep_s = self.interval_s
                if deadline is not None:
                    sleep_s = min(sleep_s, max(0.0, deadline - self._monotonic()))
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
            recent_samples=tuple(self._recent_samples),
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
            recent_samples=tuple(self._recent_samples),
            heartbeat_at=heartbeat_at,
            probe_inflight=probe_inflight,
            probe_started_at=probe_started_at,
            probe_age_s=probe_age_s,
        )
        with self._store_lock:
            self.store.save(snapshot)

    def _starting_sample(self, *, captured_at: str) -> RemoteMemoryWatchdogSample:
        """Synthesize the startup state before the first real probe sample exists."""

        required = bool(
            self.config.long_term_memory_enabled
            and self.config.long_term_memory_mode == "remote_primary"
            and self.config.long_term_memory_remote_required
        )
        return RemoteMemoryWatchdogSample(
            seq=0,
            captured_at=captured_at,
            status="starting",
            ready=False,
            mode="remote_primary" if required else "disabled",
            required=required,
            latency_ms=0.0,
            consecutive_ok=0,
            consecutive_fail=0,
            detail="Remote memory watchdog is starting.",
        )

    def _emit_transition_event(self, sample: RemoteMemoryWatchdogSample) -> None:
        previous_status = self._last_status
        previous_detail = self._last_detail
        self._last_status = sample.status
        self._last_detail = sample.detail
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
            },
        )
