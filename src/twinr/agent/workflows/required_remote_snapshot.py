"""Evaluate the external remote-memory watchdog snapshot for live runtimes.

The live button/audio loops must not perform deep remote-memory readiness
checks inside the GPIO-critical process. Those checks are expensive and can
starve button handling on the Pi. This helper treats the dedicated external
remote watchdog artifact as the authoritative runtime signal for
required-remote readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import errno
import math
import os
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogSnapshot, RemoteMemoryWatchdogStore


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse one persisted UTC timestamp from the watchdog artifact."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_is_alive(pid: int | None) -> bool:
    """Report whether one local PID is still alive."""

    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


@dataclass(frozen=True, slots=True)
class RequiredRemoteWatchdogAssessment:
    """Summarize one external-watchdog readiness evaluation."""

    ready: bool
    detail: str
    artifact_path: str
    pid_alive: bool
    sample_age_s: float | None
    max_sample_age_s: float
    sample_status: str | None
    sample_ready: bool | None
    sample_required: bool | None
    sample_latency_ms: float | None
    watchdog_pid: int | None = None
    snapshot_updated_at: str | None = None
    snapshot_stale: bool = False
    heartbeat_age_s: float | None = None
    heartbeat_updated_at: str | None = None
    probe_inflight: bool = False
    probe_age_s: float | None = None


def _watchdog_interval_s(config: TwinrConfig, snapshot: RemoteMemoryWatchdogSnapshot) -> float:
    """Return the effective watchdog heartbeat interval in seconds."""

    return max(
        0.1,
        float(getattr(snapshot, "interval_s", 0.0) or 0.0),
        float(getattr(config, "long_term_memory_remote_watchdog_interval_s", 1.0) or 1.0),
    )


def _keepalive_floor_s(config: TwinrConfig) -> float:
    """Return the minimum remote keepalive floor in seconds."""

    return max(
        5.0,
        float(getattr(config, "long_term_memory_remote_keepalive_interval_s", 5.0) or 5.0),
    )


def _max_allowed_sample_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
) -> float:
    """Derive a bounded freshness budget for the watchdog snapshot.

    The dedicated remote watchdog performs the expensive deep probe outside the
    live runtime. Each probe may legitimately take tens of seconds while large
    remote snapshots are validated, so freshness must be based on the observed
    probe latency instead of a small fixed timeout.
    """

    observed_latency_s = max(
        0.0,
        float(getattr(snapshot.current, "latency_ms", 0.0) or 0.0) / 1000.0,
    )
    watchdog_interval_s = _watchdog_interval_s(config, snapshot)
    keepalive_floor_s = _keepalive_floor_s(config)
    return max(keepalive_floor_s, watchdog_interval_s * 2.0, observed_latency_s * 2.0 + 5.0)


def _max_allowed_heartbeat_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
) -> float:
    """Return how old a persisted watchdog heartbeat may be before it is stale."""

    return max(_keepalive_floor_s(config), _watchdog_interval_s(config, snapshot) * 3.0)


def _max_allowed_inflight_probe_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
    max_sample_age_s: float,
) -> float:
    """Return the maximum bounded age for one still-running deep probe."""

    keepalive_floor_s = _keepalive_floor_s(config)
    timeout_budget_s = max(
        float(getattr(config, "chonkydb_timeout_s", 20.0) or 20.0),
        float(getattr(config, "long_term_memory_remote_read_timeout_s", 8.0) or 8.0),
        float(getattr(config, "long_term_memory_remote_write_timeout_s", 15.0) or 15.0),
    )
    return max(
        max_sample_age_s + keepalive_floor_s,
        timeout_budget_s * 2.0 + keepalive_floor_s,
    )


def assess_required_remote_watchdog_snapshot(
    config: TwinrConfig,
    *,
    now_wall: datetime | None = None,
    store: RemoteMemoryWatchdogStore | None = None,
) -> RequiredRemoteWatchdogAssessment:
    """Evaluate whether the external remote watchdog currently proves readiness."""

    watchdog_store = store or RemoteMemoryWatchdogStore.from_config(config)
    snapshot = watchdog_store.load()
    artifact_path = str(watchdog_store.path)
    if snapshot is None:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail="Remote memory watchdog snapshot is missing.",
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=None,
            sample_ready=None,
            sample_required=None,
            sample_latency_ms=None,
            watchdog_pid=None,
            snapshot_updated_at=None,
            snapshot_stale=True,
        )

    pid_alive = _pid_is_alive(getattr(snapshot, "pid", None))
    if not pid_alive:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=f"Remote memory watchdog process {snapshot.pid} is not alive.",
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=snapshot.current.status,
            sample_ready=snapshot.current.ready,
            sample_required=snapshot.current.required,
            sample_latency_ms=snapshot.current.latency_ms,
            watchdog_pid=snapshot.pid,
            snapshot_updated_at=snapshot.updated_at,
            snapshot_stale=True,
            heartbeat_updated_at=getattr(snapshot, "heartbeat_at", None),
            probe_inflight=bool(getattr(snapshot, "probe_inflight", False)),
            probe_age_s=getattr(snapshot, "probe_age_s", None),
        )

    current = snapshot.current
    if not bool(current.required):
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail="Remote memory watchdog is not enforcing a required remote dependency.",
            artifact_path=artifact_path,
            pid_alive=True,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=current.status,
            sample_ready=current.ready,
            sample_required=current.required,
            sample_latency_ms=current.latency_ms,
            watchdog_pid=snapshot.pid,
            snapshot_updated_at=snapshot.updated_at,
            heartbeat_updated_at=getattr(snapshot, "heartbeat_at", None),
            probe_inflight=bool(getattr(snapshot, "probe_inflight", False)),
            probe_age_s=getattr(snapshot, "probe_age_s", None),
        )

    captured_at = _parse_utc_timestamp(current.captured_at) or _parse_utc_timestamp(snapshot.updated_at)
    heartbeat_at = _parse_utc_timestamp(getattr(snapshot, "heartbeat_at", None)) or _parse_utc_timestamp(snapshot.updated_at)
    resolved_now = now_wall or datetime.now(timezone.utc)
    sample_age_s = None
    if captured_at is not None:
        sample_age_s = max(0.0, (resolved_now - captured_at).total_seconds())
    heartbeat_age_s = None
    if heartbeat_at is not None:
        heartbeat_age_s = max(0.0, (resolved_now - heartbeat_at).total_seconds())
    max_sample_age_s = _max_allowed_sample_age_s(config=config, snapshot=snapshot)
    max_heartbeat_age_s = _max_allowed_heartbeat_age_s(config=config, snapshot=snapshot)
    inflight_probe_age_s = getattr(snapshot, "probe_age_s", None)
    inflight_heartbeat_fresh = (
        bool(getattr(snapshot, "probe_inflight", False))
        and heartbeat_age_s is not None
        and math.isfinite(heartbeat_age_s)
        and heartbeat_age_s <= max_heartbeat_age_s
        and inflight_probe_age_s is not None
        and math.isfinite(float(inflight_probe_age_s))
        and float(inflight_probe_age_s)
        <= _max_allowed_inflight_probe_age_s(
            config=config,
            snapshot=snapshot,
            max_sample_age_s=max_sample_age_s,
        )
    )
    snapshot_stale = (
        sample_age_s is None or not math.isfinite(sample_age_s) or sample_age_s > max_sample_age_s
    ) and not inflight_heartbeat_fresh
    if snapshot_stale:
        detail = (
            f"Remote memory watchdog snapshot is stale (age={sample_age_s!r}s, max={max_sample_age_s:.1f}s)."
        )
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            pid_alive=True,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            sample_status=current.status,
            sample_ready=current.ready,
            sample_required=current.required,
            sample_latency_ms=current.latency_ms,
            watchdog_pid=snapshot.pid,
            snapshot_updated_at=snapshot.updated_at,
            snapshot_stale=True,
            heartbeat_age_s=heartbeat_age_s,
            heartbeat_updated_at=getattr(snapshot, "heartbeat_at", None),
            probe_inflight=bool(getattr(snapshot, "probe_inflight", False)),
            probe_age_s=inflight_probe_age_s,
        )

    if str(current.status or "").strip().lower() != "ok" or not bool(current.ready):
        detail = str(current.detail or "").strip() or "Remote memory watchdog reports unavailable."
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            pid_alive=True,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            sample_status=current.status,
            sample_ready=current.ready,
            sample_required=current.required,
            sample_latency_ms=current.latency_ms,
            watchdog_pid=snapshot.pid,
            snapshot_updated_at=snapshot.updated_at,
            snapshot_stale=False,
            heartbeat_age_s=heartbeat_age_s,
            heartbeat_updated_at=getattr(snapshot, "heartbeat_at", None),
            probe_inflight=bool(getattr(snapshot, "probe_inflight", False)),
            probe_age_s=inflight_probe_age_s,
        )

    return RequiredRemoteWatchdogAssessment(
        ready=True,
        detail="ok",
        artifact_path=artifact_path,
        pid_alive=True,
        sample_age_s=sample_age_s,
        max_sample_age_s=max_sample_age_s,
        sample_status=current.status,
        sample_ready=current.ready,
        sample_required=current.required,
        sample_latency_ms=current.latency_ms,
        watchdog_pid=snapshot.pid,
        snapshot_updated_at=snapshot.updated_at,
        snapshot_stale=False,
        heartbeat_age_s=heartbeat_age_s,
        heartbeat_updated_at=getattr(snapshot, "heartbeat_at", None),
        probe_inflight=bool(getattr(snapshot, "probe_inflight", False)),
        probe_age_s=inflight_probe_age_s,
    )


def _watchdog_startup_wait_s(config: TwinrConfig) -> float:
    """Return how long runtime bootstrap may wait for a starting watchdog."""

    return max(
        0.0,
        float(getattr(config, "long_term_memory_remote_watchdog_startup_wait_s", 30.0) or 30.0),
    )


def _watchdog_startup_poll_s(config: TwinrConfig) -> float:
    """Return the bounded poll cadence while the watchdog is still starting."""

    return min(
        1.0,
        max(
            0.1,
            float(getattr(config, "long_term_memory_remote_watchdog_interval_s", 1.0) or 1.0),
        ),
    )


def _assessment_allows_startup_wait(assessment: RequiredRemoteWatchdogAssessment) -> bool:
    """Return whether a non-ready watchdog state still looks like startup."""

    if assessment.ready or assessment.snapshot_stale:
        return False
    status = str(assessment.sample_status or "").strip().lower()
    detail = str(assessment.detail or "").strip().lower()
    if status == "starting":
        return True
    if "watchdog is starting" in detail:
        return True
    if "snapshot is missing" in detail:
        return True
    return False


def ensure_required_remote_watchdog_snapshot_ready(
    config: TwinrConfig,
    *,
    store: RemoteMemoryWatchdogStore | None = None,
) -> RequiredRemoteWatchdogAssessment:
    """Raise when the external remote watchdog does not prove readiness."""

    watchdog_store = store or RemoteMemoryWatchdogStore.from_config(config)
    assessment = assess_required_remote_watchdog_snapshot(config, store=watchdog_store)
    if assessment.ready:
        return assessment
    startup_wait_s = _watchdog_startup_wait_s(config)
    if startup_wait_s > 0.0 and _assessment_allows_startup_wait(assessment):
        deadline = time.monotonic() + startup_wait_s
        poll_s = _watchdog_startup_poll_s(config)
        while time.monotonic() < deadline:
            remaining_s = max(0.0, deadline - time.monotonic())
            time.sleep(min(poll_s, remaining_s))
            assessment = assess_required_remote_watchdog_snapshot(config, store=watchdog_store)
            if assessment.ready:
                return assessment
            if not _assessment_allows_startup_wait(assessment):
                break
    raise LongTermRemoteUnavailableError(assessment.detail)
