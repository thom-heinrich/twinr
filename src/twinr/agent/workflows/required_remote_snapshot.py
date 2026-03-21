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


def _recent_sample_cycle_s(snapshot: RemoteMemoryWatchdogSnapshot) -> float | None:
    """Return the largest recent completed-sample cadence in seconds."""

    recent_samples = tuple(getattr(snapshot, "recent_samples", ()) or ())
    if len(recent_samples) < 2:
        return None
    parsed_samples = []
    for sample in recent_samples[-4:]:
        captured_at = _parse_utc_timestamp(getattr(sample, "captured_at", None))
        if captured_at is not None:
            parsed_samples.append(captured_at)
    if len(parsed_samples) < 2:
        return None
    intervals_s: list[float] = []
    for previous_at, current_at in zip(parsed_samples, parsed_samples[1:]):
        interval_s = max(0.0, (current_at - previous_at).total_seconds())
        if interval_s > 0.0 and math.isfinite(interval_s):
            intervals_s.append(interval_s)
    if not intervals_s:
        return None
    return max(intervals_s)


def _max_allowed_steady_state_sample_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
    max_sample_age_s: float,
) -> float:
    """Bound how long one last-known-good sample may bridge fresh heartbeats.

    In steady state the external watchdog deliberately idles between deep
    probes. During that quiet window the persisted heartbeat keeps proving that
    the watchdog loop is alive, but ``current.captured_at`` continues to age
    until the next deep probe completes. Allow one bounded steady-state cycle
    so the runtime does not flap false-stale between healthy samples, while
    still failing closed if the watchdog stops producing fresh probe results.
    """

    watchdog_interval_s = _watchdog_interval_s(config, snapshot)
    keepalive_floor_s = _keepalive_floor_s(config)
    observed_latency_s = max(
        0.0,
        float(getattr(snapshot.current, "latency_ms", 0.0) or 0.0) / 1000.0,
    )
    idle_gap_s = max(watchdog_interval_s, keepalive_floor_s, observed_latency_s)
    recent_cycle_s = _recent_sample_cycle_s(snapshot) or 0.0
    heartbeat_slack_s = max(1.0, watchdog_interval_s * 2.0)
    sample_clock_slack_s = max(1.0, watchdog_interval_s)
    return max(
        max_sample_age_s + idle_gap_s + sample_clock_slack_s,
        max(recent_cycle_s, idle_gap_s + observed_latency_s) + heartbeat_slack_s + sample_clock_slack_s,
    )


def _max_allowed_heartbeat_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
) -> float:
    """Return how old a persisted watchdog heartbeat may be before it is stale.

    The watchdog persists timestamps with second resolution and relies on a
    Python loop plus atomic fsync-heavy file writes. On the Pi this can miss a
    nominal 1s beat by a small margin even while the watchdog process stays
    healthy and the deep remote probe is still running. A bounded slack keeps
    the live runtime from false-failing these in-flight probes while still
    treating genuinely stuck heartbeats as stale within a few seconds.
    """

    base_budget_s = max(
        _keepalive_floor_s(config),
        _watchdog_interval_s(config, snapshot) * 3.0,
    )
    timing_slack_s = max(1.0, _watchdog_interval_s(config, snapshot) * 2.0)
    return base_budget_s + timing_slack_s


def _max_allowed_steady_state_heartbeat_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
    max_sample_age_s: float,
    max_steady_state_sample_age_s: float,
) -> float:
    """Return the bounded heartbeat budget while one good sample bridges idle time.

    On the Pi, healthy watchdog heartbeats are persisted on a slower bounded
    cadence than the live runtime polls. While the last completed sample still
    fits inside the allowed steady-state bridge window, the heartbeat may
    legitimately age past the generic keepalive budget without indicating a
    dead watchdog loop.
    """

    keepalive_floor_s = _keepalive_floor_s(config)
    return max(
        _max_allowed_heartbeat_age_s(config=config, snapshot=snapshot),
        min(max_steady_state_sample_age_s, max_sample_age_s + keepalive_floor_s),
    )


def _max_allowed_inflight_heartbeat_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
    max_sample_age_s: float,
) -> float:
    """Return the heartbeat budget while one deep probe is still running.

    A healthy in-flight probe can leave the last completed sample old for a
    while, especially when current-scope refreshes take multiple seconds on the
    Pi. During that bounded window the heartbeat continues to prove forward
    progress, so the heartbeat budget must follow the in-flight bridge instead
    of the much smaller generic keepalive-only threshold.
    """

    keepalive_floor_s = _keepalive_floor_s(config)
    inflight_probe_budget_s = _max_allowed_inflight_probe_age_s(
        config=config,
        snapshot=snapshot,
        max_sample_age_s=max_sample_age_s,
    )
    return max(
        _max_allowed_heartbeat_age_s(config=config, snapshot=snapshot),
        min(inflight_probe_budget_s, max_sample_age_s + keepalive_floor_s),
    )


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
    max_steady_state_sample_age_s = _max_allowed_steady_state_sample_age_s(
        config=config,
        snapshot=snapshot,
        max_sample_age_s=max_sample_age_s,
    )
    max_heartbeat_age_s = _max_allowed_heartbeat_age_s(config=config, snapshot=snapshot)
    max_steady_state_heartbeat_age_s = _max_allowed_steady_state_heartbeat_age_s(
        config=config,
        snapshot=snapshot,
        max_sample_age_s=max_sample_age_s,
        max_steady_state_sample_age_s=max_steady_state_sample_age_s,
    )
    max_inflight_heartbeat_age_s = _max_allowed_inflight_heartbeat_age_s(
        config=config,
        snapshot=snapshot,
        max_sample_age_s=max_sample_age_s,
    )
    inflight_probe_age_s = getattr(snapshot, "probe_age_s", None)
    heartbeat_fresh = (
        heartbeat_age_s is not None
        and math.isfinite(heartbeat_age_s)
        and heartbeat_age_s <= max_heartbeat_age_s
    )
    inflight_heartbeat_fresh = (
        bool(getattr(snapshot, "probe_inflight", False))
        and heartbeat_age_s is not None
        and math.isfinite(heartbeat_age_s)
        and heartbeat_age_s <= max_inflight_heartbeat_age_s
        and inflight_probe_age_s is not None
        and math.isfinite(float(inflight_probe_age_s))
        and float(inflight_probe_age_s)
        <= _max_allowed_inflight_probe_age_s(
            config=config,
            snapshot=snapshot,
            max_sample_age_s=max_sample_age_s,
        )
    )
    steady_state_heartbeat_fresh = (
        not bool(getattr(snapshot, "probe_inflight", False))
        and str(current.status or "").strip().lower() == "ok"
        and bool(current.ready)
        and heartbeat_age_s is not None
        and math.isfinite(heartbeat_age_s)
        and heartbeat_age_s <= max_steady_state_heartbeat_age_s
        and sample_age_s is not None
        and math.isfinite(sample_age_s)
        and sample_age_s <= max_steady_state_sample_age_s
    )
    snapshot_stale = (
        sample_age_s is None or not math.isfinite(sample_age_s) or sample_age_s > max_sample_age_s
    ) and not inflight_heartbeat_fresh and not steady_state_heartbeat_fresh
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
