# CHANGELOG: 2026-03-28
# BUG-1: Fixed the in-flight heartbeat budget bug. The previous formula collapsed
#        to max_sample_age_s + keepalive_floor and could false-fail healthy,
#        long-running deep probes.
# BUG-2: Fixed startup waiting for a missing snapshot. Previously a missing
#        artifact was always marked stale and could skip the intended startup
#        grace window after recovery was requested.
# BUG-3: Malformed, missing-current, non-boolean, non-numeric, and future-dated
#        snapshot fields now fail closed instead of crashing or being silently
#        treated as fresh/true.
# SEC-1: Hardened process-identity checks. Plain kill(pid, 0) is now augmented
#        with pidfd-based existence checks and optional boot-id / process-start
#        attestation to reduce PID-reuse confusion.
# SEC-2: Refuse symlinked or group/other-writable watchdog artifacts by default.
#        # BREAKING: insecure artifact paths now fail closed unless
#        config.long_term_memory_remote_watchdog_allow_insecure_artifact is true.
# IMP-1: Added optional monotonic-timestamp support with boot-id attestation so
#        freshness survives RTC/NTP wall-clock jumps on Raspberry Pi deployments.
# IMP-2: Added optional systemd-aware recovery hooks and cached /proc lookups so
#        the helper fits 2026 edge-runtime supervision patterns more cleanly.

"""Evaluate the external remote-memory watchdog snapshot for live runtimes.

The live button/audio loops must not perform deep remote-memory readiness
checks inside the GPIO-critical process. Those checks are expensive and can
starve button handling on the Pi. This helper treats the dedicated external
remote watchdog artifact as the authoritative runtime signal for
required-remote readiness.

Frontier upgrades in this revision:

* Optional monotonic freshness fields (paired with boot_id attestation) are used
  when present, which avoids false freshness flips after wall-clock jumps.
* Linux pidfd-based existence checks are preferred over plain kill(pid, 0).
* Optional boot_id / pid start-time attestation can disambiguate PID reuse.
* The artifact path is hardened against common local-tampering footguns.
* Recovery can integrate with systemd when a unit name is configured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import functools
import math
import os
from pathlib import Path
import stat as statmod
import subprocess
import time
from typing import Any, Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogSnapshot, RemoteMemoryWatchdogStore


WatchdogRecoveryStarter = Callable[[TwinrConfig, str], int | None]


class RemoteWatchdogStoreLike(Protocol):
    """Describe the minimal persisted-watchdog store contract used here."""

    path: Path

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        """Return the newest watchdog snapshot or ``None`` when missing."""


_TRUTHY_TEXT = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY_TEXT = frozenset({"0", "false", "f", "no", "n", "off"})
_FUTURE_TIMESTAMP_DETAIL = "Remote memory watchdog snapshot has a future-dated timestamp."
_SAMPLE_MONOTONIC_PATHS = (
    ("current", "captured_monotonic_ns"),
    ("current", "captured_mono_ns"),
    ("current", "monotonic_ns"),
    ("captured_monotonic_ns",),
    ("captured_mono_ns",),
    ("updated_monotonic_ns",),
)
_HEARTBEAT_MONOTONIC_PATHS = (
    ("heartbeat_monotonic_ns",),
    ("heartbeat_mono_ns",),
    ("updated_monotonic_ns",),
)
_PROBE_STARTED_MONOTONIC_PATHS = (
    ("probe_started_monotonic_ns",),
    ("probe_started_mono_ns",),
    ("probe_monotonic_ns",),
)


def _normalize_text(value: Any) -> str:
    """Return a stripped textual representation."""

    return str(value or "").strip()


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer coercion for persisted snapshot values."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _normalize_text(value).replace("_", "")
    if not text:
        return None
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion for persisted snapshot values."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = _normalize_text(value).replace("_", "")
        if not text:
            return None
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_bool(value: Any) -> bool | None:
    """Best-effort boolean coercion for persisted snapshot values."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 0.0:
            return False
        if float(value) == 1.0:
            return True
        return None
    text = _normalize_text(value).lower()
    if not text:
        return None
    if text in _TRUTHY_TEXT:
        return True
    if text in _FALSY_TEXT:
        return False
    return None


def _deep_get(payload: Any, *path: str) -> Any | None:
    """Return one nested attribute or mapping value."""

    current: Any = payload
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def _first_present(payload: Any, *paths: tuple[str, ...]) -> Any | None:
    """Return the first non-None value from several nested lookup paths."""

    for path in paths:
        value = _deep_get(payload, *path)
        if value is not None:
            return value
    return None


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse one persisted UTC timestamp from the watchdog artifact."""

    text = _normalize_text(value)
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


def _artifact_mtime_utc(path: Path) -> datetime | None:
    """Return one artifact mtime as a UTC datetime."""

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except (FileNotFoundError, OSError, OverflowError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def _current_boot_id() -> str | None:
    """Return the current Linux boot ID when available."""

    try:
        text = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8")
    except OSError:
        return None
    boot_id = _normalize_text(text).lower()
    return boot_id or None


@functools.lru_cache(maxsize=1)
def _clock_ticks_per_second() -> float | None:
    """Return Linux clock ticks per second."""

    try:
        return float(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def _linux_boot_time_s() -> float | None:
    """Return the current kernel boot time from /proc/stat."""

    try:
        with Path("/proc/stat").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_proc_stat_start_ticks(pid: int | None) -> int | None:
    """Return /proc/<pid>/stat starttime ticks for one live PID."""

    resolved_pid = _coerce_int(pid)
    if resolved_pid is None or resolved_pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{resolved_pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = raw.rfind(")")
    if end < 0 or end + 2 >= len(raw):
        return None
    tail = raw[end + 2 :].split()
    if len(tail) < 20:
        return None
    try:
        return int(tail[19])
    except (TypeError, ValueError):
        return None


def _read_proc_create_time_s(pid: int | None) -> float | None:
    """Return one PID start time in epoch seconds using /proc data."""

    start_ticks = _read_proc_stat_start_ticks(pid)
    boot_time_s = _linux_boot_time_s()
    clk_tck = _clock_ticks_per_second()
    if start_ticks is None or boot_time_s is None or clk_tck in (None, 0.0):
        return None
    return boot_time_s + (float(start_ticks) / float(clk_tck))


def _pid_exists_via_pidfd(pid: int) -> bool | None:
    """Return a race-resistant pidfd existence result when supported."""

    if not hasattr(os, "pidfd_open"):
        return None
    fd: int | None = None
    try:
        fd = os.pidfd_open(pid, 0)
        return True
    except OSError as exc:
        if exc.errno in {errno.ESRCH, errno.ENOENT}:
            return False
        if exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP if hasattr(errno, "ENOTSUP") else errno.EPERM),
        }:
            return None
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _pid_is_alive(pid: int | None) -> bool:
    """Report whether one local PID is still alive."""

    resolved_pid = _coerce_int(pid)
    if resolved_pid is None or resolved_pid <= 0:
        return False
    pidfd_result = _pid_exists_via_pidfd(resolved_pid)
    if pidfd_result is not None:
        return pidfd_result
    try:
        os.kill(resolved_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _snapshot_current(snapshot: RemoteMemoryWatchdogSnapshot) -> Any | None:
    """Return the current sample payload when present."""

    return _deep_get(snapshot, "current")


def _sample_status_text(snapshot: RemoteMemoryWatchdogSnapshot) -> str | None:
    """Return the normalized current sample status."""

    status = _normalize_text(_deep_get(snapshot, "current", "status")).lower()
    return status or None


def _sample_ready_value(snapshot: RemoteMemoryWatchdogSnapshot) -> bool | None:
    """Return the parsed current.ready value."""

    return _coerce_bool(_deep_get(snapshot, "current", "ready"))


def _sample_required_value(snapshot: RemoteMemoryWatchdogSnapshot) -> bool | None:
    """Return the parsed current.required value."""

    return _coerce_bool(_deep_get(snapshot, "current", "required"))


def _current_latency_s(snapshot: RemoteMemoryWatchdogSnapshot) -> float:
    """Return the observed sample latency in seconds."""

    latency_ms = _coerce_float(_deep_get(snapshot, "current", "latency_ms"))
    return max(0.0, (latency_ms or 0.0) / 1000.0)


def _snapshot_boot_id(snapshot: RemoteMemoryWatchdogSnapshot) -> str | None:
    """Return an optional persisted boot ID attestation from the artifact."""

    value = _first_present(
        snapshot,
        ("boot_id",),
        ("watchdog_boot_id",),
        ("current", "boot_id"),
        ("current", "watchdog_boot_id"),
    )
    boot_id = _normalize_text(value).lower()
    return boot_id or None


def _watchdog_expected_start_ticks(snapshot: RemoteMemoryWatchdogSnapshot) -> int | None:
    """Return an optional persisted watchdog start-ticks attestation."""

    return _coerce_int(
        _first_present(
            snapshot,
            ("pid_starttime_ticks",),
            ("pid_start_ticks",),
            ("watchdog_pid_starttime_ticks",),
            ("watchdog_pid_start_ticks",),
            ("current", "pid_starttime_ticks"),
            ("current", "pid_start_ticks"),
        )
    )


def _watchdog_expected_create_time_s(snapshot: RemoteMemoryWatchdogSnapshot) -> float | None:
    """Return an optional persisted watchdog create-time attestation."""

    return _coerce_float(
        _first_present(
            snapshot,
            ("pid_create_time_s",),
            ("pid_create_time",),
            ("watchdog_pid_create_time_s",),
            ("watchdog_pid_create_time",),
            ("current", "pid_create_time_s"),
            ("current", "pid_create_time"),
        )
    )


def _watchdog_identity_problem(snapshot: RemoteMemoryWatchdogSnapshot) -> str | None:
    """Return an identity-mismatch detail for the persisted watchdog PID."""

    resolved_pid = _coerce_int(_deep_get(snapshot, "pid"))
    if resolved_pid is None or resolved_pid <= 0:
        return "Remote memory watchdog snapshot is missing a valid PID."

    expected_boot_id = _snapshot_boot_id(snapshot)
    current_boot_id = _current_boot_id()
    if expected_boot_id and current_boot_id and expected_boot_id != current_boot_id:
        return (
            "Remote memory watchdog snapshot belongs to a different boot ID and "
            "cannot be trusted."
        )

    expected_start_ticks = _watchdog_expected_start_ticks(snapshot)
    if expected_start_ticks is not None:
        current_start_ticks = _read_proc_stat_start_ticks(resolved_pid)
        if current_start_ticks is None:
            return (
                f"Remote memory watchdog process {resolved_pid} is alive, but its "
                "start ticks cannot be verified."
            )
        if current_start_ticks != expected_start_ticks:
            return (
                f"Remote memory watchdog PID {resolved_pid} no longer matches the "
                "persisted start-time attestation."
            )

    expected_create_time_s = _watchdog_expected_create_time_s(snapshot)
    if expected_create_time_s is not None:
        current_create_time_s = _read_proc_create_time_s(resolved_pid)
        if current_create_time_s is None:
            return (
                f"Remote memory watchdog process {resolved_pid} is alive, but its "
                "create time cannot be verified."
            )
        if abs(current_create_time_s - expected_create_time_s) > 1.0:
            return (
                f"Remote memory watchdog PID {resolved_pid} no longer matches the "
                "persisted create-time attestation."
            )

    return None


def _artifact_security_problem(config: TwinrConfig, path: Path) -> str | None:
    """Return one local-tampering problem for the artifact path."""

    # BREAKING: insecure artifact paths now fail closed by default.
    # Set config.long_term_memory_remote_watchdog_allow_insecure_artifact = True
    # to restore the legacy trust model for old deployments.
    if bool(getattr(config, "long_term_memory_remote_watchdog_allow_insecure_artifact", False)):
        return None
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        detail = _normalize_text(exc.strerror) or repr(exc)
        return f"Remote memory watchdog artifact metadata is unavailable: {detail}."
    if statmod.S_ISLNK(st.st_mode):
        return "Remote memory watchdog artifact path is a symlink; refusing to trust it."
    if not statmod.S_ISREG(st.st_mode):
        return "Remote memory watchdog artifact path is not a regular file."
    if st.st_mode & 0o022:
        return (
            "Remote memory watchdog artifact is group/other-writable; refusing "
            "to trust it."
        )
    return None


def _probe_payload(snapshot: RemoteMemoryWatchdogSnapshot) -> dict[str, Any] | None:
    """Return the normalized probe payload when present."""

    probe_payload = _deep_get(snapshot, "current", "probe")
    if not isinstance(probe_payload, dict):
        return None
    warm_payload = probe_payload.get("warm_result")
    if isinstance(warm_payload, dict):
        return warm_payload
    result_payload = probe_payload.get("result")
    if isinstance(result_payload, dict):
        return result_payload
    return None


def _probe_attestation_bool(snapshot: RemoteMemoryWatchdogSnapshot, field_name: str) -> bool | None:
    """Return one boolean attestation field from the watchdog probe payload."""

    probe_payload = _probe_payload(snapshot)
    if probe_payload is None or field_name not in probe_payload:
        return None
    return _coerce_bool(probe_payload.get(field_name))


def _probe_attestation_text(snapshot: RemoteMemoryWatchdogSnapshot, field_name: str) -> str | None:
    """Return one string attestation field from the watchdog probe payload."""

    probe_payload = _probe_payload(snapshot)
    if probe_payload is None:
        return None
    text = _normalize_text(probe_payload.get(field_name)).lower()
    return text or None


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
        _coerce_float(_deep_get(snapshot, "interval_s")) or 0.0,
        _coerce_float(getattr(config, "long_term_memory_remote_watchdog_interval_s", 1.0)) or 1.0,
    )


def _keepalive_floor_s(config: TwinrConfig) -> float:
    """Return the minimum remote keepalive floor in seconds."""

    return max(
        5.0,
        _coerce_float(getattr(config, "long_term_memory_remote_keepalive_interval_s", 5.0)) or 5.0,
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

    observed_latency_s = _current_latency_s(snapshot)
    watchdog_interval_s = _watchdog_interval_s(config, snapshot)
    keepalive_floor_s = _keepalive_floor_s(config)
    return max(keepalive_floor_s, watchdog_interval_s * 2.0, observed_latency_s * 2.0 + 5.0)


def _recent_sample_cycle_s(snapshot: RemoteMemoryWatchdogSnapshot) -> float | None:
    """Return the largest recent completed-sample cadence in seconds."""

    recent_samples = tuple(_deep_get(snapshot, "recent_samples") or ())
    if len(recent_samples) < 2:
        return None
    parsed_samples = []
    for sample in recent_samples[-4:]:
        captured_at = _parse_utc_timestamp(_deep_get(sample, "captured_at"))
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
    observed_latency_s = _current_latency_s(snapshot)
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

    watchdog_interval_s = _watchdog_interval_s(config, snapshot)
    base_budget_s = max(
        _keepalive_floor_s(config),
        watchdog_interval_s * 3.0,
    )
    timing_slack_s = max(1.0, watchdog_interval_s * 2.0)
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


def _max_allowed_inflight_probe_age_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
    max_sample_age_s: float,
) -> float:
    """Return the maximum bounded age for one still-running deep probe."""

    keepalive_floor_s = _keepalive_floor_s(config)
    timeout_budget_s = max(
        _coerce_float(getattr(config, "chonkydb_timeout_s", 20.0)) or 20.0,
        _coerce_float(getattr(config, "long_term_memory_remote_read_timeout_s", 8.0)) or 8.0,
        _coerce_float(getattr(config, "long_term_memory_remote_write_timeout_s", 15.0)) or 15.0,
    )
    return max(
        max_sample_age_s + keepalive_floor_s,
        timeout_budget_s * 2.0 + keepalive_floor_s,
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
        inflight_probe_budget_s,
        max_sample_age_s + keepalive_floor_s,
    )


def _max_allowed_future_skew_s(
    *,
    config: TwinrConfig,
    snapshot: RemoteMemoryWatchdogSnapshot,
) -> float:
    """Return the tolerated future skew before timestamps are rejected."""

    return max(2.0, _watchdog_interval_s(config, snapshot) * 2.0)


def _resolved_now_wall(now_wall: datetime | None) -> datetime:
    """Return one timezone-aware UTC wall clock."""

    resolved = now_wall or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _age_from_wall_clock_timestamp(
    timestamp: datetime | None,
    *,
    now_wall: datetime,
    max_future_skew_s: float,
) -> tuple[float | None, bool]:
    """Return one bounded wall-clock age and whether the timestamp is too far ahead."""

    if timestamp is None:
        return None, False
    delta_s = (now_wall - timestamp).total_seconds()
    if not math.isfinite(delta_s):
        return None, False
    if delta_s < 0.0:
        if abs(delta_s) <= max_future_skew_s:
            return 0.0, False
        return None, True
    return delta_s, False


def _monotonic_age_s(
    snapshot: RemoteMemoryWatchdogSnapshot,
    *,
    now_monotonic_ns: int,
    paths: tuple[tuple[str, ...], ...],
) -> float | None:
    """Return one artifact age using persisted monotonic timestamps when trusted."""

    snapshot_boot_id = _snapshot_boot_id(snapshot)
    current_boot_id = _current_boot_id()
    if not snapshot_boot_id or not current_boot_id or snapshot_boot_id != current_boot_id:
        return None
    persisted_ns = _coerce_int(_first_present(snapshot, *paths))
    if persisted_ns is None or persisted_ns < 0:
        return None
    delta_ns = now_monotonic_ns - persisted_ns
    if delta_ns < 0:
        return None
    return float(delta_ns) / 1_000_000_000.0


def _probe_inflight_value(snapshot: RemoteMemoryWatchdogSnapshot) -> bool:
    """Return the normalized inflight flag."""

    parsed = _coerce_bool(_deep_get(snapshot, "probe_inflight"))
    if parsed is not None:
        return parsed
    return bool(_deep_get(snapshot, "probe_inflight"))


def _probe_age_value_s(snapshot: RemoteMemoryWatchdogSnapshot, *, now_monotonic_ns: int) -> float | None:
    """Return the normalized probe age in seconds."""

    direct_age_s = _coerce_float(_deep_get(snapshot, "probe_age_s"))
    if direct_age_s is not None and direct_age_s >= 0.0:
        return direct_age_s
    started_age_s = _monotonic_age_s(
        snapshot,
        now_monotonic_ns=now_monotonic_ns,
        paths=_PROBE_STARTED_MONOTONIC_PATHS,
    )
    if started_age_s is not None and started_age_s >= 0.0:
        return started_age_s
    return None


def _assessment_with_current(
    *,
    ready: bool,
    detail: str,
    artifact_path: str,
    snapshot: RemoteMemoryWatchdogSnapshot,
    sample_age_s: float | None,
    max_sample_age_s: float,
    snapshot_stale: bool,
    heartbeat_age_s: float | None = None,
    probe_age_s: float | None = None,
) -> RequiredRemoteWatchdogAssessment:
    """Build one assessment using the current snapshot payload."""

    return RequiredRemoteWatchdogAssessment(
        ready=ready,
        detail=detail,
        artifact_path=artifact_path,
        pid_alive=True,
        sample_age_s=sample_age_s,
        max_sample_age_s=max_sample_age_s,
        sample_status=_deep_get(snapshot, "current", "status"),
        sample_ready=_sample_ready_value(snapshot),
        sample_required=_sample_required_value(snapshot),
        sample_latency_ms=_coerce_float(_deep_get(snapshot, "current", "latency_ms")),
        watchdog_pid=_coerce_int(_deep_get(snapshot, "pid")),
        snapshot_updated_at=_deep_get(snapshot, "updated_at"),
        snapshot_stale=snapshot_stale,
        heartbeat_age_s=heartbeat_age_s,
        heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
        probe_inflight=_probe_inflight_value(snapshot),
        probe_age_s=probe_age_s,
    )


def assess_required_remote_watchdog_snapshot(
    config: TwinrConfig,
    *,
    now_wall: datetime | None = None,
    store: RemoteWatchdogStoreLike | None = None,
) -> RequiredRemoteWatchdogAssessment:
    """Evaluate whether the external remote watchdog currently proves readiness."""

    watchdog_store = store or RemoteMemoryWatchdogStore.from_config(config)
    artifact_path = str(watchdog_store.path)
    artifact_problem = _artifact_security_problem(config, watchdog_store.path)
    snapshot = watchdog_store.load()
    watchdog_pid = _coerce_int(_deep_get(snapshot, "pid")) if snapshot is not None else None
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

    if artifact_problem is not None:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=artifact_problem,
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=_deep_get(snapshot, "current", "status"),
            sample_ready=_sample_ready_value(snapshot),
            sample_required=_sample_required_value(snapshot),
            sample_latency_ms=_coerce_float(_deep_get(snapshot, "current", "latency_ms")),
            watchdog_pid=watchdog_pid,
            snapshot_updated_at=_deep_get(snapshot, "updated_at"),
            snapshot_stale=True,
            heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
            probe_inflight=_probe_inflight_value(snapshot),
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    if _snapshot_current(snapshot) is None:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail="Remote memory watchdog snapshot is malformed: missing current sample.",
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=None,
            sample_ready=None,
            sample_required=None,
            sample_latency_ms=None,
            watchdog_pid=_coerce_int(_deep_get(snapshot, "pid")),
            snapshot_updated_at=_deep_get(snapshot, "updated_at"),
            snapshot_stale=True,
            heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
            probe_inflight=_probe_inflight_value(snapshot),
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    watchdog_pid = _coerce_int(_deep_get(snapshot, "pid"))
    if watchdog_pid is None or watchdog_pid <= 0:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail="Remote memory watchdog snapshot is missing a valid PID.",
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=_deep_get(snapshot, "current", "status"),
            sample_ready=_sample_ready_value(snapshot),
            sample_required=_sample_required_value(snapshot),
            sample_latency_ms=_coerce_float(_deep_get(snapshot, "current", "latency_ms")),
            watchdog_pid=None,
            snapshot_updated_at=_deep_get(snapshot, "updated_at"),
            snapshot_stale=True,
            heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
            probe_inflight=_probe_inflight_value(snapshot),
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    pid_alive = _pid_is_alive(watchdog_pid)
    if not pid_alive:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=f"Remote memory watchdog process {watchdog_pid} is not alive.",
            artifact_path=artifact_path,
            pid_alive=False,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=_deep_get(snapshot, "current", "status"),
            sample_ready=_sample_ready_value(snapshot),
            sample_required=_sample_required_value(snapshot),
            sample_latency_ms=_coerce_float(_deep_get(snapshot, "current", "latency_ms")),
            watchdog_pid=_coerce_int(_deep_get(snapshot, "pid")),
            snapshot_updated_at=_deep_get(snapshot, "updated_at"),
            snapshot_stale=True,
            heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
            probe_inflight=_probe_inflight_value(snapshot),
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    identity_problem = _watchdog_identity_problem(snapshot)
    if identity_problem is not None:
        return RequiredRemoteWatchdogAssessment(
            ready=False,
            detail=identity_problem,
            artifact_path=artifact_path,
            pid_alive=True,
            sample_age_s=None,
            max_sample_age_s=0.0,
            sample_status=_deep_get(snapshot, "current", "status"),
            sample_ready=_sample_ready_value(snapshot),
            sample_required=_sample_required_value(snapshot),
            sample_latency_ms=_coerce_float(_deep_get(snapshot, "current", "latency_ms")),
            watchdog_pid=_coerce_int(_deep_get(snapshot, "pid")),
            snapshot_updated_at=_deep_get(snapshot, "updated_at"),
            snapshot_stale=True,
            heartbeat_updated_at=_deep_get(snapshot, "heartbeat_at"),
            probe_inflight=_probe_inflight_value(snapshot),
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    sample_required = _sample_required_value(snapshot)
    if sample_required is not True:
        return _assessment_with_current(
            ready=False,
            detail="Remote memory watchdog is not enforcing a required remote dependency.",
            artifact_path=artifact_path,
            snapshot=snapshot,
            sample_age_s=None,
            max_sample_age_s=0.0,
            snapshot_stale=False,
            heartbeat_age_s=None,
            probe_age_s=_coerce_float(_deep_get(snapshot, "probe_age_s")),
        )

    resolved_now_wall = _resolved_now_wall(now_wall)
    resolved_now_mono_ns = time.monotonic_ns()
    artifact_mtime = _artifact_mtime_utc(watchdog_store.path)

    captured_at = (
        _parse_utc_timestamp(_deep_get(snapshot, "current", "captured_at"))
        or _parse_utc_timestamp(_deep_get(snapshot, "updated_at"))
        or artifact_mtime
    )
    heartbeat_at = (
        _parse_utc_timestamp(_deep_get(snapshot, "heartbeat_at"))
        or _parse_utc_timestamp(_deep_get(snapshot, "updated_at"))
        or artifact_mtime
    )

    max_sample_age_s = _max_allowed_sample_age_s(config=config, snapshot=snapshot)
    max_future_skew_s = _max_allowed_future_skew_s(config=config, snapshot=snapshot)

    sample_age_s = _monotonic_age_s(
        snapshot,
        now_monotonic_ns=resolved_now_mono_ns,
        paths=_SAMPLE_MONOTONIC_PATHS,
    )
    sample_time_future = False
    if sample_age_s is None:
        sample_age_s, sample_time_future = _age_from_wall_clock_timestamp(
            captured_at,
            now_wall=resolved_now_wall,
            max_future_skew_s=max_future_skew_s,
        )

    heartbeat_age_s = _monotonic_age_s(
        snapshot,
        now_monotonic_ns=resolved_now_mono_ns,
        paths=_HEARTBEAT_MONOTONIC_PATHS,
    )
    heartbeat_time_future = False
    if heartbeat_age_s is None:
        heartbeat_age_s, heartbeat_time_future = _age_from_wall_clock_timestamp(
            heartbeat_at,
            now_wall=resolved_now_wall,
            max_future_skew_s=max_future_skew_s,
        )

    if sample_time_future or heartbeat_time_future:
        detail = _FUTURE_TIMESTAMP_DETAIL
        if sample_time_future and heartbeat_time_future:
            detail = (
                "Remote memory watchdog snapshot has future-dated sample and "
                "heartbeat timestamps."
            )
        elif sample_time_future:
            detail = "Remote memory watchdog snapshot has a future-dated sample timestamp."
        elif heartbeat_time_future:
            detail = "Remote memory watchdog snapshot has a future-dated heartbeat timestamp."
        return _assessment_with_current(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            snapshot=snapshot,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            snapshot_stale=True,
            heartbeat_age_s=heartbeat_age_s,
            probe_age_s=_probe_age_value_s(snapshot, now_monotonic_ns=resolved_now_mono_ns),
        )

    max_steady_state_sample_age_s = _max_allowed_steady_state_sample_age_s(
        config=config,
        snapshot=snapshot,
        max_sample_age_s=max_sample_age_s,
    )
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

    probe_inflight = _probe_inflight_value(snapshot)
    inflight_probe_age_s = _probe_age_value_s(snapshot, now_monotonic_ns=resolved_now_mono_ns)
    sample_status = _sample_status_text(snapshot)
    sample_ready = _sample_ready_value(snapshot)

    inflight_heartbeat_fresh = (
        probe_inflight
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
        not probe_inflight
        and sample_status == "ok"
        and sample_ready is True
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
            f"Remote memory watchdog snapshot is stale (age={sample_age_s!r}s, "
            f"max={max_sample_age_s:.1f}s)."
        )
        return _assessment_with_current(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            snapshot=snapshot,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            snapshot_stale=True,
            heartbeat_age_s=heartbeat_age_s,
            probe_age_s=inflight_probe_age_s,
        )

    if sample_status != "ok" or sample_ready is not True:
        detail = _normalize_text(_deep_get(snapshot, "current", "detail")) or (
            "Remote memory watchdog reports unavailable."
        )
        return _assessment_with_current(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            snapshot=snapshot,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            snapshot_stale=False,
            heartbeat_age_s=heartbeat_age_s,
            probe_age_s=inflight_probe_age_s,
        )

    archive_safe = _probe_attestation_bool(snapshot, "archive_safe")
    health_tier = _probe_attestation_text(snapshot, "health_tier")
    if archive_safe is False or (health_tier is not None and health_tier != "ready"):
        detail = "Remote memory watchdog sample is not archive-safe."
        if health_tier and health_tier != "ready":
            detail = f"Remote memory watchdog sample is {health_tier}, not archive-safe."
        return _assessment_with_current(
            ready=False,
            detail=detail,
            artifact_path=artifact_path,
            snapshot=snapshot,
            sample_age_s=sample_age_s,
            max_sample_age_s=max_sample_age_s,
            snapshot_stale=False,
            heartbeat_age_s=heartbeat_age_s,
            probe_age_s=inflight_probe_age_s,
        )

    return _assessment_with_current(
        ready=True,
        detail="ok",
        artifact_path=artifact_path,
        snapshot=snapshot,
        sample_age_s=sample_age_s,
        max_sample_age_s=max_sample_age_s,
        snapshot_stale=False,
        heartbeat_age_s=heartbeat_age_s,
        probe_age_s=inflight_probe_age_s,
    )


def _watchdog_startup_wait_s(config: TwinrConfig) -> float:
    """Return how long runtime bootstrap may wait for a starting watchdog."""

    return max(
        0.0,
        _coerce_float(getattr(config, "long_term_memory_remote_watchdog_startup_wait_s", 30.0)) or 30.0,
    )


def _watchdog_startup_poll_s(config: TwinrConfig) -> float:
    """Return the bounded poll cadence while the watchdog is still starting."""

    return min(
        1.0,
        max(
            0.1,
            _coerce_float(getattr(config, "long_term_memory_remote_watchdog_interval_s", 1.0)) or 1.0,
        ),
    )


def _default_watchdog_env_file(config: TwinrConfig) -> str:
    """Return the canonical dotenv path for detached watchdog recovery."""

    project_root = Path(_normalize_text(getattr(config, "project_root", ""))).expanduser()
    return str(project_root / ".env")


def _systemd_watchdog_unit(config: TwinrConfig) -> str | None:
    """Return one configured systemd unit for the watchdog, if any."""

    for attr_name in (
        "long_term_memory_remote_watchdog_systemd_unit",
        "remote_memory_watchdog_systemd_unit",
    ):
        unit = _normalize_text(getattr(config, attr_name, ""))
        if unit:
            return unit
    return None


def _start_watchdog_via_systemd(config: TwinrConfig) -> bool:
    """Best-effort request that systemd starts the watchdog unit."""

    unit = _systemd_watchdog_unit(config)
    if not unit:
        return False
    timeout_s = max(
        1.0,
        _coerce_float(getattr(config, "long_term_memory_remote_watchdog_systemd_timeout_s", 10.0)) or 10.0,
    )
    try:
        completed = subprocess.run(
            ["systemctl", "--no-block", "start", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _default_watchdog_recovery_starter(config: TwinrConfig, env_file: str) -> int | None:
    """Best-effort spawn helper for a missing or dead external watchdog."""

    if _start_watchdog_via_systemd(config):
        return 0

    from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

    return ensure_remote_memory_watchdog_process(
        config,
        env_file=env_file,
        emit=lambda _line: None,
    )


def _assessment_warrants_watchdog_recovery(assessment: RequiredRemoteWatchdogAssessment) -> bool:
    """Return whether the current watchdog state is recoverable by respawn."""

    if assessment.ready:
        return False
    detail = _normalize_text(assessment.detail).lower()
    if "snapshot is missing" in detail:
        return True
    if "not alive" in detail:
        return True
    if "different boot id" in detail:
        return True
    if "no longer matches the persisted" in detail:
        return True
    return False


def _assessment_allows_startup_wait(assessment: RequiredRemoteWatchdogAssessment) -> bool:
    """Return whether a non-ready watchdog state still looks like startup."""

    if assessment.ready:
        return False
    status = _normalize_text(assessment.sample_status).lower()
    detail = _normalize_text(assessment.detail).lower()
    if "future-dated" in detail:
        return False
    if "artifact is group/other-writable" in detail:
        return False
    if "artifact path is a symlink" in detail:
        return False
    if status in {"starting", "initializing", "bootstrapping"}:
        return True
    if "watchdog is starting" in detail:
        return True
    if "snapshot is missing" in detail:
        return True
    if "process" in detail and "not alive" in detail:
        return True
    return False


def ensure_required_remote_watchdog_snapshot_ready(
    config: TwinrConfig,
    *,
    store: RemoteWatchdogStoreLike | None = None,
    env_file: str | Path | None = None,
    recovery_starter: WatchdogRecoveryStarter | None = None,
) -> RequiredRemoteWatchdogAssessment:
    """Raise when the external remote watchdog does not prove readiness."""

    watchdog_store = store or RemoteMemoryWatchdogStore.from_config(config)
    resolved_env_file = str(Path(env_file or _default_watchdog_env_file(config)).expanduser())
    starter = recovery_starter or _default_watchdog_recovery_starter

    assessment = assess_required_remote_watchdog_snapshot(config, store=watchdog_store)
    if assessment.ready:
        return assessment

    recovery_requested = False
    recovery_error: Exception | None = None
    if _assessment_warrants_watchdog_recovery(assessment):
        try:
            starter(config, resolved_env_file)
            recovery_requested = True
        except Exception as exc:  # pragma: no cover - defensive integration guard
            recovery_error = exc
        assessment = assess_required_remote_watchdog_snapshot(config, store=watchdog_store)
        if assessment.ready:
            return assessment

    startup_wait_s = _watchdog_startup_wait_s(config)
    if startup_wait_s > 0.0 and (recovery_requested or _assessment_allows_startup_wait(assessment)):
        deadline_ns = time.monotonic_ns() + int(startup_wait_s * 1_000_000_000.0)
        poll_s = _watchdog_startup_poll_s(config)
        while True:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                break
            time.sleep(min(poll_s, remaining_ns / 1_000_000_000.0))
            assessment = assess_required_remote_watchdog_snapshot(config, store=watchdog_store)
            if assessment.ready:
                return assessment
            if not _assessment_allows_startup_wait(assessment):
                break

    if recovery_error is not None:
        raise LongTermRemoteUnavailableError(
            f"{assessment.detail} Recovery starter failed: {type(recovery_error).__name__}: {recovery_error}"
        )
    raise LongTermRemoteUnavailableError(assessment.detail)