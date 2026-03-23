"""Supervise the productive Pi runtime under one authoritative owner.

The productive Raspberry Pi runtime should not depend on three unrelated
`systemd` units or on opportunistic companion spawning inside the conversation
process. This module owns one top-level supervisor loop that starts the remote
memory watchdog and the streaming loop as child processes, delays streaming
startup briefly while the watchdog comes up, and restarts the streaming child
when core voice/runtime health or runtime-snapshot freshness disappears.
Display degradation stays visible through ops health, but it must not tear down
the speech path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Protocol
import json
import logging
import os
import signal
import subprocess
import sys
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.agent.workflows.required_remote_snapshot import (
    RequiredRemoteWatchdogAssessment,
    assess_required_remote_watchdog_snapshot,
)
from twinr.ops.events import TwinrOpsEventStore, compact_text
from twinr.ops.health import ServiceHealth, TwinrSystemHealth, collect_system_health
from twinr.ops.locks import loop_lock_owner
from twinr.ops.remote_memory_watchdog import (
    RemoteMemoryWatchdogStore,
    build_remote_memory_watchdog_bootstrap_snapshot,
)
from twinr.ops.runtime_env import prime_user_session_audio_env


RUNTIME_SUPERVISOR_ENV_KEY = "TWINR_RUNTIME_SUPERVISOR_ACTIVE"
EXTERNAL_WATCHDOG_ENV_KEY = "TWINR_REMOTE_MEMORY_WATCHDOG_MANAGED_EXTERNALLY"
_DEFAULT_POLL_INTERVAL_S = 1.0
_DEFAULT_WATCHDOG_STARTUP_GRACE_S = 6.0
_DEFAULT_WATCHDOG_STARTUP_TIMEOUT_S = 60.0
_DEFAULT_STREAMING_STARTUP_TIMEOUT_S = 60.0
_DEFAULT_STREAMING_HEALTH_GRACE_S = 10.0
_DEFAULT_MAX_SNAPSHOT_AGE_S = 45.0
_DEFAULT_RESTART_BACKOFF_S = 5.0
_DEFAULT_STOP_TIMEOUT_S = 10.0
_STREAMING_HEALTH_FAILURE_THRESHOLD = 2

_LOGGER = logging.getLogger(__name__)


class ProcessHandle(Protocol):
    """Describe the minimal subprocess interface the supervisor needs."""

    pid: int

    def poll(self) -> int | None:
        """Return the child exit code or ``None`` while it is still running."""

    def terminate(self) -> None:
        """Ask the child process to shut down cleanly."""

    def kill(self) -> None:
        """Force-kill the child process."""

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the child process to exit."""


ProcessFactory = Callable[..., ProcessHandle]
EmitFn = Callable[[str], None]
MonotonicFn = Callable[[], float]
SleepFn = Callable[[float], None]
HealthCollectorFn = Callable[..., TwinrSystemHealth]
WatchdogAssessmentFn = Callable[[TwinrConfig], RequiredRemoteWatchdogAssessment]
UtcNowFn = Callable[[], datetime]
LoopOwnerFn = Callable[[TwinrConfig, str], int | None]
PidAliveFn = Callable[[int], bool]
PidSignalFn = Callable[[int, int], None]
PidCmdlineFn = Callable[[int], tuple[str, ...]]
ExternalWatchdogStarterFn = Callable[[TwinrConfig, str, EmitFn], int | None]


@dataclass(slots=True)
class _ManagedChild:
    """Track one supervisor-owned child process."""

    key: str
    label: str
    process: ProcessHandle | None = None
    started_at_monotonic: float = -1.0
    started_at_utc: datetime | None = None
    last_restart_at_monotonic: float = 0.0
    restart_count: int = 0
    health_issue: str | None = None
    health_issue_streak: int = 0

    def is_running(self) -> bool:
        """Report whether the child still appears alive."""

        return self.process is not None and self.process.poll() is None

    def age_s(self, now_monotonic: float) -> float:
        """Return the current child age in seconds."""

        if self.started_at_monotonic < 0.0:
            return 0.0
        return max(0.0, now_monotonic - self.started_at_monotonic)

    def clear_health_issue(self) -> None:
        """Reset the consecutive unhealthy-health counters."""

        self.health_issue = None
        self.health_issue_streak = 0

    def record_health_issue(self, issue: str) -> int:
        """Increment the unhealthy-health streak for one reason."""

        if self.health_issue == issue:
            self.health_issue_streak += 1
        else:
            self.health_issue = issue
            self.health_issue_streak = 1
        return self.health_issue_streak


def _default_emit(line: str) -> None:
    """Print one bounded supervisor line for journald/systemd capture."""

    print(line, flush=True)


def _default_monotonic() -> float:
    """Return the current monotonic clock value."""

    return time.monotonic()


def _default_sleep(seconds: float) -> None:
    """Sleep for the requested duration."""

    time.sleep(seconds)


def _default_utcnow() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _env_flag(name: str) -> bool:
    """Interpret one conventional environment flag as a boolean."""

    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _default_pid_alive(pid: int) -> bool:
    """Return whether one local PID currently appears alive."""

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _default_pid_cmdline(pid: int) -> tuple[str, ...]:
    """Return one best-effort command line tuple for a local PID."""

    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ()
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return tuple(parts)


class _PidProcessHandle:
    """Wrap an already-running PID in the minimal child-process protocol."""

    def __init__(
        self,
        pid: int,
        *,
        pid_alive: PidAliveFn,
        pid_signal: PidSignalFn,
        monotonic: MonotonicFn,
        sleep: SleepFn,
    ) -> None:
        self.pid = int(pid)
        self._pid_alive = pid_alive
        self._pid_signal = pid_signal
        self._monotonic = monotonic
        self._sleep = sleep

    def poll(self) -> int | None:
        return None if self._pid_alive(self.pid) else 0

    def terminate(self) -> None:
        self._pid_signal(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        self._pid_signal(self.pid, signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else self._monotonic() + max(0.0, float(timeout))
        while self._pid_alive(self.pid):
            if deadline is not None and self._monotonic() >= deadline:
                raise TimeoutError(f"PID {self.pid} did not exit before timeout.")
            self._sleep(0.05)
        return 0


def _default_process_factory(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
) -> ProcessHandle:
    """Launch one child process for the supervisor."""

    return subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        close_fds=True,
    )


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse one persisted UTC timestamp from a runtime snapshot."""

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


def _python_executable_for_runtime(config: TwinrConfig) -> str:
    """Resolve the preferred Python executable for one runtime tree."""

    candidate = Path(config.project_root).resolve() / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _prepend_pythonpath(existing: str | None) -> str:
    """Ensure ``src`` stays on ``PYTHONPATH`` for child commands."""

    text = str(existing or "").strip()
    if not text:
        return "src"
    parts = [part for part in text.split(os.pathsep) if part]
    if "src" not in parts:
        parts.insert(0, "src")
    return os.pathsep.join(parts)


def _default_external_watchdog_starter(
    config: TwinrConfig,
    env_file: str,
    emit: EmitFn,
) -> int | None:
    """Best-effort self-heal for externally managed watchdog ownership."""

    from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

    return ensure_remote_memory_watchdog_process(config, env_file=env_file, emit=emit)


class TwinrRuntimeSupervisor:
    """Own the productive streaming loop plus remote watchdog.

    The supervisor starts both productive child processes from the canonical
    Twinr project root. It prefers to let the watchdog artifact become ready
    before considering startup healthy, but after a short grace period it will
    also start the streaming loop so the display companion can continue to
    surface runtime failure state to the user.

    Args:
        config: Runtime configuration rooted in the productive project tree.
        env_file: Dotenv path passed to both child processes.
        emit: Best-effort telemetry sink used for journald/systemd.
        event_store: Optional ops-event store override.
        snapshot_store: Optional runtime snapshot store override.
        health_collector: Optional system-health collector override.
        watchdog_assessor: Optional external-watchdog assessment override.
        process_factory: Optional child-process launcher override for tests.
        monotonic: Optional monotonic clock override.
        sleep: Optional sleep primitive override.
        utcnow: Optional UTC wall-clock override.
        poll_interval_s: Main supervisor loop interval.
        watchdog_startup_grace_s: How long to wait before starting streaming
            without a ready watchdog snapshot.
        watchdog_startup_timeout_s: Maximum time the watchdog may stay in
            pre-sample startup before the supervisor treats watchdog startup
            itself as stalled.
        streaming_startup_timeout_s: Maximum time a streaming child may stay in
            pre-lock/pre-snapshot startup before the supervisor treats startup
            itself as stalled.
        streaming_health_grace_s: How long to ignore startup transients before
            enforcing snapshot/runtime health on the streaming child.
        max_snapshot_age_s: Maximum allowed runtime-snapshot age before a
            streaming restart is triggered.
        restart_backoff_s: Minimum delay between restarts of the same child.
        stop_timeout_s: Graceful shutdown timeout for child termination.
        manage_watchdog: When false, the supervisor consumes an externally
            managed watchdog artifact/service instead of spawning and recycling
            its own watchdog child. This keeps the remote-memory watchdog warm
            across supervisor restarts on the Pi.
    """

    def __init__(
        self,
        *,
        config: TwinrConfig,
        env_file: str | Path,
        emit: EmitFn = _default_emit,
        event_store: TwinrOpsEventStore | None = None,
        snapshot_store: RuntimeSnapshotStore | None = None,
        health_collector: HealthCollectorFn = collect_system_health,
        watchdog_assessor: WatchdogAssessmentFn = assess_required_remote_watchdog_snapshot,
        process_factory: ProcessFactory = _default_process_factory,
        monotonic: MonotonicFn = _default_monotonic,
        sleep: SleepFn = _default_sleep,
        utcnow: UtcNowFn = _default_utcnow,
        loop_owner: LoopOwnerFn = loop_lock_owner,
        pid_alive: PidAliveFn = _default_pid_alive,
        pid_signal: PidSignalFn = os.kill,
        pid_cmdline: PidCmdlineFn = _default_pid_cmdline,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        watchdog_startup_grace_s: float = _DEFAULT_WATCHDOG_STARTUP_GRACE_S,
        watchdog_startup_timeout_s: float = _DEFAULT_WATCHDOG_STARTUP_TIMEOUT_S,
        streaming_startup_timeout_s: float = _DEFAULT_STREAMING_STARTUP_TIMEOUT_S,
        streaming_health_grace_s: float = _DEFAULT_STREAMING_HEALTH_GRACE_S,
        max_snapshot_age_s: float = _DEFAULT_MAX_SNAPSHOT_AGE_S,
        restart_backoff_s: float = _DEFAULT_RESTART_BACKOFF_S,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
        manage_watchdog: bool | None = None,
        external_watchdog_starter: ExternalWatchdogStarterFn = _default_external_watchdog_starter,
    ) -> None:
        self.config = config
        self.env_file = str(env_file)
        self.emit = emit
        self.event_store = event_store or TwinrOpsEventStore.from_config(config)
        self.snapshot_store = snapshot_store or RuntimeSnapshotStore(config.runtime_state_path)
        self.health_collector = health_collector
        self.watchdog_assessor = watchdog_assessor
        self.process_factory = process_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._utcnow = utcnow
        self.loop_owner = loop_owner
        self.pid_alive = pid_alive
        self.pid_signal = pid_signal
        self.pid_cmdline = pid_cmdline
        self.poll_interval_s = max(0.1, float(poll_interval_s))
        self.watchdog_startup_grace_s = max(0.0, float(watchdog_startup_grace_s))
        self.watchdog_startup_timeout_s = max(1.0, float(watchdog_startup_timeout_s))
        self.streaming_startup_timeout_s = max(1.0, float(streaming_startup_timeout_s))
        self.streaming_health_grace_s = max(0.0, float(streaming_health_grace_s))
        self.max_snapshot_age_s = max(1.0, float(max_snapshot_age_s))
        self.restart_backoff_s = max(0.0, float(restart_backoff_s))
        self.stop_timeout_s = max(0.1, float(stop_timeout_s))
        self.manage_watchdog = (
            not _env_flag(EXTERNAL_WATCHDOG_ENV_KEY)
            if manage_watchdog is None
            else bool(manage_watchdog)
        )
        self.external_watchdog_starter = external_watchdog_starter
        self.project_root = Path(config.project_root).expanduser().resolve()
        self.remote_watchdog_store = RemoteMemoryWatchdogStore.from_config(config)
        self._watchdog = _ManagedChild(
            key="remote-memory-watchdog",
            label="remote memory watchdog",
        )
        self._streaming = _ManagedChild(
            key="streaming-loop",
            label="streaming loop",
        )
        self._last_streaming_gate_reason: str | None = None
        self._run_started_at_monotonic: float = -1.0
        self._last_external_watchdog_recovery_at_monotonic: float = -1.0

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the authoritative supervisor loop."""

        resolved_duration = None if duration_s is None else max(0.0, float(duration_s))
        self._run_started_at_monotonic = self._monotonic()
        deadline = None if resolved_duration is None else self._run_started_at_monotonic + resolved_duration
        self._emit_payload(
            "runtime_supervisor_started",
            env_file=self.env_file,
            manage_watchdog=self.manage_watchdog,
            poll_interval_s=self.poll_interval_s,
            watchdog_startup_grace_s=self.watchdog_startup_grace_s,
            watchdog_startup_timeout_s=self.watchdog_startup_timeout_s,
            streaming_startup_timeout_s=self.streaming_startup_timeout_s,
            streaming_health_grace_s=self.streaming_health_grace_s,
        )
        self._append_event(
            event="runtime_supervisor_started",
            message="Twinr runtime supervisor started.",
            data={
                "env_file": self.env_file,
                "project_root": str(self.project_root),
            },
        )
        try:
            while True:
                now_monotonic = self._monotonic()
                if self.manage_watchdog:
                    self._ensure_child_running(self._watchdog, now_monotonic, reason="startup")
                assessment = self._assess_watchdog()
                if self.manage_watchdog:
                    self._maybe_restart_watchdog_for_health(
                        now_monotonic=now_monotonic,
                        assessment=assessment,
                    )
                else:
                    self._maybe_recover_external_watchdog(
                        now_monotonic=now_monotonic,
                        assessment=assessment,
                    )
                self._ensure_streaming_running(
                    now_monotonic=now_monotonic,
                    assessment=assessment,
                )
                self._enforce_streaming_health(
                    now_monotonic=now_monotonic,
                    assessment=assessment,
                )
                if deadline is not None and now_monotonic >= deadline:
                    return 0
                sleep_s = self.poll_interval_s
                if deadline is not None:
                    sleep_s = min(sleep_s, max(0.0, deadline - now_monotonic))
                if sleep_s > 0.0:
                    self._sleep(sleep_s)
        except KeyboardInterrupt:
            return 0
        finally:
            self._stop_child(self._streaming, reason="supervisor_stop")
            if self.manage_watchdog:
                self._stop_child(self._watchdog, reason="supervisor_stop")
            self._emit_payload("runtime_supervisor_stopped")
            self._append_event(
                event="runtime_supervisor_stopped",
                message="Twinr runtime supervisor stopped.",
                data={
                    "streaming_restarts": self._streaming.restart_count,
                    "watchdog_restarts": self._watchdog.restart_count,
                },
            )

    def _ensure_streaming_running(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> None:
        self._clear_dead_child(self._streaming)
        if self._streaming.is_running():
            self._last_streaming_gate_reason = None
            return
        if self._maybe_adopt_existing_streaming_owner(now_monotonic=now_monotonic):
            self._last_streaming_gate_reason = None
            return

        gate_reason = self._streaming_gate_reason(now_monotonic=now_monotonic, assessment=assessment)
        if gate_reason is not None:
            if gate_reason != self._last_streaming_gate_reason:
                self._emit_payload(
                    "runtime_supervisor_streaming_gate_blocked",
                    detail=gate_reason,
                )
            self._last_streaming_gate_reason = gate_reason
            return

        self._last_streaming_gate_reason = None
        start_reason = "watchdog_ready" if assessment.ready else "watchdog_startup_grace_elapsed"
        self._ensure_child_running(self._streaming, now_monotonic, reason=start_reason)

    def _maybe_adopt_existing_streaming_owner(self, *, now_monotonic: float) -> bool:
        """Adopt one already-running streaming owner after supervisor restarts."""

        if self._streaming.process is not None:
            return False
        owner_pid = self.loop_owner(self.config, "streaming-loop")
        if owner_pid is None or owner_pid <= 0:
            return False
        if not self.pid_alive(owner_pid):
            return False
        owner_cmdline = self.pid_cmdline(owner_pid)
        if "--run-streaming-loop" not in owner_cmdline:
            return False
        self._streaming.process = _PidProcessHandle(
            owner_pid,
            pid_alive=self.pid_alive,
            pid_signal=self.pid_signal,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )
        self._streaming.started_at_monotonic = now_monotonic
        self._streaming.started_at_utc = self._utcnow()
        self._streaming.last_restart_at_monotonic = now_monotonic
        self._streaming.clear_health_issue()
        self._emit_payload(
            "runtime_supervisor_child_adopted",
            child=self._streaming.key,
            pid=owner_pid,
            owner_cmdline=" ".join(owner_cmdline[:6]),
            reason="existing_lock_owner",
        )
        self._append_event(
            event="runtime_supervisor_child_adopted",
            message=f"Supervisor adopted existing {self._streaming.label}.",
            data={
                "child": self._streaming.key,
                "pid": owner_pid,
                "owner_cmdline": " ".join(owner_cmdline[:6]),
                "reason": "existing_lock_owner",
            },
        )
        return True

    def _streaming_gate_reason(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> str | None:
        if assessment.ready:
            return None
        if self._watchdog_startup_age_s(now_monotonic) >= self.watchdog_startup_grace_s:
            return None
        return compact_text(
            assessment.detail or "Waiting for remote memory watchdog startup.",
            limit=160,
        )

    def _watchdog_startup_age_s(self, now_monotonic: float) -> float:
        """Return the effective startup age for the active watchdog owner."""

        if self.manage_watchdog:
            return self._watchdog.age_s(now_monotonic)
        if self._run_started_at_monotonic < 0.0:
            return 0.0
        return max(0.0, now_monotonic - self._run_started_at_monotonic)

    def _ensure_child_running(
        self,
        child: _ManagedChild,
        now_monotonic: float,
        *,
        reason: str,
    ) -> None:
        self._clear_dead_child(child)
        if child.is_running():
            return
        self._start_child(child, now_monotonic, reason=reason)

    def _clear_dead_child(self, child: _ManagedChild) -> None:
        process = child.process
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        self._emit_payload(
            "runtime_supervisor_child_exited",
            child=child.key,
            exit_code=exit_code,
            pid=getattr(process, "pid", None),
        )
        self._append_event(
            event="runtime_supervisor_child_exited",
            message=f"Supervisor child {child.label} exited.",
            level="warn",
            data={
                "child": child.key,
                "exit_code": exit_code,
                "pid": getattr(process, "pid", None),
            },
        )
        child.process = None
        child.clear_health_issue()

    def _start_child(
        self,
        child: _ManagedChild,
        now_monotonic: float,
        *,
        reason: str,
    ) -> None:
        argv = self._child_command(child.key)
        env = self._child_environment()
        process = self.process_factory(argv, cwd=self.project_root, env=env)
        child.process = process
        child.started_at_monotonic = now_monotonic
        child.started_at_utc = self._utcnow()
        child.last_restart_at_monotonic = now_monotonic
        child.restart_count += 1
        child.clear_health_issue()
        self._emit_payload(
            "runtime_supervisor_child_started",
            child=child.key,
            pid=getattr(process, "pid", None),
            reason=reason,
            argv=" ".join(argv[2:]),
        )
        self._append_event(
            event="runtime_supervisor_child_started",
            message=f"Supervisor started {child.label}.",
            data={
                "child": child.key,
                "pid": getattr(process, "pid", None),
                "reason": reason,
            },
        )
        if child.key == "remote-memory-watchdog":
            try:
                bootstrap = build_remote_memory_watchdog_bootstrap_snapshot(
                    self.config,
                    pid=getattr(process, "pid", 0) or 0,
                    artifact_path=self.remote_watchdog_store.path,
                    started_at=(
                        child.started_at_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                        if child.started_at_utc is not None
                        else None
                    ),
                )
                self.remote_watchdog_store.save(bootstrap)
            except Exception as exc:
                self._emit_payload(
                    "runtime_supervisor_watchdog_bootstrap_failed",
                    detail=compact_text(f"{type(exc).__name__}: {exc}", limit=200),
                )

    def _stop_child(self, child: _ManagedChild, *, reason: str) -> None:
        process = child.process
        if process is None:
            return
        if process.poll() is not None:
            child.process = None
            child.clear_health_issue()
            return
        self._emit_payload(
            "runtime_supervisor_child_stopping",
            child=child.key,
            pid=getattr(process, "pid", None),
            reason=reason,
        )
        try:
            process.terminate()
            process.wait(timeout=self.stop_timeout_s)
        except Exception:
            _LOGGER.warning(
                "Runtime supervisor terminate/wait failed for child %s; escalating to kill.",
                child.key,
                exc_info=True,
            )
            try:
                process.kill()
                process.wait(timeout=self.stop_timeout_s)
            except Exception:
                _LOGGER.warning(
                    "Runtime supervisor kill/wait failed for child %s.",
                    child.key,
                    exc_info=True,
                )
        child.process = None
        child.clear_health_issue()

    def _restart_child(
        self,
        child: _ManagedChild,
        now_monotonic: float,
        *,
        reason: str,
    ) -> None:
        if now_monotonic - child.last_restart_at_monotonic < self.restart_backoff_s:
            return
        self._append_event(
            event="runtime_supervisor_child_restart_requested",
            message=f"Supervisor is restarting {child.label}.",
            level="warn",
            data={
                "child": child.key,
                "reason": reason,
                "pid": getattr(child.process, "pid", None) if child.process is not None else None,
            },
        )
        self._stop_child(child, reason=reason)
        self._start_child(child, now_monotonic, reason=reason)

    def _maybe_restart_watchdog_for_health(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> None:
        if not self._watchdog.is_running():
            return
        if self._watchdog.age_s(now_monotonic) < self.watchdog_startup_grace_s:
            return
        if assessment.ready:
            return
        if not self._watchdog_startup_has_progress(assessment):
            if self._watchdog.age_s(now_monotonic) < self.watchdog_startup_timeout_s:
                return
            self._restart_child(self._watchdog, now_monotonic, reason="watchdog_startup_stalled")
            return
        if not assessment.pid_alive:
            self._restart_child(self._watchdog, now_monotonic, reason="watchdog_pid_dead")
            return
        if not self._streaming.is_running():
            return
        if assessment.sample_status is None:
            self._restart_child(self._watchdog, now_monotonic, reason="watchdog_snapshot_missing")
            return
        if assessment.sample_age_s is None:
            self._restart_child(self._watchdog, now_monotonic, reason="watchdog_snapshot_invalid")
            return
        if assessment.snapshot_stale:
            self._restart_child(self._watchdog, now_monotonic, reason="watchdog_snapshot_stale")

    def _maybe_recover_external_watchdog(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> None:
        """Start a replacement watchdog when the external owner is clearly absent."""

        if self.manage_watchdog or assessment.ready or assessment.pid_alive:
            return
        if (now_monotonic - self._last_external_watchdog_recovery_at_monotonic) < self.restart_backoff_s:
            return
        self._last_external_watchdog_recovery_at_monotonic = now_monotonic
        self._emit_payload(
            "runtime_supervisor_external_watchdog_recovery_requested",
            detail=assessment.detail,
            watchdog_pid=assessment.watchdog_pid,
            sample_status=assessment.sample_status,
            snapshot_stale=assessment.snapshot_stale,
        )
        self._append_event(
            event="runtime_supervisor_external_watchdog_recovery_requested",
            message="Supervisor requested external remote-memory watchdog recovery.",
            level="warn",
            data={
                "detail": assessment.detail,
                "watchdog_pid": assessment.watchdog_pid,
                "sample_status": assessment.sample_status,
                "snapshot_stale": assessment.snapshot_stale,
            },
        )
        try:
            owner_pid = self.external_watchdog_starter(self.config, self.env_file, self.emit)
        except Exception as exc:
            detail = compact_text(f"{type(exc).__name__}: {exc}", limit=200)
            self._emit_payload(
                "runtime_supervisor_external_watchdog_recovery_failed",
                detail=detail,
            )
            self._append_event(
                event="runtime_supervisor_external_watchdog_recovery_failed",
                message="Supervisor failed to recover the external remote-memory watchdog.",
                level="error",
                data={"detail": detail},
            )
            return
        self._emit_payload(
            "runtime_supervisor_external_watchdog_recovery_started",
            owner_pid=owner_pid,
        )
        self._append_event(
            event="runtime_supervisor_external_watchdog_recovery_started",
            message="Supervisor started or adopted an external remote-memory watchdog.",
            data={"owner_pid": owner_pid},
        )

    def _enforce_streaming_health(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> None:
        if not self._streaming.is_running():
            return
        if self._streaming.age_s(now_monotonic) < self.streaming_health_grace_s:
            self._streaming.clear_health_issue()
            return

        snapshot = self._load_snapshot()
        if not self._streaming_startup_has_progress(snapshot):
            if self._streaming.age_s(now_monotonic) < self.streaming_startup_timeout_s:
                self._streaming.clear_health_issue()
                return
            self._restart_streaming_if_persistent(
                now_monotonic=now_monotonic,
                issue="streaming_startup_stalled",
            )
            return
        if not self._streaming_snapshot_has_progress(snapshot):
            if self._streaming.age_s(now_monotonic) < self.streaming_startup_timeout_s:
                self._streaming.clear_health_issue()
                return
            self._restart_streaming_if_persistent(
                now_monotonic=now_monotonic,
                issue="streaming_startup_stalled",
            )
            return
        snapshot_issue = self._snapshot_health_issue(snapshot, assessment=assessment)
        if snapshot_issue is not None:
            self._restart_streaming_if_persistent(
                now_monotonic=now_monotonic,
                issue=snapshot_issue,
            )
            return

        health = self._collect_health(snapshot)
        if health is None:
            self._streaming.clear_health_issue()
            return

        streaming_issue = self._streaming_health_issue(health)
        if streaming_issue is None:
            self._streaming.clear_health_issue()
            return
        self._restart_streaming_if_persistent(
            now_monotonic=now_monotonic,
            issue=streaming_issue,
        )

    def _restart_streaming_if_persistent(
        self,
        *,
        now_monotonic: float,
        issue: str,
    ) -> None:
        streak = self._streaming.record_health_issue(issue)
        if streak < _STREAMING_HEALTH_FAILURE_THRESHOLD:
            return
        self._restart_child(self._streaming, now_monotonic, reason=issue)
        self._streaming.clear_health_issue()

    def _streaming_startup_has_progress(self, snapshot: RuntimeSnapshot | None) -> bool:
        del snapshot
        process = self._streaming.process
        current_pid = getattr(process, "pid", None) if process is not None else None
        if current_pid is None:
            return False
        owner_pid = self.loop_owner(self.config, "streaming-loop")
        return owner_pid == current_pid

    def _streaming_snapshot_has_progress(self, snapshot: RuntimeSnapshot | None) -> bool:
        if snapshot is None:
            return False
        updated_at = _parse_utc_timestamp(getattr(snapshot, "updated_at", None))
        if updated_at is None:
            return False
        started_at_utc = self._streaming.started_at_utc
        if started_at_utc is None:
            return False
        return updated_at >= started_at_utc

    def _watchdog_startup_has_progress(self, assessment: RequiredRemoteWatchdogAssessment) -> bool:
        if assessment.sample_status is None:
            return False
        process = self._watchdog.process
        current_pid = getattr(process, "pid", None) if process is not None else None
        if current_pid is not None and assessment.watchdog_pid == current_pid:
            return True
        started_at_utc = self._watchdog.started_at_utc
        if started_at_utc is None:
            return False
        updated_at = _parse_utc_timestamp(assessment.snapshot_updated_at)
        if updated_at is None:
            return False
        return updated_at >= started_at_utc

    def _snapshot_health_issue(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> str | None:
        if snapshot is None:
            return "runtime_snapshot_missing"
        updated_at = _parse_utc_timestamp(getattr(snapshot, "updated_at", None))
        if updated_at is None:
            return "runtime_snapshot_missing"
        runtime_status = str(getattr(snapshot, "status", "") or "").strip().lower()
        if runtime_status == "waiting":
            return None
        if runtime_status == "error" and not assessment.ready:
            return None
        age_s = max(0.0, (self._utcnow() - updated_at).total_seconds())
        if age_s > self.max_snapshot_age_s:
            return "runtime_snapshot_stale"
        return None

    def _streaming_health_issue(self, health: TwinrSystemHealth) -> str | None:
        conversation = self._service(health, "conversation_loop")
        if conversation is None or not conversation.running or conversation.count != 1:
            return "conversation_loop_unhealthy"
        return None

    @staticmethod
    def _service(health: TwinrSystemHealth, key: str) -> ServiceHealth | None:
        for service in getattr(health, "services", ()) or ():
            if getattr(service, "key", None) == key:
                return service
        return None

    def _assess_watchdog(self) -> RequiredRemoteWatchdogAssessment:
        try:
            return self.watchdog_assessor(self.config)
        except Exception as exc:
            detail = compact_text(f"{type(exc).__name__}: {exc}", limit=160)
            self._emit_payload(
                "runtime_supervisor_watchdog_assessment_failed",
                detail=detail,
            )
            return RequiredRemoteWatchdogAssessment(
                ready=False,
                detail=detail or "Remote memory watchdog assessment failed.",
                artifact_path="unknown",
                pid_alive=False,
                sample_age_s=None,
                max_sample_age_s=0.0,
                sample_status=None,
                sample_ready=None,
                sample_required=None,
                sample_latency_ms=None,
                snapshot_stale=True,
            )

    def _load_snapshot(self) -> RuntimeSnapshot | None:
        try:
            return self.snapshot_store.load()
        except Exception as exc:
            self._emit_payload(
                "runtime_supervisor_snapshot_load_failed",
                detail=compact_text(f"{type(exc).__name__}: {exc}", limit=160),
            )
            return None

    def _collect_health(self, snapshot: RuntimeSnapshot | None) -> TwinrSystemHealth | None:
        try:
            return self.health_collector(self.config, snapshot=snapshot)
        except Exception as exc:
            self._emit_payload(
                "runtime_supervisor_health_collect_failed",
                detail=compact_text(f"{type(exc).__name__}: {exc}", limit=160),
            )
            return None

    def _child_command(self, key: str) -> tuple[str, ...]:
        python_executable = _python_executable_for_runtime(self.config)
        if key == "remote-memory-watchdog":
            return (
                python_executable,
                "-u",
                "-m",
                "twinr",
                "--env-file",
                self.env_file,
                "--watch-remote-memory",
            )
        if key == "streaming-loop":
            return (
                python_executable,
                "-u",
                "-m",
                "twinr",
                "--env-file",
                self.env_file,
                "--run-streaming-loop",
            )
        raise ValueError(f"Unsupported supervisor child: {key}")

    def _child_environment(self) -> dict[str, str]:
        # Prime the supervisor process first so freshly spawned children inherit
        # the detached user-session audio sockets even before their own runtime
        # bootstrap runs.
        prime_user_session_audio_env()
        env = dict(os.environ)
        env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"))
        env[RUNTIME_SUPERVISOR_ENV_KEY] = "1"
        return env

    def _append_event(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, object] | None = None,
    ) -> None:
        try:
            self.event_store.append(
                event=event,
                message=message,
                level=level,
                data=data,
            )
        except Exception:
            _LOGGER.warning(
                "Failed to append runtime supervisor ops event %s.",
                event,
                exc_info=True,
            )
            return

    def _emit_payload(self, event: str, **data: object) -> None:
        payload = {"event": event, **data}
        self.emit(json.dumps(payload, ensure_ascii=True, sort_keys=True))


__all__ = ["RUNTIME_SUPERVISOR_ENV_KEY", "TwinrRuntimeSupervisor"]
