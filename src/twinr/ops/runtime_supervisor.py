# CHANGELOG: 2026-03-30
# BUG-1: Handle SIGTERM/SIGINT so the supervisor reaches finally and does not orphan productive child processes on normal service stop/restart.
# BUG-2: Add bounded exponential backoff for crash loops and spawn failures; the previous loop could thrash CPU/I/O on a Raspberry Pi after repeated child exits.
# BUG-3: Restart the streaming loop on real conversation-loop outages even when the worker still owns the singleton lock; previously that failure mode could stay stuck indefinitely.
# SEC-1: Harden adoption of pre-existing worker PIDs with exact cmdline/env-file/project-root validation and race-resistant PID identity tracking.
# IMP-1: Use Linux pidfd when available for race-free liveness and signalling of adopted processes.
# IMP-2: Add optional systemd sd_notify READY/STATUS/WATCHDOG/STOPPING integration for production supervision.
# IMP-3: Start children in isolated POSIX sessions when supported and stop their full process groups during internal restarts.

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

from collections.abc import Callable
from datetime import datetime
import inspect
import json
import logging
import os
from pathlib import Path
import select
import signal
import socket
import threading

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
from twinr.ops.runtime_supervisor_process import (
    ManagedChild as _ManagedChild,
    PidProcessHandle as _PidProcessHandle,
    ProcessHandle,
    default_monotonic as _default_monotonic,
    default_pid_alive as _default_pid_alive,
    default_pid_cmdline as _default_pid_cmdline,
    default_process_factory as _default_process_factory,
    default_sleep as _default_sleep,
    default_utcnow as _default_utcnow,
    parse_utc_timestamp as _parse_utc_timestamp,
    prepend_pythonpath as _prepend_pythonpath,
    python_executable_for_runtime as _python_executable_for_runtime,
)


RUNTIME_SUPERVISOR_ENV_KEY = "TWINR_RUNTIME_SUPERVISOR_ACTIVE"
EXTERNAL_WATCHDOG_ENV_KEY = "TWINR_REMOTE_MEMORY_WATCHDOG_MANAGED_EXTERNALLY"
_SYSTEMD_NOTIFY_SOCKET_ENV_KEY = "NOTIFY_SOCKET"
_SYSTEMD_WATCHDOG_USEC_ENV_KEY = "WATCHDOG_USEC"
_SYSTEMD_WATCHDOG_PID_ENV_KEY = "WATCHDOG_PID"
_DEFAULT_POLL_INTERVAL_S = 1.0
_DEFAULT_WATCHDOG_STARTUP_GRACE_S = 6.0
_DEFAULT_WATCHDOG_STARTUP_TIMEOUT_S = 60.0
_DEFAULT_STREAMING_STARTUP_TIMEOUT_S = 60.0
_DEFAULT_STREAMING_HEALTH_GRACE_S = 10.0
_DEFAULT_MAX_SNAPSHOT_AGE_S = 45.0
_DEFAULT_RESTART_BACKOFF_S = 5.0
_DEFAULT_RESTART_MAX_BACKOFF_S = 60.0
_DEFAULT_STOP_TIMEOUT_S = 10.0
_DEFAULT_STATUS_EMIT_INTERVAL_S = 5.0
_STREAMING_HEALTH_FAILURE_THRESHOLD = 2
_REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE = "Required remote long-term memory is unavailable."
_REQUIRED_REMOTE_RECOVERY_ERROR_FRAGMENTS = (
    "remote long-term memory is temporarily cooling down after recent failures.",
    "remote memory watchdog snapshot",
)
_MAX_EMIT_TEXT_LEN = 240
_MAX_EVENT_TEXT_LEN = 240
_MAX_SYSTEMD_STATUS_LEN = 180

_LOGGER = logging.getLogger(__name__)


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


def _default_emit(line: str) -> None:
    """Print one bounded supervisor line for journald/systemd capture."""

    print(line, flush=True)


def _env_flag(name: str) -> bool:
    """Interpret one conventional environment flag as a boolean."""

    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
        restart_max_backoff_s: Maximum crash-loop backoff between child start
            attempts after repeated exits or spawn failures.
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
        restart_max_backoff_s: float = _DEFAULT_RESTART_MAX_BACKOFF_S,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
        manage_watchdog: bool | None = None,
        external_watchdog_starter: ExternalWatchdogStarterFn = _default_external_watchdog_starter,
    ) -> None:
        self.config = config
        self.env_file = str(env_file)
        self._resolved_env_file = Path(self.env_file).expanduser().resolve()
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
        self.restart_max_backoff_s = max(self.restart_backoff_s, float(restart_max_backoff_s))
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
        self._terminate_requested = False
        self._shutdown_signal: int | None = None
        self._previous_signal_handlers: dict[int, object] = {}
        self._child_next_start_at_monotonic: dict[str, float] = {
            self._watchdog.key: -1.0,
            self._streaming.key: -1.0,
        }
        self._child_failure_streak: dict[str, int] = {
            self._watchdog.key: 0,
            self._streaming.key: 0,
        }
        self._child_last_start_defer_reason: dict[str, str | None] = {
            self._watchdog.key: None,
            self._streaming.key: None,
        }
        self._child_pidfds: dict[str, int | None] = {
            self._watchdog.key: None,
            self._streaming.key: None,
        }
        self._child_start_ticks: dict[str, int | None] = {
            self._watchdog.key: None,
            self._streaming.key: None,
        }
        self._child_process_group_ids: dict[str, int | None] = {
            self._watchdog.key: None,
            self._streaming.key: None,
        }
        self._child_isolated_process_group: dict[str, bool] = {
            self._watchdog.key: False,
            self._streaming.key: False,
        }
        self._systemd_ready = False
        self._systemd_restart_counter_reset = False
        self._systemd_watchdog_interval_s = self._resolve_systemd_watchdog_interval_s()
        self._last_systemd_watchdog_ping_at_monotonic = -1.0
        self._last_systemd_status: str | None = None
        self._last_systemd_status_at_monotonic = -1.0

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the authoritative supervisor loop."""

        resolved_duration = None if duration_s is None else max(0.0, float(duration_s))
        self._run_started_at_monotonic = self._monotonic()
        deadline = None if resolved_duration is None else self._run_started_at_monotonic + resolved_duration
        self._install_signal_handlers()
        self._emit_payload(
            "runtime_supervisor_started",
            env_file=self.env_file,
            manage_watchdog=self.manage_watchdog,
            poll_interval_s=self.poll_interval_s,
            watchdog_startup_grace_s=self.watchdog_startup_grace_s,
            watchdog_startup_timeout_s=self.watchdog_startup_timeout_s,
            streaming_startup_timeout_s=self.streaming_startup_timeout_s,
            streaming_health_grace_s=self.streaming_health_grace_s,
            systemd_notify_active=bool(os.environ.get(_SYSTEMD_NOTIFY_SOCKET_ENV_KEY)),
            pidfd_active=bool(getattr(os, "pidfd_open", None)),
        )
        self._append_event(
            event="runtime_supervisor_started",
            message="Twinr runtime supervisor started.",
            data={
                "env_file": self.env_file,
                "project_root": str(self.project_root),
            },
        )
        self._maybe_notify_systemd_state(
            now_monotonic=self._run_started_at_monotonic,
            assessment=None,
            snapshot=None,
        )
        try:
            while True:
                now_monotonic = self._monotonic()
                if self._terminate_requested:
                    return 0

                assessment = self._assess_watchdog()
                if self.manage_watchdog:
                    if self._ensure_watchdog_running(now_monotonic=now_monotonic, assessment=assessment):
                        assessment = self._assess_watchdog()
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
                snapshot = self._load_snapshot() if self._streaming.is_running() else None
                self._enforce_streaming_health(
                    now_monotonic=now_monotonic,
                    assessment=assessment,
                    snapshot=snapshot,
                )
                self._maybe_reset_child_failure_streak(self._watchdog, now_monotonic)
                self._maybe_reset_child_failure_streak(self._streaming, now_monotonic)
                self._maybe_notify_systemd_state(
                    now_monotonic=now_monotonic,
                    assessment=assessment,
                    snapshot=snapshot,
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
            self._notify_systemd("STOPPING=1", "STATUS=Stopping Twinr runtime supervisor.")
            self._stop_child(self._streaming, reason="supervisor_stop")
            if self.manage_watchdog:
                self._stop_child(self._watchdog, reason="supervisor_stop")
            self._emit_payload("runtime_supervisor_stopped", signal=self._shutdown_signal)
            self._append_event(
                event="runtime_supervisor_stopped",
                message="Twinr runtime supervisor stopped.",
                data={
                    "streaming_restarts": self._streaming.restart_count,
                    "watchdog_restarts": self._watchdog.restart_count,
                    "signal": self._shutdown_signal,
                },
            )
            self._restore_signal_handlers()

    def _ensure_watchdog_running(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> bool:
        self._clear_dead_child(self._watchdog, now_monotonic=now_monotonic)
        if self._watchdog.is_running():
            return False
        if self._maybe_adopt_existing_watchdog_owner(
            now_monotonic=now_monotonic,
            assessment=assessment,
        ):
            return True
        return self._ensure_child_running(self._watchdog, now_monotonic, reason="startup")

    def _ensure_streaming_running(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> None:
        self._clear_dead_child(self._streaming, now_monotonic=now_monotonic)
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
        return self._maybe_adopt_existing_child_owner(
            child=self._streaming,
            owner_pid=owner_pid,
            now_monotonic=now_monotonic,
            expected_flag="--run-streaming-loop",
            reason="existing_lock_owner",
        )

    def _maybe_adopt_existing_watchdog_owner(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment,
    ) -> bool:
        """Adopt one already-running watchdog after supervisor restarts."""

        if self._watchdog.process is not None:
            return False
        owner_pid = getattr(assessment, "watchdog_pid", None)
        if owner_pid is None or owner_pid <= 0:
            return False
        if not assessment.pid_alive:
            return False
        return self._maybe_adopt_existing_child_owner(
            child=self._watchdog,
            owner_pid=owner_pid,
            now_monotonic=now_monotonic,
            expected_flag="--watch-remote-memory",
            reason="existing_watchdog_owner",
        )

    def _maybe_adopt_existing_child_owner(
        self,
        *,
        child: _ManagedChild,
        owner_pid: int,
        now_monotonic: float,
        expected_flag: str,
        reason: str,
    ) -> bool:
        valid, owner_cmdline, detail = self._validate_existing_twinr_owner(
            owner_pid,
            expected_flag=expected_flag,
        )
        if not valid:
            if detail:
                self._emit_payload(
                    "runtime_supervisor_child_adoption_rejected",
                    child=child.key,
                    pid=owner_pid,
                    detail=detail,
                )
            return False

        pidfd = self._pidfd_open(owner_pid)
        start_ticks = self._pid_start_ticks(owner_pid)

        def exact_pid_alive(pid: int) -> bool:
            return self._pid_identity_matches(pid, pidfd=pidfd, start_ticks=start_ticks)

        def exact_pid_signal(pid: int, sig: int) -> None:
            if not self._pid_identity_matches(pid, pidfd=pidfd, start_ticks=start_ticks):
                raise ProcessLookupError(pid)
            self._send_signal_to_exact_pid(pid=pid, sig=sig, pidfd=pidfd)

        self._release_child_identity(child)
        self._child_pidfds[child.key] = pidfd
        self._child_start_ticks[child.key] = start_ticks
        self._capture_process_group_metadata(child.key, owner_pid)

        self._child_next_start_at_monotonic[child.key] = -1.0
        self._child_failure_streak[child.key] = 0
        self._child_last_start_defer_reason[child.key] = None

        self._append_event(
            event="runtime_supervisor_child_adopted",
            message=f"Supervisor adopted existing {child.label}.",
            data={
                "child": child.key,
                "pid": owner_pid,
                "owner_cmdline": " ".join(owner_cmdline[:8]),
                "reason": reason,
            },
        )
        child.process = _PidProcessHandle(
            owner_pid,
            pid_alive=exact_pid_alive,
            pid_signal=exact_pid_signal,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )
        child.started_at_monotonic = now_monotonic
        child.started_at_utc = self._utcnow()
        child.last_restart_at_monotonic = now_monotonic
        child.clear_health_issue()
        self._emit_payload(
            "runtime_supervisor_child_adopted",
            child=child.key,
            pid=owner_pid,
            owner_cmdline=" ".join(owner_cmdline[:8]),
            reason=reason,
        )
        return True

    def _validate_existing_twinr_owner(
        self,
        pid: int,
        *,
        expected_flag: str,
    ) -> tuple[bool, tuple[str, ...], str | None]:
        if pid <= 0:
            return False, (), "Invalid PID for adoption."
        if not self.pid_alive(pid):
            return False, (), "PID is not alive."
        try:
            cmdline = self.pid_cmdline(pid)
        except Exception as exc:
            return False, (), compact_text(f"pid_cmdline failed: {type(exc).__name__}: {exc}", limit=160)
        if expected_flag not in cmdline:
            return False, cmdline, f"Expected flag {expected_flag} missing."
        # BREAKING: the supervisor now only adopts existing workers whose
        # command line still matches the canonical Twinr runtime invocation.
        if "-m" not in cmdline or "twinr" not in cmdline:
            return False, cmdline, "Expected Twinr module marker missing."
        env_file_arg = self._extract_cli_option_value(cmdline, "--env-file")
        if env_file_arg is None:
            return False, cmdline, "Expected --env-file missing."
        try:
            resolved_env_file_arg = Path(env_file_arg).expanduser().resolve()
        except Exception:
            return False, cmdline, "Unable to resolve adopted process env-file."
        if resolved_env_file_arg != self._resolved_env_file:
            return False, cmdline, "Adopted process env-file does not match supervisor env-file."
        cwd = self._pid_cwd(pid)
        if cwd is None:
            return False, cmdline, "Unable to inspect adopted process cwd."
        if cwd != self.project_root:
            return False, cmdline, "Adopted process cwd does not match project root."
        return True, cmdline, None

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
    ) -> bool:
        self._clear_dead_child(child, now_monotonic=now_monotonic)
        if child.is_running():
            return False
        if not self._can_attempt_child_start(child, now_monotonic, reason=reason):
            return False
        return self._start_child(child, now_monotonic, reason=reason)

    def _clear_dead_child(self, child: _ManagedChild, *, now_monotonic: float) -> None:
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
        self._note_child_failure(
            child,
            now_monotonic=now_monotonic,
            reason=f"child_exit:{exit_code}",
        )
        self._release_child_identity(child)

    def _start_child(
        self,
        child: _ManagedChild,
        now_monotonic: float,
        *,
        reason: str,
    ) -> bool:
        argv = self._child_command(child.key)
        env = self._child_environment()
        kwargs = self._process_factory_kwargs()
        try:
            process = self.process_factory(argv, cwd=self.project_root, env=env, **kwargs)
        except TypeError as exc:
            if kwargs and self._kwargs_typeerror_is_unsupported_kwarg(exc):
                try:
                    process = self.process_factory(argv, cwd=self.project_root, env=env)
                except Exception as nested_exc:
                    return self._record_child_start_failure(
                        child,
                        now_monotonic=now_monotonic,
                        reason=reason,
                        exc=nested_exc,
                    )
            else:
                return self._record_child_start_failure(
                    child,
                    now_monotonic=now_monotonic,
                    reason=reason,
                    exc=exc,
                )
        except Exception as exc:
            return self._record_child_start_failure(
                child,
                now_monotonic=now_monotonic,
                reason=reason,
                exc=exc,
            )

        child.process = process
        child.started_at_monotonic = now_monotonic
        child.started_at_utc = self._utcnow()
        child.last_restart_at_monotonic = now_monotonic
        child.restart_count += 1
        child.clear_health_issue()

        self._child_next_start_at_monotonic[child.key] = -1.0
        self._child_last_start_defer_reason[child.key] = None
        self._capture_child_identity(child, process)

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
        return True

    def _record_child_start_failure(
        self,
        child: _ManagedChild,
        *,
        now_monotonic: float,
        reason: str,
        exc: Exception,
    ) -> bool:
        detail = compact_text(f"{type(exc).__name__}: {exc}", limit=200)
        self._emit_payload(
            "runtime_supervisor_child_start_failed",
            child=child.key,
            reason=reason,
            detail=detail,
        )
        self._append_event(
            event="runtime_supervisor_child_start_failed",
            message=f"Supervisor failed to start {child.label}.",
            level="error",
            data={
                "child": child.key,
                "reason": reason,
                "detail": detail,
            },
        )
        child.process = None
        child.clear_health_issue()
        self._release_child_identity(child)
        self._note_child_failure(child, now_monotonic=now_monotonic, reason=f"spawn_failure:{reason}")
        return False

    def _stop_child(self, child: _ManagedChild, *, reason: str) -> None:
        process = child.process
        if process is None:
            self._release_child_identity(child)
            return
        if process.poll() is not None:
            child.process = None
            child.clear_health_issue()
            self._release_child_identity(child)
            return
        self._emit_payload(
            "runtime_supervisor_child_stopping",
            child=child.key,
            pid=getattr(process, "pid", None),
            reason=reason,
        )
        try:
            self._terminate_child_tree(child, process)
            process.wait(timeout=self.stop_timeout_s)
        except Exception:
            _LOGGER.warning(
                "Runtime supervisor terminate/wait failed for child %s; escalating to kill.",
                child.key,
                exc_info=True,
            )
            try:
                self._kill_child_tree(child, process)
                process.wait(timeout=self.stop_timeout_s)
            except Exception:
                _LOGGER.warning(
                    "Runtime supervisor kill/wait failed for child %s.",
                    child.key,
                    exc_info=True,
                )
        child.process = None
        child.clear_health_issue()
        self._release_child_identity(child)

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
        self._child_next_start_at_monotonic[child.key] = -1.0
        self._child_last_start_defer_reason[child.key] = None
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
            watchdog_pid=getattr(assessment, "watchdog_pid", None),
            sample_status=assessment.sample_status,
            snapshot_stale=assessment.snapshot_stale,
        )
        self._append_event(
            event="runtime_supervisor_external_watchdog_recovery_requested",
            message="Supervisor requested external remote-memory watchdog recovery.",
            level="warn",
            data={
                "detail": assessment.detail,
                "watchdog_pid": getattr(assessment, "watchdog_pid", None),
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
        snapshot: RuntimeSnapshot | None,
    ) -> None:
        if not self._streaming.is_running():
            return
        if self._streaming.age_s(now_monotonic) < self.streaming_health_grace_s:
            self._streaming.clear_health_issue()
            return

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
        return self._current_streaming_child_owns_lock()

    def _current_streaming_child_owns_lock(self) -> bool:
        process = self._streaming.process
        if process is None or process.poll() is not None:
            return False
        current_pid = getattr(process, "pid", None)
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
        if current_pid is not None and getattr(assessment, "watchdog_pid", None) == current_pid:
            return True
        started_at_utc = self._watchdog.started_at_utc
        if started_at_utc is None:
            return False
        updated_at = _parse_utc_timestamp(getattr(assessment, "snapshot_updated_at", None))
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
        if runtime_status == "error" and self._snapshot_error_is_required_remote(snapshot):
            # Required-remote failures can strand the child inside the top-level
            # runtime error hold, which never self-recovers. Once the watchdog
            # is healthy again, recycle the child promptly instead of waiting
            # for generic snapshot staleness.
            return "required_remote_recovered"
        age_s = max(0.0, (self._utcnow() - updated_at).total_seconds())
        if age_s > self.max_snapshot_age_s:
            return "runtime_snapshot_stale"
        return None

    def _snapshot_error_is_required_remote(self, snapshot: RuntimeSnapshot) -> bool:
        error_text = " ".join(str(getattr(snapshot, "error_message", "") or "").split()).strip().lower()
        if not error_text:
            return False
        configured_public = str(
            getattr(
                self.config,
                "long_term_memory_remote_required_public_error_message",
                _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE,
            )
            or _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE
        ).strip()
        if configured_public and configured_public.lower() in error_text:
            return True
        return any(fragment in error_text for fragment in _REQUIRED_REMOTE_RECOVERY_ERROR_FRAGMENTS)

    def _streaming_health_issue(self, health: TwinrSystemHealth) -> str | None:
        # The supervisor already gates health enforcement on the current child
        # owning the singleton streaming lock and refreshing the runtime
        # snapshot. Once that authoritative contract is satisfied, auxiliary
        # ops process inventories must not tear down the healthy child because
        # transient `ps`/count drift would otherwise trigger false restarts.
        # BUGFIX: still treat an explicitly non-running conversation loop as a
        # real speech-path outage even while the process keeps the lock.
        if self._current_streaming_child_owns_lock():
            conversation = self._service(health, "conversation_loop")
            if conversation is not None and not conversation.running:
                return "conversation_loop_unhealthy"
            return None
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
        prime_user_session_audio_env(
            configured_runtime_dir=getattr(self.config, "display_wayland_runtime_dir", None)
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"))
        env[RUNTIME_SUPERVISOR_ENV_KEY] = "1"
        # BREAKING: child workers no longer inherit systemd notification
        # variables; only the top-level supervisor owns readiness/watchdog I/O.
        env.pop(_SYSTEMD_NOTIFY_SOCKET_ENV_KEY, None)
        env.pop(_SYSTEMD_WATCHDOG_USEC_ENV_KEY, None)
        env.pop(_SYSTEMD_WATCHDOG_PID_ENV_KEY, None)
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
                message=compact_text(message, limit=_MAX_EVENT_TEXT_LEN),
                level=level,
                data=self._normalize_mapping(data),
            )
        except Exception:
            _LOGGER.warning(
                "Failed to append runtime supervisor ops event %s.",
                event,
                exc_info=True,
            )
            return

    def _emit_payload(self, event: str, **data: object) -> None:
        payload = {"event": event, **self._normalize_mapping(data)}
        self.emit(json.dumps(payload, ensure_ascii=True, sort_keys=True))

    def _normalize_mapping(self, data: dict[str, object] | None) -> dict[str, object] | None:
        if data is None:
            return None
        return {str(key): self._normalize_payload_value(value) for key, value in data.items()}

    def _normalize_payload_value(self, value: object) -> object:
        if isinstance(value, str):
            return compact_text(value, limit=_MAX_EMIT_TEXT_LEN)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [self._normalize_payload_value(item) for item in value]
        if isinstance(value, list):
            return [self._normalize_payload_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._normalize_payload_value(item) for key, item in value.items()}
        return value

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_shutdown_signal)
            except Exception:
                _LOGGER.debug("Failed to install runtime supervisor signal handler for %s.", signum, exc_info=True)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                _LOGGER.debug("Failed to restore runtime supervisor signal handler for %s.", signum, exc_info=True)
        self._previous_signal_handlers.clear()

    def _handle_shutdown_signal(self, signum: int, frame: object) -> None:
        del frame
        self._shutdown_signal = signum
        self._terminate_requested = True

    def _can_attempt_child_start(self, child: _ManagedChild, now_monotonic: float, *, reason: str) -> bool:
        next_start_at = self._child_next_start_at_monotonic.get(child.key, -1.0)
        if next_start_at <= 0.0 or now_monotonic >= next_start_at:
            self._child_last_start_defer_reason[child.key] = None
            return True
        remaining_s = max(0.0, next_start_at - now_monotonic)
        detail = f"Backoff active for {child.label}; retry in {remaining_s:.1f}s."
        if self._child_last_start_defer_reason.get(child.key) != detail:
            self._child_last_start_defer_reason[child.key] = detail
            self._emit_payload(
                "runtime_supervisor_child_start_deferred",
                child=child.key,
                reason=reason,
                retry_in_s=round(remaining_s, 2),
                detail=detail,
            )
        return False

    def _note_child_failure(
        self,
        child: _ManagedChild,
        *,
        now_monotonic: float,
        reason: str,
    ) -> None:
        streak = int(self._child_failure_streak.get(child.key, 0)) + 1
        self._child_failure_streak[child.key] = streak
        delay_s = self._restart_delay_for_streak(streak)
        self._child_next_start_at_monotonic[child.key] = now_monotonic + delay_s
        self._child_last_start_defer_reason[child.key] = f"{reason}:{delay_s:.1f}s"
        self._emit_payload(
            "runtime_supervisor_child_retry_scheduled",
            child=child.key,
            reason=reason,
            failure_streak=streak,
            retry_in_s=round(delay_s, 2),
        )

    def _maybe_reset_child_failure_streak(self, child: _ManagedChild, now_monotonic: float) -> None:
        if not child.is_running():
            return
        if self._child_failure_streak.get(child.key, 0) <= 0:
            return
        stable_after_s = max(15.0, self.restart_backoff_s * 3.0)
        if child.age_s(now_monotonic) < stable_after_s:
            return
        self._child_failure_streak[child.key] = 0
        self._child_next_start_at_monotonic[child.key] = -1.0
        self._child_last_start_defer_reason[child.key] = None
        self._emit_payload(
            "runtime_supervisor_child_retry_reset",
            child=child.key,
            stable_after_s=round(stable_after_s, 2),
        )

    def _restart_delay_for_streak(self, streak: int) -> float:
        if self.restart_backoff_s <= 0.0:
            return 0.0
        exponent = max(0, int(streak) - 1)
        delay_s = self.restart_backoff_s * (2 ** exponent)
        return min(self.restart_max_backoff_s, delay_s)

    def _process_factory_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if os.name != "posix":
            return kwargs
        # BREAKING: when the process factory supports it, child workers are
        # launched in isolated sessions so internal restarts can reap their
        # whole descendant trees instead of only the direct child PID.
        if self._process_factory_supports_kwarg("start_new_session"):
            kwargs["start_new_session"] = True
        return kwargs

    def _process_factory_supports_kwarg(self, name: str) -> bool:
        try:
            signature = inspect.signature(self.process_factory)
        except (TypeError, ValueError):
            return self.process_factory is _default_process_factory
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return name in signature.parameters

    @staticmethod
    def _kwargs_typeerror_is_unsupported_kwarg(exc: TypeError) -> bool:
        message = str(exc)
        return "unexpected keyword argument" in message or "got an unexpected keyword argument" in message

    def _capture_child_identity(self, child: _ManagedChild, process: ProcessHandle) -> None:
        pid = getattr(process, "pid", None)
        if pid is None or pid <= 0:
            self._release_child_identity(child)
            return
        self._release_child_identity(child)
        self._child_pidfds[child.key] = self._pidfd_open(pid)
        self._child_start_ticks[child.key] = self._pid_start_ticks(pid)
        self._capture_process_group_metadata(child.key, pid)

    def _capture_process_group_metadata(self, child_key: str, pid: int) -> None:
        pgid: int | None = None
        isolated = False
        if os.name == "posix":
            try:
                pgid = os.getpgid(pid)
            except Exception:
                pgid = None
            if pgid is not None:
                try:
                    isolated = pgid == pid and pgid != os.getpgrp()
                except Exception:
                    isolated = pgid == pid
        self._child_process_group_ids[child_key] = pgid
        self._child_isolated_process_group[child_key] = isolated

    def _release_child_identity(self, child: _ManagedChild) -> None:
        pidfd = self._child_pidfds.get(child.key)
        if pidfd is not None:
            try:
                os.close(pidfd)
            except Exception:
                _LOGGER.debug("Failed to close pidfd for child %s.", child.key, exc_info=True)
        self._child_pidfds[child.key] = None
        self._child_start_ticks[child.key] = None
        self._child_process_group_ids[child.key] = None
        self._child_isolated_process_group[child.key] = False

    def _terminate_child_tree(self, child: _ManagedChild, process: ProcessHandle) -> None:
        if self._can_signal_child_process_group(child, process):
            os.killpg(self._child_process_group_ids[child.key] or 0, signal.SIGTERM)
            return
        process.terminate()

    def _kill_child_tree(self, child: _ManagedChild, process: ProcessHandle) -> None:
        if self._can_signal_child_process_group(child, process):
            os.killpg(self._child_process_group_ids[child.key] or 0, signal.SIGKILL)
            return
        process.kill()

    def _can_signal_child_process_group(self, child: _ManagedChild, process: ProcessHandle) -> bool:
        if os.name != "posix":
            return False
        pid = getattr(process, "pid", None)
        if pid is None or pid <= 0:
            return False
        if not self._child_isolated_process_group.get(child.key, False):
            return False
        pgid = self._child_process_group_ids.get(child.key)
        if pgid is None or pgid <= 0 or pgid != pid:
            return False
        return self._pid_identity_matches(
            pid,
            pidfd=self._child_pidfds.get(child.key),
            start_ticks=self._child_start_ticks.get(child.key),
        )

    @staticmethod
    def _extract_cli_option_value(argv: tuple[str, ...], option: str) -> str | None:
        for index, token in enumerate(argv):
            if token == option and index + 1 < len(argv):
                return argv[index + 1]
            if token.startswith(option + "="):
                return token.split("=", 1)[1]
        return None

    def _pid_cwd(self, pid: int) -> Path | None:
        try:
            return Path(os.readlink(f"/proc/{pid}/cwd")).expanduser().resolve()
        except Exception:
            return None

    def _pid_start_ticks(self, pid: int) -> int | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        end = raw.rfind(")")
        if end < 0:
            return None
        fields = raw[end + 2 :].split()
        if len(fields) <= 19:
            return None
        try:
            return int(fields[19])
        except Exception:
            return None

    def _pidfd_open(self, pid: int) -> int | None:
        pidfd_open = getattr(os, "pidfd_open", None)
        if pidfd_open is None or pid <= 0:
            return None
        flags = getattr(os, "PIDFD_NONBLOCK", 0)
        try:
            return int(pidfd_open(pid, flags))
        except TypeError:
            try:
                return int(pidfd_open(pid))
            except Exception:
                return None
        except Exception:
            return None

    def _pidfd_is_running(self, pidfd: int) -> bool:
        try:
            readable, _, _ = select.select([pidfd], [], [], 0.0)
        except Exception:
            return False
        return not bool(readable)

    def _pid_identity_matches(
        self,
        pid: int,
        *,
        pidfd: int | None,
        start_ticks: int | None,
    ) -> bool:
        if pid <= 0:
            return False
        if pidfd is not None and not self._pidfd_is_running(pidfd):
            return False
        if start_ticks is not None:
            current_start_ticks = self._pid_start_ticks(pid)
            if current_start_ticks is None or current_start_ticks != start_ticks:
                return False
            return True
        return self.pid_alive(pid)

    def _send_signal_to_exact_pid(self, *, pid: int, sig: int, pidfd: int | None) -> None:
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd is not None and pidfd_send_signal is not None:
            pidfd_send_signal(pidfd, sig)
            return
        self.pid_signal(pid, sig)

    def _resolve_systemd_watchdog_interval_s(self) -> float | None:
        notify_socket = str(os.environ.get(_SYSTEMD_NOTIFY_SOCKET_ENV_KEY, "") or "").strip()
        if not notify_socket:
            return None
        watchdog_usec_raw = str(os.environ.get(_SYSTEMD_WATCHDOG_USEC_ENV_KEY, "") or "").strip()
        if not watchdog_usec_raw:
            return None
        watchdog_pid_raw = str(os.environ.get(_SYSTEMD_WATCHDOG_PID_ENV_KEY, "") or "").strip()
        if watchdog_pid_raw:
            try:
                watchdog_pid = int(watchdog_pid_raw)
            except Exception:
                return None
            if watchdog_pid not in (0, os.getpid()):
                return None
        try:
            watchdog_usec = int(watchdog_usec_raw)
        except Exception:
            return None
        if watchdog_usec <= 0:
            return None
        return max(1.0, watchdog_usec / 1_000_000.0 / 2.0)

    def _notify_systemd(self, *assignments: str) -> None:
        notify_socket = str(os.environ.get(_SYSTEMD_NOTIFY_SOCKET_ENV_KEY, "") or "").strip()
        if not notify_socket:
            return
        state = "\n".join(part for part in assignments if part).strip()
        if not state:
            return
        address = ("\0" + notify_socket[1:]) if notify_socket.startswith("@") else notify_socket
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(state.encode("utf-8", errors="replace"), address)
        except Exception as exc:
            self._emit_payload(
                "runtime_supervisor_systemd_notify_failed",
                detail=compact_text(f"{type(exc).__name__}: {exc}", limit=180),
            )

    def _maybe_notify_systemd_state(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment | None,
        snapshot: RuntimeSnapshot | None,
    ) -> None:
        status = self._systemd_status_text(now_monotonic=now_monotonic, assessment=assessment, snapshot=snapshot)
        should_emit_status = (
            status != self._last_systemd_status
            or self._last_systemd_status_at_monotonic < 0.0
            or (now_monotonic - self._last_systemd_status_at_monotonic) >= _DEFAULT_STATUS_EMIT_INTERVAL_S
        )
        assignments: list[str] = []
        ready = self._systemd_should_report_ready(snapshot)
        if ready and not self._systemd_ready:
            self._systemd_ready = True
            assignments.append("READY=1")
        if status and should_emit_status:
            assignments.append(f"STATUS={status}")
        if ready and not self._systemd_restart_counter_reset:
            assignments.append("RESTART_RESET=1")
            self._systemd_restart_counter_reset = True
        if ready and self._systemd_watchdog_interval_s is not None:
            if (
                self._last_systemd_watchdog_ping_at_monotonic < 0.0
                or (now_monotonic - self._last_systemd_watchdog_ping_at_monotonic) >= self._systemd_watchdog_interval_s
            ):
                assignments.append("WATCHDOG=1")
                self._last_systemd_watchdog_ping_at_monotonic = now_monotonic
        if not assignments:
            return
        self._notify_systemd(*assignments)
        if status and should_emit_status:
            self._last_systemd_status = status
            self._last_systemd_status_at_monotonic = now_monotonic

    def _systemd_should_report_ready(self, snapshot: RuntimeSnapshot | None) -> bool:
        if not self._streaming.is_running():
            return False
        if not self._current_streaming_child_owns_lock():
            return False
        return self._streaming_snapshot_has_progress(snapshot)

    def _systemd_status_text(
        self,
        *,
        now_monotonic: float,
        assessment: RequiredRemoteWatchdogAssessment | None,
        snapshot: RuntimeSnapshot | None,
    ) -> str:
        if self._terminate_requested:
            return "Stopping Twinr runtime supervisor."
        if not self._streaming.is_running():
            retry_in_s = self._child_start_retry_remaining_s(self._streaming, now_monotonic)
            if retry_in_s is not None:
                return compact_text(
                    f"Streaming loop restarting after failure; next retry in {retry_in_s:.1f}s.",
                    limit=_MAX_SYSTEMD_STATUS_LEN,
                )
            if self._last_streaming_gate_reason:
                return compact_text(
                    f"Starting; streaming gate blocked: {self._last_streaming_gate_reason}",
                    limit=_MAX_SYSTEMD_STATUS_LEN,
                )
            return "Starting; waiting for streaming loop."
        if not self._current_streaming_child_owns_lock():
            return "Streaming loop launched; waiting for runtime lock."
        if not self._streaming_snapshot_has_progress(snapshot):
            return "Streaming loop owns runtime lock; waiting for fresh runtime snapshot."
        if assessment is None:
            return "Running; watchdog status unknown."
        if assessment.ready:
            return "Running; streaming loop healthy and remote-memory watchdog ready."
        detail = compact_text(
            assessment.detail or "remote memory watchdog degraded",
            limit=100,
        )
        return compact_text(
            f"Running in degraded mode; {detail}",
            limit=_MAX_SYSTEMD_STATUS_LEN,
        )

    def _child_start_retry_remaining_s(self, child: _ManagedChild, now_monotonic: float) -> float | None:
        next_start_at = self._child_next_start_at_monotonic.get(child.key, -1.0)
        if next_start_at <= 0.0 or now_monotonic >= next_start_at:
            return None
        return max(0.0, next_start_at - now_monotonic)


__all__ = ["RUNTIME_SUPERVISOR_ENV_KEY", "TwinrRuntimeSupervisor"]
