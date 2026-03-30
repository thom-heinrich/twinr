# CHANGELOG: 2026-03-30
# BUG-1: Fixed false "still running" detection for zombie PIDs, which could stall waits and restarts.
# BUG-2: Fixed session-group cleanup after the session leader exited; helper descendants such as gpiomon are now still tracked and terminable.
# BUG-3: Fixed PID-reuse / getpgid-killpg races that could previously signal the wrong process or raise spuriously during shutdown.
# SEC-1: Hardened direct process signaling with best-effort pidfd identity binding and /proc start-time checks to avoid killing unrelated reused PIDs.
# SEC-2: Hardened child spawning by resolving argv[0] deterministically instead of relying on PATH/cwd-dependent exec lookup.
# IMP-1: Upgraded supervisor liveness semantics from "leader PID only" to "leader plus surviving dedicated process-group members".
# IMP-2: Added Linux /proc-backed process identity tracking and optional pidfd support while preserving the existing drop-in public API.

"""Process and timestamp helpers for the runtime supervisor.

The productive runtime can spawn helper descendants such as ``gpiomon`` that
must die with the owning streaming child during supervisor restarts. This
module therefore launches supervisor-owned children in their own sessions and
tracks that dedicated process group, so liveness checks and shutdowns continue
to work even after the session leader itself has already exited.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import errno
import os
import select
import signal
import subprocess
import sys
import time
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig


_DEAD_PROCESS_STATES = frozenset({"Z", "X", "x"})
_PROC_ROOT = Path("/proc")


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


@dataclass(frozen=True, slots=True)
class _ProcStatSnapshot:
    """Small parsed subset of ``/proc/<pid>/stat`` used for identity checks."""

    state: str
    pgrp: int
    session: int
    starttime_ticks: int


def _read_proc_stat_snapshot(pid: int) -> _ProcStatSnapshot | None:
    """Parse the identity-relevant fields from ``/proc/<pid>/stat``."""

    try:
        raw = (_PROC_ROOT / str(int(pid)) / "stat").read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    head, separator, tail = raw.rpartition(") ")
    if not separator:
        return None
    prefix, _, _command = head.partition(" (")
    if prefix.strip() != str(int(pid)):
        return None

    fields = tail.split()
    if len(fields) < 20:
        return None

    try:
        return _ProcStatSnapshot(
            state=fields[0],
            pgrp=int(fields[2]),
            session=int(fields[3]),
            starttime_ticks=int(fields[19]),
        )
    except (TypeError, ValueError, IndexError):
        return None


def _group_signal_exists(
    pgid: int,
    *,
    group_signal,
) -> bool:
    """Return whether one process group currently appears to contain members."""

    try:
        group_signal(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


def _group_has_live_members(
    pgid: int,
    *,
    proc_stat_reader,
) -> bool:
    """Return whether one process group has any non-zombie members left."""

    try:
        entries = _PROC_ROOT.iterdir()
    except OSError:
        return False

    target = int(pgid)
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        snapshot = proc_stat_reader(int(name))
        if snapshot is None:
            continue
        if snapshot.pgrp != target:
            continue
        if snapshot.state in _DEAD_PROCESS_STATES:
            continue
        return True
    return False


def _safe_close_fd(fd: int | None) -> None:
    """Best-effort close for one optional file descriptor."""

    if fd is None or fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        return


def _try_open_pidfd(pid: int) -> int | None:
    """Open one pidfd when the kernel and Python build support it."""

    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        return None
    try:
        return int(opener(int(pid), 0))
    except OSError as exc:
        if exc.errno in {
            getattr(errno, "ENOSYS", 38),
            getattr(errno, "EINVAL", 22),
            getattr(errno, "EPERM", 1),
            getattr(errno, "ENODEV", 19),
            getattr(errno, "EOPNOTSUPP", 95),
        }:
            return None
        return None


def _pidfd_has_terminated(pidfd: int | None) -> bool:
    """Return whether one pidfd reports process termination."""

    if pidfd is None:
        return False
    try:
        readable, _, _ = select.select([pidfd], [], [], 0.0)
    except (OSError, ValueError):
        return True
    return bool(readable)


def _signal_pidfd(
    pidfd: int | None,
    sig: int,
    *,
    pidfd_send_signal,
) -> bool:
    """Signal one exact process via pidfd if that interface is available."""

    if pidfd is None or pidfd_send_signal is None:
        return False
    try:
        pidfd_send_signal(int(pidfd), int(sig))
    except ProcessLookupError:
        return True
    except OSError as exc:
        if exc.errno in {
            getattr(errno, "ESRCH", 3),
            getattr(errno, "EBADF", 9),
        }:
            return True
        if exc.errno in {
            getattr(errno, "ENOSYS", 38),
            getattr(errno, "EINVAL", 22),
            getattr(errno, "EOPNOTSUPP", 95),
        }:
            return False
        raise
    return True


def _resolve_executable(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    """Resolve one executable path deterministically for supervisor spawns."""

    text = str(command).strip()
    if not text:
        raise ValueError("Child argv[0] must not be empty.")

    # Keep explicit path semantics but make them deterministic relative to cwd.
    if os.sep in text or (os.altsep and os.altsep in text):
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = Path(cwd) / candidate
        return str(candidate)

    search_path = str(env.get("PATH") or os.defpath)
    current_dir = Path(cwd)
    for raw_entry in search_path.split(os.pathsep):
        entry = raw_entry or "."
        base = Path(entry)
        if not base.is_absolute():
            base = current_dir / base
        candidate = base / text
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(f"Executable {text!r} was not found on PATH for cwd={cwd!s}.")


def _managed_group_still_running(
    pid: int,
    *,
    managed_pgid: int | None,
    expected_starttime_ticks: int | None,
    proc_stat_reader,
    pid_alive,
    pidfd: int | None,
    pid_getpgid,
    group_signal,
) -> bool:
    """Return whether the original dedicated session still has live members."""

    if managed_pgid is None or expected_starttime_ticks is None:
        return False

    snapshot = proc_stat_reader(int(pid))
    if snapshot is not None:
        if snapshot.starttime_ticks != int(expected_starttime_ticks):
            return False
        if snapshot.pgrp != int(managed_pgid) or snapshot.session != int(managed_pgid):
            return False
        if snapshot.state in _DEAD_PROCESS_STATES:
            return _group_has_live_members(int(managed_pgid), proc_stat_reader=proc_stat_reader)
        return True

    if pidfd is not None and not _pidfd_has_terminated(pidfd):
        try:
            pgid = int(pid_getpgid(int(pid)))
        except Exception:
            return False
        return pgid == int(managed_pgid)

    if pid_alive(int(pid)):
        return False

    if not _group_signal_exists(int(managed_pgid), group_signal=group_signal):
        return False
    return _group_has_live_members(int(managed_pgid), proc_stat_reader=proc_stat_reader)


def _signal_managed_group(
    pid: int,
    sig: int,
    *,
    managed_pgid: int | None,
    expected_starttime_ticks: int | None,
    proc_stat_reader,
    pid_alive,
    pidfd: int | None,
    pid_getpgid,
    group_signal,
) -> bool:
    """Signal one stored dedicated process group when that is still safe."""

    if managed_pgid is None or expected_starttime_ticks is None:
        return False

    snapshot = proc_stat_reader(int(pid))
    if snapshot is not None:
        if (
            snapshot.starttime_ticks != int(expected_starttime_ticks)
            or snapshot.pgrp != int(managed_pgid)
            or snapshot.session != int(managed_pgid)
        ):
            return False
        try:
            group_signal(int(managed_pgid), int(sig))
        except ProcessLookupError:
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return True
            raise
        return True

    if pidfd is not None and not _pidfd_has_terminated(pidfd):
        try:
            pgid = int(pid_getpgid(int(pid)))
        except Exception:
            return False
        if pgid != int(managed_pgid):
            return False
        try:
            group_signal(int(managed_pgid), int(sig))
        except ProcessLookupError:
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return True
            raise
        return True

    if pid_alive(int(pid)):
        return False

    if not _group_has_live_members(int(managed_pgid), proc_stat_reader=proc_stat_reader):
        return True

    try:
        group_signal(int(managed_pgid), int(sig))
    except ProcessLookupError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return True
        raise
    return True


def _signal_exact_pid(
    pid: int,
    sig: int,
    *,
    expected_starttime_ticks: int | None,
    proc_stat_reader,
    pid_alive,
    pid_signal,
    pidfd: int | None,
    pidfd_send_signal,
) -> None:
    """Signal one exact process identity without trusting a reused PID."""

    if _signal_pidfd(pidfd, sig, pidfd_send_signal=pidfd_send_signal):
        return

    if expected_starttime_ticks is not None:
        snapshot = proc_stat_reader(int(pid))
        if snapshot is None:
            if pid_alive(int(pid)):
                return
            return
        if snapshot.starttime_ticks != int(expected_starttime_ticks):
            return
        if snapshot.state in _DEAD_PROCESS_STATES:
            return

    try:
        pid_signal(int(pid), int(sig))
    except ProcessLookupError:
        return
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return
        raise


@dataclass(slots=True)
class ManagedChild:
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


def default_monotonic() -> float:
    """Return the current monotonic clock value."""

    return time.monotonic()


def default_sleep(seconds: float) -> None:
    """Sleep for the requested duration."""

    time.sleep(seconds)


def default_utcnow() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def default_pid_alive(pid: int) -> bool:
    """Return whether one local PID currently appears alive.

    Zombies are treated as not alive because they have already exited and
    cannot execute further user-space code.
    """

    snapshot = _read_proc_stat_snapshot(int(pid))
    if snapshot is not None:
        return snapshot.state not in _DEAD_PROCESS_STATES

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def default_pid_cmdline(pid: int) -> tuple[str, ...]:
    """Return one best-effort command line tuple for a local PID."""

    try:
        raw = (_PROC_ROOT / str(int(pid)) / "cmdline").read_bytes()
    except OSError:
        return ()
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return tuple(parts)


class PidProcessHandle:
    """Wrap an already-running PID in the minimal child-process protocol."""

    def __init__(
        self,
        pid: int,
        *,
        pid_alive,
        pid_signal,
        pid_getpgid=os.getpgid,
        group_signal=os.killpg,
        monotonic,
        sleep,
    ) -> None:
        self.pid = int(pid)
        self._pid_alive = pid_alive
        self._pid_signal = pid_signal
        self._pid_getpgid = pid_getpgid
        self._group_signal = group_signal
        self._monotonic = monotonic
        self._sleep = sleep

        self._proc_stat_reader = _read_proc_stat_snapshot
        self._pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        self._pidfd = _try_open_pidfd(self.pid)

        snapshot = self._proc_stat_reader(self.pid)
        self._expected_starttime_ticks = None if snapshot is None else snapshot.starttime_ticks
        self._managed_pgid = None
        if (
            snapshot is not None
            and snapshot.pgrp == self.pid
            and snapshot.session == self.pid
        ):
            self._managed_pgid = self.pid

    def __del__(self) -> None:
        _safe_close_fd(getattr(self, "_pidfd", None))

    def _leader_poll(self) -> int | None:
        if self._pidfd is not None and _pidfd_has_terminated(self._pidfd):
            return 0

        if self._expected_starttime_ticks is not None:
            snapshot = self._proc_stat_reader(self.pid)
            if snapshot is not None:
                if snapshot.starttime_ticks != self._expected_starttime_ticks:
                    return 0
                if snapshot.state in _DEAD_PROCESS_STATES:
                    return 0
                return None

        return None if self._pid_alive(self.pid) else 0

    def poll(self) -> int | None:
        # BREAKING: for dedicated supervisor-owned sessions, poll() now stays
        # None until the managed process group has no live helper descendants.
        leader_exit = self._leader_poll()
        if self._managed_pgid is not None and _managed_group_still_running(
            self.pid,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return None
        if leader_exit is not None:
            _safe_close_fd(self._pidfd)
            self._pidfd = None
        return leader_exit

    def terminate(self) -> None:
        if _signal_managed_group(
            self.pid,
            signal.SIGTERM,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return
        _signal_exact_pid(
            self.pid,
            signal.SIGTERM,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pid_signal=self._pid_signal,
            pidfd=self._pidfd,
            pidfd_send_signal=self._pidfd_send_signal,
        )

    def kill(self) -> None:
        if _signal_managed_group(
            self.pid,
            signal.SIGKILL,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return
        _signal_exact_pid(
            self.pid,
            signal.SIGKILL,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pid_signal=self._pid_signal,
            pidfd=self._pidfd,
            pidfd_send_signal=self._pidfd_send_signal,
        )

    def wait(self, timeout: float | None = None) -> int:
        # BREAKING: wait() now waits for the managed dedicated process group to
        # drain, not only for the original session leader PID to disappear.
        deadline = None if timeout is None else self._monotonic() + max(0.0, float(timeout))
        while True:
            exit_code = self.poll()
            if exit_code is not None:
                return int(exit_code)
            if deadline is not None and self._monotonic() >= deadline:
                raise TimeoutError(f"PID {self.pid} did not exit before timeout.")
            self._sleep(0.05)


class SessionPopenProcessHandle:
    """Wrap one supervisor-owned ``Popen`` child with group-aware signals."""

    def __init__(
        self,
        process: subprocess.Popen[object],
        *,
        pid_getpgid=os.getpgid,
        group_signal=os.killpg,
    ) -> None:
        self._process = process
        self.pid = int(process.pid)
        self._pid_getpgid = pid_getpgid
        self._group_signal = group_signal

        self._proc_stat_reader = _read_proc_stat_snapshot
        self._pid_alive = default_pid_alive
        self._pid_signal = os.kill
        self._pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        self._pidfd = _try_open_pidfd(self.pid)

        snapshot = self._proc_stat_reader(self.pid)
        self._expected_starttime_ticks = None if snapshot is None else snapshot.starttime_ticks
        self._managed_pgid = self.pid
        self._leader_exit_code: int | None = None

    def __del__(self) -> None:
        _safe_close_fd(getattr(self, "_pidfd", None))

    def poll(self) -> int | None:
        # BREAKING: for dedicated supervisor-owned sessions, poll() now stays
        # None until the managed process group has no live helper descendants.
        leader_exit = self._process.poll()
        if leader_exit is not None:
            self._leader_exit_code = int(leader_exit)
            _safe_close_fd(self._pidfd)
            self._pidfd = None

        if _managed_group_still_running(
            self.pid,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return None

        if self._leader_exit_code is not None:
            return int(self._leader_exit_code)
        if leader_exit is not None:
            return int(leader_exit)
        return None

    def terminate(self) -> None:
        if _signal_managed_group(
            self.pid,
            signal.SIGTERM,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return
        _signal_exact_pid(
            self.pid,
            signal.SIGTERM,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pid_signal=self._pid_signal,
            pidfd=self._pidfd,
            pidfd_send_signal=self._pidfd_send_signal,
        )

    def kill(self) -> None:
        if _signal_managed_group(
            self.pid,
            signal.SIGKILL,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            return
        _signal_exact_pid(
            self.pid,
            signal.SIGKILL,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pid_signal=self._pid_signal,
            pidfd=self._pidfd,
            pidfd_send_signal=self._pidfd_send_signal,
        )

    def wait(self, timeout: float | None = None) -> int:
        # BREAKING: wait() now waits for the managed dedicated process group to
        # drain, not only for the Popen leader process to exit.
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        leader_exit = int(self._process.wait(timeout=remaining))
        self._leader_exit_code = leader_exit
        _safe_close_fd(self._pidfd)
        self._pidfd = None

        while _managed_group_still_running(
            self.pid,
            managed_pgid=self._managed_pgid,
            expected_starttime_ticks=self._expected_starttime_ticks,
            proc_stat_reader=self._proc_stat_reader,
            pid_alive=self._pid_alive,
            pidfd=self._pidfd,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Process group {self._managed_pgid} did not drain before timeout."
                )
            time.sleep(0.05)
        return leader_exit


def default_process_factory(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
) -> ProcessHandle:
    """Launch one child process for the supervisor.

    Each child runs in its own session so the supervisor can later terminate the
    entire child process group, including helper descendants spawned by the
    runtime itself.
    """

    if not argv:
        raise ValueError("Child argv must not be empty.")

    resolved_argv = list(argv)
    # BREAKING: argv[0] is now resolved deterministically up front so accidental
    # PATH/cwd hijacks that used to work implicitly now fail fast instead.
    resolved_argv[0] = _resolve_executable(str(resolved_argv[0]), cwd=Path(cwd), env=env)

    process = subprocess.Popen(
        resolved_argv,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        start_new_session=True,
        close_fds=True,
    )
    return SessionPopenProcessHandle(process)


def parse_utc_timestamp(value: str | None) -> datetime | None:
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


def python_executable_for_runtime(config: TwinrConfig) -> str:
    """Resolve the preferred Python executable for one runtime tree."""

    candidate = Path(config.project_root).resolve() / ".venv" / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return sys.executable


def prepend_pythonpath(existing: str | None) -> str:
    """Ensure ``src`` stays on ``PYTHONPATH`` for child commands."""

    text = str(existing or "").strip()
    if not text:
        return "src"

    normalized = []
    seen = set()
    for part in text.split(os.pathsep):
        entry = os.path.normpath(part)
        if not entry or entry in seen:
            continue
        seen.add(entry)
        normalized.append(entry)

    if os.path.normpath("src") not in seen:
        normalized.insert(0, "src")
    return os.pathsep.join(normalized)


__all__ = [
    "ManagedChild",
    "PidProcessHandle",
    "ProcessHandle",
    "default_monotonic",
    "default_pid_alive",
    "default_pid_cmdline",
    "default_process_factory",
    "default_sleep",
    "default_utcnow",
    "parse_utc_timestamp",
    "prepend_pythonpath",
    "python_executable_for_runtime",
]