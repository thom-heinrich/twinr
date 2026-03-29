"""Process and timestamp helpers for the runtime supervisor.

The productive runtime can spawn helper descendants such as ``gpiomon`` that
must die with the owning streaming child during supervisor restarts. This
module therefore launches supervisor-owned children in their own sessions and
prefers signaling that dedicated process group when it is safe to do so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig


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


def _signal_dedicated_process_group(
    pid: int,
    sig: int,
    *,
    pid_getpgid,
    group_signal,
) -> bool:
    """Signal one owned process group when the PID leads its own session."""

    try:
        pgid = int(pid_getpgid(int(pid)))
    except Exception:
        return False
    if pgid <= 0 or pgid != int(pid):
        return False
    group_signal(pgid, sig)
    return True


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


def default_pid_cmdline(pid: int) -> tuple[str, ...]:
    """Return one best-effort command line tuple for a local PID."""

    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
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

    def poll(self) -> int | None:
        return None if self._pid_alive(self.pid) else 0

    def terminate(self) -> None:
        if not _signal_dedicated_process_group(
            self.pid,
            signal.SIGTERM,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            self._pid_signal(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        if not _signal_dedicated_process_group(
            self.pid,
            signal.SIGKILL,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            self._pid_signal(self.pid, signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else self._monotonic() + max(0.0, float(timeout))
        while self._pid_alive(self.pid):
            if deadline is not None and self._monotonic() >= deadline:
                raise TimeoutError(f"PID {self.pid} did not exit before timeout.")
            self._sleep(0.05)
        return 0


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

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        if not _signal_dedicated_process_group(
            self.pid,
            signal.SIGTERM,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            self._process.terminate()

    def kill(self) -> None:
        if not _signal_dedicated_process_group(
            self.pid,
            signal.SIGKILL,
            pid_getpgid=self._pid_getpgid,
            group_signal=self._group_signal,
        ):
            self._process.kill()

    def wait(self, timeout: float | None = None) -> int:
        return int(self._process.wait(timeout=timeout))


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

    process = subprocess.Popen(
        list(argv),
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
    if candidate.exists():
        return str(candidate)
    return sys.executable


def prepend_pythonpath(existing: str | None) -> str:
    """Ensure ``src`` stays on ``PYTHONPATH`` for child commands."""

    text = str(existing or "").strip()
    if not text:
        return "src"
    parts = [part for part in text.split(os.pathsep) if part]
    if "src" not in parts:
        parts.insert(0, "src")
    return os.pathsep.join(parts)


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
