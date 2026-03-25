"""Process and timestamp helpers for the runtime supervisor."""

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
        monotonic,
        sleep,
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


def default_process_factory(
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
