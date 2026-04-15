"""Ensure the external remote-memory watchdog process is running on the Pi.

The live button/audio runtime must not spend its own CPU budget on deep remote
snapshot validation. Instead it relies on a dedicated watchdog process that
keeps probing required remote readiness and writes one rolling artifact.
"""

from __future__ import annotations

# CHANGELOG: 2026-04-11
# BUG-1: Remove the raw-spawn fallback when a dedicated watchdog systemd unit is
# BUG-1: configured. Pi runtimes now keep one authoritative owner lane for the
# BUG-1: remote-memory watchdog instead of silently starting a second detached
# BUG-1: process after a failed or pending systemd start request.

from pathlib import Path
from typing import Callable
import os
import subprocess
import sys
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.locks import loop_lock_owner
from twinr.ops.remote_memory_watchdog_state import (
    RemoteMemoryWatchdogStore,
    build_remote_memory_watchdog_bootstrap_snapshot,
)


EmitFn = Callable[[str], None]
_PI_RUNTIME_ROOT = Path("/twinr")
_PI_REMOTE_MEMORY_WATCHDOG_SYSTEMD_UNIT = "twinr-remote-memory-watchdog.service"
_DEFAULT_SYSTEMD_START_TIMEOUT_S = 10.0


def _default_emit(line: str) -> None:
    """Print one bounded lifecycle line."""

    print(line, flush=True)


def _normalize_text(value: object) -> str:
    """Return one stripped text value or an empty string."""

    return str(value or "").strip()


def _coerce_positive_float(value: object, *, default: float) -> float:
    """Return one finite positive float or the provided default."""

    if isinstance(value, bool):
        parsed = float(value)
    elif isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return float(default)
    else:
        return float(default)
    if parsed <= 0.0:
        return float(default)
    return parsed


def _remote_required(config: TwinrConfig) -> bool:
    """Report whether this runtime must fail closed on remote memory loss."""

    return bool(
        config.long_term_memory_enabled
        and str(config.long_term_memory_mode or "").strip().lower() == "remote_primary"
        and config.long_term_memory_remote_required
    )


def _python_executable_for_runtime(config: TwinrConfig) -> str:
    """Resolve the preferred Python executable for the Pi runtime tree."""

    candidate = Path(config.project_root) / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _systemd_watchdog_unit(config: TwinrConfig) -> str | None:
    """Return the dedicated watchdog unit name when one is configured."""

    for attr_name in (
        "long_term_memory_remote_watchdog_systemd_unit",
        "remote_memory_watchdog_systemd_unit",
    ):
        unit = _normalize_text(getattr(config, attr_name, ""))
        if unit:
            return unit
    try:
        if Path(config.project_root).expanduser().resolve() == _PI_RUNTIME_ROOT:
            return _PI_REMOTE_MEMORY_WATCHDOG_SYSTEMD_UNIT
    except Exception:
        return None
    return None


def _request_watchdog_systemd_start(
    config: TwinrConfig,
    *,
    emit: EmitFn,
) -> bool:
    """Ask systemd to start the canonical watchdog unit when available."""

    unit = _systemd_watchdog_unit(config)
    if not unit:
        return False
    timeout_s = _coerce_positive_float(
        getattr(config, "long_term_memory_remote_watchdog_systemd_timeout_s", _DEFAULT_SYSTEMD_START_TIMEOUT_S),
        default=_DEFAULT_SYSTEMD_START_TIMEOUT_S,
    )
    try:
        completed = subprocess.run(
            ["systemctl", "--no-block", "start", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        emit(f"remote_memory_watchdog=systemd_start_failed:{type(exc).__name__}:{exc}")
        return False
    if completed.returncode == 0:
        emit(f"remote_memory_watchdog=systemd_start_requested:{unit}")
        return True
    emit(f"remote_memory_watchdog=systemd_start_exit:{completed.returncode}:{unit}")
    return False


def _seed_watchdog_bootstrap_snapshot(config: TwinrConfig, *, owner_pid: int) -> None:
    """Persist a fresh startup snapshot when the owner PID changed."""

    store = RemoteMemoryWatchdogStore.from_config(config)
    try:
        snapshot = store.load()
    except Exception:
        snapshot = None
    if snapshot is not None and int(getattr(snapshot, "pid", 0) or 0) == int(owner_pid):
        return
    bootstrap = build_remote_memory_watchdog_bootstrap_snapshot(
        config,
        pid=owner_pid,
        artifact_path=store.path,
        previous_snapshot=snapshot,
    )
    store.save(bootstrap)


def _maybe_seed_watchdog_bootstrap_snapshot(
    config: TwinrConfig,
    *,
    owner_pid: int,
    emit: EmitFn,
) -> None:
    """Best-effort seed for external-watchdog handoffs."""

    try:
        _seed_watchdog_bootstrap_snapshot(config, owner_pid=owner_pid)
    except Exception as exc:
        emit(f"remote_memory_watchdog=bootstrap_seed_failed:{type(exc).__name__}:{exc}")


def ensure_remote_memory_watchdog_process(
    config: TwinrConfig,
    *,
    env_file: str | Path,
    emit: EmitFn = _default_emit,
    startup_timeout_s: float = 3.0,
    allow_spawn: bool = True,
) -> int | None:
    """Ensure the external watchdog owner exists without violating ownership.

    On the productive Pi, the dedicated systemd unit is the canonical owner and
    this helper requests that unit first. Only environments without any
    configured unit may use the detached direct-spawn lane.
    """

    if not _remote_required(config):
        return None

    owner = loop_lock_owner(config, "remote-memory-watchdog")
    if owner is not None:
        _maybe_seed_watchdog_bootstrap_snapshot(config, owner_pid=owner, emit=emit)
        emit(f"remote_memory_watchdog=running:{owner}")
        return owner

    deadline = time.monotonic() + max(0.1, float(startup_timeout_s))
    systemd_unit = _systemd_watchdog_unit(config)
    if systemd_unit and _request_watchdog_systemd_start(config, emit=emit):
        while time.monotonic() < deadline:
            owner = loop_lock_owner(config, "remote-memory-watchdog")
            if owner is not None:
                _maybe_seed_watchdog_bootstrap_snapshot(config, owner_pid=owner, emit=emit)
                emit(f"remote_memory_watchdog=ready:{owner}")
                return owner
            time.sleep(0.1)
        emit("remote_memory_watchdog=systemd_start_pending")
        return None
    if systemd_unit:
        emit(f"remote_memory_watchdog=systemd_required_no_spawn:{systemd_unit}")
        return None

    if not allow_spawn:
        emit("remote_memory_watchdog=spawn_disallowed")
        return None

    project_root = Path(config.project_root).resolve()
    python_executable = _python_executable_for_runtime(config)
    env_path = str(env_file)
    log_path = project_root / "state" / "logs" / "remote-memory-watchdog.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        subprocess.Popen(
            [
                python_executable,
                "-u",
                "-m",
                "twinr",
                "--env-file",
                env_path,
                "--watch-remote-memory",
            ],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": "src"},
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    emit("remote_memory_watchdog=spawned")

    while time.monotonic() < deadline:
        owner = loop_lock_owner(config, "remote-memory-watchdog")
        if owner is not None:
            _maybe_seed_watchdog_bootstrap_snapshot(config, owner_pid=owner, emit=emit)
            emit(f"remote_memory_watchdog=ready:{owner}")
            return owner
        time.sleep(0.1)
    emit("remote_memory_watchdog=startup_timeout")
    return None
