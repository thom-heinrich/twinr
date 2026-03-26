"""Ensure the external remote-memory watchdog process is running on the Pi.

The live button/audio runtime must not spend its own CPU budget on deep remote
snapshot validation. Instead it relies on a dedicated watchdog process that
keeps probing required remote readiness and writes one rolling artifact.
"""

from __future__ import annotations

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


def _default_emit(line: str) -> None:
    """Print one bounded lifecycle line."""

    print(line, flush=True)


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
) -> int | None:
    """Start the external watchdog process when it is not already running."""

    if not _remote_required(config):
        return None

    owner = loop_lock_owner(config, "remote-memory-watchdog")
    if owner is not None:
        _maybe_seed_watchdog_bootstrap_snapshot(config, owner_pid=owner, emit=emit)
        emit(f"remote_memory_watchdog=running:{owner}")
        return owner

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

    deadline = time.monotonic() + max(0.1, float(startup_timeout_s))
    while time.monotonic() < deadline:
        owner = loop_lock_owner(config, "remote-memory-watchdog")
        if owner is not None:
            _maybe_seed_watchdog_bootstrap_snapshot(config, owner_pid=owner, emit=emit)
            emit(f"remote_memory_watchdog=ready:{owner}")
            return owner
        time.sleep(0.1)
    emit("remote_memory_watchdog=startup_timeout")
    return None
