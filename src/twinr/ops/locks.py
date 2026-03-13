from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import fcntl
import os

from twinr.agent.base_agent.config import TwinrConfig


def loop_lock_path(config: TwinrConfig, loop_name: str) -> Path:
    runtime_state_path = Path(config.runtime_state_path)
    if not runtime_state_path.is_absolute():
        runtime_state_path = (Path(config.project_root) / runtime_state_path).resolve()
    return runtime_state_path.parent / f"twinr-{loop_name}.lock"


@dataclass(slots=True)
class TwinrInstanceLock:
    path: Path
    label: str
    _handle: object | None = field(default=None, init=False, repr=False)

    def acquire(self) -> "TwinrInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            owner_suffix = f" (pid {owner})" if owner.isdigit() else ""
            raise RuntimeError(
                f"Another Twinr {self.label} is already running{owner_suffix}."
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "TwinrInstanceLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def loop_instance_lock(config: TwinrConfig, loop_name: str, *, label: str | None = None) -> TwinrInstanceLock:
    return TwinrInstanceLock(
        path=loop_lock_path(config, loop_name),
        label=label or loop_name.replace("-", " "),
    )
