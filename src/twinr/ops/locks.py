from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import errno
import fcntl
import io
import os
import re
import stat
from typing import TextIO

from twinr.agent.base_agent.config import TwinrConfig


# AUDIT-FIX(#1): Restrict lock names to a single safe filename component.
_LOCK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# AUDIT-FIX(#1): Refuse silent fallback to insecure lock-file opens on platforms without O_NOFOLLOW.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", None)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _validated_loop_name(loop_name: str) -> str:
    if not _LOCK_NAME_RE.fullmatch(loop_name):
        raise ValueError(
            "loop_name must contain only ASCII letters, digits, '.', '_' or '-'."
        )
    return loop_name


def _resolved_runtime_state_path(config: TwinrConfig) -> Path:
    # AUDIT-FIX(#1): Canonicalize both relative and absolute state paths so every caller lands on one lock directory.
    runtime_state_path = Path(config.runtime_state_path).expanduser()
    project_root = Path(config.project_root).expanduser().resolve()
    if not runtime_state_path.is_absolute():
        runtime_state_path = project_root / runtime_state_path
    return runtime_state_path.resolve()


def loop_lock_path(config: TwinrConfig, loop_name: str) -> Path:
    runtime_state_path = _resolved_runtime_state_path(config)
    safe_loop_name = _validated_loop_name(loop_name)
    return runtime_state_path.parent / f"twinr-{safe_loop_name}.lock"


def _read_owner_text(handle: TextIO) -> str:
    # AUDIT-FIX(#1): Bound the read so a maliciously large file cannot be slurped into memory.
    handle.seek(0)
    return handle.readline(128).strip()


def _close_quietly(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except OSError:
        pass


def _unlock_and_close_quietly(handle: TextIO) -> None:
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        _close_quietly(handle)


def _open_lock_file(path: Path, *, create: bool) -> TextIO:
    # AUDIT-FIX(#1): Open relative to the canonical parent directory and refuse symlink lock files.
    if _O_NOFOLLOW is None:
        raise RuntimeError(
            "Secure lock-file handling requires os.O_NOFOLLOW on this platform."
        )

    parent_flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY
    parent_fd = os.open(path.parent, parent_flags)
    try:
        file_flags = _O_CLOEXEC | _O_NOFOLLOW | (os.O_RDWR if create else os.O_RDONLY)
        if create:
            file_flags |= os.O_CREAT

        try:
            fd = os.open(path.name, file_flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(
                    f"Refusing to follow a symlink for lock file: {path}"
                ) from exc
            raise

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"Lock path is not a regular file: {path}")
            return io.open(
                fd,
                mode="r+" if create else "r",
                encoding="utf-8",
                closefd=True,
            )
        except Exception:
            os.close(fd)
            raise
    finally:
        os.close(parent_fd)


@dataclass(slots=True)
class TwinrInstanceLock:
    path: Path
    label: str
    _handle: TextIO | None = field(default=None, init=False, repr=False)  # AUDIT-FIX(#6): Use a concrete handle type for safer static checking.

    def acquire(self) -> "TwinrInstanceLock":
        # AUDIT-FIX(#2): Prevent re-acquiring the same instance and leaking the original lock FD.
        if self._handle is not None:
            raise RuntimeError(f"The Twinr {self.label} lock is already acquired.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = _open_lock_file(self.path, create=True)

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = _read_owner_text(handle)
            _close_quietly(handle)
            owner_suffix = f" (pid {owner})" if owner.isdigit() else ""
            raise RuntimeError(
                f"Another Twinr {self.label} is already running{owner_suffix}."
            ) from exc
        except Exception:
            # AUDIT-FIX(#3): Close the FD on non-contention flock failures so we do not leak descriptors.
            _close_quietly(handle)
            raise

        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except Exception:
            # AUDIT-FIX(#3): If owner metadata write fails after flock succeeds, release the kernel lock before bubbling up.
            _unlock_and_close_quietly(handle)
            raise

        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            # AUDIT-FIX(#4): Cleanup is best-effort; metadata truncation must not mask the real failure path.
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
            except (OSError, ValueError):
                pass

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, ValueError):
                pass
        finally:
            _close_quietly(handle)
            self._handle = None

    def __enter__(self) -> "TwinrInstanceLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def loop_instance_lock(
    config: TwinrConfig, loop_name: str, *, label: str | None = None
) -> TwinrInstanceLock:
    return TwinrInstanceLock(
        path=loop_lock_path(config, loop_name),
        label=label or loop_name.replace("-", " "),
    )


def loop_lock_owner(config: TwinrConfig, loop_name: str) -> int | None:
    path = loop_lock_path(config, loop_name)

    try:
        # AUDIT-FIX(#5): Open directly and treat a missing file as "no owner" to remove the exists/open race.
        handle = _open_lock_file(path, create=False)
    except FileNotFoundError:
        return None

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = _read_owner_text(handle)
            return int(owner) if owner.isdigit() else None
        return None
    finally:
        _unlock_and_close_quietly(handle)