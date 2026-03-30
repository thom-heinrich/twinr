"""Coordinate single-instance Twinr loops through secure file locks.

This module derives lock-file paths from runtime configuration and exposes
helpers to acquire or inspect per-loop ownership.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-30
# BUG-1: Fixed a false-positive race in loop_lock_owner(); the old implementation
# BUG-1: briefly acquired the same exclusive flock used for ownership, so an
# BUG-1: inspection call could make a concurrent real startup fail spuriously.
# BUG-2: Improved owner diagnostics under contention by resolving the owning PID
# BUG-2: from /proc/locks instead of relying only on the lock-file payload.
# SEC-1: Hardened pathname resolution with Linux openat2(RESOLVE_NO_SYMLINKS)
# SEC-1: when available, plus stricter directory and file validation to reduce
# SEC-1: practical symlink / unsafe-directory lock-file attacks on Raspberry Pi.
# SEC-2: Tightened existing lock-file permissions to 0600 and reject non-regular
# SEC-2: or multiply-linked lock files to reduce tampering / corruption risk.
# IMP-1: Upgraded owner metadata from PID-only text to a backward-compatible
# IMP-1: first-line PID + JSON sidecar line carrying boot ID, start ticks, UID,
# IMP-1: hostname, and label for stronger process identity and observability.
# IMP-2: Metadata writes / clears are now fsync()'d best-effort so diagnostics
# IMP-2: survive crashes and SD-card write reordering more reliably.
# BREAKING: lock acquisition now rejects group/world-writable lock directories
# BREAKING: and insecure preexisting lock files instead of silently proceeding.

import ctypes
import errno
import fcntl
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import socket
import stat
from typing import Any, TextIO

from twinr.agent.base_agent.config import TwinrConfig


_LOCK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_OWNER_READ_LIMIT = 4096
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_PROC_LOCKS_PATH = Path("/proc/locks")
_PROC_LOCKS_UNAVAILABLE = object()

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", None)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_AT_FDCWD = getattr(os, "AT_FDCWD", -100)

_RESOLVE_NO_MAGICLINKS = 0x02
_RESOLVE_NO_SYMLINKS = 0x04

_SYS_OPENAT2 = 437


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def _load_libc() -> Any | None:
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None


_LIBC = _load_libc()
_OPENAT2_DISABLED = False


class TwinrInstanceAlreadyRunningError(RuntimeError):
    """Signal a non-authoritative Twinr start attempt for an already-owned loop."""

    def __init__(self, *, label: str, owner_pid: int | None = None) -> None:
        self.label = str(label)
        self.owner_pid = int(owner_pid) if owner_pid is not None else None
        owner_suffix = f" (pid {self.owner_pid})" if self.owner_pid is not None else ""
        super().__init__(f"Another Twinr {self.label} is already running{owner_suffix}.")


@dataclass(frozen=True, slots=True)
class _OwnerMetadata:
    pid: int
    boot_id: str | None
    start_ticks: int | None
    uid: int | None
    hostname: str | None
    label: str | None
    version: int = 2

    def to_file_text(self) -> str:
        payload = {
            "v": self.version,
            "pid": self.pid,
            "boot_id": self.boot_id,
            "start_ticks": self.start_ticks,
            "uid": self.uid,
            "hostname": self.hostname,
            "label": self.label,
        }
        return f"{self.pid}\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n"


def _validated_loop_name(loop_name: str) -> str:
    if not _LOCK_NAME_RE.fullmatch(loop_name):
        raise ValueError(
            "loop_name must contain only ASCII letters, digits, '.', '_' or '-'."
        )
    return loop_name


def _resolved_runtime_state_path(config: TwinrConfig) -> Path:
    runtime_state_path = Path(config.runtime_state_path).expanduser()
    project_root = Path(config.project_root).expanduser().resolve()
    if not runtime_state_path.is_absolute():
        runtime_state_path = project_root / runtime_state_path
    return runtime_state_path.resolve()


def loop_lock_path(config: TwinrConfig, loop_name: str) -> Path:
    """Return the canonical lock-file path for one Twinr loop."""

    runtime_state_path = _resolved_runtime_state_path(config)
    safe_loop_name = _validated_loop_name(loop_name)
    return runtime_state_path.parent / f"twinr-{safe_loop_name}.lock"


def _close_quietly(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except (OSError, ValueError):
        pass


def _unlock_and_close_quietly(handle: TextIO) -> None:
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
    finally:
        _close_quietly(handle)


def _fsync_quietly(handle: TextIO) -> None:
    try:
        handle.flush()
        os.fsync(handle.fileno())
    except (OSError, ValueError):
        pass


def _read_owner_lines(handle: TextIO) -> list[str]:
    handle.seek(0)
    return handle.read(_OWNER_READ_LIMIT).splitlines()


def _read_owner_payload(handle: TextIO) -> tuple[int | None, dict[str, Any] | None]:
    lines = _read_owner_lines(handle)
    pid: int | None = None
    payload: dict[str, Any] | None = None

    if lines:
        first = lines[0].strip()
        if first.isdigit():
            pid = int(first)

    if len(lines) >= 2:
        try:
            raw_payload = json.loads(lines[1])
        except json.JSONDecodeError:
            raw_payload = None
        if isinstance(raw_payload, dict):
            payload = raw_payload

    return pid, payload


def _read_owner_text(handle: TextIO) -> str:
    lines = _read_owner_lines(handle)
    return lines[0].strip() if lines else ""


def _current_boot_id() -> str | None:
    try:
        return _BOOT_ID_PATH.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _process_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    right_paren = text.rfind(")")
    if right_paren < 0:
        return None

    fields = text[right_paren + 2 :].split()
    if len(fields) <= 19:
        return None

    try:
        return int(fields[19])
    except ValueError:
        return None


def _current_owner_metadata(*, label: str) -> _OwnerMetadata:
    pid = os.getpid()
    return _OwnerMetadata(
        pid=pid,
        boot_id=_current_boot_id(),
        start_ticks=_process_start_ticks(pid),
        uid=os.geteuid() if hasattr(os, "geteuid") else None,
        hostname=socket.gethostname(),
        label=label,
    )


def _metadata_pid_if_same_process(
    pid: int | None,
    payload: dict[str, Any] | None,
    *,
    allow_pid_only: bool,
) -> int | None:
    if pid is None or pid <= 0:
        return None

    expected_boot_id: str | None = None
    expected_start_ticks: int | None = None
    if payload is not None:
        boot_id = payload.get("boot_id")
        if isinstance(boot_id, str) and boot_id:
            expected_boot_id = boot_id
        start_ticks = payload.get("start_ticks")
        if isinstance(start_ticks, int) and start_ticks >= 0:
            expected_start_ticks = start_ticks

    has_strong_identity = expected_boot_id is not None or expected_start_ticks is not None
    if not has_strong_identity and not allow_pid_only:
        return None

    if expected_boot_id is not None:
        current_boot_id = _current_boot_id()
        if current_boot_id is None or current_boot_id != expected_boot_id:
            return None

    if hasattr(os, "pidfd_open"):
        try:
            fd = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            return None
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return None
            fd = None
        if fd is not None:
            os.close(fd)

    if expected_start_ticks is not None:
        actual_start_ticks = _process_start_ticks(pid)
        if actual_start_ticks != expected_start_ticks:
            return None

    return pid


def _lock_identity_from_fd(fd: int) -> tuple[int, int, int] | None:
    try:
        file_stat = os.fstat(fd)
    except OSError:
        return None
    return (os.major(file_stat.st_dev), os.minor(file_stat.st_dev), file_stat.st_ino)


def _pid_from_proc_locks(fd: int) -> int | None | object:
    identity = _lock_identity_from_fd(fd)
    if identity is None:
        return None

    want_major, want_minor, want_inode = identity

    try:
        with _PROC_LOCKS_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 8:
                    continue

                lock_type = parts[1]
                lock_mode = parts[3]
                pid_text = parts[4]
                dev_inode_text = parts[5]

                if lock_type != "FLOCK" or lock_mode != "WRITE":
                    continue

                dev_parts = dev_inode_text.split(":")
                if len(dev_parts) != 3:
                    continue

                try:
                    major_num = int(dev_parts[0], 16)
                    minor_num = int(dev_parts[1], 16)
                    inode_num = int(dev_parts[2], 10)
                except ValueError:
                    continue

                if (major_num, minor_num, inode_num) != (
                    want_major,
                    want_minor,
                    want_inode,
                ):
                    continue

                if pid_text.isdigit():
                    pid = int(pid_text)
                    if pid > 0:
                        return pid
                return None
    except OSError:
        return _PROC_LOCKS_UNAVAILABLE

    return None


def _openat2_fd(path: Path, *, flags: int, mode: int = 0, resolve: int = 0) -> int:
    if _LIBC is None or _OPENAT2_DISABLED:
        raise OSError(errno.ENOSYS, "openat2 unavailable", os.fspath(path))

    how = _OpenHow(flags=flags, mode=mode, resolve=resolve)
    raw_path = os.fsencode(path)
    result = _LIBC.syscall(
        ctypes.c_long(_SYS_OPENAT2),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(raw_path),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), os.fspath(path))
    return int(result)


def _ensure_lock_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)

    try:
        dir_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Lock directory vanished during creation: {path}") from exc

    if not stat.S_ISDIR(dir_stat.st_mode):
        raise RuntimeError(f"Lock directory is not a directory: {path}")

    if dir_stat.st_mode & 0o022:
        raise RuntimeError(
            f"Refusing insecure lock directory with group/world write permissions: {path}"
        )


def _secure_open_fallback(path: Path, *, create: bool) -> int:
    if _O_NOFOLLOW is None:
        raise RuntimeError(
            "Secure lock-file handling requires either Linux openat2() or os.O_NOFOLLOW."
        )

    parent_flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY
    parent_fd = os.open(path.parent, parent_flags)
    try:
        file_flags = _O_CLOEXEC | _O_NOFOLLOW | (os.O_RDWR if create else os.O_RDONLY)
        if create:
            file_flags |= os.O_CREAT

        try:
            return os.open(path.name, file_flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(
                    f"Refusing to follow a symlink for lock file: {path}"
                ) from exc
            raise
    finally:
        os.close(parent_fd)


def _open_lock_fd(path: Path, *, create: bool) -> int:
    global _OPENAT2_DISABLED

    file_flags = _O_CLOEXEC | (os.O_RDWR if create else os.O_RDONLY)
    file_mode = 0o600 if create else 0
    if _O_NOFOLLOW is not None:
        file_flags |= _O_NOFOLLOW
    if create:
        file_flags |= os.O_CREAT

    try:
        fd = _openat2_fd(
            path,
            flags=file_flags,
            mode=file_mode,
            resolve=_RESOLVE_NO_SYMLINKS | _RESOLVE_NO_MAGICLINKS,
        )
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EPERM}:
            _OPENAT2_DISABLED = True
            fd = _secure_open_fallback(path, create=create)
        elif exc.errno == errno.ELOOP:
            raise RuntimeError(
                f"Refusing to follow a symlink anywhere in lock path: {path}"
            ) from exc
        else:
            raise

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Lock path is not a regular file: {path}")
        if file_stat.st_nlink != 1:
            raise RuntimeError(f"Refusing multiply-linked lock file: {path}")

        try:
            os.fchmod(fd, 0o600)
        except PermissionError as exc:
            raise RuntimeError(
                f"Refusing lock file with insecure ownership or permissions: {path}"
            ) from exc

        return fd
    except Exception:
        os.close(fd)
        raise


def _open_lock_file(path: Path, *, create: bool) -> TextIO:
    fd = _open_lock_fd(path, create=create)
    try:
        return io.open(
            fd,
            mode="r+" if create else "r",
            encoding="utf-8",
            errors="replace",
            closefd=True,
        )
    except Exception:
        os.close(fd)
        raise


@dataclass(slots=True)
class TwinrInstanceLock:
    """Guard one Twinr runtime loop against concurrent local execution."""

    path: Path
    label: str
    _handle: TextIO | None = field(default=None, init=False, repr=False)

    def acquire(self) -> "TwinrInstanceLock":
        """Acquire the lock without waiting and record the current PID."""

        if self._handle is not None:
            raise RuntimeError(f"The Twinr {self.label} lock is already acquired.")

        _ensure_lock_directory(self.path.parent)
        handle = _open_lock_file(self.path, create=True)

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner_pid = _pid_from_proc_locks(handle.fileno())
            if owner_pid is _PROC_LOCKS_UNAVAILABLE:
                owner_pid = None

            if owner_pid is None:
                pid, payload = _read_owner_payload(handle)
                owner_pid = _metadata_pid_if_same_process(pid, payload, allow_pid_only=True)
                if owner_pid is None:
                    owner_text = _read_owner_text(handle)
                    owner_pid = int(owner_text) if owner_text.isdigit() else None

            _close_quietly(handle)
            raise TwinrInstanceAlreadyRunningError(
                label=self.label,
                owner_pid=owner_pid,
            ) from exc
        except Exception:
            _close_quietly(handle)
            raise

        try:
            metadata = _current_owner_metadata(label=self.label)
            handle.seek(0)
            handle.truncate()
            handle.write(metadata.to_file_text())
            _fsync_quietly(handle)
        except Exception:
            _unlock_and_close_quietly(handle)
            raise

        self._handle = handle
        return self

    def release(self) -> None:
        """Release the lock and clear the owner metadata best-effort."""

        handle = self._handle
        if handle is None:
            return

        try:
            try:
                handle.seek(0)
                handle.truncate()
                _fsync_quietly(handle)
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
    """Build a ``TwinrInstanceLock`` for one named Twinr loop."""

    return TwinrInstanceLock(
        path=loop_lock_path(config, loop_name),
        label=label or loop_name.replace("-", " "),
    )


def loop_lock_owner(config: TwinrConfig, loop_name: str) -> int | None:
    """Return the owning PID for a held loop lock, if any."""

    path = loop_lock_path(config, loop_name)

    try:
        handle = _open_lock_file(path, create=False)
    except FileNotFoundError:
        return None

    try:
        owner_pid = _pid_from_proc_locks(handle.fileno())
        if owner_pid is _PROC_LOCKS_UNAVAILABLE:
            pid, payload = _read_owner_payload(handle)
            return _metadata_pid_if_same_process(pid, payload, allow_pid_only=False)
        if isinstance(owner_pid, int):
            return owner_pid
        return None
    finally:
        _close_quietly(handle)
