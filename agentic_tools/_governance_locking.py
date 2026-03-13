from __future__ import annotations

import errno
import json
import os
import random
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


try:
    import fcntl as _fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows-only branch
    _fcntl = None

try:
    import msvcrt as _msvcrt  # type: ignore
except Exception:  # pragma: no cover - POSIX-only branch
    _msvcrt = None


class FileLockTimeout(TimeoutError):
    """Raised when a governance store lock cannot be acquired within budget."""


@dataclass(frozen=True)
class GovernanceLockSettings:
    timeout_sec: float
    poll_sec: float
    stale_after_sec: int
    heartbeat_sec: float


_LOCK_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()
_INPROCESS_LOCKS: Dict[str, threading.Lock] = {}
_INPROCESS_LOCKS_GUARD = threading.Lock()
_MAX_RETRY_BACKOFF_SEC = 5.0


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def resolve_lock_settings(
    *,
    timeout_envs: Sequence[str],
    timeout_default: float,
    poll_env: str,
    poll_default: float,
    stale_env: str,
    stale_default: int,
    heartbeat_env: str,
    heartbeat_default: Optional[float] = None,
) -> GovernanceLockSettings:
    timeout_sec = float(timeout_default)
    for name in list(timeout_envs or ()):
        timeout_sec = env_float(name, timeout_sec)
        if os.getenv(name) is not None:
            break
    stale_after_sec = max(1, env_int(stale_env, int(stale_default)))
    default_heartbeat = heartbeat_default
    if default_heartbeat is None:
        default_heartbeat = max(1.0, float(stale_after_sec) / 4.0)
    return GovernanceLockSettings(
        timeout_sec=max(0.01, float(timeout_sec)),
        poll_sec=max(0.01, env_float(poll_env, float(poll_default))),
        stale_after_sec=int(stale_after_sec),
        heartbeat_sec=max(0.0, env_float(heartbeat_env, float(default_heartbeat))),
    )


def resolve_retry_budget(
    *,
    retry_timeout_env: str,
    lock_timeout_envs: Sequence[str],
    default_lock_timeout: float,
    minimum_budget_sec: float = 75.0,
) -> float:
    base_lock_timeout = float(default_lock_timeout)
    for name in list(lock_timeout_envs or ()):
        base_lock_timeout = env_float(name, base_lock_timeout)
        if os.getenv(name) is not None:
            break
    return max(
        0.0,
        env_float(
            retry_timeout_env,
            max(float(minimum_budget_sec), float(base_lock_timeout) * 2.0),
        ),
    )


def resolve_retry_poll(*, retry_poll_env: str, default_poll_sec: float = 0.25) -> float:
    return max(0.01, env_float(retry_poll_env, float(default_poll_sec)))


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _read_boot_id() -> Optional[str]:
    if _is_windows_platform():  # pragma: no cover - POSIX-only branch
        return None
    try:
        text = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8", errors="replace")
            .strip()
        )
    except Exception:
        return None
    return text or None


def _read_linux_proc_start_ticks(pid: int) -> Optional[int]:
    if _is_windows_platform():  # pragma: no cover - POSIX-only branch
        return None
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except Exception:
        return None
    rparen = raw.rfind(")")
    if rparen < 0:
        return None
    tail = raw[rparen + 2 :].split()
    if len(tail) <= 19:
        return None
    try:
        return int(tail[19])
    except Exception:
        return None


def _process_identity_token_for_pid(pid: int) -> Optional[str]:
    start_ticks = _read_linux_proc_start_ticks(pid)
    if start_ticks is None:
        return None
    boot_id = _read_boot_id() or "unknown-boot"
    return f"linux:{boot_id}:{start_ticks}"


_PROCESS_IDENTITY_TOKEN = (
    _process_identity_token_for_pid(os.getpid())
    or f"pid:{os.getpid()}:{time.time_ns()}"
)


def _pid_matches_identity_token(pid: int, token: Optional[str]) -> Optional[bool]:
    token_text = str(token or "").strip()
    if not token_text:
        return None
    current_token = _process_identity_token_for_pid(pid)
    if not current_token:
        return None
    return current_token == token_text


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if (
        _is_windows_platform()
    ):  # pragma: no cover - exercised via targeted monkeypatching
        return _pid_is_alive_windows(pid)
    pidfd_open = getattr(os, "pidfd_open", None)
    if callable(pidfd_open):
        try:
            fd = pidfd_open(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            pass
        else:
            try:
                return True
            finally:
                _close_fd(int(fd))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _pid_is_alive_windows(pid: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return True

    process_query_limited_information = 0x1000
    still_active = 259
    access_denied = 5
    invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_exit_code_process = kernel32.GetExitCodeProcess
    get_exit_code_process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code_process.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_query_limited_information, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        if err == access_denied:
            return True
        if err == invalid_parameter:
            return False
        return False

    exit_code = wintypes.DWORD()
    try:
        if not get_exit_code_process(handle, ctypes.byref(exit_code)):
            err = ctypes.get_last_error()
            if err == access_denied:
                return True
            return False
        return int(exit_code.value) == int(still_active)
    finally:
        close_handle(handle)


def _parse_lock_metadata_map(raw: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    text = str(raw or "").strip()
    if not text:
        return meta

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, Mapping):
        for key, value in parsed.items():
            key_text = str(key or "").strip().lower()
            if not key_text or value is None:
                continue
            value_text = str(value).strip()
            if value_text:
                meta[key_text] = value_text
        if meta:
            return meta

    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key_text = key.strip().lower()
        value_text = value.strip()
        if key_text and value_text:
            meta[key_text] = value_text

    if "pid" not in meta:
        first = text.split()[0]
        try:
            meta["pid"] = str(int(first))
        except Exception:
            pass
    return meta


def _parse_lock_metadata(raw: str) -> Tuple[Optional[str], Optional[int]]:
    meta = _parse_lock_metadata_map(raw)
    host = str(meta.get("host") or "").strip() or None
    pid_raw = meta.get("pid")
    try:
        pid = int(pid_raw) if pid_raw is not None else None
    except Exception:
        pid = None
    return host, pid


def _lock_age_seconds(path: Path) -> Optional[float]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return max(0.0, time.time() - float(stat.st_mtime))


def _get_inprocess_lock(key: str) -> threading.Lock:
    with _INPROCESS_LOCKS_GUARD:
        lock = _INPROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INPROCESS_LOCKS[key] = lock
        return lock


def _open_lock_fd(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # pragma: no cover - Windows only
    return os.open(str(path), flags, 0o644)


def _prepare_windows_lock_target(fd: int) -> None:
    if os.name != "nt":  # pragma: no cover - POSIX only
        return
    if _msvcrt is None:  # pragma: no cover - defensive
        raise RuntimeError("governance_locking_missing_msvcrt")
    try:
        os.lseek(fd, 0, os.SEEK_END)
        if os.lseek(fd, 0, os.SEEK_CUR) == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
    except Exception:
        os.lseek(fd, 0, os.SEEK_SET)


def _acquire_os_lock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - Windows only
        if _msvcrt is None:
            raise RuntimeError("governance_locking_missing_msvcrt")
        _prepare_windows_lock_target(fd)
        _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
        return
    if _fcntl is None:  # pragma: no cover - defensive
        raise RuntimeError("governance_locking_missing_fcntl")
    _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)


def _release_os_lock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - Windows only
        if _msvcrt is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except Exception:
            pass
        _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        return
    if _fcntl is None:  # pragma: no cover - defensive
        return
    _fcntl.flock(fd, _fcntl.LOCK_UN)


def _close_fd(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except Exception:
        pass


def _is_lock_contended_error(exc: BaseException) -> bool:
    if isinstance(exc, BlockingIOError):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EBUSY}
    return False


def _read_fd_snapshot(fd: int, *, max_bytes: int = 65536) -> bytes:
    try:
        size = max(0, int(os.fstat(fd).st_size))
    except Exception:
        size = max_bytes
    read_size = min(max_bytes, size)
    os.lseek(fd, 0, os.SEEK_SET)
    if read_size <= 0:
        return b""
    return os.read(fd, read_size)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise OSError(
                errno.EIO, "short write while updating governance lock metadata"
            )
        offset += written


def _restore_fd_snapshot(fd: int, snapshot: bytes) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    if snapshot:
        _write_all(fd, snapshot)
    os.fsync(fd)


def _write_lock_metadata(
    *,
    fd: int,
    host: str,
    pid: int,
    claim_token: str,
    process_identity_token: str,
) -> None:
    payload = (
        f"host={host}\n"
        f"pid={pid}\n"
        f"claim_token={claim_token}\n"
        f"process_identity_token={process_identity_token}\n"
        f"created_at_utc={_utc_now_z()}\n"
        f"lock_kind=os_advisory\n"
    )
    data = payload.encode("utf-8", errors="ignore")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    _write_all(fd, data)
    os.fsync(fd)


def _start_registry_heartbeat(
    *, path: Path, heartbeat_sec: float
) -> Tuple[Optional[threading.Event], Optional[threading.Thread]]:
    if heartbeat_sec <= 0:
        return None, None
    stop = threading.Event()

    def _run() -> None:
        while not stop.wait(heartbeat_sec):
            try:
                os.utime(str(path), None)
            except Exception:
                return

    thread = threading.Thread(
        target=_run,
        name=f"governance_lock_heartbeat:{path.name}",
        daemon=True,
    )
    thread.start()
    return stop, thread


def _stop_registry_heartbeat(reg: Mapping[str, Any]) -> None:
    stop = reg.get("heartbeat_stop")
    thread = reg.get("heartbeat_thread")
    if isinstance(stop, threading.Event):
        try:
            stop.set()
        except Exception:
            pass
    if isinstance(thread, threading.Thread):
        try:
            thread.join(timeout=0.5)
        except Exception:
            pass


@dataclass
class GovernanceFileLock:
    path: Path
    timeout_sec: float = 30.0
    poll_sec: float = 0.05
    stale_after_sec: int = 120
    heartbeat_sec: float = 30.0
    reentrant: bool = True
    _host: str = field(init=False, default="", repr=False)
    _pid: int = field(init=False, default=0, repr=False)
    _key: str = field(init=False, default="", repr=False)
    _acquired: bool = field(init=False, default=False, repr=False)
    _acquire_depth: int = field(init=False, default=0, repr=False)
    _owner_thread_id: Optional[int] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.timeout_sec = max(0.01, float(self.timeout_sec))
        self.poll_sec = max(0.01, float(self.poll_sec))
        self.stale_after_sec = max(1, int(self.stale_after_sec))
        self.heartbeat_sec = max(0.0, float(self.heartbeat_sec))
        self._host = _get_hostname()
        self._pid = int(os.getpid())
        self._key = str(self.path.expanduser().resolve())

    def _timeout_detail(self) -> str:
        raw = ""
        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            raw = ""
        meta = _parse_lock_metadata_map(raw)
        host = str(meta.get("host") or "").strip() or None
        pid_raw = meta.get("pid")
        try:
            pid = int(pid_raw) if pid_raw is not None else None
        except Exception:
            pid = None
        age_s = _lock_age_seconds(self.path)
        owner_parts = []
        if host:
            owner_parts.append(f"holder_host={host}")
        if pid is not None:
            owner_parts.append(f"holder_pid={pid}")
            if host is None or host == self._host:
                owner_parts.append(f"holder_alive={str(_pid_is_alive(pid)).lower()}")
                identity_match = _pid_matches_identity_token(
                    pid, meta.get("process_identity_token")
                )
                if identity_match is not None:
                    owner_parts.append(
                        f"holder_identity_match={str(identity_match).lower()}"
                    )
        claim_token = str(meta.get("claim_token") or "").strip()
        if claim_token:
            owner_parts.append(f"holder_claim_token={claim_token[:12]}")
        if age_s is not None:
            owner_parts.append(f"holder_age_s={age_s:.3f}")
        owner_parts.append(f"stale_after_s={self.stale_after_sec}")
        if not owner_parts:
            return ""
        return " (" + " ".join(owner_parts) + ")"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner_thread = threading.get_ident()

        with _LOCK_REGISTRY_GUARD:
            reg = _LOCK_REGISTRY.get(self._key)
            if (
                self.reentrant
                and reg is not None
                and reg.get("pid") == self._pid
                and reg.get("thread_id") == owner_thread
            ):
                reg["count"] = int(reg.get("count", 0)) + 1
                self._acquire_depth += 1
                self._owner_thread_id = owner_thread
                self._acquired = True
                return

        deadline = time.monotonic() + float(self.timeout_sec)
        local_lock = _get_inprocess_lock(self._key)
        remaining = max(0.0, deadline - time.monotonic())
        if not local_lock.acquire(timeout=remaining):
            raise FileLockTimeout(
                f"timed out acquiring lock: {self.path}{self._timeout_detail()}"
            )

        fd: Optional[int] = None
        prior_snapshot = b""
        os_lock_acquired = False
        hb_stop: Optional[threading.Event] = None
        hb_thread: Optional[threading.Thread] = None
        try:
            fd = _open_lock_fd(self.path)
            while True:
                try:
                    _acquire_os_lock(fd)
                    os_lock_acquired = True
                    break
                except Exception as exc:
                    if not _is_lock_contended_error(exc):
                        raise
                    if time.monotonic() >= deadline:
                        raise FileLockTimeout(
                            f"timed out acquiring lock: {self.path}{self._timeout_detail()}"
                        ) from exc
                    time.sleep(self.poll_sec)

            prior_snapshot = _read_fd_snapshot(fd)
            _write_lock_metadata(
                fd=fd,
                host=self._host,
                pid=self._pid,
                claim_token=uuid.uuid4().hex,
                process_identity_token=_PROCESS_IDENTITY_TOKEN,
            )
            hb_stop, hb_thread = _start_registry_heartbeat(
                path=self.path,
                heartbeat_sec=self.heartbeat_sec,
            )
            with _LOCK_REGISTRY_GUARD:
                _LOCK_REGISTRY[self._key] = {
                    "pid": self._pid,
                    "thread_id": owner_thread,
                    "count": 1,
                    "fd": fd,
                    "local_lock": local_lock,
                    "heartbeat_stop": hb_stop,
                    "heartbeat_thread": hb_thread,
                }
            self._acquire_depth = 1
            self._owner_thread_id = owner_thread
            self._acquired = True
        except Exception:
            if hb_stop is not None or hb_thread is not None:
                _stop_registry_heartbeat(
                    {
                        "heartbeat_stop": hb_stop,
                        "heartbeat_thread": hb_thread,
                    }
                )
            if fd is not None and os_lock_acquired:
                try:
                    _restore_fd_snapshot(fd, prior_snapshot)
                except Exception:
                    pass
                try:
                    _release_os_lock(fd)
                except Exception:
                    pass
            self._acquire_depth = 0
            self._owner_thread_id = None
            self._acquired = False
            _close_fd(fd)
            try:
                local_lock.release()
            except Exception:
                pass
            raise

    def release(self) -> None:
        if self._acquire_depth <= 0:
            return
        owner_thread = threading.get_ident()
        if self._owner_thread_id is not None and self._owner_thread_id != owner_thread:
            raise RuntimeError(
                f"lock release attempted by non-owner thread: {self.path}"
            )

        reg: Optional[Dict[str, Any]] = None
        with _LOCK_REGISTRY_GUARD:
            current = _LOCK_REGISTRY.get(self._key)
            if (
                current is None
                or current.get("pid") != self._pid
                or current.get("thread_id") != owner_thread
            ):
                raise RuntimeError(f"lock state lost before release: {self.path}")
            current["count"] = int(current.get("count", 0)) - 1
            if int(current.get("count", 0)) < 0:
                raise RuntimeError(f"lock recursion count went negative: {self.path}")
            self._acquire_depth -= 1
            self._acquired = self._acquire_depth > 0
            if self._acquire_depth <= 0:
                self._owner_thread_id = None
            if int(current.get("count", 0)) > 0:
                return
            reg = dict(current)
            _LOCK_REGISTRY.pop(self._key, None)

        if reg is None:
            return

        _stop_registry_heartbeat(reg)
        fd = reg.get("fd")
        if isinstance(fd, int):
            try:
                _release_os_lock(fd)
            except Exception:
                pass
        _close_fd(fd if isinstance(fd, int) else None)

        local_lock = reg.get("local_lock")
        if local_lock is not None:
            try:
                local_lock.release()
            except Exception:
                pass

    def __enter__(self) -> "GovernanceFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def run_mutation_with_retry(
    *,
    action: str,
    op: Callable[[], T],
    classify_exception: Callable[[BaseException], Tuple[str, Optional[bool]]],
    retry_budget_sec: float,
    retry_poll_sec: float,
    emit_retry: Optional[Callable[[str, int, float, BaseException], None]] = None,
) -> Tuple[T, Dict[str, Any]]:
    """Retry only atomic or idempotent mutations with capped exponential full jitter."""
    started = time.monotonic()
    deadline = started + float(retry_budget_sec)
    retries = 0

    while True:
        try:
            result = op()
            retry_meta: Dict[str, Any] = {}
            if retries > 0:
                retry_meta = {
                    "attempts": int(retries + 1),
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                }
            return result, retry_meta
        except SystemExit:
            raise
        except Exception as exc:
            error_code, retryable = classify_exception(exc)
            if not retryable:
                raise
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = round((now - started) * 1000.0, 3)
                raise FileLockTimeout(
                    f"{action} exhausted retry budget after {retries + 1} attempts ({elapsed_ms} ms): {exc}"
                ) from exc
            retries += 1
            sleep_s = _compute_retry_sleep_sec(
                base_poll_sec=retry_poll_sec,
                retry_number=retries,
                remaining_sec=max(0.0, deadline - now),
            )
            if emit_retry is not None:
                emit_retry(str(error_code or "").strip(), retries, sleep_s, exc)
            time.sleep(sleep_s)


def _compute_retry_sleep_sec(
    *,
    base_poll_sec: float,
    retry_number: int,
    remaining_sec: float,
    max_backoff_sec: float = _MAX_RETRY_BACKOFF_SEC,
) -> float:
    remaining = max(0.0, float(remaining_sec))
    if remaining <= 0.0:
        return 0.0
    base = max(0.01, float(base_poll_sec))
    retry_idx = max(1, int(retry_number))
    exponential_cap = base * (2 ** (retry_idx - 1))
    cap = min(max(0.01, float(max_backoff_sec)), exponential_cap, remaining)
    jitter = random.random() * cap
    return min(max(0.01, jitter), remaining)
