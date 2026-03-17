"""Apply best-effort OS hardening inside sandbox child processes.

The goal here is to raise the safety floor without breaking the skill-package
capability surface. The baseline hardening is always applied when possible,
while Linux Landlock enforcement is enabled only when the running kernel can do
so without requiring extra packaged dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import errno
import os
from pathlib import Path
import platform
import stat
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None

_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_SECCOMP = 21
_SECCOMP_MODE_NONE = 0
_SECCOMP_MODE_STRICT = 1
_SECCOMP_MODE_FILTER = 2
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
_LANDLOCK_RULE_PATH_BENEATH = 1
_PROC_SELF_FD = "/proc/self/fd"
_PROC_SELF_STATUS = "/proc/self/status"
_DEFAULT_SANITIZED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_MIN_HARDENING_FD_HEADROOM = 4


def _syscall_number(name: str) -> int | None:
    machine = platform.machine().lower()
    syscall_numbers = {
        "x86_64": {
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "amd64": {
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "aarch64": {
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "arm64": {
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "arm": {  # AUDIT-FIX(#3): Support 32-bit ARM targets used by Raspberry Pi deployments.
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "armv6l": {  # AUDIT-FIX(#3): Support 32-bit ARM targets used by Raspberry Pi deployments.
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "armv7l": {  # AUDIT-FIX(#3): Support 32-bit ARM targets used by Raspberry Pi deployments.
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
        "armv8l": {  # AUDIT-FIX(#3): Support 32-bit ARM targets used by Raspberry Pi deployments.
            "landlock_create_ruleset": 444,
            "landlock_add_rule": 445,
            "landlock_restrict_self": 446,
        },
    }
    arch_numbers = syscall_numbers.get(machine)
    if arch_numbers is None:
        return None
    return arch_numbers.get(name)


@dataclass(frozen=True, slots=True)
class SandboxHardeningReport:
    """Summarize which hardening layers were actually applied in one child."""

    no_new_privs: str = "unavailable"
    seccomp: str = "unknown"
    landlock: str = "not_attempted"
    closed_fd_count: int = 0
    working_directory: str | None = None
    notes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload for operator status and run metadata."""

        return {
            "no_new_privs": self.no_new_privs,
            "seccomp": self.seccomp,
            "landlock": self.landlock,
            "closed_fd_count": self.closed_fd_count,
            "working_directory": self.working_directory,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SandboxHardeningLimits:
    """Pass the numeric sandbox ceilings without coupling to runner internals."""

    cpu_seconds: int
    address_space_bytes: int
    max_open_files: int


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int),
        ("reserved", ctypes.c_uint32),
    ]


def apply_baseline_os_hardening(
    *,
    limits: SandboxHardeningLimits,
    working_directory: Path,
    keep_fds: tuple[int, ...] = (),
) -> SandboxHardeningReport:
    """Apply resource ceilings and process-level baseline hardening."""

    notes: list[str] = []
    normalized_keep_fds, keep_notes = _normalize_keep_fds(keep_fds)  # AUDIT-FIX(#5): Reject invalid keep_fds values without crashing hardening.
    notes.extend(keep_notes)
    closed_fd_count, close_notes = _close_extra_fds(normalized_keep_fds)  # AUDIT-FIX(#5): Close only currently open descriptors and avoid unbounded scans.
    notes.extend(close_notes)
    working_directory_status = None
    working_directory_fd = -1
    try:
        working_directory_fd = _open_existing_directory_fd(working_directory, use_o_path=False)  # AUDIT-FIX(#1): Open directories component-by-component without following symlinks before changing cwd.
        os.fchdir(working_directory_fd)  # AUDIT-FIX(#1): Switch cwd through a verified directory file descriptor.
        working_directory_status = os.getcwd()  # AUDIT-FIX(#1): Report the actual cwd rather than the unresolved input path.
    except OSError as exc:
        notes.append(f"chdir_failed:{exc.errno or 'unknown'}")
    finally:
        if working_directory_fd >= 0:
            try:
                os.close(working_directory_fd)
            except OSError:
                pass
    _sanitize_environment()  # AUDIT-FIX(#8): Preserve a safe runtime subset instead of dropping every operational environment variable.
    effective_limits, limit_notes = _normalize_limits(limits, normalized_keep_fds)  # AUDIT-FIX(#4): Clamp invalid/tight RLIMIT_NOFILE and skip impossible resource ceilings.
    notes.extend(limit_notes)
    notes.extend(_apply_resource_limits(effective_limits))  # AUDIT-FIX(#7): Surface resource-limit failures instead of silently downgrading hardening.
    no_new_privs_status = _apply_no_new_privs()
    seccomp_status = _read_seccomp_mode()
    return SandboxHardeningReport(
        no_new_privs=no_new_privs_status,
        seccomp=seccomp_status,
        landlock="pending",
        closed_fd_count=closed_fd_count,
        working_directory=working_directory_status,
        notes=tuple(notes),
    )


def apply_post_load_landlock(
    *,
    report: SandboxHardeningReport,
    readable_root: Path,
) -> SandboxHardeningReport:
    """Apply a filesystem Landlock policy after the trusted module has loaded."""

    landlock_status, note = _apply_landlock_ruleset(readable_root)
    notes = tuple(report.notes) if note is None else tuple((*report.notes, note))
    return replace(report, landlock=landlock_status, notes=notes)


def _apply_resource_limits(limits: SandboxHardeningLimits) -> tuple[str, ...]:
    notes: list[str] = []  # AUDIT-FIX(#7): Collect downgrade notes so callers can see what hardening actually stuck.
    if resource is None:  # pragma: no cover - non-POSIX fallback
        notes.append("resource_unavailable")
        return tuple(notes)
    cpu_note = _set_limit(getattr(resource, "RLIMIT_CPU", None), limits.cpu_seconds, "cpu")
    if cpu_note is not None:
        notes.append(cpu_note)
    address_space_note = _set_limit(getattr(resource, "RLIMIT_AS", None), limits.address_space_bytes, "address_space")
    if address_space_note is not None:
        notes.append(address_space_note)
    nofile_note = _set_limit(getattr(resource, "RLIMIT_NOFILE", None), limits.max_open_files, "nofile")
    if nofile_note is not None:
        notes.append(nofile_note)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError as exc:
            notes.append(f"nice_failed:{exc.errno or 'unknown'}")  # AUDIT-FIX(#7): Report priority-hardening failure instead of swallowing it.
    return tuple(notes)


def _set_limit(limit_name: int | None, value: int, label: str) -> str | None:
    if limit_name is None:
        return f"{label}_limit_unavailable"
    if value <= 0:  # AUDIT-FIX(#4): Skip non-positive limits that would otherwise brick the child immediately.
        return f"{label}_limit_skipped_invalid"
    try:
        resource.setrlimit(limit_name, (int(value), int(value)))  # type: ignore[union-attr]
    except (OSError, ValueError) as exc:
        errno_value = exc.errno if isinstance(exc, OSError) else "invalid"
        return f"{label}_limit_failed:{errno_value}"
    return None


def _close_extra_fds(keep_fds: tuple[int, ...]) -> tuple[int, tuple[str, ...]]:
    open_fds = _list_open_fds()
    if open_fds is not None:  # AUDIT-FIX(#5): Enumerate real open descriptors instead of looping to a potentially massive SC_OPEN_MAX.
        closed_count = 0
        for fd in open_fds:
            if fd < 3 or fd in keep_fds:
                continue
            try:
                os.close(fd)
                closed_count += 1
            except OSError:
                continue
        return closed_count, ()
    max_fd = _max_fd_scan_limit()
    keep = set(keep_fds)
    sorted_keep = sorted(fd for fd in keep if 3 <= fd < max_fd)
    cursor = 3
    for kept_fd in sorted_keep:
        if cursor < kept_fd:
            os.closerange(cursor, kept_fd)  # AUDIT-FIX(#5): Use the C-level closerange fast path when /proc fd listing is unavailable.
        cursor = kept_fd + 1
    if cursor < max_fd:
        os.closerange(cursor, max_fd)  # AUDIT-FIX(#5): Avoid a Python-level close loop across large fd ranges.
    return 0, ("closed_fd_count_unavailable",)


def _sanitize_environment() -> None:
    inherited = dict(os.environ)  # AUDIT-FIX(#8): Snapshot selected runtime variables before clearing the process environment.
    allowed = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PATH": _safe_path_value(inherited.get("PATH")),  # AUDIT-FIX(#8): Preserve absolute PATH entries so subprocess lookup keeps working.
    }
    for variable_name in ("HOME", "TMPDIR", "TEMP", "TMP", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"):
        variable_value = inherited.get(variable_name)
        if variable_value and os.path.isabs(variable_value):  # AUDIT-FIX(#8): Keep only absolute filesystem-backed values to avoid relative-path injection.
            allowed[variable_name] = variable_value
    os.environ.clear()
    os.environ.update(allowed)


def _libc() -> ctypes.CDLL | None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:  # pragma: no cover - defensive fallback
        return None
    if hasattr(libc, "prctl"):
        libc.prctl.restype = ctypes.c_int  # AUDIT-FIX(#7): Use explicit return types so errno-based failure reporting stays reliable.
    if hasattr(libc, "syscall"):
        libc.syscall.restype = ctypes.c_long  # AUDIT-FIX(#7): System calls return long, not int, on 64-bit Linux.
    return libc


def _apply_no_new_privs() -> str:
    libc = _libc()
    if libc is None or not hasattr(libc, "prctl"):
        return "unavailable"
    ctypes.set_errno(0)  # AUDIT-FIX(#7): Clear stale errno before prctl so failure status reflects the current call.
    result = libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if result == 0:
        return "enforced"
    return f"failed_errno_{ctypes.get_errno() or 'unknown'}"


def _read_seccomp_mode() -> str:
    if platform.system() != "Linux":
        return "unavailable"
    try:
        with open(_PROC_SELF_STATUS, "rt", encoding="utf-8", errors="replace") as status_file:  # AUDIT-FIX(#2): Read seccomp state from /proc to avoid PR_GET_SECCOMP killing the process.
            for line in status_file:
                if not line.startswith("Seccomp:"):
                    continue
                raw_mode = line.partition(":")[2].strip()
                if raw_mode == "0":
                    return "mode_none"
                if raw_mode == "1":
                    return "mode_strict"
                if raw_mode == "2":
                    return "mode_filter"
                return f"mode_unknown_{raw_mode}"
    except OSError as exc:
        return f"unavailable_errno_{exc.errno or 'unknown'}"
    return "unavailable"


def _apply_landlock_ruleset(readable_root: Path) -> tuple[str, str | None]:
    if platform.system() != "Linux":
        return "unavailable_non_linux", None
    create_ruleset_number = _syscall_number("landlock_create_ruleset")
    add_rule_number = _syscall_number("landlock_add_rule")
    restrict_self_number = _syscall_number("landlock_restrict_self")
    if create_ruleset_number is None or add_rule_number is None or restrict_self_number is None:
        return "unavailable_architecture", None

    libc = _libc()
    if libc is None or not hasattr(libc, "syscall"):
        return "unavailable_libc", None

    abi_version = _syscall(libc, create_ruleset_number, ctypes.c_void_p(), 0, _LANDLOCK_CREATE_RULESET_VERSION)
    if abi_version < 0:
        error = ctypes.get_errno()
        if error in {errno.ENOSYS, errno.EOPNOTSUPP}:
            return "unsupported_kernel", None
        return f"failed_probe_errno_{error or 'unknown'}", None

    handled_access = _handled_access_mask(int(abi_version))
    allowed_access = _LANDLOCK_ACCESS_FS_EXECUTE | _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_READ_DIR
    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled_access)
    ruleset_fd = _syscall(libc, create_ruleset_number, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
    if ruleset_fd < 0:
        return f"failed_create_errno_{ctypes.get_errno() or 'unknown'}", None

    root_fd = -1
    try:
        try:
            root_fd = _open_existing_directory_fd(readable_root, use_o_path=True)  # AUDIT-FIX(#6): Fail closed on invalid roots before Landlock setup proceeds.
        except OSError as exc:
            return f"failed_open_root_errno_{exc.errno or 'unknown'}", None
        path_attr = _LandlockPathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=root_fd,
            reserved=0,
        )
        add_result = _syscall(
            libc,
            add_rule_number,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(path_attr),
            0,
        )
        if add_result < 0:
            return f"failed_add_rule_errno_{ctypes.get_errno() or 'unknown'}", None
        restrict_result = _syscall(libc, restrict_self_number, ruleset_fd, 0)
        if restrict_result < 0:
            return f"failed_restrict_errno_{ctypes.get_errno() or 'unknown'}", None
    finally:
        try:
            os.close(ruleset_fd)
        except OSError:
            pass
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except OSError:
                pass
    return "enforced", f"landlock_abi_{int(abi_version)}"


def _handled_access_mask(abi_version: int) -> int:
    handled = (
        _LANDLOCK_ACCESS_FS_EXECUTE
        | _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_READ_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi_version >= 2:
        handled |= _LANDLOCK_ACCESS_FS_REFER
    if abi_version >= 3:
        handled |= _LANDLOCK_ACCESS_FS_TRUNCATE
    return handled


def _list_open_fds() -> list[int] | None:
    for fd_directory in (_PROC_SELF_FD, "/dev/fd"):
        try:
            return sorted({int(entry_name) for entry_name in os.listdir(fd_directory) if entry_name.isdigit()})
        except OSError:
            continue
    return None


def _max_fd_scan_limit() -> int:
    if resource is not None and hasattr(resource, "RLIMIT_NOFILE"):
        try:
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if isinstance(soft_limit, int) and 0 < soft_limit < (1 << 20):
                return max(256, int(soft_limit))
        except (OSError, ValueError):
            pass
    try:
        sysconf_limit = int(os.sysconf("SC_OPEN_MAX"))
    except (AttributeError, OSError, ValueError):
        return 256
    if sysconf_limit <= 0:
        return 256
    return max(256, min(sysconf_limit, 1 << 20))


def _normalize_keep_fds(keep_fds: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    keep: set[int] = {0, 1, 2}
    notes: list[str] = []
    for raw_fd in keep_fds:
        try:
            fd = int(raw_fd)
        except (TypeError, ValueError):
            notes.append("invalid_keep_fd_ignored")
            continue
        if fd < 0:
            notes.append("negative_keep_fd_ignored")
            continue
        keep.add(fd)
    return tuple(sorted(keep)), tuple(notes)


def _normalize_limits(
    limits: SandboxHardeningLimits,
    keep_fds: tuple[int, ...],
) -> tuple[SandboxHardeningLimits, tuple[str, ...]]:
    notes: list[str] = []
    cpu_seconds = _coerce_int(limits.cpu_seconds)
    address_space_bytes = _coerce_int(limits.address_space_bytes)
    max_open_files = _coerce_int(limits.max_open_files)
    minimum_nofile = len(keep_fds) + _MIN_HARDENING_FD_HEADROOM
    if cpu_seconds <= 0:
        notes.append("cpu_limit_invalid")
        cpu_seconds = 0
    if address_space_bytes <= 0:
        notes.append("address_space_limit_invalid")
        address_space_bytes = 0
    if max_open_files < minimum_nofile:  # AUDIT-FIX(#4): Keep enough descriptor headroom for /proc reads, secure chdir, and later Landlock setup.
        notes.append(f"nofile_limit_clamped:{minimum_nofile}")
        max_open_files = minimum_nofile
    return (
        SandboxHardeningLimits(
            cpu_seconds=cpu_seconds,
            address_space_bytes=address_space_bytes,
            max_open_files=max_open_files,
        ),
        tuple(notes),
    )


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_path_value(raw_path: str | None) -> str:
    if not raw_path:
        return _DEFAULT_SANITIZED_PATH
    absolute_entries = [entry for entry in raw_path.split(os.pathsep) if entry and os.path.isabs(entry)]
    return os.pathsep.join(absolute_entries) or _DEFAULT_SANITIZED_PATH


def _open_existing_directory_fd(path: Path, *, use_o_path: bool) -> int:
    absolute_path = os.path.abspath(os.fspath(path))
    open_flags = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if hasattr(os, "O_PATH"):  # AUDIT-FIX(#1): O_PATH|O_NOFOLLOW lets us inspect path components without traversing symlinks.
        open_flags |= getattr(os, "O_PATH")
    elif use_o_path:
        open_flags |= os.O_RDONLY
    else:
        open_flags |= os.O_RDONLY
    current_fd = os.open(os.path.sep, open_flags)
    try:
        current_stat = os.fstat(current_fd)
        if not stat.S_ISDIR(current_stat.st_mode):
            raise OSError(errno.ENOTDIR, "root_is_not_directory", absolute_path)
        normalized_path = Path(absolute_path)
        if normalized_path == Path(os.path.sep):
            return current_fd
        for component in normalized_path.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError(errno.EINVAL, "unsafe_directory_component", absolute_path)  # AUDIT-FIX(#1): Refuse lexical traversal components when opening sandbox roots.
            next_fd = os.open(component, open_flags, dir_fd=current_fd)
            next_stat = os.fstat(next_fd)
            if stat.S_ISLNK(next_stat.st_mode):  # AUDIT-FIX(#1): Reject symlink components instead of silently broadening the sandbox root.
                os.close(next_fd)
                raise OSError(errno.ELOOP, "symlink_component_rejected", absolute_path)
            if not stat.S_ISDIR(next_stat.st_mode):  # AUDIT-FIX(#6): Return a controlled failure when the requested root is not a directory.
                os.close(next_fd)
                raise OSError(errno.ENOTDIR, "non_directory_component", absolute_path)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        try:
            os.close(current_fd)
        except OSError:
            pass
        raise


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    ctypes.set_errno(0)
    return int(libc.syscall(number, *args))


__all__ = [
    "SandboxHardeningLimits",
    "SandboxHardeningReport",
    "apply_baseline_os_hardening",
    "apply_post_load_landlock",
]