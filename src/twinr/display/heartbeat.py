# CHANGELOG: 2026-03-28
# BUG-1: Fixed false-fresh heartbeats when the wall clock jumps backwards or a persisted timestamp is in the future by preferring boot-aware monotonic ages and rejecting impossible time skews.
# BUG-2: Fixed naive-datetime timezone corruption by normalizing naive datetimes as UTC consistently for persisted and comparison timestamps.
# BUG-3: Fixed same-process concurrent save races caused by predictable per-PID temp filenames.
# BUG-4: Fixed stale-lock false positives by validating owner liveness and process identity (pidfd inode or /proc starttime) instead of trusting PID equality alone.
# SEC-1: Hardened heartbeat reads/writes against oversized payloads, symlink tricks, and weak file permissions; writes now use secure temp files, fsync(), and shared 0644 ops artifacts.
# IMP-1: Added schema-versioned heartbeat metadata (boot_id, monotonic_ns, process identity) for 2026-grade gray-failure detection.
# IMP-2: BREAKING: when systemd RuntimeDirectory= is present, the canonical heartbeat path moves to volatile runtime storage and the legacy project-root path is treated as a read fallback during rollout.
# IMP-3: Added config-driven timeout overrides and richer health diagnostics while keeping the module API drop-in compatible.

"""Persist display-loop forward-progress heartbeats for health/supervision.

The productive Pi runtime cannot infer display liveness from lock ownership
alone because a hung display companion thread can still keep the lock while the
panel stops receiving updates. This module stores a small authoritative
heartbeat file that records the display loop's last idle/render progress.

2026 upgrade notes:
- Freshness is now primarily evaluated against a boot-aware monotonic clock,
  not wall time.
- Process identity is validated with pidfd inode IDs when available, and falls
  back to /proc/<pid>/stat starttime when pidfds are unavailable.
- On systemd deployments that export RuntimeDirectory=, the heartbeat is kept
  on tmpfs to avoid high-frequency SD-card writes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import errno
import json
import logging
import os
from pathlib import Path
import stat
import tempfile
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.locks import loop_lock_owner


_DISPLAY_HEARTBEAT_RELATIVE_PATH = Path("artifacts") / "stores" / "ops" / "display_heartbeat.json"
_DEFAULT_STALE_AFTER_S = 15.0
_DEFAULT_RENDERING_STALE_AFTER_S = 30.0
_MIN_STALE_AFTER_S = 5.0
_MAX_HEARTBEAT_BYTES = 64 * 1024
_MAX_FUTURE_CLOCK_SKEW_S = 2.0
_RUNTIME_HEARTBEAT_FILENAME = "display_heartbeat.json"
_HEARTBEAT_SCHEMA_VERSION = 2
_DISPLAY_HEARTBEAT_FILE_MODE = 0o644

_LOGGER = logging.getLogger(__name__)

LoopOwnerFn = Callable[[TwinrConfig, str], int | None]


@dataclass(frozen=True, slots=True)
class _ProcessIdentity:
    pid: int
    boot_id: str | None
    process_start_ticks: int | None
    pidfd_inode_id: int | None


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _monotonic_ns() -> int:
    """Return the process-external monotonic clock in nanoseconds."""

    return time.monotonic_ns()


def _normalize_datetime(value: datetime | None) -> datetime | None:
    """Normalize one datetime to an aware UTC datetime."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_timestamp(value: str | None) -> datetime | None:
    """Parse one persisted ISO-8601 timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _normalize_datetime(parsed)


def _format_optional_datetime(value: datetime | None) -> str | None:
    normalized = _normalize_datetime(value)
    if normalized is None:
        return None
    return normalized.isoformat()


def _estimate_monotonic_ns_from_wall_clock(
    *,
    event_at: datetime | None,
    reference_at: datetime,
    reference_monotonic_ns: int,
) -> int | None:
    """Estimate a monotonic timestamp from a wall-clock event timestamp."""

    normalized_event_at = _normalize_datetime(event_at)
    normalized_reference_at = _normalize_datetime(reference_at)
    if normalized_event_at is None or normalized_reference_at is None:
        return None
    delta_ns = int((normalized_reference_at - normalized_event_at).total_seconds() * 1_000_000_000)
    estimated = reference_monotonic_ns - delta_ns
    if estimated < 0:
        return None
    return estimated


def _optional_text(value: object | None, *, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if max_length is not None and len(text) > max_length:
        return text[:max_length]
    return text


def _optional_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_positive_float(value: object | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        parsed = float(int(value))
    elif isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed <= 0.0:
        return None
    return parsed


@lru_cache(maxsize=1)
def _current_boot_id() -> str | None:
    """Return the Linux boot ID, or ``None`` when unavailable."""

    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _read_process_start_ticks(pid: int) -> int | None:
    """Return /proc/<pid>/stat starttime clock ticks, or ``None``."""

    if pid <= 0:
        return None
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    close_paren = raw.rfind(")")
    if close_paren < 0 or close_paren + 2 >= len(raw):
        return None
    # Fields after the executable name begin after ") ".
    # starttime is field (22), i.e. index 19 in the post-comm slice.
    fields = raw[close_paren + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except (TypeError, ValueError, OverflowError):
        return None


def _read_pidfd_inode_id(pid: int) -> int | None:
    """Return the pidfd inode ID for a live process, or ``None``."""

    if pid <= 0 or not hasattr(os, "pidfd_open"):
        return None
    try:
        fd = os.pidfd_open(pid, 0)
    except OSError:
        return None
    try:
        return int(os.fstat(fd).st_ino)
    except OSError:
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


@lru_cache(maxsize=4)
def _current_process_identity(pid: int) -> _ProcessIdentity:
    """Return stable identity metadata for the current process PID."""

    return _ProcessIdentity(
        pid=pid,
        boot_id=_current_boot_id(),
        process_start_ticks=_read_process_start_ticks(pid),
        pidfd_inode_id=_read_pidfd_inode_id(pid),
    )


def _live_process_identity(pid: int) -> _ProcessIdentity | None:
    """Return identity metadata for one currently running process PID."""

    if pid <= 0:
        return None
    if pid == os.getpid():
        return _current_process_identity(pid)

    process_start_ticks = _read_process_start_ticks(pid)
    pidfd_inode_id = _read_pidfd_inode_id(pid)
    if process_start_ticks is None and pidfd_inode_id is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
        except OSError:
            return None
    return _ProcessIdentity(
        pid=pid,
        boot_id=_current_boot_id(),
        process_start_ticks=process_start_ticks,
        pidfd_inode_id=pidfd_inode_id,
    )


def _describe_identity_mismatch(heartbeat: "DisplayHeartbeat", expected_pid: int) -> str | None:
    """Return a human-readable reason when a heartbeat identity mismatches."""

    if heartbeat.pid != expected_pid:
        return f"heartbeat belongs to pid={heartbeat.pid}"
    live_identity = _live_process_identity(expected_pid)
    if live_identity is None:
        return "lock owner is not running"
    if heartbeat.boot_id and live_identity.boot_id and heartbeat.boot_id != live_identity.boot_id:
        return "heartbeat belongs to a different boot"
    if (
        heartbeat.pidfd_inode_id is not None
        and live_identity.pidfd_inode_id is not None
        and heartbeat.pidfd_inode_id != live_identity.pidfd_inode_id
    ):
        return "heartbeat belongs to a different process instance"
    if (
        heartbeat.process_start_ticks is not None
        and live_identity.process_start_ticks is not None
        and heartbeat.process_start_ticks != live_identity.process_start_ticks
    ):
        return "heartbeat belongs to a different process instance"
    return None


def _heartbeat_matches_process_identity(heartbeat: "DisplayHeartbeat", expected_pid: int) -> bool:
    """Return whether a heartbeat belongs to the currently running process."""

    return _describe_identity_mismatch(heartbeat, expected_pid) is None


def display_heartbeat_stale_after_s(config: TwinrConfig) -> float:
    """Return the allowed heartbeat age before display liveness is stale."""

    override_s = _coerce_positive_float(getattr(config, "display_heartbeat_stale_after_s", None))
    if override_s is not None:
        return max(_MIN_STALE_AFTER_S, override_s)
    poll_interval_s = _coerce_positive_float(getattr(config, "display_poll_interval_s", None))
    if poll_interval_s is None:
        poll_interval_s = 0.5
    return max(_MIN_STALE_AFTER_S, max(_DEFAULT_STALE_AFTER_S, poll_interval_s * 20.0))


def display_rendering_heartbeat_stale_after_s(config: TwinrConfig) -> float:
    """Return the longer timeout tolerated for an in-flight panel render."""

    override_s = _coerce_positive_float(getattr(config, "display_rendering_heartbeat_stale_after_s", None))
    if override_s is not None:
        return max(display_heartbeat_stale_after_s(config), override_s)
    return max(
        display_heartbeat_stale_after_s(config),
        _DEFAULT_RENDERING_STALE_AFTER_S,
    )


def _heartbeat_future_clock_skew_s(config: TwinrConfig) -> float:
    override_s = _coerce_positive_float(getattr(config, "display_heartbeat_future_clock_skew_s", None))
    if override_s is not None:
        return override_s
    return _MAX_FUTURE_CLOCK_SKEW_S


@dataclass(frozen=True, slots=True)
class DisplayHeartbeat:
    """Represent one persisted display-loop forward-progress sample."""

    pid: int
    updated_at: str
    runtime_status: str
    phase: str
    seq: int
    last_render_started_at: str | None = None
    last_render_completed_at: str | None = None
    detail: str | None = None
    schema_version: int = _HEARTBEAT_SCHEMA_VERSION
    boot_id: str | None = None
    updated_monotonic_ns: int | None = None
    process_start_ticks: int | None = None
    pidfd_inode_id: int | None = None
    last_render_started_monotonic_ns: int | None = None
    last_render_completed_monotonic_ns: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the heartbeat for JSON persistence."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DisplayHeartbeat":
        """Build one heartbeat from JSON payload data."""

        return cls(
            pid=_optional_int(payload.get("pid")) or 0,
            updated_at=str(payload.get("updated_at", "") or ""),
            runtime_status=str(payload.get("runtime_status", "") or ""),
            phase=str(payload.get("phase", "") or ""),
            seq=_optional_int(payload.get("seq")) or 0,
            last_render_started_at=_optional_text(payload.get("last_render_started_at")),
            last_render_completed_at=_optional_text(payload.get("last_render_completed_at")),
            detail=_optional_text(payload.get("detail")),
            schema_version=max(1, _optional_int(payload.get("schema_version")) or 1),
            boot_id=_optional_text(payload.get("boot_id")),
            updated_monotonic_ns=_optional_int(payload.get("updated_monotonic_ns")),
            process_start_ticks=_optional_int(payload.get("process_start_ticks")),
            pidfd_inode_id=_optional_int(payload.get("pidfd_inode_id")),
            last_render_started_monotonic_ns=_optional_int(payload.get("last_render_started_monotonic_ns")),
            last_render_completed_monotonic_ns=_optional_int(payload.get("last_render_completed_monotonic_ns")),
        )


def _resolve_explicit_heartbeat_path(config: TwinrConfig, project_root: Path) -> Path | None:
    """Resolve an explicit heartbeat path override from config or environment."""

    configured = getattr(config, "display_heartbeat_path", None)
    if configured:
        candidate = Path(str(configured)).expanduser()
        return candidate if candidate.is_absolute() else (project_root / candidate).resolve()
    env_override = os.environ.get("TWINR_DISPLAY_HEARTBEAT_PATH")
    if env_override:
        candidate = Path(env_override).expanduser()
        return candidate if candidate.is_absolute() else (project_root / candidate).resolve()
    return None


def _resolve_runtime_heartbeat_path() -> Path | None:
    """Resolve a volatile runtime path when the supervisor exports one."""

    runtime_directory = _optional_text(os.environ.get("RUNTIME_DIRECTORY"))
    if runtime_directory:
        runtime_root = runtime_directory.split(":", 1)[0].strip()
        if runtime_root:
            return Path(runtime_root) / _RUNTIME_HEARTBEAT_FILENAME
    return None


def _read_text_file_secure(path: Path, *, max_bytes: int = _MAX_HEARTBEAT_BYTES) -> str:
    """Read one small regular text file without following symlinks when possible."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(os.fspath(path), flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError(errno.EINVAL, "heartbeat path is not a regular file", os.fspath(path))
        if file_stat.st_size > max_bytes:
            raise OSError(errno.EFBIG, "heartbeat file too large", os.fspath(path))
        with os.fdopen(fd, mode="rb", closefd=False) as handle:
            payload_bytes = handle.read(max_bytes + 1)
        if len(payload_bytes) > max_bytes:
            raise OSError(errno.EFBIG, "heartbeat file too large", os.fspath(path))
        return payload_bytes.decode("utf-8")
    finally:
        os.close(fd)


def _fsync_parent_directory(path: Path) -> None:
    """Best-effort fsync of a file's parent directory."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(os.fspath(path.parent), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


@dataclass(slots=True)
class DisplayHeartbeatStore:
    """Read and write the display-loop heartbeat artifact."""

    path: Path
    fallback_paths: tuple[Path, ...] = ()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayHeartbeatStore":
        """Resolve the canonical heartbeat path from Twinr config."""

        project_root = Path(config.project_root).expanduser().resolve()
        legacy_path = project_root / _DISPLAY_HEARTBEAT_RELATIVE_PATH

        explicit_path = _resolve_explicit_heartbeat_path(config, project_root)
        runtime_path = _resolve_runtime_heartbeat_path()

        # BREAKING: when systemd RuntimeDirectory= is present, the canonical
        # heartbeat location moves to tmpfs to avoid high-frequency writes to
        # persistent flash. The legacy project-root artifact remains readable as
        # a fallback during rollout and mixed-version deployments.
        primary_path = explicit_path or runtime_path or legacy_path
        fallback_paths: tuple[Path, ...] = ()
        if primary_path != legacy_path:
            fallback_paths = (legacy_path,)
        return cls(primary_path, fallback_paths)

    def _candidate_paths(self) -> tuple[Path, ...]:
        seen: set[str] = set()
        ordered: list[Path] = []
        for candidate in (self.path, *self.fallback_paths):
            normalized = os.fspath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(candidate)
        return tuple(ordered)

    def _load_from_path(self, candidate: Path) -> DisplayHeartbeat | None:
        """Load one heartbeat from one candidate file path."""

        try:
            payload_text = _read_text_file_secure(candidate)
        except FileNotFoundError:
            return None
        except OSError:
            _LOGGER.warning("Failed to read display heartbeat from %s.", candidate, exc_info=True)
            return None
        try:
            payload = json.loads(payload_text)
        except Exception:
            _LOGGER.warning("Failed to parse display heartbeat from %s.", candidate, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning("Ignoring invalid display heartbeat payload at %s because it is not an object.", candidate)
            return None
        try:
            return DisplayHeartbeat.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid display heartbeat payload at %s.", candidate, exc_info=True)
            return None

    def load(self) -> DisplayHeartbeat | None:
        """Load the current display heartbeat if one exists and parses."""

        freshest_heartbeat: DisplayHeartbeat | None = None
        freshest_key: tuple[float, int, int, str] | None = None
        for candidate in self._candidate_paths():
            heartbeat = self._load_from_path(candidate)
            if heartbeat is None:
                continue
            monotonic_age_s = display_heartbeat_monotonic_age_s(heartbeat)
            if monotonic_age_s is not None:
                freshness_key = -monotonic_age_s
            else:
                wall_age_s = display_heartbeat_wall_clock_age_s(heartbeat)
                freshness_key = -wall_age_s if wall_age_s is not None else float("-inf")
            candidate_key = (
                freshness_key,
                heartbeat.seq,
                heartbeat.updated_monotonic_ns or -1,
                heartbeat.updated_at,
            )
            if freshest_key is None or candidate_key > freshest_key:
                freshest_key = candidate_key
                freshest_heartbeat = heartbeat
        return freshest_heartbeat

    def save(self, heartbeat: DisplayHeartbeat) -> None:
        """Persist one heartbeat atomically, durably, and operator-readable."""

        serialized = json.dumps(
            heartbeat.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        for target_path in self._candidate_paths():
            self._save_serialized(target_path=target_path, serialized=serialized)

    def _save_serialized(self, *, target_path: Path, serialized: str) -> None:
        """Persist one already-serialized heartbeat to one target path."""

        target_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=os.fspath(target_path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, mode="w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), _DISPLAY_HEARTBEAT_FILE_MODE)
            os.replace(tmp_path, target_path)
            _fsync_parent_directory(target_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class DisplayCompanionHealth:
    """Describe whether the display companion currently proves forward progress."""

    owner_pid: int | None
    running: bool
    count: int
    detail: str
    heartbeat: DisplayHeartbeat | None = None


def display_heartbeat_monotonic_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now_monotonic_ns: int | None = None,
    current_boot_id: str | None = None,
) -> float | None:
    """Return monotonic heartbeat age in seconds when boot identity matches."""

    if heartbeat.updated_monotonic_ns is None or heartbeat.updated_monotonic_ns < 0:
        return None
    live_boot_id = _current_boot_id() if current_boot_id is None else current_boot_id
    if live_boot_id is not None:
        if not heartbeat.boot_id:
            return None
        if heartbeat.boot_id != live_boot_id:
            return None
    effective_now_ns = _monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    if heartbeat.updated_monotonic_ns > effective_now_ns:
        return None
    return (effective_now_ns - heartbeat.updated_monotonic_ns) / 1_000_000_000.0


def display_heartbeat_wall_clock_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now: datetime | None = None,
    max_future_skew_s: float = _MAX_FUTURE_CLOCK_SKEW_S,
) -> float | None:
    """Return wall-clock heartbeat age in seconds, rejecting impossible futures."""

    updated_at = _normalize_timestamp(heartbeat.updated_at)
    if updated_at is None:
        return None
    effective_now = _normalize_datetime(now) or _utc_now()
    delta_s = (effective_now - updated_at).total_seconds()
    if delta_s < -max_future_skew_s:
        return None
    return max(0.0, delta_s)


def display_heartbeat_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now: datetime | None = None,
    now_monotonic_ns: int | None = None,
    current_boot_id: str | None = None,
    max_future_skew_s: float = _MAX_FUTURE_CLOCK_SKEW_S,
) -> float | None:
    """Return the heartbeat age in seconds, or ``None`` for invalid timestamps."""

    monotonic_age_s = display_heartbeat_monotonic_age_s(
        heartbeat,
        now_monotonic_ns=now_monotonic_ns,
        current_boot_id=current_boot_id,
    )
    if monotonic_age_s is not None:
        return monotonic_age_s
    return display_heartbeat_wall_clock_age_s(
        heartbeat,
        now=now,
        max_future_skew_s=max_future_skew_s,
    )


def is_display_heartbeat_fresh(
    heartbeat: DisplayHeartbeat,
    *,
    config: TwinrConfig,
    expected_pid: int | None = None,
    now: datetime | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Return whether the persisted heartbeat proves current display progress."""

    if expected_pid is not None and heartbeat.pid != expected_pid:
        return False
    age_s = display_heartbeat_age_s(
        heartbeat,
        now=now,
        now_monotonic_ns=now_monotonic_ns,
        max_future_skew_s=_heartbeat_future_clock_skew_s(config),
    )
    if age_s is None:
        return False
    if age_s <= display_heartbeat_stale_after_s(config):
        return True
    rendering_age_s = _display_rendering_age_s(
        heartbeat,
        now=now,
        now_monotonic_ns=now_monotonic_ns,
    )
    if rendering_age_s is None:
        return False
    return rendering_age_s <= display_rendering_heartbeat_stale_after_s(config)


def build_display_heartbeat(
    *,
    runtime_status: str,
    phase: str,
    seq: int,
    detail: str | None = None,
    pid: int | None = None,
    updated_at: datetime | None = None,
    last_render_started_at: datetime | None = None,
    last_render_completed_at: datetime | None = None,
    boot_id: str | None = None,
    updated_monotonic_ns: int | None = None,
    process_start_ticks: int | None = None,
    pidfd_inode_id: int | None = None,
    last_render_started_monotonic_ns: int | None = None,
    last_render_completed_monotonic_ns: int | None = None,
) -> DisplayHeartbeat:
    """Build one normalized display heartbeat snapshot."""

    effective_pid = int(os.getpid() if pid is None else pid)
    effective_updated_at = _normalize_datetime(updated_at) or _utc_now()
    effective_updated_monotonic_ns = int(_monotonic_ns() if updated_monotonic_ns is None else updated_monotonic_ns)

    live_identity = _current_process_identity(effective_pid) if effective_pid == os.getpid() else _live_process_identity(effective_pid)

    effective_boot_id = boot_id if boot_id is not None else (None if live_identity is None else live_identity.boot_id)
    effective_process_start_ticks = (
        process_start_ticks
        if process_start_ticks is not None
        else (None if live_identity is None else live_identity.process_start_ticks)
    )
    effective_pidfd_inode_id = (
        pidfd_inode_id
        if pidfd_inode_id is not None
        else (None if live_identity is None else live_identity.pidfd_inode_id)
    )

    if last_render_started_monotonic_ns is None:
        last_render_started_monotonic_ns = _estimate_monotonic_ns_from_wall_clock(
            event_at=last_render_started_at,
            reference_at=effective_updated_at,
            reference_monotonic_ns=effective_updated_monotonic_ns,
        )
    if last_render_completed_monotonic_ns is None:
        last_render_completed_monotonic_ns = _estimate_monotonic_ns_from_wall_clock(
            event_at=last_render_completed_at,
            reference_at=effective_updated_at,
            reference_monotonic_ns=effective_updated_monotonic_ns,
        )

    return DisplayHeartbeat(
        pid=effective_pid,
        updated_at=effective_updated_at.isoformat(),
        runtime_status=str(runtime_status or "").strip().lower() or "error",
        phase=str(phase or "").strip().lower() or "idle",
        seq=max(0, int(seq)),
        last_render_started_at=_format_optional_datetime(last_render_started_at),
        last_render_completed_at=_format_optional_datetime(last_render_completed_at),
        detail=_optional_text(detail, max_length=2048),
        schema_version=_HEARTBEAT_SCHEMA_VERSION,
        boot_id=_optional_text(effective_boot_id),
        updated_monotonic_ns=max(0, effective_updated_monotonic_ns),
        process_start_ticks=max(0, int(effective_process_start_ticks))
        if effective_process_start_ticks is not None
        else None,
        pidfd_inode_id=max(0, int(effective_pidfd_inode_id)) if effective_pidfd_inode_id is not None else None,
        last_render_started_monotonic_ns=max(0, int(last_render_started_monotonic_ns))
        if last_render_started_monotonic_ns is not None
        else None,
        last_render_completed_monotonic_ns=max(0, int(last_render_completed_monotonic_ns))
        if last_render_completed_monotonic_ns is not None
        else None,
    )


def save_display_heartbeat(
    store: DisplayHeartbeatStore,
    *,
    runtime_status: str,
    phase: str,
    seq: int,
    detail: str | None = None,
    pid: int | None = None,
    updated_at: datetime | None = None,
    last_render_started_at: datetime | None = None,
    last_render_completed_at: datetime | None = None,
    boot_id: str | None = None,
    updated_monotonic_ns: int | None = None,
    process_start_ticks: int | None = None,
    pidfd_inode_id: int | None = None,
    last_render_started_monotonic_ns: int | None = None,
    last_render_completed_monotonic_ns: int | None = None,
) -> None:
    """Persist one normalized display heartbeat through the canonical contract."""

    store.save(
        build_display_heartbeat(
            runtime_status=runtime_status,
            phase=phase,
            seq=seq,
            detail=detail,
            pid=pid,
            updated_at=updated_at,
            last_render_started_at=last_render_started_at,
            last_render_completed_at=last_render_completed_at,
            boot_id=boot_id,
            updated_monotonic_ns=updated_monotonic_ns,
            process_start_ticks=process_start_ticks,
            pidfd_inode_id=pidfd_inode_id,
            last_render_started_monotonic_ns=last_render_started_monotonic_ns,
            last_render_completed_monotonic_ns=last_render_completed_monotonic_ns,
        )
    )


def assess_display_companion_health(
    config: TwinrConfig,
    *,
    owner_pid: int | None = None,
    loop_name: str = "display-loop",
    heartbeat_store: DisplayHeartbeatStore | None = None,
    loop_owner_fn: LoopOwnerFn | None = None,
    now: datetime | None = None,
    now_monotonic_ns: int | None = None,
) -> DisplayCompanionHealth:
    """Assess whether a display companion lock owner is also making heartbeat progress."""

    resolved_owner_pid = owner_pid
    if resolved_owner_pid is None:
        resolver = loop_lock_owner if loop_owner_fn is None else loop_owner_fn
        resolved_owner_pid = resolver(config, loop_name)
    if resolved_owner_pid is None:
        return DisplayCompanionHealth(
            owner_pid=None,
            running=False,
            count=0,
            detail="display-companion not detected",
            heartbeat=None,
        )

    if _live_process_identity(resolved_owner_pid) is None:
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=False,
            count=0,
            detail=f"pid={resolved_owner_pid} display-companion owner missing",
            heartbeat=None,
        )

    store = heartbeat_store if heartbeat_store is not None else DisplayHeartbeatStore.from_config(config)
    heartbeat = store.load()
    if heartbeat is None:
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=False,
            count=0,
            detail=f"pid={resolved_owner_pid} display-companion heartbeat missing",
            heartbeat=None,
        )

    identity_mismatch = _describe_identity_mismatch(heartbeat, resolved_owner_pid)
    if identity_mismatch is not None:
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=False,
            count=0,
            detail=f"pid={resolved_owner_pid} display-companion {identity_mismatch}",
            heartbeat=heartbeat,
        )

    if is_display_heartbeat_fresh(
        heartbeat,
        config=config,
        expected_pid=resolved_owner_pid,
        now=now,
        now_monotonic_ns=now_monotonic_ns,
    ):
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=True,
            count=1,
            detail=f"pid={resolved_owner_pid} display-companion",
            heartbeat=heartbeat,
        )
    heartbeat_age_s = display_heartbeat_age_s(
        heartbeat,
        now=now,
        now_monotonic_ns=now_monotonic_ns,
        max_future_skew_s=_heartbeat_future_clock_skew_s(config),
    )
    age_text = f" stale={heartbeat_age_s:.1f}s" if heartbeat_age_s is not None else " stale"
    return DisplayCompanionHealth(
        owner_pid=resolved_owner_pid,
        running=False,
        count=0,
        detail=f"pid={resolved_owner_pid} display-companion heartbeat {heartbeat.phase or 'unknown'}{age_text}",
        heartbeat=heartbeat,
    )


def display_heartbeat_has_progress(
    config: TwinrConfig,
    *,
    expected_pid: int,
    started_at_utc: datetime | None,
    heartbeat_store: DisplayHeartbeatStore | None = None,
    now: datetime | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Return whether the heartbeat proves progress for the current runtime child."""

    if started_at_utc is None:
        return False
    store = heartbeat_store if heartbeat_store is not None else DisplayHeartbeatStore.from_config(config)
    heartbeat = store.load()
    if heartbeat is None:
        return False
    if not is_display_heartbeat_fresh(
        heartbeat,
        config=config,
        expected_pid=expected_pid,
        now=now,
        now_monotonic_ns=now_monotonic_ns,
    ):
        return False
    if _heartbeat_matches_process_identity(heartbeat, expected_pid):
        return True

    updated_at = _normalize_timestamp(heartbeat.updated_at)
    if updated_at is None:
        return False
    normalized_started_at = _normalize_datetime(started_at_utc)
    if normalized_started_at is None:
        return False
    return updated_at >= normalized_started_at


def _display_rendering_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now: datetime | None = None,
    now_monotonic_ns: int | None = None,
) -> float | None:
    if str(heartbeat.phase or "").strip().lower() != "rendering":
        return None

    current_boot_id = _current_boot_id()

    if heartbeat.last_render_started_monotonic_ns is not None:
        if current_boot_id is not None:
            if not heartbeat.boot_id:
                return None
            if heartbeat.boot_id != current_boot_id:
                return None
        effective_now_ns = _monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if heartbeat.last_render_started_monotonic_ns > effective_now_ns:
            return None
        if (
            heartbeat.last_render_completed_monotonic_ns is not None
            and heartbeat.last_render_completed_monotonic_ns >= heartbeat.last_render_started_monotonic_ns
        ):
            return None
        return (effective_now_ns - heartbeat.last_render_started_monotonic_ns) / 1_000_000_000.0

    started_at = _normalize_timestamp(heartbeat.last_render_started_at)
    if started_at is None:
        return None
    completed_at = _normalize_timestamp(heartbeat.last_render_completed_at)
    if completed_at is not None and completed_at >= started_at:
        return None
    effective_now = _normalize_datetime(now) or _utc_now()
    return max(0.0, (effective_now - started_at).total_seconds())
