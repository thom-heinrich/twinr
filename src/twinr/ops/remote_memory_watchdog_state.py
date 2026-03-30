# CHANGELOG: 2026-03-30
# BUG-1: build_starting_sample reported optional remote_primary deployments as "disabled"; startup mode now reflects the effective Twinr config.
# BUG-2: float/int coercion accepted NaN/Infinity and _coerce_int_value could crash on overflow; numeric coercion is now finite-safe and crash-safe.
# BUG-3: persisted history was neither compacted nor capped despite existing constants; save()/to_dict() now strip heavy probe payloads from history and enforce the persisted sample cap.
# BUG-4: persistence relied on dataclasses.asdict() deep copies and a single low-level os.write path; serialization is now explicit, bounded, and fully flushed before replace().
# SEC-1: predictable temp filenames plus 0644 permissions made local clobbering and metadata leakage practical on shared Raspberry Pi deployments; writes now use secure temp files and 0600 permissions.
# SEC-2: load() followed symlinks/special files and parsed unbounded JSON; loads are now size-bounded, regular-file-only, and O_NOFOLLOW-hardened where supported.
# SEC-3: rename-only persistence was not durable across sudden power loss; the containing directory is now fsync()'d after replace().
# IMP-1: optional orjson fast-path enables faster, stricter JSON IO on ARM/Pi builds when installed, while preserving a stdlib fallback.
# IMP-2: persistence now sanitizes nested JSON-ish payloads and validates snapshot shape earlier, making the watchdog state file far more self-healing in real deployments.

"""Persisted state models and storage helpers for the remote-memory watchdog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import stat
import tempfile

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.events import compact_text
from twinr.ops.paths import resolve_ops_paths_for_config

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - optional dependency
    _orjson = None


DEFAULT_WATCHDOG_INTERVAL_S = 1.0
DEFAULT_HISTORY_LIMIT = 3600
SNAPSHOT_SCHEMA_VERSION = 1
PERSISTED_RECENT_SAMPLE_LIMIT = 64
STARTING_SAMPLE_DETAIL = "Remote memory watchdog is starting."

# BREAKING: persisted watchdog snapshots are now owner-readable only (0600) instead of world-readable (0644).
_DEFAULT_SNAPSHOT_FILE_MODE = 0o600
_MAX_SNAPSHOT_BYTES = 1 * 1024 * 1024
_MAX_TEXT_FIELD_CHARS = 240
_MAX_JSON_SANITIZE_DEPTH = 16
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})
_ORJSON_DUMP_OPTIONS = (
    (_orjson.OPT_APPEND_NEWLINE | _orjson.OPT_INDENT_2 | _orjson.OPT_SORT_KEYS)
    if _orjson is not None
    else 0
)
_OVERSIZE_CURRENT_PROBE_DETAIL = (
    "Current probe payload was omitted from the persisted watchdog snapshot "
    "because it exceeded the persistence size budget."
)


def _coerce_float_value(value: object, *, default: float) -> float:
    """Return one finite float from JSON-like payload values."""

    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _coerce_int_value(value: object, *, default: int) -> int:
    """Return one integer from JSON-like payload values."""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return default
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text, 10)
        except ValueError:
            try:
                parsed_float = float(text)
            except (TypeError, ValueError, OverflowError):
                return default
            if not math.isfinite(parsed_float) or not parsed_float.is_integer():
                return default
            return int(parsed_float)
        except OverflowError:
            return default
    return default


def _coerce_bool_value(value: object, *, default: bool) -> bool:
    """Return one conservative boolean from JSON-like payload values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        if value in (0.0, 1.0):
            return bool(int(value))
        return default
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default


def _compact_text_field(value: object, *, default: str | None = None) -> str | None:
    text = compact_text(str(value or "").strip(), limit=_MAX_TEXT_FIELD_CHARS)
    if text:
        return text
    return default


def _sanitize_json_value(
    value: object,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> object:
    """Return one JSON-serializable, finite-only copy of a nested value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if _depth >= _MAX_JSON_SANITIZE_DEPTH:
        return _compact_text_field(value, default=None)

    if isinstance(value, dict):
        obj_id = id(value)
        seen = _seen or set()
        if obj_id in seen:
            return "<cycle>"
        seen.add(obj_id)
        try:
            return {
                str(key): _sanitize_json_value(item, _depth=_depth + 1, _seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.remove(obj_id)

    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        seen = _seen or set()
        if obj_id in seen:
            return ["<cycle>"]
        seen.add(obj_id)
        try:
            return [_sanitize_json_value(item, _depth=_depth + 1, _seen=seen) for item in value]
        finally:
            seen.remove(obj_id)

    if isinstance(value, (set, frozenset)):
        obj_id = id(value)
        seen = _seen or set()
        if obj_id in seen:
            return ["<cycle>"]
        seen.add(obj_id)
        try:
            ordered_items = sorted(value, key=repr)
            return [_sanitize_json_value(item, _depth=_depth + 1, _seen=seen) for item in ordered_items]
        finally:
            seen.remove(obj_id)

    return _compact_text_field(value, default=None)


def _coerce_object_dict(value: object) -> dict[str, object] | None:
    """Copy one generic mapping into a JSON-serializable object dict."""

    if not isinstance(value, dict):
        return None
    sanitized = _sanitize_json_value(value)
    if not isinstance(sanitized, dict):
        return None
    return sanitized


def _json_dumps_bytes(payload: dict[str, object]) -> bytes:
    """Encode one payload to deterministic JSON bytes."""

    if _orjson is not None:
        return _orjson.dumps(payload, option=_ORJSON_DUMP_OPTIONS)
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_loads_bytes(data: bytes) -> object:
    """Decode one JSON payload from bytes."""

    if _orjson is not None:
        return _orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def _append_detail_suffix(detail: str | None, suffix: str) -> str:
    """Append one operational suffix to a compact detail string once."""

    if not detail:
        return suffix
    if suffix in detail:
        return detail
    return f"{detail} {suffix}"


def _bound_snapshot_payload_size(
    payload: dict[str, object],
    *,
    current: "RemoteMemoryWatchdogSample",
) -> dict[str, object]:
    """Ensure the persisted snapshot stays within the bounded on-disk budget."""

    encoded = _json_dumps_bytes(payload)
    if len(encoded) <= _MAX_SNAPSHOT_BYTES:
        return payload

    if current.probe is not None:
        trimmed_payload = dict(payload)
        trimmed_current = compact_history_sample(current).to_dict()
        trimmed_current["detail"] = RemoteMemoryWatchdogSample._coerce_optional_text(
            _append_detail_suffix(
                RemoteMemoryWatchdogSample._coerce_optional_text(trimmed_current.get("detail")),
                _OVERSIZE_CURRENT_PROBE_DETAIL,
            )
        )
        trimmed_payload["current"] = trimmed_current
        encoded = _json_dumps_bytes(trimmed_payload)
        if len(encoded) <= _MAX_SNAPSHOT_BYTES:
            return trimmed_payload

    raise RuntimeError(
        f"Remote memory watchdog snapshot exceeds the maximum supported size of {_MAX_SNAPSHOT_BYTES} bytes after compaction."
    )


def utc_now_iso() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_interval_s(value: object, *, default: float) -> float:
    """Normalize the watchdog interval to a finite positive float."""

    interval_s = _coerce_float_value(value, default=default)
    if interval_s <= 0.0:
        return default
    return interval_s


def coerce_history_limit(value: object, *, default: int) -> int:
    """Normalize the rolling sample-history length."""

    history_limit = _coerce_int_value(value, default=default)
    if history_limit <= 0:
        return default
    return history_limit


def _fsync_directory(path: Path) -> None:
    """Force the containing directory entry to disk."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if directory_flag:
        flags |= directory_flag
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically and durably to disk."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = _json_dumps_bytes(payload)
    if len(encoded) > _MAX_SNAPSHOT_BYTES:
        raise RuntimeError(
            f"Remote memory watchdog snapshot exceeds the maximum supported size of {_MAX_SNAPSHOT_BYTES} bytes."
        )

    fd: int | None = None
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=False,
        )
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(fd)
        fd = None

        os.replace(tmp_name, path)
        os.chmod(path, _DEFAULT_SNAPSHOT_FILE_MODE)
        _fsync_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def _read_snapshot_bytes(path: Path) -> bytes:
    """Read one bounded snapshot file without following a final symlink."""

    # BREAKING: the final snapshot path component may no longer be a symlink.
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if nofollow_flag:
        flags |= nofollow_flag

    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Remote memory watchdog snapshot is not a regular file: {path}")
        if file_stat.st_size > _MAX_SNAPSHOT_BYTES:
            raise RuntimeError(
                f"Remote memory watchdog snapshot is too large ({file_stat.st_size} bytes > {_MAX_SNAPSHOT_BYTES} bytes): {path}"
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(_MAX_SNAPSHOT_BYTES + 1)
    finally:
        os.close(fd)

    if len(raw) > _MAX_SNAPSHOT_BYTES:
        raise RuntimeError(
            f"Remote memory watchdog snapshot exceeds the maximum supported size of {_MAX_SNAPSHOT_BYTES} bytes: {path}"
        )
    return raw


def _effective_long_term_memory_mode(config: TwinrConfig) -> str:
    """Return the effective long-term-memory mode exposed to operators."""

    enabled = bool(getattr(config, "long_term_memory_enabled", False))
    if not enabled:
        return "disabled"
    mode = str(getattr(config, "long_term_memory_mode", "unknown") or "unknown").strip()
    return mode or "unknown"


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSample:
    """Capture one remote-memory probe result."""

    seq: int
    captured_at: str
    status: str
    ready: bool
    mode: str
    required: bool
    latency_ms: float
    consecutive_ok: int
    consecutive_fail: int
    detail: str | None = None
    probe: dict[str, object] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSample":
        """Hydrate one sample from persisted JSON data."""

        probe_payload = _coerce_object_dict(payload.get("probe"))
        return cls(
            seq=max(0, _coerce_int_value(payload.get("seq", 0), default=0)),
            captured_at=str(payload.get("captured_at", "") or ""),
            status=str(payload.get("status", "unknown") or "unknown"),
            ready=_coerce_bool_value(payload.get("ready", False), default=False),
            mode=str(payload.get("mode", "unknown") or "unknown"),
            required=_coerce_bool_value(payload.get("required", False), default=False),
            latency_ms=max(0.0, _coerce_float_value(payload.get("latency_ms", 0.0), default=0.0)),
            consecutive_ok=max(0, _coerce_int_value(payload.get("consecutive_ok", 0), default=0)),
            consecutive_fail=max(0, _coerce_int_value(payload.get("consecutive_fail", 0), default=0)),
            detail=cls._coerce_optional_text(payload.get("detail")),
            probe=probe_payload,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "seq": max(0, int(self.seq)),
            "captured_at": str(self.captured_at or ""),
            "status": str(self.status or "unknown"),
            "ready": bool(self.ready),
            "mode": str(self.mode or "unknown"),
            "required": bool(self.required),
            "latency_ms": max(0.0, _coerce_float_value(self.latency_ms, default=0.0)),
            "consecutive_ok": max(0, _coerce_int_value(self.consecutive_ok, default=0)),
            "consecutive_fail": max(0, _coerce_int_value(self.consecutive_fail, default=0)),
            "detail": self._coerce_optional_text(self.detail),
            "probe": _coerce_object_dict(self.probe),
        }

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        return _compact_text_field(value, default=None)


def compact_history_sample(sample: RemoteMemoryWatchdogSample) -> RemoteMemoryWatchdogSample:
    """Drop heavy nested probe payloads from persisted history samples."""

    if sample.probe is None:
        return sample
    return replace(sample, probe=None)


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSnapshot:
    """Persisted rolling watchdog state."""

    schema_version: int
    started_at: str
    updated_at: str
    hostname: str
    pid: int
    interval_s: float
    history_limit: int
    sample_count: int
    failure_count: int
    last_ok_at: str | None
    last_failure_at: str | None
    artifact_path: str
    current: RemoteMemoryWatchdogSample
    recent_samples: tuple[RemoteMemoryWatchdogSample, ...]
    heartbeat_at: str | None = None
    probe_inflight: bool = False
    probe_started_at: str | None = None
    probe_age_s: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSnapshot":
        """Hydrate one persisted snapshot from JSON data."""

        schema_version = _coerce_int_value(payload.get("schema_version", 0), default=0)
        if schema_version > SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Remote memory watchdog snapshot schema version {schema_version} is newer than this runtime supports ({SNAPSHOT_SCHEMA_VERSION})."
            )

        current_payload = payload.get("current")
        if not isinstance(current_payload, dict):
            raise ValueError("Remote memory watchdog snapshot is missing `current`.")

        raw_recent_samples = payload.get("recent_samples")
        if raw_recent_samples is None:
            raw_recent_samples = ()
        if not isinstance(raw_recent_samples, list):
            raise ValueError("Remote memory watchdog snapshot has malformed `recent_samples`.")

        recent_samples: list[RemoteMemoryWatchdogSample] = []
        for item in raw_recent_samples[-PERSISTED_RECENT_SAMPLE_LIMIT:]:
            if isinstance(item, dict):
                recent_samples.append(RemoteMemoryWatchdogSample.from_dict(item))

        return cls(
            schema_version=schema_version or SNAPSHOT_SCHEMA_VERSION,
            started_at=str(payload.get("started_at", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
            hostname=str(payload.get("hostname", "") or ""),
            pid=max(0, _coerce_int_value(payload.get("pid", 0), default=0)),
            interval_s=coerce_interval_s(
                payload.get("interval_s", DEFAULT_WATCHDOG_INTERVAL_S),
                default=DEFAULT_WATCHDOG_INTERVAL_S,
            ),
            history_limit=coerce_history_limit(
                payload.get("history_limit", DEFAULT_HISTORY_LIMIT),
                default=DEFAULT_HISTORY_LIMIT,
            ),
            sample_count=max(0, _coerce_int_value(payload.get("sample_count", 0), default=0)),
            failure_count=max(0, _coerce_int_value(payload.get("failure_count", 0), default=0)),
            last_ok_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_ok_at")),
            last_failure_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_failure_at")),
            artifact_path=str(payload.get("artifact_path", "") or ""),
            current=RemoteMemoryWatchdogSample.from_dict(current_payload),
            recent_samples=tuple(recent_samples),
            heartbeat_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("heartbeat_at")),
            probe_inflight=_coerce_bool_value(payload.get("probe_inflight", False), default=False),
            probe_started_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("probe_started_at")),
            probe_age_s=cls._coerce_optional_float(payload.get("probe_age_s")),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        recent_samples = tuple(
            compact_history_sample(sample) for sample in self.recent_samples[-PERSISTED_RECENT_SAMPLE_LIMIT:]
        )
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "started_at": str(self.started_at or ""),
            "updated_at": str(self.updated_at or ""),
            "hostname": str(self.hostname or ""),
            "pid": max(0, _coerce_int_value(self.pid, default=0)),
            "interval_s": coerce_interval_s(self.interval_s, default=DEFAULT_WATCHDOG_INTERVAL_S),
            "history_limit": coerce_history_limit(self.history_limit, default=DEFAULT_HISTORY_LIMIT),
            "sample_count": max(0, _coerce_int_value(self.sample_count, default=0)),
            "failure_count": max(0, _coerce_int_value(self.failure_count, default=0)),
            "last_ok_at": RemoteMemoryWatchdogSample._coerce_optional_text(self.last_ok_at),
            "last_failure_at": RemoteMemoryWatchdogSample._coerce_optional_text(self.last_failure_at),
            "artifact_path": str(self.artifact_path or ""),
            "current": self.current.to_dict(),
            "recent_samples": [sample.to_dict() for sample in recent_samples],
            "heartbeat_at": RemoteMemoryWatchdogSample._coerce_optional_text(self.heartbeat_at),
            "probe_inflight": bool(self.probe_inflight),
            "probe_started_at": RemoteMemoryWatchdogSample._coerce_optional_text(self.probe_started_at),
            "probe_age_s": self._coerce_optional_float(self.probe_age_s),
        }
        return _bound_snapshot_payload_size(payload, current=self.current)

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        parsed = _coerce_float_value(value, default=-1.0)
        if parsed < 0.0:
            return None
        return parsed


class RemoteMemoryWatchdogStore:
    """Persist the current rolling watchdog snapshot to disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteMemoryWatchdogStore":
        """Build the canonical watchdog-state store for one Twinr config."""

        ops_paths = resolve_ops_paths_for_config(config)
        return cls(ops_paths.ops_store_root / "remote_memory_watchdog.json")

    def save(self, snapshot: RemoteMemoryWatchdogSnapshot) -> None:
        """Persist the latest rolling snapshot atomically."""

        atomic_write_json(self.path, snapshot.to_dict())

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        """Load the persisted rolling snapshot when it exists."""

        try:
            raw_payload = _read_snapshot_bytes(self.path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is unreadable: {self.path}") from exc

        try:
            payload = _json_loads_bytes(raw_payload)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is not valid JSON: {self.path}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Remote memory watchdog snapshot has an invalid top-level payload: {self.path}")

        try:
            return RemoteMemoryWatchdogSnapshot.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is malformed: {self.path}") from exc


def build_starting_sample(config: TwinrConfig, *, captured_at: str) -> RemoteMemoryWatchdogSample:
    """Return the synthetic startup sample used before the first deep probe."""

    mode = _effective_long_term_memory_mode(config)
    required = bool(
        bool(getattr(config, "long_term_memory_enabled", False))
        and mode == "remote_primary"
        and bool(getattr(config, "long_term_memory_remote_required", False))
    )
    return RemoteMemoryWatchdogSample(
        seq=0,
        captured_at=captured_at,
        status="starting",
        ready=False,
        mode=mode,
        required=required,
        latency_ms=0.0,
        consecutive_ok=0,
        consecutive_fail=0,
        detail=STARTING_SAMPLE_DETAIL,
    )


def build_remote_memory_watchdog_bootstrap_snapshot(
    config: TwinrConfig,
    *,
    pid: int,
    artifact_path: str | Path,
    started_at: str | None = None,
    captured_at: str | None = None,
) -> RemoteMemoryWatchdogSnapshot:
    """Build the supervisor-seeded startup snapshot for a fresh watchdog child."""

    resolved_started_at = str(started_at or utc_now_iso())
    resolved_captured_at = str(captured_at or resolved_started_at)
    interval_s = coerce_interval_s(
        getattr(config, "long_term_memory_remote_watchdog_interval_s", DEFAULT_WATCHDOG_INTERVAL_S),
        default=DEFAULT_WATCHDOG_INTERVAL_S,
    )
    history_limit = coerce_history_limit(
        getattr(config, "long_term_memory_remote_watchdog_history_limit", DEFAULT_HISTORY_LIMIT),
        default=DEFAULT_HISTORY_LIMIT,
    )
    return RemoteMemoryWatchdogSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        started_at=resolved_started_at,
        updated_at=resolved_captured_at,
        hostname=socket.gethostname(),
        pid=max(0, int(pid)),
        interval_s=interval_s,
        history_limit=history_limit,
        sample_count=0,
        failure_count=0,
        last_ok_at=None,
        last_failure_at=None,
        artifact_path=str(artifact_path),
        current=build_starting_sample(config, captured_at=resolved_captured_at),
        recent_samples=(),
        heartbeat_at=resolved_captured_at,
        probe_inflight=True,
        probe_started_at=resolved_captured_at,
        probe_age_s=0.0,
    )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_WATCHDOG_INTERVAL_S",
    "PERSISTED_RECENT_SAMPLE_LIMIT",
    "RemoteMemoryWatchdogSample",
    "RemoteMemoryWatchdogSnapshot",
    "RemoteMemoryWatchdogStore",
    "SNAPSHOT_SCHEMA_VERSION",
    "STARTING_SAMPLE_DETAIL",
    "atomic_write_json",
    "build_remote_memory_watchdog_bootstrap_snapshot",
    "build_starting_sample",
    "coerce_history_limit",
    "coerce_interval_s",
    "compact_history_sample",
    "utc_now_iso",
]
