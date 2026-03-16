"""Persist and tail Twinr operational events as sanitized JSON lines.

The file-backed store keeps event payloads JSON-safe, secret-aware, and
bounded for dashboards, support bundles, and runtime audit trails.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math  # AUDIT-FIX(#3): Normalize NaN/Inf payload values into strict-JSON-safe representations.
import os  # AUDIT-FIX(#1,#2,#5,#6): Use secure low-level file descriptors for locking, safe append, and bounded tail reads.
import stat  # AUDIT-FIX(#1): Reject non-regular files as event-store targets.
import fcntl  # AUDIT-FIX(#2,#5): Coordinate concurrent readers and writers with advisory file locks.
from collections.abc import Mapping, Sequence  # AUDIT-FIX(#3,#4): Recursively sanitize arbitrary mapping/sequence payloads.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config

_REDACTED_VALUE = "[REDACTED]"  # AUDIT-FIX(#4): Prevent obvious secrets from being persisted verbatim.
_SENSITIVE_KEYWORDS = (
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "password",
    "passwd",
    "pwd",
    "refresh_token",
    "secret",
    "session",
    "token",
)  # AUDIT-FIX(#4): Cover the common secret-bearing field names seen in ops payloads.
_REVERSE_READ_CHUNK_SIZE = 4096  # AUDIT-FIX(#6): Bound per-chunk memory during tail reads on RPi-class hardware.
_MAX_REPR_CHARS = 1024  # AUDIT-FIX(#3): Cap fallback repr payload size so malformed objects cannot explode log lines.


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_text(value: str | None, *, limit: int = 160) -> str:
    """Collapse whitespace and truncate text to a bounded single line."""

    text = " ".join((value or "").split())  # AUDIT-FIX(#7): Preserve whitespace compaction while fixing small-limit truncation semantics.
    try:
        normalized_limit = max(int(limit), 0)  # AUDIT-FIX(#7): Clamp negative or malformed limits instead of returning strings longer than requested.
    except (TypeError, ValueError):
        normalized_limit = 0
    if len(text) <= normalized_limit:
        return text
    if normalized_limit <= 3:
        return text[:normalized_limit]  # AUDIT-FIX(#7): For tiny limits, return a hard truncation that never exceeds the requested length.
    return text[: normalized_limit - 3].rstrip() + "..."


def _safe_text(value: object, *, fallback: str, lower: bool = False) -> str:
    text = value if isinstance(value, str) else ("" if value is None else str(value))  # AUDIT-FIX(#8): Accept runtime-non-str inputs without crashing on .strip().
    normalized = text.strip()
    if not normalized:
        normalized = fallback
    return normalized.lower() if lower else normalized


def _safe_key(value: object) -> str:
    return compact_text(" ".join(_safe_text(value, fallback="unknown").split()), limit=128) or "unknown"  # AUDIT-FIX(#3,#4): Keep JSON keys stable, bounded, and string-only.


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(token in lowered for token in _SENSITIVE_KEYWORDS)  # AUDIT-FIX(#4): Use key-based redaction for common credential-bearing fields.


def _normalize_number(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)  # AUDIT-FIX(#3): Strict JSON forbids NaN/Infinity, so normalize them before serialization.


def _normalize_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat(timespec="seconds")  # AUDIT-FIX(#3): Preserve naive datetimes without inventing a timezone.
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")  # AUDIT-FIX(#3): Canonicalize aware datetimes to UTC for stable event payloads.


def _json_safe(value: object, *, seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _normalize_number(value)
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, Path):
        return str(value)

    object_id = id(value)
    if seen is None:
        seen = set()

    if isinstance(value, Mapping):
        if object_id in seen:
            return "[CIRCULAR]"  # AUDIT-FIX(#3): Prevent recursive payloads from blowing up append().
        seen.add(object_id)
        try:
            sanitized: dict[str, object] = {}
            for raw_key, raw_item in value.items():
                key = _safe_key(raw_key)
                sanitized[key] = _REDACTED_VALUE if _looks_sensitive_key(key) else _json_safe(raw_item, seen=seen)  # AUDIT-FIX(#4): Redact sensitive fields before they hit disk.
            return sanitized
        finally:
            seen.discard(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if object_id in seen:
            return ["[CIRCULAR]"]  # AUDIT-FIX(#3): Keep circular sequences serializable without recursion errors.
        seen.add(object_id)
        try:
            return [_json_safe(item, seen=seen) for item in value]
        finally:
            seen.discard(object_id)

    if isinstance(value, (set, frozenset)):
        if object_id in seen:
            return ["[CIRCULAR]"]  # AUDIT-FIX(#3): Normalize recursive sets safely.
        seen.add(object_id)
        try:
            return [_json_safe(item, seen=seen) for item in sorted(value, key=repr)]
        finally:
            seen.discard(object_id)

    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")  # AUDIT-FIX(#3): Decode binary payloads deterministically instead of crashing json.dumps().

    return compact_text(repr(value), limit=_MAX_REPR_CHARS)  # AUDIT-FIX(#3): Fallback to bounded repr() for arbitrary objects.


def _normalize_data(data: object) -> dict[str, object]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        normalized = _json_safe(data)
        return normalized if isinstance(normalized, dict) else {"value": normalized}
    return {"value": _json_safe(data)}  # AUDIT-FIX(#3): Preserve append() contract even when callers pass non-mapping payloads by mistake.


def _normalize_limit(limit: object) -> int:
    try:
        normalized = int(limit)
    except (TypeError, ValueError):
        return 0  # AUDIT-FIX(#5,#6): Invalid caller input should degrade to an empty tail instead of raising at runtime.
    return max(normalized, 0)


def _assert_no_symlink_components(path: Path) -> None:
    anchor = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    current = anchor
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Refusing to use symlinked event-store path component: {current}")  # AUDIT-FIX(#1): Block symlink traversal in parent directories, including broken links.
        if not current.exists():
            break


def _lock_fd(fd: int, *, exclusive: bool) -> None:
    fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)  # AUDIT-FIX(#2,#5): Serialize append() and stabilize tail() snapshots.


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"Short write while appending event-store payload to fd={fd}")  # AUDIT-FIX(#2): Fail loudly on torn writes instead of silently truncating JSONL entries.
        view = view[written:]


def _iter_lines_reversed(fd: int):
    position = os.lseek(fd, 0, os.SEEK_END)
    remainder = b""
    while position > 0:
        read_size = min(_REVERSE_READ_CHUNK_SIZE, position)
        position -= read_size
        os.lseek(fd, position, os.SEEK_SET)
        chunk = os.read(fd, read_size)
        if not chunk:
            break
        parts = (chunk + remainder).split(b"\n")
        remainder = parts[0]
        for line in reversed(parts[1:]):
            yield line  # AUDIT-FIX(#6): Stream the file backwards so tail() does not load the entire JSONL file into memory.
    if remainder:
        yield remainder


class TwinrOpsEventStore:
    """Store sanitized Twinr ops events in a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()  # AUDIT-FIX(#1): Normalize user-home paths once so all subsequent file checks target the same location.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrOpsEventStore":
        """Build an ops event store rooted in Twinr's configured project."""

        return cls(resolve_ops_paths_for_config(config).events_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrOpsEventStore":
        """Build an ops event store rooted in a given Twinr project tree."""

        return cls(resolve_ops_paths(project_root).events_path)

    def _open_fd(self, *, write: bool) -> int:
        _assert_no_symlink_components(self.path.parent)  # AUDIT-FIX(#1): Reject symlinked parent components before mkdir/open follows them.
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        flags = getattr(os, "O_CLOEXEC", 0)
        if write:
            flags |= os.O_APPEND | os.O_CREAT | os.O_WRONLY
        else:
            flags |= os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW  # AUDIT-FIX(#1): Refuse symlink targets for the event-store file itself on Linux/RPi.
        elif self.path.is_symlink():
            raise RuntimeError(f"Refusing to use symlinked event-store file: {self.path}")  # AUDIT-FIX(#1): Fallback protection when O_NOFOLLOW is unavailable, including broken symlinks.

        fd = os.open(self.path, flags, 0o600)  # AUDIT-FIX(#1): Create new files with owner-only permissions because ops logs may contain sensitive metadata.
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"Event-store path must be a regular file: {self.path}")  # AUDIT-FIX(#1): Block device nodes, FIFOs, and directories.
            return fd
        except Exception:
            os.close(fd)
            raise

    def append(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Append one sanitized event record and return the stored payload."""

        normalized_event = _safe_text(event, fallback="unknown")  # AUDIT-FIX(#8): Normalize runtime-non-str event values without raising AttributeError.
        entry = {
            "created_at": _utc_now_iso_z(),
            "level": _safe_text(level, fallback="info", lower=True) or "info",  # AUDIT-FIX(#8): Keep level normalization robust under bad caller input.
            "event": normalized_event,
            "message": _safe_text(message, fallback=normalized_event),  # AUDIT-FIX(#8): Preserve message fallback semantics while tolerating None/object inputs.
            "data": _normalize_data(data),  # AUDIT-FIX(#3,#4): Guarantee JSON-safe, secret-aware structured payloads.
        }
        payload = (json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")  # AUDIT-FIX(#3): Emit strict JSON lines after payload normalization.

        fd = self._open_fd(write=True)
        try:
            _lock_fd(fd, exclusive=True)  # AUDIT-FIX(#2): Prevent interleaved concurrent writers from corrupting JSONL records.
            _write_all(fd, payload)
        finally:
            os.close(fd)

        return entry

    def tail(self, *, limit: int = 100) -> list[dict[str, object]]:
        """Return the most recent event records in chronological order."""

        normalized_limit = _normalize_limit(limit)  # AUDIT-FIX(#5,#6): Clamp bad inputs before filesystem work.
        if normalized_limit <= 0:
            return []

        try:
            fd = self._open_fd(write=False)
        except FileNotFoundError:
            return []
        except OSError:
            return []  # AUDIT-FIX(#5): Read-side IO races and transient filesystem errors should degrade to an empty tail, not crash callers.

        try:
            _lock_fd(fd, exclusive=False)  # AUDIT-FIX(#2,#5): Read under a shared lock to avoid parsing half-written lines.
            entries_reversed: list[dict[str, object]] = []
            for raw_line in _iter_lines_reversed(fd):
                line = raw_line.decode("utf-8", errors="replace").strip()  # AUDIT-FIX(#5): Corrupt bytes should be contained to the affected line, not abort the whole tail.
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    entries_reversed.append(parsed)
                    if len(entries_reversed) >= normalized_limit:
                        break
            return list(reversed(entries_reversed))  # AUDIT-FIX(#6): Preserve the original chronological tail() contract while using reverse scanning.
        finally:
            os.close(fd)
