# CHANGELOG: 2026-03-30
# BUG-1: Repair a missing trailing newline before append so a torn/corrupt last record cannot poison the next valid JSONL entry.
# BUG-2: Bound event/message/data payload sizes and rotate/compress the file with retention so the store cannot grow without limit on Raspberry Pi 4 deployments.
# BUG-3: Keep the live current ops event stream writable for both Pi services and operator probes; 0644
# BUG-3: on a root-owned current file caused real PermissionError failures in the operator-facing probe path.
# SEC-1: Keep dirfd-based O_NOFOLLOW traversal while moving the sanitized ops-event file onto an explicit cross-service mode contract so Pi runtime and operator diagnostics can share the same store safely.
# SEC-2: Extend secret redaction from key names to free-text values (headers, URLs, bearer tokens, cookies, passwords, JWTs) so sensitive material is not persisted verbatim.
# IMP-1: Add OTel-aligned metadata (schema_version, record_id, observed_at, severity_number, trace/span promotion) while preserving the original append()/tail() contract.
# IMP-2: tail() now spans rotated archives, and archive compression is configurable (gzip by default for filelog compatibility, zstd optional on Python 3.14+).

"""Persist and tail Twinr operational events as sanitized JSON lines.

The store is designed for edge deployments: it keeps payloads JSON-safe,
secret-aware, size-bounded, and multi-process safe, while retaining recent
history in rotated archives that tail() can still read.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
import fcntl
import gzip
import json
import math
import os
import re
import shutil
import stat
import time
import uuid

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config

try:  # Python 3.14+
    from compression import zstd as _zstd  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - older Python / distributor without zstd
    _zstd = None


_SCHEMA_VERSION: Final[int] = 2
_REDACTED_VALUE: Final[str] = "[REDACTED]"
_NOFOLLOW_FLAG: Final[int] = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC_FLAG: Final[int] = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAG: Final[int] = getattr(os, "O_DIRECTORY", 0)

_DIR_MODE: Final[int] = 0o755
_CURRENT_FILE_MODE: Final[int] = 0o666
_ARCHIVE_FILE_MODE: Final[int] = 0o644
_LOCK_FILE_MODE: Final[int] = 0o666

_REVERSE_READ_CHUNK_SIZE: Final[int] = 4096
_STREAM_COPY_CHUNK_SIZE: Final[int] = 1024 * 1024

_DEFAULT_MAX_FILE_BYTES: Final[int] = 4 * 1024 * 1024
_DEFAULT_BACKUP_COUNT: Final[int] = 6
_DEFAULT_COMPRESSION: Final[str] = "gzip"  # gzip keeps OTel filelog compatibility; zstd is opt-in.
_DEFAULT_FSYNC: Final[bool] = True

_MAX_EVENT_CHARS: Final[int] = 128
_MAX_MESSAGE_CHARS: Final[int] = 4096
_MAX_KEY_CHARS: Final[int] = 128
_MAX_STRING_CHARS: Final[int] = 2048
_MAX_REPR_CHARS: Final[int] = 1024
_MAX_COLLECTION_ITEMS: Final[int] = 64
_MAX_DEPTH: Final[int] = 8
_MAX_PARSE_LINE_BYTES: Final[int] = 256 * 1024

_SENSITIVE_KEYWORDS: Final[tuple[str, ...]] = (
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "id_token",
    "jwt",
    "password",
    "passwd",
    "private_key",
    "pwd",
    "refresh_token",
    "secret",
    "session",
    "sessionid",
    "token",
)

_REDACTION_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"(?i)\b(authorization)\s*([:=])\s*(bearer|basic)\s+([A-Za-z0-9._~+/=-]+)"),
        r"\1\2 \3 " + _REDACTED_VALUE,
    ),
    (
        re.compile(
            r"(?i)\b((?:access|refresh|id)?_?token|api[_-]?key|apikey|client[_-]?secret|password|passwd|pwd|session(?:id)?|cookie)\b\s*([:=])\s*([^\s,;]+)"
        ),
        r"\1\2" + _REDACTED_VALUE,
    ),
    (
        re.compile(
            r"(?i)([?&](?:access_token|api_key|apikey|token|key|password|passwd|pwd|session|sessionid)=)([^&#\s]+)"
        ),
        r"\1" + _REDACTED_VALUE,
    ),
    (
        re.compile(r"(?P<prefix>\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)(?P<secret>[^/\s@]+)(?P<suffix>@)"),
        r"\g<prefix>" + _REDACTED_VALUE + r"\g<suffix>",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
        _REDACTED_VALUE,
    ),
)

_SEVERITY_NUMBERS: Final[dict[str, int]] = {
    "trace": 1,
    "debug": 5,
    "info": 9,
    "notice": 9,
    "warning": 13,
    "warn": 13,
    "error": 17,
    "critical": 21,
    "fatal": 21,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_choice(name: str, default: str, *, allowed: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    return normalized if normalized in allowed else default


def _utc_now() -> tuple[str, int]:
    unix_ns = time.time_ns()
    iso_value = datetime.fromtimestamp(unix_ns / 1_000_000_000, tz=timezone.utc).isoformat(timespec="milliseconds")
    return iso_value.replace("+00:00", "Z"), unix_ns


def compact_text(value: str | None, *, limit: int = 160) -> str:
    """Collapse whitespace and truncate text to a bounded single line."""

    text = " ".join((value or "").split())
    try:
        normalized_limit = max(int(limit), 0)
    except (TypeError, ValueError):
        normalized_limit = 0
    if len(text) <= normalized_limit:
        return text
    if normalized_limit <= 3:
        return text[:normalized_limit]
    return text[: normalized_limit - 3].rstrip() + "..."


def _truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _redact_text(text: str) -> str:
    redacted = text
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _sanitize_string(value: object, *, limit: int, compact: bool = False) -> str:
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    if compact:
        text = " ".join(text.split())
    # Only scan a small overrun because any data beyond the persisted prefix will be dropped anyway.
    scan_limit = max(limit, 0) + 256
    if len(text) > scan_limit:
        text = text[:scan_limit]
    text = _redact_text(text)
    return _truncate_text(text, limit=limit)


def _safe_text(value: object, *, fallback: str, lower: bool = False, limit: int = _MAX_STRING_CHARS) -> str:
    normalized = _sanitize_string(value, limit=limit).strip()
    if not normalized:
        normalized = fallback
    return normalized.casefold() if lower else normalized


def _safe_key(value: object) -> str:
    return _safe_text(value, fallback="unknown", limit=_MAX_KEY_CHARS) or "unknown"


def _unique_key(target: dict[str, object], key: str) -> str:
    if key not in target:
        return key
    index = 2
    while f"{key}__{index}" in target:
        index += 1
    return f"{key}__{index}"


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(token in lowered for token in _SENSITIVE_KEYWORDS)


def _normalize_number(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def _normalize_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat(timespec="seconds")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_safe(value: object, *, seen: set[int] | None = None, depth: int = 0) -> object:
    if depth >= _MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _normalize_number(value)
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, Path):
        return _sanitize_string(str(value), limit=_MAX_STRING_CHARS)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str):
        return _sanitize_string(value, limit=_MAX_STRING_CHARS)

    object_id = id(value)
    if seen is None:
        seen = set()

    if isinstance(value, Mapping):
        if object_id in seen:
            return "[CIRCULAR]"
        seen.add(object_id)
        try:
            sanitized: dict[str, object] = {}
            for index, (raw_key, raw_item) in enumerate(value.items()):
                if index >= _MAX_COLLECTION_ITEMS:
                    sanitized["_truncated_items"] = True
                    break
                key = _unique_key(sanitized, _safe_key(raw_key))
                if _looks_sensitive_key(key):
                    sanitized[key] = _REDACTED_VALUE
                else:
                    sanitized[key] = _json_safe(raw_item, seen=seen, depth=depth + 1)
            return sanitized
        finally:
            seen.discard(object_id)

    if isinstance(value, (set, frozenset)):
        if object_id in seen:
            return ["[CIRCULAR]"]
        seen.add(object_id)
        try:
            sanitized_list: list[object] = []
            for index, item in enumerate(sorted(value, key=repr)):
                if index >= _MAX_COLLECTION_ITEMS:
                    sanitized_list.append("[TRUNCATED]")
                    break
                sanitized_list.append(_json_safe(item, seen=seen, depth=depth + 1))
            return sanitized_list
        finally:
            seen.discard(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if object_id in seen:
            return ["[CIRCULAR]"]
        seen.add(object_id)
        try:
            sanitized_list = []
            for index, item in enumerate(value):
                if index >= _MAX_COLLECTION_ITEMS:
                    sanitized_list.append("[TRUNCATED]")
                    break
                sanitized_list.append(_json_safe(item, seen=seen, depth=depth + 1))
            return sanitized_list
        finally:
            seen.discard(object_id)

    if isinstance(value, (bytes, bytearray)):
        return _sanitize_string(bytes(value).decode("utf-8", errors="replace"), limit=_MAX_STRING_CHARS)

    return _sanitize_string(repr(value), limit=_MAX_REPR_CHARS)


def _normalize_data(data: object) -> dict[str, object]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        normalized = _json_safe(data)
        return normalized if isinstance(normalized, dict) else {"value": normalized}
    return {"value": _json_safe(data)}


def _normalize_limit(limit: object) -> int:
    try:
        normalized = int(limit)
    except (TypeError, ValueError):
        return 0
    return max(normalized, 0)


def _severity_number(level: str) -> int:
    return _SEVERITY_NUMBERS.get(level.casefold(), _SEVERITY_NUMBERS["info"])


def _filename_timestamp() -> str:
    unix_ns = time.time_ns()
    dt = datetime.fromtimestamp(unix_ns / 1_000_000_000, tz=timezone.utc)
    return f"{dt.strftime('%Y%m%dT%H%M%S')}{unix_ns % 1_000_000_000:09d}Z"


def _time_ordered_id() -> str:
    uuid7 = getattr(uuid, "uuid7", None)
    if callable(uuid7):  # Python 3.14+
        return str(uuid7())  # pylint: disable=not-callable
    return str(uuid.uuid4())


def _normalize_trace_hex(value: object, *, length: int) -> str | None:
    if value is None:
        return None
    text = _sanitize_string(value, limit=length, compact=True).strip().lower()
    if len(text) != length:
        return None
    if all(character in "0123456789abcdef" for character in text):
        return text
    return None


def _extract_trace_context(data: Mapping[str, object]) -> tuple[str | None, str | None, str | None]:
    trace_id = _normalize_trace_hex(data.get("trace_id"), length=32)
    span_id = _normalize_trace_hex(data.get("span_id"), length=16)
    trace_flags = _normalize_trace_hex(data.get("trace_flags"), length=2)

    if trace_id and span_id:
        return trace_id, span_id, trace_flags

    traceparent = data.get("traceparent")
    if isinstance(traceparent, str):
        match = re.fullmatch(r"(?i)\s*([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\s*", traceparent)
        if match:
            return match.group(2).lower(), match.group(3).lower(), match.group(4).lower()

    return trace_id, span_id, trace_flags


def _lock_fd(fd: int, *, exclusive: bool) -> None:
    fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"Short write while appending event-store payload to fd={fd}")
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
            yield line
    if remainder:
        yield remainder


def _parse_entry(raw_line: bytes) -> dict[str, object] | None:
    if not raw_line:
        return None
    if len(raw_line) > _MAX_PARSE_LINE_BYTES:
        return None
    try:
        text = raw_line.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fdatasync(fd: int) -> None:
    sync_fn = getattr(os, "fdatasync", None) or os.fsync
    sync_fn(fd)


def _ensure_directory_fd(fd: int) -> None:
    file_stat = os.fstat(fd)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise RuntimeError("Expected an open directory file descriptor.")


def _ensure_regular_fd(fd: int, *, path_hint: str) -> None:
    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"Event-store path must be a regular file: {path_hint}")


class TwinrOpsEventStore:
    """Store sanitized Twinr ops events in a JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_file_bytes: int | None = None,
        backup_count: int | None = None,
        compression: str | None = None,
        fsync: bool | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        # BREAKING: once max_file_bytes is exceeded, older records rotate into archives instead of the active file growing forever.
        self.max_file_bytes = max(
            int(_DEFAULT_MAX_FILE_BYTES if max_file_bytes is None else max_file_bytes),
            0,
        )
        if max_file_bytes is None:
            self.max_file_bytes = _env_int("TWINR_OPS_EVENTSTORE_MAX_FILE_BYTES", self.max_file_bytes)
        self.backup_count = max(
            int(_DEFAULT_BACKUP_COUNT if backup_count is None else backup_count),
            0,
        )
        if backup_count is None:
            self.backup_count = _env_int("TWINR_OPS_EVENTSTORE_BACKUP_COUNT", self.backup_count)
        selected_compression = _DEFAULT_COMPRESSION if compression is None else str(compression).strip().casefold()
        if compression is None:
            selected_compression = _env_choice(
                "TWINR_OPS_EVENTSTORE_COMPRESSION",
                selected_compression,
                allowed={"none", "gzip", "zstd"},
            )
        self.compression = selected_compression if selected_compression in {"none", "gzip", "zstd"} else _DEFAULT_COMPRESSION
        self.fsync_enabled = _DEFAULT_FSYNC if fsync is None else bool(fsync)
        if fsync is None:
            self.fsync_enabled = _env_bool("TWINR_OPS_EVENTSTORE_FSYNC", self.fsync_enabled)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrOpsEventStore":
        """Build an ops event store rooted in Twinr's configured project."""

        return cls(resolve_ops_paths_for_config(config).events_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrOpsEventStore":
        """Build an ops event store rooted in a given Twinr project tree."""

        return cls(resolve_ops_paths(project_root).events_path)

    @property
    def _lock_filename(self) -> str:
        return f".{self.path.name}.lock"

    @property
    def _archive_prefix(self) -> str:
        return f"{self.path.name}."

    def _open_parent_dir_fd(self, *, create: bool) -> int:
        path_open_flags = getattr(os, "O_PATH", os.O_RDONLY) | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
        current_fd = os.open("/" if self.path.is_absolute() else ".", path_open_flags)
        parts = self.path.parent.parts[1:] if self.path.is_absolute() else self.path.parent.parts
        traversed = Path(self.path.anchor) if self.path.is_absolute() else Path(".")
        try:
            for part in parts:
                if part in {"", "."}:
                    continue
                traversed = traversed / part
                try:
                    next_fd = os.open(part, path_open_flags, dir_fd=current_fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, _DIR_MODE, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                    next_fd = os.open(part, path_open_flags, dir_fd=current_fd)
                file_stat = os.fstat(next_fd)
                # BREAKING: symlinked parent components are now refused instead of followed.
                if stat.S_ISLNK(file_stat.st_mode):
                    os.close(next_fd)
                    raise RuntimeError(f"Refusing to use symlinked event-store path component: {traversed}")
                if not stat.S_ISDIR(file_stat.st_mode):
                    os.close(next_fd)
                    raise RuntimeError(f"Event-store parent path must contain only directories: {traversed}")
                os.close(current_fd)
                current_fd = next_fd
            final_fd = os.open(".", os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG, dir_fd=current_fd)
            os.close(current_fd)
            return final_fd
        except Exception:
            os.close(current_fd)
            raise

    def _open_regular_file_at(
        self,
        parent_fd: int,
        name: str,
        *,
        file_mode: int,
        read: bool,
        write: bool,
        create: bool = False,
        append: bool = False,
        truncate: bool = False,
        exclusive_create: bool = False,
    ) -> int:
        flags = _CLOEXEC_FLAG | _NOFOLLOW_FLAG
        if read and write:
            flags |= os.O_RDWR
        elif write:
            flags |= os.O_WRONLY
        else:
            flags |= os.O_RDONLY
        if create:
            flags |= os.O_CREAT
        if append:
            flags |= os.O_APPEND
        if truncate:
            flags |= os.O_TRUNC
        if exclusive_create:
            flags |= os.O_EXCL

        fd = os.open(name, flags, file_mode, dir_fd=parent_fd)
        try:
            _ensure_regular_fd(fd, path_hint=str(self.path.parent / name))
            if write:
                try:
                    os.fchmod(fd, file_mode)
                except OSError:
                    pass
            return fd
        except Exception:
            os.close(fd)
            raise

    def _open_lock_fd(self, parent_fd: int) -> int:
        return self._open_regular_file_at(
            parent_fd,
            self._lock_filename,
            file_mode=_LOCK_FILE_MODE,
            read=True,
            write=True,
            create=True,
        )

    def _chmod_path_at(self, parent_fd: int, name: str, mode: int) -> None:
        """Refresh one regular file mode without following symlinks."""

        fd = self._open_regular_file_at(parent_fd, name, file_mode=mode, read=True, write=False)
        try:
            os.fchmod(fd, mode)
        finally:
            os.close(fd)

    def _archive_names(self, parent_fd: int) -> list[str]:
        names: list[str] = []
        for name in os.listdir(parent_fd):
            if not isinstance(name, str):
                continue
            if not name.startswith(self._archive_prefix):
                continue
            if name == self.path.name:
                continue
            names.append(name)
        names.sort(reverse=True)
        return names

    def _repair_trailing_newline_if_needed(self, fd: int) -> bool:
        file_size = os.lseek(fd, 0, os.SEEK_END)
        if file_size <= 0:
            return False
        os.lseek(fd, -1, os.SEEK_END)
        last_byte = os.read(fd, 1)
        if last_byte == b"\n":
            return False
        os.lseek(fd, 0, os.SEEK_END)
        _write_all(fd, b"\n")
        return True

    def _should_rotate(self, current_size: int, incoming_bytes: int) -> bool:
        if self.max_file_bytes <= 0:
            return False
        if current_size <= 0:
            return False
        return current_size + incoming_bytes > self.max_file_bytes

    def _compression_suffix(self) -> str:
        if self.compression == "zstd" and _zstd is not None:
            return ".zst"
        if self.compression in {"gzip", "zstd"}:
            return ".gz"
        return ""

    def _compress_archive_locked(self, parent_fd: int, source_name: str) -> str:
        suffix = self._compression_suffix()
        if not suffix:
            return source_name

        target_name = source_name + suffix
        source_fd = self._open_regular_file_at(parent_fd, source_name, file_mode=_ARCHIVE_FILE_MODE, read=True, write=False)
        try:
            target_fd = self._open_regular_file_at(
                parent_fd,
                target_name,
                file_mode=_ARCHIVE_FILE_MODE,
                read=False,
                write=True,
                create=True,
                truncate=True,
                exclusive_create=True,
            )
        except Exception:
            os.close(source_fd)
            return source_name

        try:
            with os.fdopen(source_fd, "rb", closefd=True) as source_file:
                with os.fdopen(target_fd, "wb", closefd=True) as target_file:
                    if suffix == ".zst" and _zstd is not None:
                        with _zstd.open(target_file, "wb") as compressed_file:
                            shutil.copyfileobj(source_file, compressed_file, length=_STREAM_COPY_CHUNK_SIZE)
                    elif suffix == ".gz":
                        with gzip.GzipFile(fileobj=target_file, mode="wb", compresslevel=6) as compressed_file:
                            shutil.copyfileobj(source_file, compressed_file, length=_STREAM_COPY_CHUNK_SIZE)
                    else:
                        return source_name
            if self.fsync_enabled:
                compressed_fd = self._open_regular_file_at(
                    parent_fd,
                    target_name,
                    file_mode=_ARCHIVE_FILE_MODE,
                    read=True,
                    write=False,
                )
                try:
                    _fdatasync(compressed_fd)
                finally:
                    os.close(compressed_fd)
            os.unlink(source_name, dir_fd=parent_fd)
            return target_name
        except Exception:
            try:
                os.unlink(target_name, dir_fd=parent_fd)
            except OSError:
                pass
            return source_name

    def _rotate_locked(self, parent_fd: int) -> bool:
        try:
            current_fd = self._open_regular_file_at(
                parent_fd,
                self.path.name,
                file_mode=_CURRENT_FILE_MODE,
                read=True,
                write=False,
            )
        except FileNotFoundError:
            return False

        try:
            current_size = os.fstat(current_fd).st_size
        finally:
            os.close(current_fd)

        if current_size <= 0:
            return False

        rotated_name = f"{self.path.name}.{_filename_timestamp()}.{_time_ordered_id()}"
        os.rename(self.path.name, rotated_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        self._chmod_path_at(parent_fd, rotated_name, _ARCHIVE_FILE_MODE)
        final_name = self._compress_archive_locked(parent_fd, rotated_name)
        del final_name  # reserved for future hooks / metrics

        for stale_name in self._archive_names(parent_fd)[self.backup_count :]:
            try:
                os.unlink(stale_name, dir_fd=parent_fd)
            except FileNotFoundError:
                continue

        if self.fsync_enabled:
            os.fsync(parent_fd)
        return True

    def _tail_from_fd(self, fd: int, *, limit: int) -> list[dict[str, object]]:
        entries_reversed: list[dict[str, object]] = []
        for raw_line in _iter_lines_reversed(fd):
            parsed = _parse_entry(raw_line)
            if parsed is None:
                continue
            entries_reversed.append(parsed)
            if len(entries_reversed) >= limit:
                break
        return list(reversed(entries_reversed))

    def _tail_from_stream(self, stream, *, limit: int) -> list[dict[str, object]]:
        entries: deque[dict[str, object]] = deque(maxlen=limit)
        for raw_line in stream:
            parsed = _parse_entry(raw_line if isinstance(raw_line, bytes) else str(raw_line).encode("utf-8", errors="replace"))
            if parsed is not None:
                entries.append(parsed)
        return list(entries)

    def _read_archive_tail(self, parent_fd: int, name: str, *, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        if name.endswith(".gz"):
            archive_fd = self._open_regular_file_at(parent_fd, name, file_mode=_ARCHIVE_FILE_MODE, read=True, write=False)
            with os.fdopen(archive_fd, "rb", closefd=True) as raw_file:
                with gzip.GzipFile(fileobj=raw_file, mode="rb") as compressed_file:
                    return self._tail_from_stream(compressed_file, limit=limit)
        if name.endswith(".zst") and _zstd is not None:
            archive_fd = self._open_regular_file_at(parent_fd, name, file_mode=_ARCHIVE_FILE_MODE, read=True, write=False)
            with os.fdopen(archive_fd, "rb", closefd=True) as raw_file:
                with _zstd.open(raw_file, "rb") as compressed_file:
                    return self._tail_from_stream(compressed_file, limit=limit)
        archive_fd = self._open_regular_file_at(parent_fd, name, file_mode=_ARCHIVE_FILE_MODE, read=True, write=False)
        try:
            return self._tail_from_fd(archive_fd, limit=limit)
        finally:
            os.close(archive_fd)

    def append(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Append one sanitized event record and return the stored payload."""

        normalized_data = _normalize_data(data)
        normalized_event = _safe_text(event, fallback="unknown", limit=_MAX_EVENT_CHARS)
        normalized_level = _safe_text(level, fallback="info", lower=True, limit=32) or "info"
        created_at, created_at_ns = _utc_now()
        observed_at, observed_at_ns = _utc_now()
        trace_id, span_id, trace_flags = _extract_trace_context(normalized_data)

        # BREAKING: stored records now include schema_version/record_id/observed_at/severity_number for OTel-grade correlation.
        entry: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "record_id": _time_ordered_id(),
            "created_at": created_at,
            "created_at_ns": created_at_ns,
            "observed_at": observed_at,
            "observed_at_ns": observed_at_ns,
            "level": normalized_level,
            "severity_number": _severity_number(normalized_level),
            "event": normalized_event,
            "message": _safe_text(message, fallback=normalized_event, limit=_MAX_MESSAGE_CHARS),
            "data": normalized_data,
        }
        if trace_id:
            entry["trace_id"] = trace_id
        if span_id:
            entry["span_id"] = span_id
        if trace_flags:
            entry["trace_flags"] = trace_flags

        payload = (json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")

        parent_fd = self._open_parent_dir_fd(create=True)
        try:
            lock_fd = self._open_lock_fd(parent_fd)
            try:
                _lock_fd(lock_fd, exclusive=True)

                try:
                    current_fd = self._open_regular_file_at(
                        parent_fd,
                        self.path.name,
                        file_mode=_CURRENT_FILE_MODE,
                        read=True,
                        write=True,
                        create=True,
                        append=True,
                    )
                except FileNotFoundError:
                    current_fd = self._open_regular_file_at(
                        parent_fd,
                        self.path.name,
                        file_mode=_CURRENT_FILE_MODE,
                        read=True,
                        write=True,
                        create=True,
                        append=True,
                    )

                try:
                    original_size = os.fstat(current_fd).st_size
                finally:
                    os.close(current_fd)

                rotated = False
                if self._should_rotate(original_size, len(payload)):
                    rotated = self._rotate_locked(parent_fd)

                current_fd = self._open_regular_file_at(
                    parent_fd,
                    self.path.name,
                    file_mode=_CURRENT_FILE_MODE,
                    read=True,
                    write=True,
                    create=True,
                    append=True,
                )
                try:
                    repair_performed = self._repair_trailing_newline_if_needed(current_fd)
                    _write_all(current_fd, payload)
                    if self.fsync_enabled:
                        _fdatasync(current_fd)
                        if rotated or repair_performed or original_size == 0:
                            os.fsync(parent_fd)
                finally:
                    os.close(current_fd)
            finally:
                os.close(lock_fd)
        finally:
            os.close(parent_fd)

        return entry

    def tail(self, *, limit: int = 100) -> list[dict[str, object]]:
        """Return the most recent event records in chronological order."""

        normalized_limit = _normalize_limit(limit)
        if normalized_limit <= 0:
            return []

        try:
            parent_fd = self._open_parent_dir_fd(create=False)
        except FileNotFoundError:
            return []
        except OSError:
            return []

        try:
            lock_fd = self._open_lock_fd(parent_fd)
        except OSError:
            os.close(parent_fd)
            return []

        try:
            _lock_fd(lock_fd, exclusive=False)
            entries: deque[dict[str, object]] = deque(maxlen=normalized_limit)

            try:
                    current_fd = self._open_regular_file_at(
                        parent_fd,
                        self.path.name,
                        file_mode=_CURRENT_FILE_MODE,
                        read=True,
                        write=False,
                    )
            except FileNotFoundError:
                current_fd = None
            except OSError:
                current_fd = None

            if current_fd is not None:
                try:
                    for entry in self._tail_from_fd(current_fd, limit=normalized_limit):
                        entries.append(entry)
                except OSError:
                    pass
                finally:
                    os.close(current_fd)

            if len(entries) < normalized_limit:
                needed = normalized_limit - len(entries)
                for archive_name in self._archive_names(parent_fd):
                    try:
                        archive_entries = self._read_archive_tail(parent_fd, archive_name, limit=needed)
                    except OSError:
                        continue
                    for entry in reversed(archive_entries):
                        entries.appendleft(entry)
                        if len(entries) > normalized_limit:
                            entries.popleft()
                    needed = normalized_limit - len(entries)
                    if needed <= 0:
                        break

            return list(entries)
        finally:
            os.close(lock_fd)
            os.close(parent_fd)
