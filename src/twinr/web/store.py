from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs

import errno
import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Twinr target is Linux/RPi, but keep import-safe.
    fcntl = None


# AUDIT-FIX(#3): Bound form parsing to prevent malformed payload crashes and simple field-count/body-size abuse.
_MAX_FORM_BODY_BYTES = 1024 * 1024
_MAX_FORM_FIELDS = 128
# AUDIT-FIX(#5): Only allow portable .env variable names on write to prevent config-line injection.
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# AUDIT-FIX(#1): Serialize same-path writes in-process before taking an OS-level advisory file lock.
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def parse_urlencoded_form(body: bytes) -> dict[str, str]:
    # AUDIT-FIX(#3): Malformed UTF-8 must not crash the settings flow; use replacement decoding and field caps.
    if len(body) > _MAX_FORM_BODY_BYTES:
        return {}

    try:
        parsed = parse_qs(
            body.decode("utf-8", errors="replace"),
            keep_blank_values=True,
            strict_parsing=False,
            encoding="utf-8",
            errors="replace",
            max_num_fields=_MAX_FORM_FIELDS,
            separator="&",
        )
    except ValueError:
        return {}
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def read_env_values(path: Path) -> dict[str, str]:
    # AUDIT-FIX(#4): Read through a hardened helper that avoids exists()-then-open races and rejects symlink targets.
    content = _read_utf8_text(path)
    if content == "":
        return {}

    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        parsed = _split_env_assignment(raw_line)
        if parsed is None:
            continue
        key, raw_value = parsed
        # AUDIT-FIX(#2): Parse quoted/unquoted values without lossy strip()-based corruption.
        values[key] = _parse_env_value(raw_value)
    return values


def write_env_updates(path: Path, updates: dict[str, str]) -> None:
    # AUDIT-FIX(#5): Reject invalid keys and non-string values before touching the .env file.
    validated_updates = _validate_env_updates(updates)
    if not validated_updates:
        return

    # AUDIT-FIX(#1): Protect the full read-modify-write cycle with a lock and atomic replace.
    with _locked_path(path):
        existing_lines = _read_utf8_text(path).splitlines()

        result: list[str] = []
        seen: set[str] = set()
        for line in existing_lines:
            parsed = _split_env_assignment(line)
            if parsed is None:
                result.append(line)
                continue
            key, _ = parsed
            if key in validated_updates:
                if key not in seen:
                    # AUDIT-FIX(#2): Re-emit values with reversible escaping so reads and writes round-trip safely.
                    result.append(f"{key}={_quote_env_value(validated_updates[key])}")
                    seen.add(key)
                continue
            result.append(line)

        for key, value in validated_updates.items():
            if key in seen:
                continue
            result.append(f"{key}={_quote_env_value(value)}")

        _atomic_write_text(path, "\n".join(result).rstrip("\n") + "\n")


def read_text_file(path: Path) -> str:
    # AUDIT-FIX(#4): Avoid exists()-then-open races and refuse symlink/non-regular file targets.
    return _read_utf8_text(path)


def write_text_file(path: Path, content: str) -> None:
    # AUDIT-FIX(#6): Preserve trailing spaces/tabs; normalize only line endings and terminal newlines.
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n") + "\n"
    # AUDIT-FIX(#1): Use the same atomic, durable write path as .env updates to avoid torn writes on power loss.
    with _locked_path(path):
        _atomic_write_text(path, normalized)


def mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    # AUDIT-FIX(#7): Reduce disclosure to suffix-only masking; prefix+suffix reveals too much key material.
    if len(value) <= 8:
        return "Configured"
    return f"****{value[-4:]}"


@dataclass(frozen=True, slots=True)
class FileBackedSetting:
    key: str
    label: str
    value: str
    help_text: str = ""
    tooltip_text: str = ""
    input_type: str = "text"
    options: tuple[tuple[str, str], ...] = ()
    placeholder: str = ""
    rows: int = 4
    wide: bool = False
    secret: bool = False


def _quote_env_value(value: str) -> str:
    # AUDIT-FIX(#2): Preserve leading/trailing spaces and escape control characters/newlines to prevent file corruption.
    _validate_env_value(value)
    if value == "":
        return '""'

    needs_quotes = (
        value != value.strip()
        or any(char.isspace() for char in value)
        or any(char in value for char in ["#", '"', "'", "\\"])
    )
    if not needs_quotes:
        return value

    escaped = (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )
    return f'"{escaped}"'


def _validate_env_updates(updates: dict[str, str]) -> dict[str, str]:
    # AUDIT-FIX(#5): Normalize and validate all keys before persisting anything to disk.
    validated: dict[str, str] = {}
    for key, value in updates.items():
        normalized_key = _validate_env_key(key)
        validated[normalized_key] = _validate_env_value(value)
    return validated


def _validate_env_key(key: str) -> str:
    # AUDIT-FIX(#5): Reject keys that could break .env syntax or create injected assignments.
    if not isinstance(key, str):
        raise TypeError("Environment variable names must be strings")
    normalized = key.strip()
    if not _ENV_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    return normalized


def _validate_env_value(value: str) -> str:
    # AUDIT-FIX(#2): NUL bytes are not valid in text-backed config files and must be rejected.
    if not isinstance(value, str):
        raise TypeError("Environment variable values must be strings")
    if "\x00" in value:
        raise ValueError("Environment variable values may not contain NUL bytes")
    return value


def _split_env_assignment(raw_line: str) -> tuple[str, str] | None:
    # AUDIT-FIX(#2): Keep parsing lenient for existing files while separating key/value safely.
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#") or "=" not in raw_line:
        return None

    key_part, value_part = raw_line.split("=", 1)
    key = key_part.strip()
    if key.startswith("export "):
        key = key[7:].strip()
    if not key:
        return None
    return key, value_part


def _parse_env_value(raw_value: str) -> str:
    # AUDIT-FIX(#2): Support quoted values and inline comments without destroying significant whitespace.
    value = raw_value.lstrip()
    if value == "":
        return ""

    if value[0] == '"':
        return _parse_double_quoted_env_value(value)
    if value[0] == "'":
        end_index = value.find("'", 1)
        if end_index == -1:
            return value[1:]
        return value[1:end_index]

    result: list[str] = []
    previous_was_whitespace = False
    for char in value:
        if char == "#" and previous_was_whitespace:
            break
        result.append(char)
        previous_was_whitespace = char.isspace()
    return "".join(result).rstrip()


def _parse_double_quoted_env_value(value: str) -> str:
    # AUDIT-FIX(#2): Reverse the escape format emitted by _quote_env_value for reliable round-trips.
    result: list[str] = []
    escaped = False
    for char in value[1:]:
        if escaped:
            if char == "n":
                result.append("\n")
            elif char == "r":
                result.append("\r")
            elif char == "t":
                result.append("\t")
            else:
                result.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            break
        result.append(char)

    if escaped:
        result.append("\\")
    return "".join(result)


def _get_path_lock(path: Path) -> threading.RLock:
    # AUDIT-FIX(#1): Reuse one re-entrant mutex per path to prevent in-process lost-update races.
    lock_key = os.path.abspath(os.fspath(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[lock_key] = lock
        return lock


@contextmanager
def _locked_path(path: Path) -> Iterator[None]:
    # AUDIT-FIX(#1): Combine in-process serialization with an advisory file lock for cross-task safety.
    in_process_lock = _get_path_lock(path)
    with in_process_lock:
        _ensure_parent_directory(path.parent)
        if fcntl is None:
            yield
            return

        lock_path = path.parent / f".{path.name}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            with os.fdopen(lock_fd, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            try:
                os.close(lock_fd)
            except OSError:
                pass
            raise


def _read_utf8_text(path: Path) -> str:
    # AUDIT-FIX(#4): Centralize hardened UTF-8 reads so callers do not repeat racy exists()-checks.
    try:
        fd = _open_regular_file_for_read(path)
    except FileNotFoundError:
        return ""

    with os.fdopen(fd, "r", encoding="utf-8", errors="replace", newline="") as file_obj:
        return file_obj.read()


def _open_regular_file_for_read(path: Path) -> int:
    # AUDIT-FIX(#4): Refuse symlink and non-regular-file reads to reduce file-target confusion attacks.
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"Refusing to read symlink target: {path}") from exc
        raise

    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"Refusing to read non-regular file: {path}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    # AUDIT-FIX(#1): Write to a secure temp file, fsync it, then replace atomically to avoid torn files.
    _ensure_parent_directory(path.parent)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _ensure_parent_directory(directory: Path) -> None:
    # AUDIT-FIX(#4): Ensure the immediate parent is a real directory, not a symlink hop or special file.
    directory.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"Refusing to write through symlink directory: {directory}") from exc
        raise
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Write target parent is not a directory: {directory}")
    finally:
        os.close(fd)


def _fsync_directory(directory: Path) -> None:
    # AUDIT-FIX(#1): Flush the directory entry so the rename survives sudden power loss more reliably.
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(fd)