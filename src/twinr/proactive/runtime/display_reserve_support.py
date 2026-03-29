# CHANGELOG: 2026-03-29
# BUG-1: default_local_now() no longer returns a fixed-offset local tz that can mis-handle future DST-aware scheduling.
# BUG-2: format_timestamp() no longer silently serializes naive datetimes as if they were system-local time.
# BUG-3: compact_text() now enforces a real length bound even for invalid max_len values and no longer returns oversized text.
# SEC-1: compact_text() no longer uses split()-join over unbounded input, reducing memory-amplification risk from oversized untrusted text on Pi-class hardware.
# IMP-1: local timezone resolution now prefers IANA ZoneInfo via Pi/Linux configuration (TZ, /etc/timezone, /etc/localtime), with optional tzlocal fallback.
# IMP-2: parse_timestamp()/parse_local_time() now use modern ISO parsing, accept bytes-like inputs, expose strict mode, and emit canonical UTC 'Z' timestamps by default.

"""Shared support helpers for right-lane reserve runtime modules.

These helpers stay tiny and side-effect light, but they now fail fast on
ambiguous datetime inputs, use DST-correct local zones on Pi/Linux, and keep
text normalization bounded under oversized input.
"""

from __future__ import annotations

import os
from datetime import datetime, time as LocalTime, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc

try:
    # Optional: preferred when installed for non-standard local timezone setups.
    from tzlocal import get_localzone as _tzlocal_get_localzone
except Exception:  # pragma: no cover - optional dependency
    _tzlocal_get_localzone = None

_ZONEINFO_PATH_PREFIXES = (
    "/usr/share/zoneinfo/",
    "/usr/lib/zoneinfo/",
    "/var/db/timezone/zoneinfo/",
)
_ELLIPSIS = "…"


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _force_truncated(text: str, *, max_len: int) -> str:
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if max_len == 1:
        return _ELLIPSIS
    trimmed = text[: max_len - 1].rstrip()
    return (trimmed or "") + _ELLIPSIS


def _zoneinfo_from_key(key: str) -> tzinfo | None:
    candidate = key.strip()
    if not candidate:
        return None
    if candidate.startswith(":"):
        candidate = candidate[1:]
    for prefix in _ZONEINFO_PATH_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        return None


def _zoneinfo_from_local_files() -> tzinfo | None:
    env_tz = _zoneinfo_from_key(os.environ.get("TZ", ""))
    if env_tz is not None:
        return env_tz

    timezone_file = Path("/etc/timezone")
    try:
        text = timezone_file.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if text:
        zone = _zoneinfo_from_key(text.splitlines()[0])
        if zone is not None:
            return zone

    localtime = Path("/etc/localtime")
    try:
        resolved = localtime.resolve(strict=True).as_posix()
    except OSError:
        resolved = ""
    for prefix in _ZONEINFO_PATH_PREFIXES:
        if resolved.startswith(prefix):
            zone = _zoneinfo_from_key(resolved[len(prefix) :])
            if zone is not None:
                return zone

    return None


def _resolved_local_tz() -> tzinfo:
    zone = _zoneinfo_from_local_files()
    if zone is not None:
        return zone

    if _tzlocal_get_localzone is not None:
        try:
            zone = _tzlocal_get_localzone()
            if zone is not None:
                return zone
        except Exception:
            pass

    fallback = datetime.now().astimezone().tzinfo
    return fallback or UTC


def default_local_now() -> datetime:
    """Return the current local wall clock as an aware datetime."""

    # BREAKING: tzinfo may now be a DST-aware ZoneInfo object instead of a
    # fixed-offset datetime.timezone instance.
    return datetime.now(_resolved_local_tz())


def utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(UTC)


def compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text."""

    if value is None:
        return ""

    # BREAKING: non-positive max_len now raises instead of returning incorrect
    # or effectively unbounded output.
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    text = _coerce_text(value)
    if not text:
        return ""

    out: list[str] = []
    out_len = 0
    saw_non_whitespace = False
    pending_space = False
    truncated = False

    for char in text:
        if char.isspace():
            if saw_non_whitespace:
                pending_space = True
            continue

        if pending_space:
            if out_len >= max_len:
                truncated = True
                break
            out.append(" ")
            out_len += 1
            pending_space = False

        if out_len >= max_len:
            truncated = True
            break

        out.append(char)
        out_len += 1
        saw_non_whitespace = True

    compact = "".join(out).rstrip()
    if not truncated:
        return compact
    return _force_truncated(compact, max_len=max_len)


def _parse_iso_datetime(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # Compatibility fallback for older runtimes where trailing Z is not
        # accepted by fromisoformat().
        if text.endswith("Z"):
            return datetime.fromisoformat(text[:-1] + "+00:00")
        raise


def parse_timestamp(
    value: object | None,
    *,
    assume_tz: tzinfo | None = UTC,
    strict: bool = False,
) -> datetime | None:
    """Parse one optional ISO-8601 timestamp into an aware UTC datetime.

    Naive timestamps are interpreted in ``assume_tz``. The legacy default stays
    UTC for drop-in compatibility; callers with local wall-clock timestamps can
    pass an explicit local zone.
    """

    text = _coerce_text(value).strip()
    if not text:
        return None

    try:
        parsed = _parse_iso_datetime(text)
    except ValueError:
        if strict:
            raise
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if assume_tz is None:
            if strict:
                raise ValueError("naive timestamps require assume_tz")
            return None
        parsed = parsed.replace(tzinfo=assume_tz)

    return parsed.astimezone(UTC)


def format_timestamp(value: datetime, *, use_z: bool = True) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    # BREAKING: naive datetimes are now rejected instead of being silently
    # reinterpreted as system-local time by astimezone().
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("format_timestamp() requires an aware datetime")

    rendered = value.astimezone(UTC).isoformat()

    # BREAKING: UTC text now defaults to RFC3339-style trailing 'Z' instead of
    # '+00:00'. Pass use_z=False to preserve the legacy textual form.
    if use_z and rendered.endswith("+00:00"):
        return rendered[:-6] + "Z"
    return rendered


def _normalize_local_time(value: LocalTime) -> LocalTime:
    return LocalTime(hour=value.hour, minute=value.minute, fold=value.fold)


@lru_cache(maxsize=32)
def _parse_fallback_local_time(fallback: str) -> LocalTime:
    parsed = LocalTime.fromisoformat(fallback)
    if parsed.tzinfo is not None:
        raise ValueError("fallback local time must not include a timezone")
    return _normalize_local_time(parsed)


def parse_local_time(
    value: object | None,
    *,
    fallback: str,
    strict: bool = False,
) -> LocalTime:
    """Parse one local time string into minute precision with a stable fallback.

    Accepts ISO-like local times such as ``HH:MM`` and ``HH:MM:SS`` but always
    normalizes back to minute precision because the planners schedule by minute.
    """

    fallback_time = _parse_fallback_local_time(fallback)
    text = _coerce_text(value).strip() or fallback

    try:
        parsed = LocalTime.fromisoformat(text)
    except ValueError:
        if strict:
            raise
        return fallback_time

    if parsed.tzinfo is not None:
        if strict:
            raise ValueError("local time values must not include a timezone")
        return fallback_time

    return _normalize_local_time(parsed)


__all__ = [
    "compact_text",
    "default_local_now",
    "format_timestamp",
    "parse_local_time",
    "parse_timestamp",
    "utc_now",
]