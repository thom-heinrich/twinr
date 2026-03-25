"""Shared support helpers for right-lane reserve runtime modules.

These helpers are intentionally tiny and side-effect free. They keep the
reserve publisher, day planner, and nightly planner aligned on the same text,
timestamp, and local-time normalization rules without regrowing those small
utilities in each module.
"""

from __future__ import annotations

from datetime import datetime, time as LocalTime, timezone


def default_local_now() -> datetime:
    """Return the current local wall clock as an aware datetime."""

    return datetime.now().astimezone()


def utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional ISO-8601 timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def parse_local_time(value: object | None, *, fallback: str) -> LocalTime:
    """Parse one ``HH:MM`` local time string with a stable fallback."""

    text = str(value or "").strip() or fallback
    hour_text, separator, minute_text = text.partition(":")
    if separator != ":":
        hour_text, minute_text = fallback.split(":", 1)
    try:
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    return LocalTime(hour=hour, minute=minute)


__all__ = [
    "compact_text",
    "default_local_now",
    "format_timestamp",
    "parse_local_time",
    "parse_timestamp",
    "utc_now",
]
