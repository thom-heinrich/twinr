"""Provide shared helper logic for automation tool handlers and background use.

The helpers here keep automation-specific parsing and normalization close to
the handler boundary without forcing unrelated modules to import the full
automation handler implementation.
"""

from __future__ import annotations

from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_PRINT_DELIVERY_ALIASES = {"print", "printed", "printer"}
_SPOKEN_DELIVERY_ALIASES = {"say", "speak", "spoken", "speech", "voice", "audio"}
_MAX_TAG_LENGTH = 64


def automation_name_key(raw_value: object) -> str:
    """Normalize an automation name for case-insensitive comparisons."""

    return _text_or_empty(raw_value).lower()


def resolve_timezone_name(owner: Any, raw_value: object, *, fallback: str | None = None) -> str:
    """Validate and resolve a timezone name for automation scheduling."""

    candidate = (
        _text_or_empty(raw_value)
        or _text_or_empty(fallback)
        or _text_or_empty(getattr(owner.config, "local_timezone_name", ""))
        or "UTC"
    )
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Unknown timezone_name `{candidate}`") from exc
    return candidate


def validate_non_negative_finite(field_name: str, raw_value: object) -> float:
    """Coerce a numeric field and reject negative or non-finite values."""

    if isinstance(raw_value, bool):
        raise RuntimeError(f"{field_name} must be a number")
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise RuntimeError(f"{field_name} must be a number")
        try:
            value = float(stripped)
        except ValueError as exc:
            raise RuntimeError(f"{field_name} must be a number") from exc
    else:
        raise RuntimeError(f"{field_name} must be a number")
    if not isfinite(value):
        raise RuntimeError(f"{field_name} must be a finite number")
    if value < 0:
        raise RuntimeError(f"{field_name} must be zero or greater")
    return value


def parse_delivery(raw_value: object, *, default: str) -> str:
    """Normalize a spoken or printed delivery selector."""

    normalized = _lower_text_or_empty(raw_value)
    if not normalized:
        return default
    if normalized in _PRINT_DELIVERY_ALIASES:
        return "printed"
    if normalized in _SPOKEN_DELIVERY_ALIASES:
        return "spoken"
    raise RuntimeError("delivery must be `spoken` or `printed`")


def parse_content_mode(raw_value: object, *, default: str) -> str:
    """Normalize the automation content-mode selector."""

    normalized = _lower_text_or_empty(raw_value)
    if not normalized:
        return default
    if normalized in {"static_text", "llm_prompt"}:
        return normalized
    raise RuntimeError("content_mode must be `static_text` or `llm_prompt`")


def normalize_delivery(raw_value: object) -> str:
    """Best-effort normalize a delivery alias for existing action payloads."""

    normalized = _lower_text_or_empty(raw_value)
    if normalized in _PRINT_DELIVERY_ALIASES:
        return "printed"
    if normalized in _SPOKEN_DELIVERY_ALIASES:
        return "spoken"
    return "spoken"


def parse_weekdays(raw_value: object) -> tuple[int, ...]:
    """Parse weekday values into a unique sorted tuple of integers."""

    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("weekdays must be an array of weekday numbers 0-6")
    weekdays: list[int] = []
    for item in raw_value:
        if isinstance(item, bool):
            raise RuntimeError("weekdays must be integers 0-6")
        if isinstance(item, int):
            weekday = item
        elif isinstance(item, str):
            normalized = item.strip()
            if not normalized or not normalized.lstrip("+-").isdigit():
                raise RuntimeError("weekdays must be integers 0-6")
            weekday = int(normalized)
        else:
            raise RuntimeError("weekdays must be integers 0-6")
        if weekday < 0 or weekday > 6:
            raise RuntimeError("weekdays must use integers 0-6")
        weekdays.append(weekday)
    return tuple(sorted(set(weekdays)))


def parse_tags(raw_value: object) -> tuple[str, ...]:
    """Parse user-provided automation tags into a bounded unique tuple."""

    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("tags must be an array of short strings")
    tags: list[str] = []
    for item in raw_value:
        if item is None:
            continue
        if not isinstance(item, str):
            raise RuntimeError("tags must be an array of short strings")
        tag = item.strip()
        if not tag:
            continue
        if len(tag) > _MAX_TAG_LENGTH:
            raise RuntimeError(f"tags must be at most {_MAX_TAG_LENGTH} characters long")
        if any(ord(char) < 32 for char in tag):
            raise RuntimeError("tags must not contain control characters")
        tags.append(tag)
    return tuple(dict.fromkeys(tags))


def _text_or_empty(raw_value: object) -> str:
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _lower_text_or_empty(raw_value: object) -> str:
    return _text_or_empty(raw_value).lower()
