"""Pure helper functions shared across the environment-profile package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone, tzinfo
from hashlib import sha256
import logging
import math
from typing import cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .constants import _DEFAULT_TIMEZONE_NAME

logger = logging.getLogger("twinr.memory.longterm.ingestion.environment_profile")


def _normalize_text(value: object | None) -> str:
    """Collapse arbitrary input into one bounded line of text."""

    return " ".join(str(value or "").split()).strip()


def _normalize_slug(value: object | None, *, fallback: str) -> str:
    """Return one storage-safe token for identifiers and memory IDs."""

    normalized = _normalize_text(value).lower()
    if not normalized:
        return fallback
    slug_chars = [character if character.isalnum() else "_" for character in normalized]
    slug = "".join(slug_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def _tokenize_identifier(value: str, *, fallback: str) -> str:
    """Return one compact token safe for use inside long-term memory IDs."""

    slug = _normalize_slug(value, fallback=fallback)
    if len(slug) <= 48:
        return slug
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{fallback}_{digest}"


def _coerce_mapping(value: object) -> dict[str, object]:
    """Coerce one mapping-like value into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one numeric input to float with fallback."""

    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return default
    try:
        return float(cast(str | bytes | bytearray | int | float, value))
    except (TypeError, ValueError):
        return default


def _weekday_class(value: date) -> str:
    """Return the shared weekday bucket for one date."""

    return "weekend" if value.weekday() >= 5 else "weekday"


def _resolve_timezone(name: str) -> tzinfo:
    """Resolve one timezone name with bounded fallback behavior."""

    normalized = _normalize_text(name) or _DEFAULT_TIMEZONE_NAME
    try:
        return ZoneInfo(normalized)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Falling back from invalid timezone %r to %r.", normalized, _DEFAULT_TIMEZONE_NAME)
        try:
            return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
        except (ValueError, ZoneInfoNotFoundError):
            return timezone.utc


def _normalize_datetime(value: datetime | None, *, timezone: tzinfo) -> datetime | None:
    """Normalize one datetime into the configured timezone."""

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _parse_source_event_datetime(event_id: str) -> datetime | None:
    """Extract the timestamp component from one synthetic source event ID."""

    clean = _normalize_text(event_id)
    if not clean:
        return None
    parts = clean.split(":", 2)
    if len(parts) < 2:
        return None
    timestamp = parts[1]
    for pattern in ("%Y%m%dT%H%M%S%f%z", "%Y%m%dT%H%M%S%z"):
        try:
            return datetime.strptime(timestamp, pattern)
        except ValueError:
            continue
    return None


def _ewma(values: Sequence[float], *, alpha: float = 0.35) -> float:
    """Return one simple exponentially weighted moving average."""

    if not values:
        raise ValueError("values must not be empty.")
    result = float(values[0])
    for value in values[1:]:
        result = (alpha * float(value)) + ((1.0 - alpha) * result)
    return result


def _entropy_from_counts(counts: Mapping[object, int]) -> float:
    """Return the Shannon entropy for one count distribution."""

    total = sum(int(value) for value in counts.values() if int(value) > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return the cosine similarity between two equal-length vectors."""

    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / (left_norm * right_norm)
