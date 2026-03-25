"""Shared payload-normalization helpers for personality model modules.

The root ``models.py`` file and the ``intelligence/models.py`` submodule both
deserialize persisted remote-state payloads. Keeping these normalization rules
in one place avoids silent drift between the two typed model layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def clean_text(value: object | None) -> str:
    """Collapse arbitrary text-like input into a single trimmed string."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def optional_text(value: object | None) -> str | None:
    """Return normalized text or ``None`` when the input is blank."""

    normalized = clean_text(value)
    return normalized or None


def normalize_string_tuple(value: object | None, *, field_name: str) -> tuple[str, ...]:
    """Normalize a sequence of text items into a non-blank tuple."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    items: list[str] = []
    for index, item in enumerate(value):
        normalized = clean_text(item)
        if not normalized:
            raise ValueError(f"{field_name}[{index}] cannot be blank.")
        items.append(normalized)
    return tuple(items)


def normalize_float(value: object | None, *, field_name: str, default: float) -> float:
    """Normalize a confidence/salience-like value onto the inclusive 0..1 band."""

    if value is None:
        return default
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def normalize_int(
    value: object | None,
    *,
    field_name: str,
    default: int,
    minimum: int = 0,
) -> int:
    """Normalize an integer field onto a bounded lower limit."""

    if value is None:
        return default
    try:
        parsed = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    return max(minimum, parsed)


def normalize_mapping(value: object | None, *, field_name: str) -> Mapping[str, object] | None:
    """Normalize a plain JSON-like mapping field."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = clean_text(raw_key)
        if not key:
            raise ValueError(f"{field_name} cannot contain blank keys.")
        normalized[key] = raw_value
    return normalized


def mapping_items(value: object | None, *, field_name: str) -> tuple[Mapping[str, object], ...]:
    """Normalize a payload field into a tuple of mappings."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of mappings.")
    items: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping.")
        items.append(item)
    return tuple(items)


def required_mapping_text(
    payload: Mapping[str, object],
    *,
    field_name: str,
    aliases: tuple[str, ...] = (),
) -> str:
    """Read one required normalized text field from a payload mapping."""

    for key in (field_name,) + aliases:
        normalized = clean_text(payload.get(key))
        if normalized:
            return normalized
    raise ValueError(f"{field_name} is required.")
