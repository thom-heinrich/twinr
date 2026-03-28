# CHANGELOG: 2026-03-27
# BUG-1: normalize_float now rejects NaN/Infinity instead of leaking/clamping non-finite values into valid score fields.
# BUG-2: normalize_int now rejects booleans and non-integral floats instead of silently coercing or truncating them.
# BUG-3: mapping_items now reuses normalize_mapping so key cleanup and nested JSON-like validation stay consistent.
# BUG-4: normalize_mapping now detects key collisions introduced by whitespace normalization instead of silently overwriting data.
# BUG-5: clean_text now decodes UTF-8 bytes and rejects containers/custom objects instead of stringifying schema drift into bogus text.
# SEC-1: Added configurable caps for text size, collection size, and nesting depth to reduce memory/CPU denial-of-service risk on Raspberry Pi 4 deployments.
# SEC-2: Nested mapping values are now validated as JSON-like data, numeric coercion is limited to primitive JSON-compatible types, and mapping keys must be textual.
# IMP-1: Added precise field-path validation errors and centralized recursive normalization so both typed model layers apply the same rules.
# IMP-2: Added optional msgspec bridge helpers for schema-first, Pi-friendly strict validation and JSON decoding.

"""Shared payload-normalization helpers for personality model modules.

The root ``models.py`` file and the ``intelligence/models.py`` submodule both
deserialize persisted remote-state payloads. Keeping these normalization rules
in one place avoids silent drift between the two typed model layers.

This revision keeps the existing public helpers, but hardens them to behave
like a 2026 boundary-validation layer:

* fail fast on schema drift instead of silently coercing containers/objects;
* reject non-finite numbers;
* cap payload sizes for constrained Raspberry Pi deployments;
* validate nested mapping payloads as JSON-like data;
* optionally expose ``msgspec`` fast paths for schema-first ingestion.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar, cast

try:  # Optional frontier fast-path for schema-first validation.
    import msgspec  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    msgspec = None  # type: ignore[assignment]

__all__ = [
    "JSONScalar",
    "JSONValue",
    "MAX_KEY_CHARS",
    "MAX_MAPPING_ITEMS",
    "MAX_NESTING_DEPTH",
    "MAX_SEQUENCE_ITEMS",
    "MAX_TEXT_CHARS",
    "clean_text",
    "decode_json_with_msgspec",
    "mapping_items",
    "normalize_float",
    "normalize_int",
    "normalize_mapping",
    "normalize_string_tuple",
    "optional_text",
    "required_mapping_text",
    "validate_with_msgspec",
]

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | tuple["JSONValue", ...] | dict[str, "JSONValue"]

_T = TypeVar("_T")

_BYTES_LIKE_TYPES = (bytes, bytearray, memoryview)


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """Read a positive integer limit from the environment."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer environment variable.") from exc
    if parsed < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}.")
    return parsed


MAX_TEXT_CHARS = _env_int("TWINR_NORMALIZE_MAX_TEXT_CHARS", 16_384, minimum=1)
MAX_KEY_CHARS = _env_int("TWINR_NORMALIZE_MAX_KEY_CHARS", 256, minimum=1)
MAX_SEQUENCE_ITEMS = _env_int("TWINR_NORMALIZE_MAX_SEQUENCE_ITEMS", 512, minimum=1)
MAX_MAPPING_ITEMS = _env_int("TWINR_NORMALIZE_MAX_MAPPING_ITEMS", 256, minimum=1)
MAX_NESTING_DEPTH = _env_int("TWINR_NORMALIZE_MAX_NESTING_DEPTH", 16, minimum=1)


def _raise_too_long(*, field_name: str, limit: int) -> None:
    raise ValueError(f"{field_name} exceeds the maximum allowed length of {limit} characters.")


def _ensure_text_limit(text: str, *, field_name: str, limit: int) -> None:
    if len(text) > limit:
        _raise_too_long(field_name=field_name, limit=limit)


def _decode_text_bytes(value: bytes | bytearray | memoryview, *, field_name: str) -> str:
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 text.") from exc


def _clean_text(value: object | None, *, field_name: str, max_chars: int) -> str:
    """Normalize a text-like scalar into one trimmed single-line string."""

    if value is None:
        return ""

    if isinstance(value, str):
        raw_text = value
    elif isinstance(value, _BYTES_LIKE_TYPES):
        raw_text = _decode_text_bytes(cast(bytes | bytearray | memoryview, value), field_name=field_name)
    elif isinstance(value, bool):
        # BREAKING: booleans no longer stringify into text because that hides schema drift.
        raise ValueError(f"{field_name} must be text-like, not boolean.")
    elif isinstance(value, int):
        raw_text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite text-like data.")
        raw_text = str(value)
    else:
        # BREAKING: containers/custom objects no longer stringify into text.
        raise ValueError(f"{field_name} must be a text-like scalar.")

    _ensure_text_limit(raw_text, field_name=field_name, limit=max_chars)
    normalized = " ".join(raw_text.split()).strip()
    _ensure_text_limit(normalized, field_name=field_name, limit=max_chars)
    return normalized


def _clean_mapping_key(raw_key: object, *, field_name: str) -> str:
    """Normalize one JSON-like mapping key."""

    if isinstance(raw_key, str):
        raw_text = raw_key
    elif isinstance(raw_key, _BYTES_LIKE_TYPES):
        raw_text = _decode_text_bytes(cast(bytes | bytearray | memoryview, raw_key), field_name=field_name)
    else:
        # BREAKING: JSON-like mappings now require textual keys.
        raise ValueError(f"{field_name} keys must be strings.")

    _ensure_text_limit(raw_text, field_name=field_name, limit=MAX_KEY_CHARS)
    normalized = " ".join(raw_text.split()).strip()
    _ensure_text_limit(normalized, field_name=field_name, limit=MAX_KEY_CHARS)
    return normalized


def _normalize_bounded_float(value: object | None, *, field_name: str, default: float) -> float:
    """Normalize a score-like value onto the inclusive 0..1 band."""

    if isinstance(default, bool) or not isinstance(default, (int, float)):
        raise ValueError(f"{field_name} default must be numeric.")

    if value is None:
        parsed = float(default)
    else:
        if isinstance(value, bool):
            # BREAKING: booleans no longer coerce to 0/1 for score fields.
            raise ValueError(f"{field_name} must be numeric, not boolean.")
        if isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"{field_name} must be numeric.")
            try:
                parsed = float(stripped)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be numeric.") from exc
        else:
            # BREAKING: custom objects with __float__ are no longer accepted.
            raise ValueError(f"{field_name} must be numeric.")

    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_integer(value: object | None, *, field_name: str, default: int, minimum: int) -> int:
    """Normalize an integer field onto a bounded lower limit."""

    if isinstance(default, bool) or not isinstance(default, int):
        raise ValueError(f"{field_name} default must be an integer.")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise ValueError(f"{field_name} minimum must be an integer.")

    if value is None:
        parsed = default
    elif isinstance(value, bool):
        # BREAKING: booleans no longer silently coerce to 0/1 for integer fields.
        raise ValueError(f"{field_name} must be an integer, not boolean.")
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer.")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must be an integer.")
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    else:
        raise ValueError(f"{field_name} must be an integer.")

    return max(minimum, parsed)


def _normalize_json_like(value: object | None, *, field_name: str, depth: int) -> JSONValue:
    """Validate nested JSON-like values for mapping payloads."""

    if depth > MAX_NESTING_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum nesting depth of {MAX_NESTING_DEPTH}.")

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return cast(JSONValue, value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats.")
        return value

    if isinstance(value, str):
        _ensure_text_limit(value, field_name=field_name, limit=MAX_TEXT_CHARS)
        return value

    if isinstance(value, list):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ValueError(f"{field_name} exceeds the maximum allowed size of {MAX_SEQUENCE_ITEMS} items.")
        return [_normalize_json_like(item, field_name=f"{field_name}[{index}]", depth=depth + 1) for index, item in enumerate(value)]

    if isinstance(value, tuple):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ValueError(f"{field_name} exceeds the maximum allowed size of {MAX_SEQUENCE_ITEMS} items.")
        return tuple(_normalize_json_like(item, field_name=f"{field_name}[{index}]", depth=depth + 1) for index, item in enumerate(value))

    if isinstance(value, Mapping):
        return _normalize_mapping_object(value, field_name=field_name, depth=depth)

    # BREAKING: arbitrary runtime objects are rejected; persisted payloads must stay JSON-like.
    raise ValueError(f"{field_name} must be JSON-like data.")


def _normalize_mapping_object(
    value: Mapping[object, object],
    *,
    field_name: str,
    depth: int,
) -> dict[str, JSONValue]:
    """Normalize a JSON-like mapping and its nested values."""

    if depth > MAX_NESTING_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum nesting depth of {MAX_NESTING_DEPTH}.")
    if len(value) > MAX_MAPPING_ITEMS:
        raise ValueError(f"{field_name} exceeds the maximum allowed size of {MAX_MAPPING_ITEMS} items.")

    normalized: dict[str, JSONValue] = {}
    for raw_key, raw_value in value.items():
        key = _clean_mapping_key(raw_key, field_name=f"{field_name}.<key>")
        if not key:
            raise ValueError(f"{field_name} cannot contain blank keys.")
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate key {key!r} after normalization.")
        normalized[key] = _normalize_json_like(raw_value, field_name=f"{field_name}.{key}", depth=depth + 1)
    return normalized


def clean_text(value: object | None) -> str:
    """Collapse a text-like scalar into a single trimmed string."""

    return _clean_text(value, field_name="value", max_chars=MAX_TEXT_CHARS)


def optional_text(value: object | None) -> str | None:
    """Return normalized text or ``None`` when the input is blank."""

    normalized = _clean_text(value, field_name="value", max_chars=MAX_TEXT_CHARS)
    return normalized or None


def normalize_string_tuple(value: object | None, *, field_name: str) -> tuple[str, ...]:
    """Normalize a sequence of text items into a non-blank tuple."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, *_BYTES_LIKE_TYPES)):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    if len(value) > MAX_SEQUENCE_ITEMS:
        raise ValueError(f"{field_name} exceeds the maximum allowed size of {MAX_SEQUENCE_ITEMS} items.")

    items: list[str] = []
    for index, item in enumerate(value):
        normalized = _clean_text(item, field_name=f"{field_name}[{index}]", max_chars=MAX_TEXT_CHARS)
        if not normalized:
            raise ValueError(f"{field_name}[{index}] cannot be blank.")
        items.append(normalized)
    return tuple(items)


def normalize_float(value: object | None, *, field_name: str, default: float) -> float:
    """Normalize a confidence/salience-like value onto the inclusive 0..1 band."""

    return _normalize_bounded_float(value, field_name=field_name, default=default)


def normalize_int(
    value: object | None,
    *,
    field_name: str,
    default: int,
    minimum: int = 0,
) -> int:
    """Normalize an integer field onto a bounded lower limit."""

    return _normalize_integer(value, field_name=field_name, default=default, minimum=minimum)


def normalize_mapping(value: object | None, *, field_name: str) -> Mapping[str, JSONValue] | None:
    """Normalize a plain JSON-like mapping field."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return _normalize_mapping_object(value, field_name=field_name, depth=0)


def mapping_items(value: object | None, *, field_name: str) -> tuple[Mapping[str, JSONValue], ...]:
    """Normalize a payload field into a tuple of JSON-like mappings."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, *_BYTES_LIKE_TYPES)):
        raise ValueError(f"{field_name} must be a sequence of mappings.")
    if len(value) > MAX_SEQUENCE_ITEMS:
        raise ValueError(f"{field_name} exceeds the maximum allowed size of {MAX_SEQUENCE_ITEMS} items.")

    items: list[Mapping[str, JSONValue]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping.")
        items.append(_normalize_mapping_object(item, field_name=f"{field_name}[{index}]", depth=0))
    return tuple(items)


def required_mapping_text(
    payload: Mapping[str, object],
    *,
    field_name: str,
    aliases: tuple[str, ...] = (),
) -> str:
    """Read one required normalized text field from a payload mapping."""

    for key in (field_name,) + aliases:
        normalized = _clean_text(payload.get(key), field_name=key, max_chars=MAX_TEXT_CHARS)
        if normalized:
            return normalized
    raise ValueError(f"{field_name} is required.")


def validate_with_msgspec(value: object, *, schema: type[_T]) -> _T:
    """Validate a Python object against a typed schema using optional ``msgspec``.

    This is a frontier fast-path for callers that want schema-first validation
    at the ingestion boundary while keeping this module usable without extra
    dependencies.
    """

    if msgspec is None:  # pragma: no cover - optional dependency
        raise RuntimeError("msgspec is not installed. Install msgspec to use validate_with_msgspec().")
    try:
        return cast(_T, msgspec.convert(value, type=schema))
    except msgspec.ValidationError as exc:  # type: ignore[union-attr]
        raise ValueError(str(exc)) from exc


def decode_json_with_msgspec(data: str | bytes | bytearray, *, schema: type[_T]) -> _T:
    """Decode and validate JSON directly into ``schema`` using optional ``msgspec``."""

    if msgspec is None:  # pragma: no cover - optional dependency
        raise RuntimeError("msgspec is not installed. Install msgspec to use decode_json_with_msgspec().")
    try:
        return cast(_T, msgspec.json.decode(data, type=schema))
    except msgspec.ValidationError as exc:  # type: ignore[union-attr]
        raise ValueError(str(exc)) from exc