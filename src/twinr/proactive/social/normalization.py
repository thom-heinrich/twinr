# CHANGELOG: 2026-03-29
# BUG-1: coerce_enum_member unterstützt jetzt echte Enum-Value-Lookups
#        (z. B. IntEnum-Codes) vor dem String-Fallback; vorher fielen valide
#        Nicht-String-Values still auf `unknown`.
# BUG-2: coerce_non_negative_int_or_default lehnt jetzt lossy, nicht-ganzzahlige
#        Numerics ab statt sie still mit int() zu truncieren.
# BUG-3: coerce_spatial_box_coordinates verwirft jetzt nicht-endliche und
#        invertierte Boxen; vorher konnten NaN/Inf/negative Geometrien downstream
#        zu falschem Verhalten führen.
# SEC-1: Begrenzte Token-/Numerik-Textlängen reduzieren Oversized-Payload-DoS
#        auf Edge-Geräten.
# IMP-1: Enum-Coercion unterstützt jetzt zusätzlich case-insensitive Member-Namen,
#        stringifizierte skalare Enum-Werte und Alias-Namen via Cache.
# IMP-2: Box-Coercion akzeptiert generische Mapping/Sequence-Payloads und erzwingt
#        kanonische top,left,bottom,right-Ordnung.

"""Shared low-risk normalization helpers for the proactive social package.

This module owns generic coercion reused across multiple social submodules.
Contract-specific policy and higher-level interpretation stay in caller modules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import lru_cache
from operator import index as index_operator
from typing import Final, SupportsFloat, SupportsIndex, SupportsInt, TypeAlias, TypeVar, cast


EnumT = TypeVar("EnumT", bound=Enum)
_FloatLike: TypeAlias = str | bytes | bytearray | SupportsFloat | SupportsIndex
_IntLike: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex

_MAX_ENUM_TOKEN_LENGTH: Final[int] = 128
_MAX_NUMERIC_TEXT_LENGTH: Final[int] = 256


def coerce_enum_member(
    value: object,
    enum_type: type[EnumT],
    *,
    unknown: EnumT,
    allow_stringify: bool = False,
) -> EnumT:
    """Coerce one token-like value into an enum member with a safe fallback.

    Resolution order:
    1. Already-correct enum member instance.
    2. Exact enum value lookup via ``enum_type(value)``.
    3. Case-insensitive lookup over enum names and string-like/scalar values.
    """

    if isinstance(value, enum_type):
        return value

    try:
        return enum_type(cast(object, value))
    except (TypeError, ValueError):
        pass

    token = _coerce_text(
        value,
        allow_stringify=allow_stringify,
        normalize_case=True,
        max_length=_MAX_ENUM_TOKEN_LENGTH,
    )
    if token is None:
        return unknown

    return cast(EnumT, _enum_normalized_lookup(enum_type).get(token, unknown))


def coerce_non_negative_int_or_default(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with caller-chosen fallback."""

    number = _coerce_exact_non_negative_int(value)
    return default if number is None else number


def coerce_spatial_box_coordinates(
    value: object,
) -> tuple[float, float, float, float] | None:
    """Coerce one box-like payload into canonical ``top,left,bottom,right`` coordinates."""

    candidate = _extract_spatial_box_candidate(value)
    if candidate is None:
        return None

    coordinates = tuple(_coerce_coordinate(item) for item in candidate)
    if any(item is None for item in coordinates):
        return None

    top, left, bottom, right = cast(tuple[float, float, float, float], coordinates)

    # BREAKING: inverted or otherwise non-canonical boxes are now rejected
    # instead of being passed through as silently wrong geometry.
    if bottom < top or right < left:
        return None

    return top, left, bottom, right


def _coerce_exact_non_negative_int(value: object) -> int | None:
    """Coerce one value to an exact non-negative integer or ``None``."""

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 0 else None

    if isinstance(value, float):
        # BREAKING: non-integral floats now reject instead of truncating.
        if not math.isfinite(value) or not value.is_integer():
            return None
        number = int(value)
        return number if number >= 0 else None

    if isinstance(value, (str, bytes, bytearray)):
        token = _coerce_text(
            value,
            normalize_case=False,
            max_length=_MAX_NUMERIC_TEXT_LENGTH,
        )
        if token is None:
            return None
        try:
            number = int(token, 10)
        except ValueError:
            return None
        return number if number >= 0 else None

    try:
        number = index_operator(cast(SupportsIndex, value))
    except TypeError:
        try:
            number = int(cast(_IntLike, value))
        except (TypeError, ValueError, OverflowError):
            return None

        # Reject lossy __int__ coercions where we can prove the source value is
        # not exactly equal to the resulting integer.
        try:
            if value != number:
                return None
        except Exception:
            pass

    return number if number >= 0 else None


def _extract_spatial_box_candidate(
    value: object,
) -> tuple[object, object, object, object] | None:
    """Extract one 4-tuple candidate from mapping- or sequence-like payloads."""

    if isinstance(value, Mapping):
        return (
            value.get("top"),
            value.get("left"),
            value.get("bottom"),
            value.get("right"),
        )

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 4
    ):
        return cast(tuple[object, object, object, object], tuple(value))

    return None


def _coerce_coordinate(value: object) -> float | None:
    """Coerce one coordinate-like value to a finite float or ``None``."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (str, bytes, bytearray)):
        value = _coerce_text(
            value,
            normalize_case=False,
            max_length=_MAX_NUMERIC_TEXT_LENGTH,
        )
        if value is None:
            return None

    try:
        coordinate = float(cast(_FloatLike, value))
    except (TypeError, ValueError, OverflowError):
        return None

    # BREAKING: NaN/Inf are now rejected instead of propagating poison values.
    if not math.isfinite(coordinate):
        return None

    return coordinate


def _coerce_text(
    value: object,
    *,
    allow_stringify: bool = False,
    normalize_case: bool = False,
    max_length: int,
) -> str | None:
    """Coerce one small text-like value to stripped text or ``None``."""

    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif allow_stringify and value is not None:
        text = str(value)
    else:
        return None

    text = text.strip()
    if not text:
        return None

    # BREAKING: oversized tokens now reject to bound parse cost on edge devices.
    if len(text) > max_length:
        return None

    if normalize_case:
        text = text.casefold()

    return text


def _normalize_enum_member_token(value: object) -> str | None:
    """Normalize one enum name/value into one case-insensitive token."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return _coerce_text(
            str(value),
            normalize_case=True,
            max_length=_MAX_ENUM_TOKEN_LENGTH,
        )

    return _coerce_text(
        value,
        normalize_case=True,
        max_length=_MAX_ENUM_TOKEN_LENGTH,
    )


@lru_cache(maxsize=None)
def _enum_normalized_lookup(enum_type: type[Enum]) -> dict[str, Enum]:
    """Build one cached case-insensitive lookup table for enum names/values."""

    lookup: dict[str, Enum] = {}

    # __members__ includes aliases; keeping the first match preserves stable,
    # definition-ordered behavior while still allowing alias names to resolve.
    for name, member in enum_type.__members__.items():
        normalized_name = _coerce_text(
            name,
            normalize_case=True,
            max_length=_MAX_ENUM_TOKEN_LENGTH,
        )
        if normalized_name is not None:
            lookup.setdefault(normalized_name, member)

        normalized_value = _normalize_enum_member_token(member.value)
        if normalized_value is not None:
            lookup.setdefault(normalized_value, member)

    return lookup


__all__ = [
    "coerce_enum_member",
    "coerce_non_negative_int_or_default",
    "coerce_spatial_box_coordinates",
]