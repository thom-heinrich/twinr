"""Shared low-risk normalization helpers for the proactive social package.

This module only owns generic coercion that is reused across multiple social
submodules. Contract-specific policy and higher-level interpretation stay in
the caller modules.
"""

from __future__ import annotations

from enum import Enum
from typing import SupportsFloat, SupportsIndex, SupportsInt, TypeAlias, TypeVar, cast


EnumT = TypeVar("EnumT", bound=Enum)
_FloatLike: TypeAlias = str | bytes | bytearray | SupportsFloat | SupportsIndex
_IntLike: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex


def coerce_enum_member(
    value: object,
    enum_type: type[EnumT],
    *,
    unknown: EnumT,
    allow_stringify: bool = False,
) -> EnumT:
    """Coerce one token-like value into an enum member with a safe fallback."""

    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
    elif allow_stringify and value is not None:
        token = str(value).strip().lower()
    else:
        return unknown
    if not token:
        return unknown
    try:
        return enum_type(token)
    except ValueError:
        return unknown


def coerce_non_negative_int_or_default(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with caller-chosen fallback."""

    if isinstance(value, bool):
        return default
    try:
        number = int(cast(_IntLike, value))
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def coerce_spatial_box_coordinates(
    value: object,
) -> tuple[float, float, float, float] | None:
    """Coerce one box-like payload into ``top,left,bottom,right`` coordinates."""

    if isinstance(value, dict):
        candidate = (
            value.get("top"),
            value.get("left"),
            value.get("bottom"),
            value.get("right"),
        )
    elif isinstance(value, (tuple, list)) and len(value) == 4:
        candidate = tuple(value)
    else:
        return None
    coordinates = tuple(_coerce_coordinate(item) for item in candidate)
    if any(item is None for item in coordinates):
        return None
    top, left, bottom, right = cast(tuple[float, float, float, float], coordinates)
    return top, left, bottom, right


def _coerce_coordinate(value: object) -> float | None:
    """Coerce one coordinate-like value to a float or ``None``."""

    if isinstance(value, bool):
        return None
    try:
        return float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "coerce_enum_member",
    "coerce_non_negative_int_or_default",
    "coerce_spatial_box_coordinates",
]
