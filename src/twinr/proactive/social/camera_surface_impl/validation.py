"""Numeric validation and coercion helpers for the camera surface package."""

from __future__ import annotations

import math
from typing import SupportsIndex, SupportsInt, TypeAlias, cast

from ..normalization import coerce_non_negative_int_or_default

IntLike: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex


def coerce_non_negative_int(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with fallback."""

    return coerce_non_negative_int_or_default(value, default=default)


def coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce one positive integer with a safe fallback."""

    if isinstance(value, bool):
        return default
    try:
        number = int(cast(IntLike, value))
    except (TypeError, ValueError):
        return default
    return default if number < 1 else number


def coerce_positive_float(value: object, *, default: float) -> float:
    """Return a finite positive float, falling back to ``default`` when needed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def coerce_non_negative_float(value: object, *, default: float) -> float:
    """Return one finite non-negative float, falling back to ``default``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


def require_positive_int(value: object, *, field_name: str) -> int:
    """Validate and return one positive integer config value."""

    number = coerce_non_negative_int(value, default=0)
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def require_non_negative_float(value: object, *, field_name: str) -> float:
    """Validate and return one non-negative float config value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative float")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a non-negative float")
    return number


def require_bounded_float(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return one bounded float config value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite float")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return number


def require_bounded_ratio(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return one bounded ratio-like float config value."""

    return require_bounded_float(
        value,
        field_name=field_name,
        minimum=minimum,
        maximum=maximum,
    )
