# CHANGELOG: 2026-03-29
# BUG-1: Fixed uncaught OverflowError escapes for non-finite or out-of-range numeric conversions.
# BUG-2: Stopped silent truncation of non-integral numerics in integer coercion and validation paths.
# SEC-1: Reject pathological textual numerics early and reject non-finite float spellings to reduce config-driven crash/DoS risk.
# IMP-1: Unified all parsing behind exact-integer and finite-float helpers that accept standard numeric objects and numeric text.
# IMP-2: Added a Python 3.14 float.from_number fast-path, explicit API exports, and defensive bound validation.

"""Numeric validation and coercion helpers for the camera surface package.

This module keeps a zero-dependency runtime footprint for Raspberry Pi class
deployments while enforcing predictable numeric semantics:

- integer helpers only accept *exact* integers; fractional values are rejected
  instead of being silently truncated
- float helpers only return finite floats
- coercion helpers fall back to the provided default
- require_* helpers raise ValueError with field-specific messages

2026 frontier note:
Structured schema validation is best enforced at the configuration boundary
(e.g. msgspec Structs or strict Pydantic models). These helpers remain the
lightweight fallback for internal and hot-path code.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Final, TypeAlias

NumericText: TypeAlias = str | bytes | bytearray
IntLike: TypeAlias = NumericText | Real

__all__ = [
    "IntLike",
    "coerce_non_negative_int",
    "coerce_positive_int",
    "coerce_positive_float",
    "coerce_non_negative_float",
    "require_positive_int",
    "require_non_negative_float",
    "require_bounded_float",
    "require_bounded_ratio",
]

# BREAKING: textual numerics longer than this are rejected intentionally.
# Camera-facing configuration values do not need arbitrarily long numerics, and
# this keeps coercion latency predictable on Raspberry Pi-class edge hardware.
_MAX_NUMERIC_TEXT_LENGTH: Final[int] = 256
_FLOAT_FROM_NUMBER = getattr(float, "from_number", None)


def _normalize_numeric_text(value: NumericText) -> str | None:
    """Return stripped numeric text or ``None`` when the input is unusable."""

    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = bytes(value).decode("ascii").strip()
        except UnicodeDecodeError:
            return None

    if not text or len(text) > _MAX_NUMERIC_TEXT_LENGTH:
        return None
    return text


def _parse_exact_int(value: object) -> int | None:
    """Return one exact integer or ``None`` when coercion would be unsafe."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        text = _normalize_numeric_text(value)
        if text is None:
            return None
        try:
            return int(text)
        except (TypeError, ValueError, OverflowError):
            return None

    if not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _parse_finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` when coercion is invalid."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (str, bytes, bytearray)):
        text = _normalize_numeric_text(value)
        if text is None:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, Real):
        try:
            if _FLOAT_FROM_NUMBER is not None:
                number = _FLOAT_FROM_NUMBER(value)
            else:
                number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None

    if not math.isfinite(number):
        return None
    return number


def _require_finite_bounds(*, field_name: str, minimum: float, maximum: float) -> None:
    """Validate bounded-float helper bounds before comparing values."""

    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"{field_name} bounds must be finite")
    if minimum > maximum:
        raise ValueError(f"{field_name} minimum cannot exceed maximum")


def coerce_non_negative_int(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with fallback."""

    number = _parse_exact_int(value)
    return default if number is None or number < 0 else number


def coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce one value to a positive integer with a safe fallback."""

    number = _parse_exact_int(value)
    return default if number is None or number < 1 else number


def coerce_positive_float(value: object, *, default: float) -> float:
    """Return one finite positive float, falling back to ``default`` when needed."""

    number = _parse_finite_float(value)
    return default if number is None or number <= 0.0 else number


def coerce_non_negative_float(value: object, *, default: float) -> float:
    """Return one finite non-negative float, falling back to ``default``."""

    number = _parse_finite_float(value)
    return default if number is None or number < 0.0 else number


def require_positive_int(value: object, *, field_name: str) -> int:
    """Validate and return one positive integer config value."""

    number = _parse_exact_int(value)
    if number is None or number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def require_non_negative_float(value: object, *, field_name: str) -> float:
    """Validate and return one finite non-negative float config value."""

    number = _parse_finite_float(value)
    if number is None or number < 0.0:
        raise ValueError(f"{field_name} must be a non-negative float")
    return number


def require_bounded_float(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return one finite bounded float config value."""

    _require_finite_bounds(field_name=field_name, minimum=minimum, maximum=maximum)

    number = _parse_finite_float(value)
    if number is None or number < minimum or number > maximum:
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
