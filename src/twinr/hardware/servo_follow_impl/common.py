"""Share small numeric and GPIO helpers for servo-follow modules."""

from __future__ import annotations

import math

def _clamp_ratio(value: float) -> float:
    if value != value:
        return 0.5
    return max(0.0, min(1.0, value))

def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    raw_value = default if value is None else value
    if not isinstance(raw_value, (int, float, str)):
        normalized = default
    else:
        try:
            normalized = float(raw_value)
        except (TypeError, ValueError):
            normalized = default
    if not isinstance(normalized, float) or not math.isfinite(normalized):
        normalized = default
    return max(minimum, min(maximum, normalized))

def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    raw_value = default if value is None else value
    if not isinstance(raw_value, (int, float, str)):
        normalized = default
    else:
        try:
            normalized = int(raw_value)
        except (TypeError, ValueError):
            normalized = default
    return max(minimum, min(maximum, normalized))

def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

def _normalize_chip_index(chip_name: str) -> int:
    normalized = str(chip_name or "gpiochip0").strip()
    if not normalized:
        raise ValueError("GPIO chip name must not be empty")
    if normalized.startswith("/dev/"):
        normalized = normalized.rsplit("/", 1)[-1]
    if normalized.startswith("gpiochip"):
        suffix = normalized[len("gpiochip") :]
        if suffix and suffix.isdigit():
            return int(suffix)
        raise ValueError(f"Unsupported GPIO chip name for servo output: {chip_name}")
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"Unsupported GPIO chip name for servo output: {chip_name}")
