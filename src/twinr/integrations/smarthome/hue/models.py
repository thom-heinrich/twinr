"""Define configuration models for the Hue smart-home provider."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


def _ensure_non_empty_text(field_name: str, value: object) -> str:
    """Return one stripped non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def _ensure_bool(field_name: str, value: object) -> bool:
    """Require a real boolean value."""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _ensure_positive_number(field_name: str, value: object) -> float:
    """Require one positive finite float."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive number.")
    if isinstance(value, Real):
        parsed = float(value)
    elif isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{field_name} must be a positive number.") from exc
    else:
        raise TypeError(f"{field_name} must be a positive number.")
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return parsed


@dataclass(frozen=True, slots=True)
class HueBridgeConfig:
    """Describe one local Hue bridge connection."""

    bridge_host: str
    application_key: str
    verify_tls: bool = True
    timeout_s: float = 10.0
    max_response_bytes: int = 2 * 1024 * 1024
    max_event_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        """Validate connection and response-size settings."""

        object.__setattr__(self, "bridge_host", _ensure_non_empty_text("bridge_host", self.bridge_host))
        object.__setattr__(self, "application_key", _ensure_non_empty_text("application_key", self.application_key))
        object.__setattr__(self, "verify_tls", _ensure_bool("verify_tls", self.verify_tls))
        object.__setattr__(self, "timeout_s", _ensure_positive_number("timeout_s", self.timeout_s))
        max_response_bytes = int(self.max_response_bytes)
        max_event_bytes = int(self.max_event_bytes)
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be >= 1024.")
        if max_event_bytes < 1024:
            raise ValueError("max_event_bytes must be >= 1024.")
        object.__setattr__(self, "max_response_bytes", max_response_bytes)
        object.__setattr__(self, "max_event_bytes", max_event_bytes)

    @property
    def base_url(self) -> str:
        """Return the local HTTPS base URL for the bridge."""

        return f"https://{self.bridge_host}"


__all__ = ["HueBridgeConfig"]
