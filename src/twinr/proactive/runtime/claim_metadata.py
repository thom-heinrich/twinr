"""Shared metadata helpers for conservative multimodal runtime claims.

This module keeps the common claim-metadata contract for proactive runtime
inferences out of individual policy modules. The claims that consume it remain
small and explicit, while downstream automation facts receive the same compact
`confidence` / `source` / `requires_confirmation` surface everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class RuntimeClaimMetadata:
    """Describe provenance for one conservative runtime claim."""

    confidence: float
    source: str
    source_type: str = "observed"
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        """Normalize one claim metadata record into a bounded immutable form."""

        object.__setattr__(self, "confidence", _require_ratio(self.confidence))
        normalized_source = _normalize_text(self.source)
        if not normalized_source:
            raise ValueError("source must not be blank")
        object.__setattr__(self, "source", normalized_source)
        normalized_source_type = _normalize_text(self.source_type) or "observed"
        object.__setattr__(self, "source_type", normalized_source_type)
        object.__setattr__(self, "requires_confirmation", self.requires_confirmation is True)

    def to_payload(self) -> dict[str, object]:
        """Serialize one claim metadata record into plain automation facts."""

        return {
            "confidence": self.confidence,
            "source": self.source,
            "source_type": self.source_type,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
        *,
        default_confidence: float = 0.0,
        default_source: str,
        default_source_type: str = "observed",
        default_requires_confirmation: bool = False,
    ) -> "RuntimeClaimMetadata":
        """Parse serialized claim metadata while preserving conservative defaults."""

        mapping = coerce_mapping(payload)
        confidence = coerce_optional_ratio(mapping.get("confidence"))
        return cls(
            confidence=(default_confidence if confidence is None else confidence),
            source=normalize_text(mapping.get("source")) or default_source,
            source_type=normalize_text(mapping.get("source_type")) or default_source_type,
            requires_confirmation=(
                coerce_optional_bool(mapping.get("requires_confirmation"))
                if "requires_confirmation" in mapping
                else default_requires_confirmation
            )
            is True,
        )


def mean_confidence(values: tuple[float | None, ...]) -> float | None:
    """Return the arithmetic mean of available confidence values."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def coerce_mapping(value: object | None) -> dict[str, object]:
    """Coerce one optional mapping into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def coerce_optional_bool(value: object | None) -> bool | None:
    """Parse one optional conservative boolean value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = _normalize_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return None


def coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer value."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def coerce_optional_float(value: object | None) -> float | None:
    """Parse one optional finite float value."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        numeric = float(value)
    elif isinstance(value, (str, bytes, bytearray)):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def coerce_optional_ratio(value: object | None) -> float | None:
    """Parse one optional ratio in ``[0.0, 1.0]``."""

    numeric = coerce_optional_float(value)
    if numeric is None:
        return None
    return max(0.0, min(1.0, numeric))


def normalize_text(value: object | None) -> str:
    """Collapse one optional value into compact single-line text."""

    return _normalize_text(value)


def _require_ratio(value: object) -> float:
    """Return one validated confidence ratio."""

    parsed = coerce_optional_ratio(value)
    if parsed is None:
        raise ValueError("confidence must be a finite ratio between 0.0 and 1.0")
    return parsed


def _normalize_text(value: object | None) -> str:
    """Collapse one optional value into compact single-line text."""

    return " ".join(str(value or "").split()).strip()


__all__ = [
    "RuntimeClaimMetadata",
    "coerce_mapping",
    "coerce_optional_bool",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_optional_ratio",
    "mean_confidence",
    "normalize_text",
]
