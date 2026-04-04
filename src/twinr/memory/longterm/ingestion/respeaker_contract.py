"""Normalize ReSpeaker claim payloads for long-term memory ingestion.

This module keeps the ReSpeaker-specific claim-contract parsing and memory
attribute helpers separate from the higher-level audio pattern extractor. It
accepts only structured claim payloads already exported by the proactive
runtime and never touches raw audio bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real


RESPEAKER_EPHEMERAL_STATE = "ephemeral_state"
RESPEAKER_SESSION_MEMORY = "session_memory"
RESPEAKER_OBSERVED_PREFERENCE = "observed_preference"
RESPEAKER_CONFIRMED_PREFERENCE = "confirmed_preference"


def _normalize_text(value: object | None) -> str:
    """Collapse one arbitrary value into a bounded single-line string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_optional_float(value: object | None) -> float | None:
    """Return one finite float value or ``None``."""

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


def _coerce_optional_int(value: object | None) -> int | None:
    """Return one integer value or ``None``."""

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


def _coerce_bool(value: object | None) -> bool | None:
    """Return one conservative boolean value or ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _normalize_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_mapping(value: object | None) -> dict[str, object]:
    """Return one shallow plain mapping or an empty dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _string_tuple(value: object | None) -> tuple[str, ...]:
    """Normalize one optional scalar or sequence into a string tuple."""

    raw_items: tuple[object, ...]
    if value is None:
        raw_items = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raw_items = (value,)
    elif isinstance(value, Sequence):
        raw_items = tuple(value)
    else:
        raw_items = (value,)
    normalized: list[str] = []
    for item in raw_items:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ReSpeakerClaimEvidence:
    """Hold one parsed claim-contract record for memory ingestion."""

    captured_at: float
    source: str
    source_type: str
    confidence: float
    sensor_window_ms: int
    memory_class: str
    session_id: int | None = None
    requires_confirmation: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ReSpeakerClaimEvidence | None":
        """Build one claim-evidence record from a fact payload mapping."""

        captured_at = _coerce_optional_float(payload.get("captured_at"))
        source = _normalize_text(payload.get("source"))
        source_type = _normalize_text(payload.get("source_type"))
        confidence = _coerce_optional_float(payload.get("confidence"))
        sensor_window_ms = _coerce_optional_int(payload.get("sensor_window_ms"))
        memory_class = _normalize_text(payload.get("memory_class")) or RESPEAKER_EPHEMERAL_STATE
        requires_confirmation = _coerce_bool(payload.get("requires_confirmation"))
        if (
            captured_at is None
            or not source
            or not source_type
            or confidence is None
            or sensor_window_ms is None
        ):
            return None
        return cls(
            captured_at=captured_at,
            source=source,
            source_type=source_type,
            confidence=max(0.0, min(1.0, confidence)),
            sensor_window_ms=max(0, sensor_window_ms),
            memory_class=memory_class,
            session_id=_coerce_optional_int(payload.get("session_id")),
            requires_confirmation=requires_confirmation is True,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize one parsed claim record back into plain attributes."""

        return {
            "captured_at": round(self.captured_at, 3),
            "source": self.source,
            "source_type": self.source_type,
            "confidence": round(self.confidence, 4),
            "sensor_window_ms": self.sensor_window_ms,
            "memory_class": self.memory_class,
            "session_id": self.session_id,
            "requires_confirmation": self.requires_confirmation,
        }


def coerce_respeaker_claim_evidence_map(value: object | None) -> dict[str, ReSpeakerClaimEvidence]:
    """Parse one nested ReSpeaker claim-contract payload into typed evidence."""

    payload = _coerce_mapping(value)
    evidence_map: dict[str, ReSpeakerClaimEvidence] = {}
    for key, item in payload.items():
        text_key = _normalize_text(key)
        if not text_key:
            continue
        item_mapping = _coerce_mapping(item)
        if not item_mapping:
            continue
        evidence = ReSpeakerClaimEvidence.from_payload(item_mapping)
        if evidence is not None:
            evidence_map[text_key] = evidence
    return evidence_map


def claim_contract_payload_subset(
    evidence_map: Mapping[str, ReSpeakerClaimEvidence],
    *,
    claim_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Return one plain nested payload containing only the requested claims."""

    subset: dict[str, dict[str, object]] = {}
    for claim_name in claim_names:
        evidence = evidence_map.get(claim_name)
        if evidence is not None:
            subset[claim_name] = evidence.to_payload()
    return subset


def claim_confidence_summary(
    evidence_map: Mapping[str, ReSpeakerClaimEvidence],
    *,
    claim_names: Sequence[str],
) -> float | None:
    """Return the average confidence for the requested claim subset."""

    confidences = [
        evidence_map[claim_name].confidence
        for claim_name in claim_names
        if claim_name in evidence_map
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def claim_session_id(
    evidence_map: Mapping[str, ReSpeakerClaimEvidence],
    *,
    claim_names: Sequence[str],
    fallback_session_id: int | None = None,
) -> int | None:
    """Return the first available session identifier in one claim subset."""

    for claim_name in claim_names:
        evidence = evidence_map.get(claim_name)
        if evidence is not None and evidence.session_id is not None:
            return evidence.session_id
    return fallback_session_id


def claim_sources(
    evidence_map: Mapping[str, ReSpeakerClaimEvidence],
    *,
    claim_names: Sequence[str],
) -> tuple[str | None, str | None]:
    """Return the first available source/source-type pair for one claim subset."""

    for claim_name in claim_names:
        evidence = evidence_map.get(claim_name)
        if evidence is not None:
            return evidence.source, evidence.source_type
    return None, None


def claim_requires_confirmation(
    evidence_map: Mapping[str, ReSpeakerClaimEvidence],
    *,
    claim_names: Sequence[str],
    default: bool,
) -> bool:
    """Return whether any selected claim already demands confirmation."""

    for claim_name in claim_names:
        evidence = evidence_map.get(claim_name)
        if evidence is not None and evidence.requires_confirmation:
            return True
    return default


__all__ = [
    "RESPEAKER_CONFIRMED_PREFERENCE",
    "RESPEAKER_EPHEMERAL_STATE",
    "RESPEAKER_OBSERVED_PREFERENCE",
    "RESPEAKER_SESSION_MEMORY",
    "ReSpeakerClaimEvidence",
    "claim_confidence_summary",
    "claim_contract_payload_subset",
    "claim_requires_confirmation",
    "claim_session_id",
    "claim_sources",
    "coerce_respeaker_claim_evidence_map",
]
