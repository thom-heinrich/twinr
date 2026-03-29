"""Build and serialize confidence-bearing ReSpeaker claim metadata.

This module keeps the per-field provenance contract for XVF3800-derived
runtime facts separate from signal acquisition and policy orchestration. It
turns one typed signal snapshot into inspectable claim metadata that later
runtime and memory layers can export without re-inventing source, confidence,
session semantics, or provenance annotations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from twinr.hardware.respeaker.models import (
    FrozenDict,
    ReSpeakerClaimMetadata,
    ReSpeakerMemoryClass,
    ReSpeakerSignalSnapshot,
)

# CHANGELOG: 2026-03-28
# BUG-1: Fixed falsy-zero handling for direction_confidence so 0.0 no longer inflates
#        azimuth/direction metadata confidence.
# BUG-2: Rebuild the canonical claim contract from the snapshot during serialization and
#        preserve only non-canonical precomputed extras, preventing stale or missing
#        snapshot.claim_contract data from silently leaking into payloads.
# SEC-1: Sanitize non-finite / out-of-range confidence values before payload emission to
#        prevent malformed JSON-like payloads and broken downstream trust decisions when
#        transport data is corrupt or compromised.
# IMP-1: Low-confidence derived claims now set requires_confirmation=True instead of being
#        emitted as action-ready by default.
# IMP-2: Serialized payloads now include a versioned twinr_meta provenance envelope for
#        cross-run analytics and schema evolution.
# BREAKING: payload dictionaries now include a top-level "twinr_meta" key.
# BREAKING: low-confidence derived claims may now flip requires_confirmation to True.

_DIRECT_HARDWARE_CONFIDENCE: Final[float] = 0.99
_DIRECT_SIGNAL_CONFIDENCE: Final[float] = 0.76
_DERIVED_SIGNAL_CONFIDENCE: Final[float] = 0.72
_BEAM_ACTIVITY_CONFIDENCE: Final[float] = 0.68
_AMBIENT_ACTIVITY_TRUE_CONFIDENCE: Final[float] = 0.82
_AMBIENT_ACTIVITY_FALSE_CONFIDENCE: Final[float] = 0.68

_REQUIRES_CONFIRMATION_CONFIDENCE: Final[float] = 0.75
_DOA_REQUIRES_CONFIRMATION_CONFIDENCE: Final[float] = 0.60

_SCHEMA_FAMILY: Final[str] = "twinr.respeaker.claim-metadata"
_SCHEMA_VERSION: Final[str] = "2.0.0"
_CONFIDENCE_POLICY_VERSION: Final[str] = "twinr.respeaker.confidence-policy.v2"

_SESSION_SCOPED_CLAIMS = frozenset(
    {
        "speech_detected",
        "room_quiet",
        "recent_speech_age_s",
        "azimuth_deg",
        "direction_confidence",
        "beam_activity",
        "speech_overlap_likely",
        "barge_in_detected",
        "non_speech_audio_likely",
        "background_media_likely",
    }
)

_CONFIRMATION_GATED_CLAIMS = frozenset(
    {
        "azimuth_deg",
        "beam_activity",
        "speech_overlap_likely",
        "barge_in_detected",
        "non_speech_audio_likely",
        "background_media_likely",
    }
)

_CLAIM_PROVENANCE_KIND: Final[dict[str, str]] = {
    "device_runtime_mode": "direct_hardware",
    "host_control_ready": "direct_hardware",
    "transport_reason": "direct_hardware",
    "firmware_version": "direct_hardware",
    "assistant_output_active": "direct_hardware",
    "mute_active": "direct_hardware",
    "gpo_logic_levels": "direct_hardware",
    "speech_detected": "direct_signal",
    "room_quiet": "direct_signal",
    "recent_speech_age_s": "direct_signal",
    "azimuth_deg": "derived_signal",
    "direction_confidence": "direct_signal",
    "beam_activity": "derived_signal",
    "speech_overlap_likely": "derived_signal",
    "barge_in_detected": "derived_signal",
    "non_speech_audio_likely": "ambient_inference",
    "background_media_likely": "ambient_inference",
}

_CLAIM_PROVENANCE_INPUTS: Final[dict[str, tuple[str, ...]]] = {
    "device_runtime_mode": ("device_runtime_mode",),
    "host_control_ready": ("host_control_ready",),
    "transport_reason": ("transport_reason",),
    "firmware_version": ("firmware_version",),
    "assistant_output_active": ("assistant_output_active",),
    "mute_active": ("mute_active",),
    "gpo_logic_levels": ("gpo_logic_levels",),
    "speech_detected": ("speech_detected",),
    "room_quiet": ("room_quiet",),
    "recent_speech_age_s": ("recent_speech_age_s",),
    "azimuth_deg": ("azimuth_deg", "direction_confidence"),
    "direction_confidence": ("direction_confidence",),
    "beam_activity": ("beam_activity",),
    "speech_overlap_likely": ("speech_overlap_likely",),
    "barge_in_detected": ("barge_in_detected",),
    "non_speech_audio_likely": ("non_speech_audio_likely",),
    "background_media_likely": ("background_media_likely",),
}


def build_respeaker_signal_claim_contract(
    snapshot: ReSpeakerSignalSnapshot,
) -> FrozenDict[ReSpeakerClaimMetadata]:
    """Build the base per-claim metadata mapping for one signal snapshot."""

    claims: dict[str, ReSpeakerClaimMetadata] = {}
    direction_confidence = _optional_finite_float(getattr(snapshot, "direction_confidence", None))

    base_kwargs = {
        "captured_at": snapshot.captured_at,
        "source": snapshot.source,
        "source_type": snapshot.source_type,
        "sensor_window_ms": snapshot.sensor_window_ms,
        "memory_class": ReSpeakerMemoryClass.EPHEMERAL_STATE,
    }

    _add_claim(
        claims,
        "device_runtime_mode",
        snapshot.device_runtime_mode,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "host_control_ready",
        snapshot.host_control_ready,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "transport_reason",
        snapshot.transport_reason,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "firmware_version",
        snapshot.firmware_version,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "assistant_output_active",
        snapshot.assistant_output_active,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "mute_active",
        snapshot.mute_active,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "gpo_logic_levels",
        snapshot.gpo_logic_levels,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "speech_detected",
        snapshot.speech_detected,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "room_quiet",
        snapshot.room_quiet,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "recent_speech_age_s",
        snapshot.recent_speech_age_s,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "azimuth_deg",
        snapshot.azimuth_deg,
        confidence=direction_confidence if direction_confidence is not None else 0.0,
        requires_confirmation=_doa_requires_confirmation(direction_confidence),
        **base_kwargs,
    )
    _add_claim(
        claims,
        "direction_confidence",
        snapshot.direction_confidence,
        confidence=direction_confidence if direction_confidence is not None else 0.0,
        requires_confirmation=False,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "beam_activity",
        snapshot.beam_activity,
        confidence=_BEAM_ACTIVITY_CONFIDENCE,
        requires_confirmation=True,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "speech_overlap_likely",
        snapshot.speech_overlap_likely,
        confidence=_DERIVED_SIGNAL_CONFIDENCE,
        requires_confirmation=True,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "barge_in_detected",
        snapshot.barge_in_detected,
        confidence=_DERIVED_SIGNAL_CONFIDENCE,
        requires_confirmation=True,
        **base_kwargs,
    )
    return FrozenDict(claims)


def build_respeaker_claim_payloads(
    *,
    signal_snapshot: ReSpeakerSignalSnapshot | None,
    session_id: int | None = None,
    non_speech_audio_likely: bool | None = None,
    background_media_likely: bool | None = None,
) -> dict[str, dict[str, object]]:
    """Serialize ReSpeaker claim metadata for runtime facts and memory ingestion."""

    if signal_snapshot is None:
        return {}

    normalized_session_id = _normalize_session_id(session_id)
    payloads = claim_contract_to_payload(
        _resolve_claim_contract(signal_snapshot),
        session_id=normalized_session_id,
    )

    if non_speech_audio_likely is not None:
        payloads["non_speech_audio_likely"] = _build_ambient_claim_payload(
            claim_name="non_speech_audio_likely",
            signal_snapshot=signal_snapshot,
            session_id=normalized_session_id,
            value=non_speech_audio_likely,
        )
    if background_media_likely is not None:
        payloads["background_media_likely"] = _build_ambient_claim_payload(
            claim_name="background_media_likely",
            signal_snapshot=signal_snapshot,
            session_id=normalized_session_id,
            value=background_media_likely,
        )
    return payloads


def claim_contract_to_payload(
    claim_contract: Mapping[str, ReSpeakerClaimMetadata],
    *,
    session_id: int | None = None,
) -> dict[str, dict[str, object]]:
    """Serialize one claim contract mapping into plain nested dictionaries."""

    normalized_session_id = _normalize_session_id(session_id)
    payloads: dict[str, dict[str, object]] = {}
    for claim_name, metadata in claim_contract.items():
        claim_name_str = str(claim_name)
        claim_metadata = metadata
        if normalized_session_id is not None and claim_name_str in _SESSION_SCOPED_CLAIMS:
            claim_metadata = metadata.with_session_id(normalized_session_id)
        payloads[claim_name_str] = _annotate_payload(
            claim_name_str,
            claim_metadata.to_payload(),
        )
    return payloads


def _add_claim(
    claims: dict[str, ReSpeakerClaimMetadata],
    claim_name: str,
    value: object,
    *,
    confidence: float,
    captured_at: float,
    source: str,
    source_type: str,
    sensor_window_ms: int,
    memory_class: ReSpeakerMemoryClass,
    requires_confirmation: bool,
) -> None:
    """Insert one claim metadata record when the underlying value is known."""

    if not _is_known_value(value):
        return

    claims[claim_name] = ReSpeakerClaimMetadata(
        captured_at=captured_at,
        source=source,
        source_type=source_type,
        confidence=_sanitize_confidence(confidence, fallback=0.0),
        sensor_window_ms=sensor_window_ms,
        memory_class=memory_class,
        requires_confirmation=requires_confirmation,
    )


def _build_ambient_claim_payload(
    *,
    claim_name: str,
    signal_snapshot: ReSpeakerSignalSnapshot,
    session_id: int | None,
    value: bool,
) -> dict[str, object]:
    """Build one ad-hoc ambient-activity claim payload."""

    confidence = (
        _AMBIENT_ACTIVITY_TRUE_CONFIDENCE
        if value
        else _AMBIENT_ACTIVITY_FALSE_CONFIDENCE
    )
    metadata = ReSpeakerClaimMetadata(
        captured_at=signal_snapshot.captured_at,
        source=signal_snapshot.source,
        source_type=signal_snapshot.source_type,
        confidence=_sanitize_confidence(confidence, fallback=0.0),
        sensor_window_ms=signal_snapshot.sensor_window_ms,
        memory_class=ReSpeakerMemoryClass.EPHEMERAL_STATE,
        session_id=session_id if claim_name in _SESSION_SCOPED_CLAIMS else None,
        requires_confirmation=_ambient_requires_confirmation(value),
    )
    return _annotate_payload(claim_name, metadata.to_payload())


def _resolve_claim_contract(
    signal_snapshot: ReSpeakerSignalSnapshot,
) -> FrozenDict[ReSpeakerClaimMetadata]:
    """Build canonical claims from the snapshot and preserve non-canonical extras."""

    canonical_claims = dict(build_respeaker_signal_claim_contract(signal_snapshot))

    existing_contract = getattr(signal_snapshot, "claim_contract", None)
    if isinstance(existing_contract, Mapping):
        for claim_name, metadata in existing_contract.items():
            claim_name_str = str(claim_name)
            canonical_claims.setdefault(claim_name_str, metadata)

    return FrozenDict(canonical_claims)


def _annotate_payload(claim_name: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Attach stable Twinr provenance annotations and sanitize confidence."""

    annotated = dict(payload)
    raw_confidence = annotated.get("confidence")
    sanitized_confidence = _sanitize_confidence(raw_confidence, fallback=0.0)
    confidence_was_sanitized = _confidence_needs_sanitization(raw_confidence)
    annotated["confidence"] = sanitized_confidence

    requires_confirmation = bool(annotated.get("requires_confirmation", False))
    if not requires_confirmation and claim_name in _CONFIRMATION_GATED_CLAIMS:
        requires_confirmation = _claim_requires_confirmation(
            claim_name,
            sanitized_confidence,
        )
        annotated["requires_confirmation"] = requires_confirmation

    twinr_meta: dict[str, object] = {
        "schema_family": _SCHEMA_FAMILY,
        "schema_version": _SCHEMA_VERSION,
        "confidence_policy_version": _CONFIDENCE_POLICY_VERSION,
        "claim_name": claim_name,
        "claim_scope": "session" if claim_name in _SESSION_SCOPED_CLAIMS else "device",
        "provenance_kind": _CLAIM_PROVENANCE_KIND.get(claim_name, "external"),
        "provenance_inputs": list(_CLAIM_PROVENANCE_INPUTS.get(claim_name, ())),
        "confidence_band": _confidence_band(sanitized_confidence),
    }
    if confidence_was_sanitized:
        twinr_meta["confidence_sanitized"] = True
    if requires_confirmation:
        twinr_meta["confirmation_reason"] = _confirmation_reason(
            claim_name,
            sanitized_confidence,
        )

    annotated["twinr_meta"] = twinr_meta
    return annotated


def _is_known_value(value: object) -> bool:
    """Return True when a signal value is present and numerically sane."""

    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _optional_finite_float(value: object) -> float | None:
    """Coerce one value to a finite float when possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _sanitize_confidence(value: object, *, fallback: float) -> float:
    """Clamp one confidence-like value to the closed interval [0.0, 1.0]."""

    coerced = _optional_finite_float(value)
    if coerced is None:
        return fallback
    if coerced < 0.0:
        return 0.0
    if coerced > 1.0:
        return 1.0
    return coerced


def _confidence_needs_sanitization(value: object) -> bool:
    """Return True when a confidence-like value was invalid or out of range."""

    coerced = _optional_finite_float(value)
    if coerced is None:
        return value is not None
    return coerced < 0.0 or coerced > 1.0


def _normalize_session_id(session_id: int | None) -> int | None:
    """Reject bools and negative session identifiers."""

    if session_id is None or isinstance(session_id, bool):
        return None
    if session_id < 0:
        return None
    return session_id


def _claim_requires_confirmation(claim_name: str, confidence: float) -> bool:
    """Return True when a claim should be treated as verify-before-use."""

    threshold = (
        _DOA_REQUIRES_CONFIRMATION_CONFIDENCE
        if claim_name == "azimuth_deg"
        else _REQUIRES_CONFIRMATION_CONFIDENCE
    )
    return confidence < threshold


def _doa_requires_confirmation(direction_confidence: float | None) -> bool:
    """Return True when DoA-derived claims should not be treated as action-ready."""

    if direction_confidence is None:
        return True
    return direction_confidence < _DOA_REQUIRES_CONFIRMATION_CONFIDENCE


def _ambient_requires_confirmation(value: bool) -> bool:
    """Return confirmation policy for ambient-activity heuristics."""

    confidence = (
        _AMBIENT_ACTIVITY_TRUE_CONFIDENCE
        if value
        else _AMBIENT_ACTIVITY_FALSE_CONFIDENCE
    )
    return _claim_requires_confirmation("non_speech_audio_likely", confidence)


def _confidence_band(confidence: float) -> str:
    """Bucket one normalized confidence for cheap analytics and routing."""

    if confidence >= 0.9:
        return "very_high"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.6:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "none"


def _confirmation_reason(claim_name: str, confidence: float) -> str:
    """Emit a stable reason string for verify-before-use claims."""

    if claim_name == "azimuth_deg":
        if confidence <= 0.0:
            return "direction_confidence_unavailable"
        return "direction_confidence_below_threshold"
    if claim_name in {
        "beam_activity",
        "speech_overlap_likely",
        "barge_in_detected",
        "non_speech_audio_likely",
        "background_media_likely",
    }:
        return "derived_or_heuristic_confidence_below_threshold"
    return "confidence_below_threshold"


__all__ = [
    "build_respeaker_claim_payloads",
    "build_respeaker_signal_claim_contract",
    "claim_contract_to_payload",
]