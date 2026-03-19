"""Build and serialize confidence-bearing ReSpeaker claim metadata.

This module keeps the per-field provenance contract for XVF3800-derived
runtime facts separate from signal acquisition and policy orchestration. It
turns one typed signal snapshot into inspectable claim metadata that later
runtime and memory layers can export without re-inventing source, confidence,
or session semantics.
"""

from __future__ import annotations

from collections.abc import Mapping

from twinr.hardware.respeaker.models import (
    FrozenDict,
    ReSpeakerClaimMetadata,
    ReSpeakerMemoryClass,
    ReSpeakerSignalSnapshot,
)


_DIRECT_HARDWARE_CONFIDENCE = 0.99
_DIRECT_SIGNAL_CONFIDENCE = 0.76
_DERIVED_SIGNAL_CONFIDENCE = 0.72
_BEAM_ACTIVITY_CONFIDENCE = 0.68
_AMBIENT_ACTIVITY_TRUE_CONFIDENCE = 0.82
_AMBIENT_ACTIVITY_FALSE_CONFIDENCE = 0.68
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


def build_respeaker_signal_claim_contract(
    snapshot: ReSpeakerSignalSnapshot,
) -> FrozenDict[ReSpeakerClaimMetadata]:
    """Build the base per-claim metadata mapping for one signal snapshot."""

    claims: dict[str, ReSpeakerClaimMetadata] = {}
    base_kwargs = {
        "captured_at": snapshot.captured_at,
        "source": snapshot.source,
        "source_type": snapshot.source_type,
        "sensor_window_ms": snapshot.sensor_window_ms,
        "memory_class": ReSpeakerMemoryClass.EPHEMERAL_STATE,
        "requires_confirmation": False,
    }

    _add_claim(
        claims,
        "device_runtime_mode",
        snapshot.device_runtime_mode,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "host_control_ready",
        snapshot.host_control_ready,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "transport_reason",
        snapshot.transport_reason,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "firmware_version",
        snapshot.firmware_version,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "assistant_output_active",
        snapshot.assistant_output_active,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "mute_active",
        snapshot.mute_active,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "gpo_logic_levels",
        snapshot.gpo_logic_levels,
        confidence=_DIRECT_HARDWARE_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "speech_detected",
        snapshot.speech_detected,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "room_quiet",
        snapshot.room_quiet,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "recent_speech_age_s",
        snapshot.recent_speech_age_s,
        confidence=_DIRECT_SIGNAL_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "azimuth_deg",
        snapshot.azimuth_deg,
        confidence=max(0.4, snapshot.direction_confidence or 0.0),
        **base_kwargs,
    )
    _add_claim(
        claims,
        "direction_confidence",
        snapshot.direction_confidence,
        confidence=snapshot.direction_confidence or _DERIVED_SIGNAL_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "beam_activity",
        snapshot.beam_activity,
        confidence=_BEAM_ACTIVITY_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "speech_overlap_likely",
        snapshot.speech_overlap_likely,
        confidence=_DERIVED_SIGNAL_CONFIDENCE,
        **base_kwargs,
    )
    _add_claim(
        claims,
        "barge_in_detected",
        snapshot.barge_in_detected,
        confidence=_DERIVED_SIGNAL_CONFIDENCE,
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
    payloads = claim_contract_to_payload(
        signal_snapshot.claim_contract,
        session_id=session_id,
    )
    base_confidence_true = _AMBIENT_ACTIVITY_TRUE_CONFIDENCE
    base_confidence_false = _AMBIENT_ACTIVITY_FALSE_CONFIDENCE
    if non_speech_audio_likely is not None:
        payloads["non_speech_audio_likely"] = ReSpeakerClaimMetadata(
            captured_at=signal_snapshot.captured_at,
            source=signal_snapshot.source,
            source_type=signal_snapshot.source_type,
            confidence=base_confidence_true if non_speech_audio_likely else base_confidence_false,
            sensor_window_ms=signal_snapshot.sensor_window_ms,
            memory_class=ReSpeakerMemoryClass.EPHEMERAL_STATE,
            session_id=session_id if "non_speech_audio_likely" in _SESSION_SCOPED_CLAIMS else None,
            requires_confirmation=False,
        ).to_payload()
    if background_media_likely is not None:
        payloads["background_media_likely"] = ReSpeakerClaimMetadata(
            captured_at=signal_snapshot.captured_at,
            source=signal_snapshot.source,
            source_type=signal_snapshot.source_type,
            confidence=base_confidence_true if background_media_likely else base_confidence_false,
            sensor_window_ms=signal_snapshot.sensor_window_ms,
            memory_class=ReSpeakerMemoryClass.EPHEMERAL_STATE,
            session_id=session_id if "background_media_likely" in _SESSION_SCOPED_CLAIMS else None,
            requires_confirmation=False,
        ).to_payload()
    return payloads


def claim_contract_to_payload(
    claim_contract: Mapping[str, ReSpeakerClaimMetadata],
    *,
    session_id: int | None = None,
) -> dict[str, dict[str, object]]:
    """Serialize one claim contract mapping into plain nested dictionaries."""

    payloads: dict[str, dict[str, object]] = {}
    for claim_name, metadata in claim_contract.items():
        claim_metadata = metadata
        if session_id is not None and claim_name in _SESSION_SCOPED_CLAIMS:
            claim_metadata = metadata.with_session_id(session_id)
        payloads[str(claim_name)] = claim_metadata.to_payload()
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

    if value is None:
        return
    claims[claim_name] = ReSpeakerClaimMetadata(
        captured_at=captured_at,
        source=source,
        source_type=source_type,
        confidence=confidence,
        sensor_window_ms=sensor_window_ms,
        memory_class=memory_class,
        requires_confirmation=requires_confirmation,
    )


__all__ = [
    "build_respeaker_claim_payloads",
    "build_respeaker_signal_claim_contract",
    "claim_contract_to_payload",
]
