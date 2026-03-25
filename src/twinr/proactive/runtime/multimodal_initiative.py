"""Derive conservative multimodal initiative state from camera and ReSpeaker.

This layer does not replace the social trigger engine or the proactive
delivery policy. It provides one explicit, confidence-bearing gate that later
proactive behaviors can consult before speaking into an ambiguous room.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from twinr.proactive.runtime.claim_metadata import (
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


_INITIATIVE_READY_MIN_CONFIDENCE = 0.72


@dataclass(frozen=True, slots=True)
class ReSpeakerMultimodalInitiativeSnapshot:
    """Describe whether current room context is clear enough for initiative."""

    observed_at: float | None = None
    ready: bool = False
    confidence: float | None = None
    block_reason: str | None = None
    recommended_channel: str = "display"
    speaker_association_state: str | None = None
    speaker_association_confidence: float | None = None

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the snapshot into automation-friendly facts."""

        return {
            "observed_at": self.observed_at,
            "ready": self.ready,
            "confidence": self.confidence,
            "block_reason": self.block_reason,
            "recommended_channel": self.recommended_channel,
            "speaker_association_state": self.speaker_association_state,
            "speaker_association_confidence": self.speaker_association_confidence,
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the snapshot into flat event fields."""

        return {
            "multimodal_initiative_ready": self.ready,
            "multimodal_initiative_confidence": self.confidence,
            "multimodal_initiative_block_reason": self.block_reason,
            "multimodal_initiative_recommended_channel": self.recommended_channel,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "ReSpeakerMultimodalInitiativeSnapshot | None":
        """Parse one serialized multimodal-initiative fact payload."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        return cls(
            observed_at=coerce_optional_float(payload.get("observed_at")),
            ready=coerce_optional_bool(payload.get("ready")) is True,
            confidence=coerce_optional_ratio(payload.get("confidence")),
            block_reason=normalize_text(payload.get("block_reason")) or None,
            recommended_channel=_normalize_channel(payload.get("recommended_channel")),
            speaker_association_state=normalize_text(payload.get("speaker_association_state")) or None,
            speaker_association_confidence=coerce_optional_ratio(payload.get("speaker_association_confidence")),
        )


def derive_respeaker_multimodal_initiative(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object],
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
) -> ReSpeakerMultimodalInitiativeSnapshot:
    """Return one conservative multimodal initiative snapshot."""

    camera = coerce_mapping(live_facts.get("camera"))
    audio_policy = coerce_mapping(live_facts.get("audio_policy"))
    association = speaker_association or derive_respeaker_speaker_association(
        observed_at=observed_at,
        live_facts=live_facts,
    )

    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    person_count = coerce_optional_int(camera.get("person_count"))
    room_busy = coerce_optional_bool(audio_policy.get("room_busy_or_overlapping")) is True
    defer_reason = normalize_text(audio_policy.get("speech_delivery_defer_reason")) or None
    initiative_block_reason = normalize_text(audio_policy.get("initiative_block_reason")) or None

    if room_busy or (person_count is not None and person_count > 1):
        return ReSpeakerMultimodalInitiativeSnapshot(
            observed_at=observed_at,
            confidence=association.confidence,
            block_reason="multi_person_context",
            recommended_channel="display",
            speaker_association_state=association.state,
            speaker_association_confidence=association.confidence,
        )
    if not person_visible:
        return ReSpeakerMultimodalInitiativeSnapshot(
            observed_at=observed_at,
            confidence=association.confidence,
            block_reason="no_visible_person",
            recommended_channel="display",
            speaker_association_state=association.state,
            speaker_association_confidence=association.confidence,
        )
    if defer_reason:
        return ReSpeakerMultimodalInitiativeSnapshot(
            observed_at=observed_at,
            confidence=association.confidence,
            block_reason=defer_reason,
            recommended_channel="display",
            speaker_association_state=association.state,
            speaker_association_confidence=association.confidence,
        )
    if initiative_block_reason in {"mute_blocks_voice_capture", "respeaker_unavailable", "respeaker_unready"}:
        return ReSpeakerMultimodalInitiativeSnapshot(
            observed_at=observed_at,
            confidence=association.confidence,
            block_reason=initiative_block_reason,
            recommended_channel="display",
            speaker_association_state=association.state,
            speaker_association_confidence=association.confidence,
        )
    if not association.associated:
        return ReSpeakerMultimodalInitiativeSnapshot(
            observed_at=observed_at,
            confidence=association.confidence,
            block_reason="low_confidence_speaker_association",
            recommended_channel="display",
            speaker_association_state=association.state,
            speaker_association_confidence=association.confidence,
        )

    confidence = _initiative_confidence(camera=camera, audio_policy=audio_policy, association=association)
    ready = confidence is not None and confidence >= _INITIATIVE_READY_MIN_CONFIDENCE
    return ReSpeakerMultimodalInitiativeSnapshot(
        observed_at=observed_at,
        ready=ready,
        confidence=confidence,
        block_reason=(None if ready else "low_multimodal_initiative_confidence"),
        recommended_channel="speech" if ready else "display",
        speaker_association_state=association.state,
        speaker_association_confidence=association.confidence,
    )


def _initiative_confidence(
    *,
    camera: Mapping[str, object],
    audio_policy: Mapping[str, object],
    association: ReSpeakerSpeakerAssociationSnapshot,
) -> float | None:
    """Return one conservative multimodal initiative confidence score."""

    values: list[float] = []
    if association.confidence is not None:
        values.append(association.confidence)

    visual_attention = coerce_optional_ratio(camera.get("visual_attention_score"))
    if visual_attention is not None:
        values.append(max(0.6, visual_attention))
    elif coerce_optional_bool(camera.get("engaged_with_device")) is True:
        values.append(0.88)
    elif coerce_optional_bool(camera.get("looking_toward_device")) is True:
        values.append(0.84)
    else:
        values.append(0.66)

    if coerce_optional_bool(audio_policy.get("presence_audio_active")) is True:
        values.append(0.86)
    elif coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True:
        values.append(0.82)
    elif coerce_optional_bool(audio_policy.get("quiet_window_open")) is True:
        values.append(0.72)

    return mean_confidence(tuple(values))


def _normalize_channel(value: object | None) -> str:
    """Normalize one channel token into the small proactive vocabulary."""

    normalized = normalize_text(value).lower()
    if normalized not in {"speech", "display", "print"}:
        return "display"
    return normalized


__all__ = [
    "ReSpeakerMultimodalInitiativeSnapshot",
    "derive_respeaker_multimodal_initiative",
]
