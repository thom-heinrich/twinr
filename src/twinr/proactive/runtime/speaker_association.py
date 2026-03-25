"""Fuse ReSpeaker and camera facts into a conservative speaker association.

Twinr does not treat the XVF3800 as an identity sensor. This module therefore
only associates current speech with the single primary visible person when the
room context is simple enough and both the audio and camera anchors are
strong. Multi-person scenes or weak direction hints fail closed to explicit
ambiguous states.
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


_DIRECTION_MIN_CONFIDENCE = 0.75
_ASSOCIATION_MIN_CONFIDENCE = 0.72


@dataclass(frozen=True, slots=True)
class ReSpeakerSpeakerAssociationSnapshot:
    """Describe the best current audio-to-camera speaker association."""

    observed_at: float | None = None
    state: str = "unknown"
    associated: bool = False
    target_id: str | None = None
    confidence: float | None = None
    camera_person_count: int | None = None
    direction_confidence: float | None = None
    azimuth_deg: int | None = None
    primary_person_zone: str | None = None

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the snapshot into automation-friendly facts."""

        return {
            "observed_at": self.observed_at,
            "state": self.state,
            "associated": self.associated,
            "target_id": self.target_id,
            "confidence": self.confidence,
            "camera_person_count": self.camera_person_count,
            "direction_confidence": self.direction_confidence,
            "azimuth_deg": self.azimuth_deg,
            "primary_person_zone": self.primary_person_zone,
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the snapshot into flat event fields."""

        return {
            "speaker_association_state": self.state,
            "speaker_association_associated": self.associated,
            "speaker_association_target_id": self.target_id,
            "speaker_association_confidence": self.confidence,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "ReSpeakerSpeakerAssociationSnapshot | None":
        """Parse a serialized speaker-association fact payload."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        return cls(
            observed_at=coerce_optional_float(payload.get("observed_at")),
            state=normalize_text(payload.get("state")) or "unknown",
            associated=coerce_optional_bool(payload.get("associated")) is True,
            target_id=normalize_text(payload.get("target_id")) or None,
            confidence=coerce_optional_ratio(payload.get("confidence")),
            camera_person_count=coerce_optional_int(payload.get("camera_person_count")),
            direction_confidence=coerce_optional_ratio(payload.get("direction_confidence")),
            azimuth_deg=coerce_optional_int(payload.get("azimuth_deg")),
            primary_person_zone=normalize_text(payload.get("primary_person_zone")) or None,
        )


def derive_respeaker_speaker_association(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object],
) -> ReSpeakerSpeakerAssociationSnapshot:
    """Return one conservative speaker-association snapshot from live facts."""

    camera = coerce_mapping(live_facts.get("camera"))
    respeaker = coerce_mapping(live_facts.get("respeaker"))
    audio_policy = coerce_mapping(live_facts.get("audio_policy"))

    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    person_count = coerce_optional_int(camera.get("person_count"))
    person_count_unknown = coerce_optional_bool(camera.get("person_count_unknown")) is True
    primary_person_zone = normalize_text(camera.get("primary_person_zone")) or None
    primary_person_center_x = coerce_optional_ratio(camera.get("primary_person_center_x"))
    direction_confidence = coerce_optional_ratio(respeaker.get("direction_confidence"))
    azimuth_deg = coerce_optional_int(respeaker.get("azimuth_deg"))
    speaker_direction_stable = coerce_optional_bool(audio_policy.get("speaker_direction_stable"))

    if not person_visible:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="no_visible_person",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )
    if person_count_unknown:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="camera_person_count_unknown",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )
    if person_count is not None and person_count > 1:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="multi_person_context",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )
    if azimuth_deg is None or direction_confidence is None:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="audio_direction_unavailable",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )
    if speaker_direction_stable is not True or direction_confidence < _DIRECTION_MIN_CONFIDENCE:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="audio_direction_unstable",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )
    if primary_person_zone in {"", "unknown"} and primary_person_center_x is None:
        return ReSpeakerSpeakerAssociationSnapshot(
            observed_at=observed_at,
            state="no_camera_anchor",
            camera_person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
        )

    visual_anchor_confidence = _visual_anchor_confidence(camera)
    confidence = mean_confidence((direction_confidence, visual_anchor_confidence))
    associated = confidence is not None and confidence >= _ASSOCIATION_MIN_CONFIDENCE
    return ReSpeakerSpeakerAssociationSnapshot(
        observed_at=observed_at,
        state="primary_visible_person_associated" if associated else "primary_visible_person_low_confidence",
        associated=associated,
        target_id="primary_visible_person" if associated else None,
        confidence=confidence,
        camera_person_count=person_count,
        direction_confidence=direction_confidence,
        azimuth_deg=azimuth_deg,
        primary_person_zone=primary_person_zone,
    )


def _visual_anchor_confidence(camera: Mapping[str, object]) -> float:
    """Return one conservative confidence score for the primary camera anchor."""

    explicit_attention = coerce_optional_ratio(camera.get("visual_attention_score"))
    if explicit_attention is not None:
        return max(0.6, explicit_attention)
    if coerce_optional_bool(camera.get("engaged_with_device")) is True:
        return 0.88
    if coerce_optional_bool(camera.get("looking_toward_device")) is True:
        return 0.84
    if coerce_optional_bool(camera.get("person_near_device")) is True:
        return 0.78
    return 0.68


__all__ = [
    "ReSpeakerSpeakerAssociationSnapshot",
    "derive_respeaker_speaker_association",
]
