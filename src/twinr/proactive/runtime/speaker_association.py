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

        payload = _coerce_mapping(value)
        if not payload:
            return None
        return cls(
            observed_at=_coerce_optional_float(payload.get("observed_at")),
            state=_normalize_text(payload.get("state")) or "unknown",
            associated=_coerce_optional_bool(payload.get("associated")) is True,
            target_id=_normalize_text(payload.get("target_id")) or None,
            confidence=_coerce_optional_ratio(payload.get("confidence")),
            camera_person_count=_coerce_optional_int(payload.get("camera_person_count")),
            direction_confidence=_coerce_optional_ratio(payload.get("direction_confidence")),
            azimuth_deg=_coerce_optional_int(payload.get("azimuth_deg")),
            primary_person_zone=_normalize_text(payload.get("primary_person_zone")) or None,
        )


def derive_respeaker_speaker_association(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object],
) -> ReSpeakerSpeakerAssociationSnapshot:
    """Return one conservative speaker-association snapshot from live facts."""

    camera = _coerce_mapping(live_facts.get("camera"))
    respeaker = _coerce_mapping(live_facts.get("respeaker"))
    audio_policy = _coerce_mapping(live_facts.get("audio_policy"))

    person_visible = _coerce_optional_bool(camera.get("person_visible")) is True
    person_count = _coerce_optional_int(camera.get("person_count"))
    person_count_unknown = _coerce_optional_bool(camera.get("person_count_unknown")) is True
    primary_person_zone = _normalize_text(camera.get("primary_person_zone")) or None
    primary_person_center_x = _coerce_optional_ratio(camera.get("primary_person_center_x"))
    direction_confidence = _coerce_optional_ratio(respeaker.get("direction_confidence"))
    azimuth_deg = _coerce_optional_int(respeaker.get("azimuth_deg"))
    speaker_direction_stable = _coerce_optional_bool(audio_policy.get("speaker_direction_stable"))

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
    confidence = _mean_confidence((direction_confidence, visual_anchor_confidence))
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

    explicit_attention = _coerce_optional_ratio(camera.get("visual_attention_score"))
    if explicit_attention is not None:
        return max(0.6, explicit_attention)
    if _coerce_optional_bool(camera.get("engaged_with_device")) is True:
        return 0.88
    if _coerce_optional_bool(camera.get("looking_toward_device")) is True:
        return 0.84
    if _coerce_optional_bool(camera.get("person_near_device")) is True:
        return 0.78
    return 0.68


def _mean_confidence(values: tuple[float | None, ...]) -> float | None:
    """Return the arithmetic mean of available confidence values."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def _normalize_text(value: object | None) -> str:
    """Collapse one optional value into compact single-line text."""

    return " ".join(str(value or "").split()).strip()


def _coerce_mapping(value: object | None) -> dict[str, object]:
    """Coerce one optional mapping into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return dict(value or {})
    except (TypeError, ValueError):
        return {}


def _coerce_optional_bool(value: object | None) -> bool | None:
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


def _coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object | None) -> float | None:
    """Parse one optional finite float value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _coerce_optional_ratio(value: object | None) -> float | None:
    """Parse one optional ratio in ``[0.0, 1.0]``."""

    numeric = _coerce_optional_float(value)
    if numeric is None:
        return None
    return max(0.0, min(1.0, numeric))


__all__ = [
    "ReSpeakerSpeakerAssociationSnapshot",
    "derive_respeaker_speaker_association",
]
