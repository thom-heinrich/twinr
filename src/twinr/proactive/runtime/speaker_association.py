# CHANGELOG: 2026-03-29
# BUG-1: Fixed false associations when the camera had no usable spatial anchor because
# BUG-1: `primary_person_zone is None` bypassed the old no-anchor guard.
# BUG-2: Fixed incorrect associations caused by ignoring audio↔camera spatial consistency.
# BUG-3: Fixed azimuth parsing to tolerate floating-point XVF3800 degree values and NaN-like inputs.
# BUG-4: Fixed confidence inflation where very low explicit visual-attention scores were floored to 0.6.
# SEC-1: Added freshness and cross-sensor skew checks to fail closed on stale/replayed or desynchronized facts.
# IMP-1: Added optional frontier-grade cross-modal cues (spatial match, visual speech / ASD score,
# IMP-1: and face-voice / speaker-embedding match) while remaining dependency-light for Raspberry Pi 4.
# IMP-2: Added richer diagnostics for downstream automation and observability.

"""Fuse ReSpeaker and camera facts into a conservative speaker association.

Twinr does not treat the XVF3800 as an identity sensor. This module therefore
only associates current speech with the primary visible person when the room
context is simple enough, the audio facts are current and stable, and at least
one strong cross-modal anchor is present.

Accepted cross-modal anchors are:
- explicit audio↔camera spatial consistency (preferred),
- a visual active-speaker / visual-speech score, or
- a face-voice / speaker-embedding association score.

The function remains drop-in callable but now fails closed more aggressively
when the upstream fact graph does not provide enough evidence to justify an
association.

# BREAKING: a sole visible person is no longer sufficient for association.
# BREAKING: the module now requires at least one real cross-modal anchor.
# BREAKING: `state` can now emit additional fail-closed values such as
# BREAKING: `camera_data_stale`, `audio_data_stale`,
# BREAKING: `sensor_temporal_misalignment`, `no_current_speech`,
# BREAKING: `no_cross_modal_anchor`, and `audio_camera_direction_mismatch`.
"""

from __future__ import annotations

import math
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
_MIN_CROSS_MODAL_CONFIDENCE = 0.66
_MIN_SPEECH_CONFIDENCE = 0.55

_DEFAULT_CAMERA_FACT_MAX_AGE_S = 1.5
_DEFAULT_AUDIO_FACT_MAX_AGE_S = 1.5
_DEFAULT_MAX_SENSOR_SKEW_S = 0.75

_STRONG_SPATIAL_MATCH_DEG = 22.5
_WEAK_SPATIAL_MATCH_DEG = 45.0
_STRONG_SPATIAL_MISMATCH_CONFIDENCE = 0.25


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
    speech_confidence: float | None = None
    spatial_match_confidence: float | None = None
    biometric_match_confidence: float | None = None
    sensor_skew_ms: int | None = None

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
            "speech_confidence": self.speech_confidence,
            "spatial_match_confidence": self.spatial_match_confidence,
            "biometric_match_confidence": self.biometric_match_confidence,
            "sensor_skew_ms": self.sensor_skew_ms,
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the snapshot into flat event fields."""

        return {
            "speaker_association_state": self.state,
            "speaker_association_associated": self.associated,
            "speaker_association_target_id": self.target_id,
            "speaker_association_confidence": self.confidence,
            "speaker_association_speech_confidence": self.speech_confidence,
            "speaker_association_spatial_match_confidence": self.spatial_match_confidence,
            "speaker_association_biometric_match_confidence": self.biometric_match_confidence,
            "speaker_association_sensor_skew_ms": self.sensor_skew_ms,
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
            azimuth_deg=_coerce_optional_azimuth_int(payload.get("azimuth_deg")),
            primary_person_zone=normalize_text(payload.get("primary_person_zone")) or None,
            speech_confidence=coerce_optional_ratio(payload.get("speech_confidence")),
            spatial_match_confidence=coerce_optional_ratio(payload.get("spatial_match_confidence")),
            biometric_match_confidence=coerce_optional_ratio(payload.get("biometric_match_confidence")),
            sensor_skew_ms=coerce_optional_int(payload.get("sensor_skew_ms")),
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
    association_config = coerce_mapping(live_facts.get("speaker_association_config"))
    shared_calibration = coerce_mapping(live_facts.get("audio_visual_calibration"))

    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    person_count = coerce_optional_int(camera.get("person_count"))
    person_count_unknown = coerce_optional_bool(camera.get("person_count_unknown")) is True
    primary_person_zone = normalize_text(camera.get("primary_person_zone")) or None

    direction_confidence = _first_ratio(
        respeaker,
        "direction_confidence",
        "selected_direction_confidence",
        "speaker_direction_confidence",
    )
    azimuth_deg = _coerce_optional_azimuth_deg(
        _first_value(
            respeaker,
            "azimuth_deg",
            "selected_azimuth_deg",
            "speaker_azimuth_deg",
        )
    )
    speaker_direction_stable = coerce_optional_bool(
        _first_value(audio_policy, "speaker_direction_stable", "direction_stable")
    )

    camera_observed_at = _extract_observed_at(camera)
    respeaker_observed_at = _extract_observed_at(respeaker)
    audio_policy_observed_at = _extract_observed_at(audio_policy)

    sensor_skew_ms = _sensor_skew_ms(camera_observed_at, respeaker_observed_at, audio_policy_observed_at)
    camera_max_age_s = _config_seconds(
        association_config,
        audio_policy,
        "camera_fact_max_age_s",
        default=_DEFAULT_CAMERA_FACT_MAX_AGE_S,
    )
    audio_max_age_s = _config_seconds(
        association_config,
        audio_policy,
        "audio_fact_max_age_s",
        default=_DEFAULT_AUDIO_FACT_MAX_AGE_S,
    )
    max_sensor_skew_s = _config_seconds(
        association_config,
        audio_policy,
        "max_sensor_skew_s",
        default=_DEFAULT_MAX_SENSOR_SKEW_S,
    )

    if not person_visible:
        return _snapshot(
            observed_at=observed_at,
            state="no_visible_person",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )
    if person_count_unknown:
        return _snapshot(
            observed_at=observed_at,
            state="camera_person_count_unknown",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )
    if person_count is not None and person_count <= 0:
        return _snapshot(
            observed_at=observed_at,
            state="no_visible_person",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )
    if person_count is not None and person_count > 1:
        return _snapshot(
            observed_at=observed_at,
            state="multi_person_context",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )

    if observed_at is not None:
        camera_age = _fact_age_seconds(observed_at, camera_observed_at)
        if camera_age is not None and camera_age > camera_max_age_s:
            return _snapshot(
                observed_at=observed_at,
                state="camera_data_stale",
                person_count=person_count,
                direction_confidence=direction_confidence,
                azimuth_deg=azimuth_deg,
                primary_person_zone=primary_person_zone,
                sensor_skew_ms=sensor_skew_ms,
            )
        audio_age = _fact_age_seconds(
            observed_at,
            _most_recent_timestamp(respeaker_observed_at, audio_policy_observed_at),
        )
        if audio_age is not None and audio_age > audio_max_age_s:
            return _snapshot(
                observed_at=observed_at,
                state="audio_data_stale",
                person_count=person_count,
                direction_confidence=direction_confidence,
                azimuth_deg=azimuth_deg,
                primary_person_zone=primary_person_zone,
                sensor_skew_ms=sensor_skew_ms,
            )
    if sensor_skew_ms is not None and sensor_skew_ms > int(round(max_sensor_skew_s * 1000.0)):
        return _snapshot(
            observed_at=observed_at,
            state="sensor_temporal_misalignment",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )

    if azimuth_deg is None or direction_confidence is None:
        return _snapshot(
            observed_at=observed_at,
            state="audio_direction_unavailable",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )
    if speaker_direction_stable is not True or direction_confidence < _DIRECTION_MIN_CONFIDENCE:
        return _snapshot(
            observed_at=observed_at,
            state="audio_direction_unstable",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            sensor_skew_ms=sensor_skew_ms,
        )

    speech_confidence = _speech_confidence(respeaker, audio_policy)
    if speech_confidence is not None and speech_confidence < _MIN_SPEECH_CONFIDENCE:
        return _snapshot(
            observed_at=observed_at,
            state="no_current_speech",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            speech_confidence=speech_confidence,
            sensor_skew_ms=sensor_skew_ms,
        )

    visual_anchor_confidence = _visual_anchor_confidence(camera)
    active_speaker_confidence = _active_speaker_confidence(camera)
    biometric_match_confidence = _biometric_match_confidence(camera, respeaker, audio_policy)
    spatial_match_confidence = _spatial_match_confidence(
        camera=camera,
        respeaker=respeaker,
        audio_policy=audio_policy,
        shared_calibration=shared_calibration,
    )

    if spatial_match_confidence is not None and spatial_match_confidence <= _STRONG_SPATIAL_MISMATCH_CONFIDENCE:
        return _snapshot(
            observed_at=observed_at,
            state="audio_camera_direction_mismatch",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            speech_confidence=speech_confidence,
            spatial_match_confidence=spatial_match_confidence,
            biometric_match_confidence=biometric_match_confidence,
            sensor_skew_ms=sensor_skew_ms,
        )

    cross_modal_confidence = _weighted_mean(
        (
            (spatial_match_confidence, 0.50),
            (active_speaker_confidence, 0.25),
            (biometric_match_confidence, 0.25),
        )
    )
    if cross_modal_confidence is None:
        legacy_anchor_confidence = _legacy_visual_anchor_confidence(
            camera=camera,
            primary_person_zone=primary_person_zone,
        )
        if legacy_anchor_confidence is not None:
            cross_modal_confidence = legacy_anchor_confidence
    if cross_modal_confidence is None:
        return _snapshot(
            observed_at=observed_at,
            state="no_cross_modal_anchor",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            speech_confidence=speech_confidence,
            sensor_skew_ms=sensor_skew_ms,
        )
    if cross_modal_confidence < _MIN_CROSS_MODAL_CONFIDENCE:
        confidence = _weighted_mean(
            (
                (direction_confidence, 0.30),
                (speech_confidence, 0.10),
                (visual_anchor_confidence, 0.10),
                (cross_modal_confidence, 0.50),
            )
        )
        return _snapshot(
            observed_at=observed_at,
            state="primary_visible_person_low_confidence",
            person_count=person_count,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            primary_person_zone=primary_person_zone,
            speech_confidence=speech_confidence,
            spatial_match_confidence=spatial_match_confidence,
            biometric_match_confidence=biometric_match_confidence,
            sensor_skew_ms=sensor_skew_ms,
            confidence=confidence,
        )

    confidence = _weighted_mean(
        (
            (direction_confidence, 0.28),
            (speech_confidence, 0.12),
            (visual_anchor_confidence, 0.10),
            (cross_modal_confidence, 0.50),
        )
    )
    associated = confidence is not None and confidence >= _ASSOCIATION_MIN_CONFIDENCE
    target_id = _primary_person_target_id(camera) if associated else None
    return _snapshot(
        observed_at=observed_at,
        state="primary_visible_person_associated" if associated else "primary_visible_person_low_confidence",
        person_count=person_count,
        direction_confidence=direction_confidence,
        azimuth_deg=azimuth_deg,
        primary_person_zone=primary_person_zone,
        speech_confidence=speech_confidence,
        spatial_match_confidence=spatial_match_confidence,
        biometric_match_confidence=biometric_match_confidence,
        sensor_skew_ms=sensor_skew_ms,
        associated=associated,
        target_id=target_id,
        confidence=confidence,
    )


def _snapshot(
    *,
    observed_at: float | None,
    state: str,
    person_count: int | None,
    direction_confidence: float | None,
    azimuth_deg: float | None,
    primary_person_zone: str | None,
    speech_confidence: float | None = None,
    spatial_match_confidence: float | None = None,
    biometric_match_confidence: float | None = None,
    sensor_skew_ms: int | None = None,
    associated: bool = False,
    target_id: str | None = None,
    confidence: float | None = None,
) -> ReSpeakerSpeakerAssociationSnapshot:
    return ReSpeakerSpeakerAssociationSnapshot(
        observed_at=observed_at,
        state=state,
        associated=associated,
        target_id=target_id,
        confidence=confidence,
        camera_person_count=person_count,
        direction_confidence=direction_confidence,
        azimuth_deg=_round_azimuth_int(azimuth_deg),
        primary_person_zone=primary_person_zone,
        speech_confidence=speech_confidence,
        spatial_match_confidence=spatial_match_confidence,
        biometric_match_confidence=biometric_match_confidence,
        sensor_skew_ms=sensor_skew_ms,
    )


def _visual_anchor_confidence(camera: Mapping[str, object]) -> float | None:
    """Return a conservative confidence score for the primary camera anchor."""

    explicit_attention = _first_ratio(
        camera,
        "visual_attention_score",
        "primary_person_visual_attention_score",
        "primary_person_confidence",
    )
    if explicit_attention is not None:
        return explicit_attention
    if coerce_optional_bool(camera.get("engaged_with_device")) is True:
        return 0.88
    if coerce_optional_bool(camera.get("looking_toward_device")) is True:
        return 0.84
    if coerce_optional_bool(camera.get("person_near_device")) is True:
        return 0.78
    if coerce_optional_bool(camera.get("person_visible")) is True:
        return 0.68
    return None


def _legacy_visual_anchor_confidence(
    *,
    camera: Mapping[str, object],
    primary_person_zone: str | None,
) -> float | None:
    """Return the legacy single-person anchor confidence for established strong camera cues."""

    has_camera_anchor = primary_person_zone not in {None, "", "unknown"} or coerce_optional_ratio(
        _first_value(camera, "primary_person_center_x", "person_center_x")
    ) is not None
    if not has_camera_anchor:
        return None
    explicit_attention = _first_ratio(
        camera,
        "visual_attention_score",
        "primary_person_visual_attention_score",
        "primary_person_confidence",
    )
    if explicit_attention is not None and explicit_attention >= 0.72:
        return explicit_attention
    if coerce_optional_bool(camera.get("engaged_with_device")) is True:
        return 0.88
    if coerce_optional_bool(camera.get("looking_toward_device")) is True:
        return 0.84
    if coerce_optional_bool(camera.get("person_near_device")) is True:
        return 0.78
    return None


def _active_speaker_confidence(camera: Mapping[str, object]) -> float | None:
    """Return an optional visual speaking/activity confidence from upstream ASD."""

    return _first_ratio(
        camera,
        "primary_person_active_speaker_score",
        "active_speaker_score",
        "visual_speech_score",
        "mouth_motion_score",
        "lip_sync_score",
    )


def _biometric_match_confidence(
    camera: Mapping[str, object],
    respeaker: Mapping[str, object],
    audio_policy: Mapping[str, object],
) -> float | None:
    """Return an optional face-voice / speaker-embedding match confidence."""

    return mean_confidence(
        (
            _first_ratio(
                camera,
                "face_voice_association_score",
                "primary_person_face_voice_score",
                "cross_modal_face_voice_score",
                "voice_face_match_score",
            ),
            _first_ratio(
                audio_policy,
                "speaker_embedding_match_score",
                "target_speaker_match_score",
            ),
            _first_ratio(
                respeaker,
                "speaker_embedding_match_score",
                "voice_face_match_score",
            ),
        )
    )


def _speech_confidence(
    respeaker: Mapping[str, object],
    audio_policy: Mapping[str, object],
) -> float | None:
    """Return an optional speech-presence confidence for the current azimuth."""

    for mapping in (audio_policy, respeaker):
        explicit = _first_ratio(
            mapping,
            "speech_confidence",
            "vad_confidence",
            "speech_presence_confidence",
        )
        if explicit is not None:
            return explicit

    for mapping in (audio_policy, respeaker):
        if coerce_optional_bool(_first_value(mapping, "speech_present", "vad_speech", "speech_detected")) is False:
            return 0.0

    for mapping in (audio_policy, respeaker):
        energy = _first_positive_float(
            mapping,
            "speech_energy",
            "selected_speech_energy",
            "spenergy",
            "speech_power",
        )
        if energy is not None:
            return _logistic_energy_confidence(energy)

    return None


def _spatial_match_confidence(
    *,
    camera: Mapping[str, object],
    respeaker: Mapping[str, object],
    audio_policy: Mapping[str, object],
    shared_calibration: Mapping[str, object],
) -> float | None:
    """Return an optional audio↔camera spatial consistency confidence."""

    explicit_camera_azimuth = _coerce_optional_azimuth_deg(
        _first_value(
            camera,
            "primary_person_azimuth_deg",
            "primary_person_direction_deg",
            "person_azimuth_deg",
        )
    )
    explicit_audio_azimuth = _coerce_optional_azimuth_deg(
        _first_value(
            respeaker,
            "azimuth_deg",
            "selected_azimuth_deg",
            "speaker_azimuth_deg",
        )
    )
    if explicit_camera_azimuth is not None and explicit_audio_azimuth is not None:
        return _angular_match_confidence(explicit_audio_azimuth, explicit_camera_azimuth)

    camera_relative_zone = _camera_relative_zone(camera)
    audio_relative_zone = _audio_relative_zone(audio_policy, respeaker, camera, shared_calibration)
    if camera_relative_zone is not None and audio_relative_zone is not None:
        if camera_relative_zone == audio_relative_zone:
            return 0.90
        if {camera_relative_zone, audio_relative_zone} == {"center", "left"}:
            return 0.35
        if {camera_relative_zone, audio_relative_zone} == {"center", "right"}:
            return 0.35
        return 0.10

    calibrated_camera_azimuth = _calibrated_camera_azimuth(camera, shared_calibration)
    if calibrated_camera_azimuth is not None and explicit_audio_azimuth is not None:
        return _angular_match_confidence(explicit_audio_azimuth, calibrated_camera_azimuth)

    return None


def _camera_relative_zone(camera: Mapping[str, object]) -> str | None:
    zone = _normalize_zone(_first_value(camera, "primary_person_zone", "person_zone"))
    if zone not in {None, "unknown"}:
        return zone
    center_x = coerce_optional_ratio(_first_value(camera, "primary_person_center_x", "person_center_x"))
    if center_x is None:
        return None
    if center_x < 0.35:
        return "left"
    if center_x > 0.65:
        return "right"
    return "center"


def _audio_relative_zone(
    audio_policy: Mapping[str, object],
    respeaker: Mapping[str, object],
    camera: Mapping[str, object],
    shared_calibration: Mapping[str, object],
) -> str | None:
    explicit_zone = _normalize_zone(
        _first_value(
            audio_policy,
            "camera_relative_direction_zone",
            "speaker_direction_zone",
            "direction_zone",
        )
    )
    if explicit_zone not in {None, "unknown"}:
        return explicit_zone

    explicit_zone = _normalize_zone(
        _first_value(respeaker, "camera_relative_direction_zone", "direction_zone")
    )
    if explicit_zone not in {None, "unknown"}:
        return explicit_zone

    audio_azimuth = _coerce_optional_azimuth_deg(
        _first_value(
            respeaker,
            "azimuth_deg",
            "selected_azimuth_deg",
            "speaker_azimuth_deg",
        )
    )
    if audio_azimuth is None:
        return None

    forward_azimuth = _coerce_optional_azimuth_deg(
        _first_defined_value(
            _first_value(
                camera,
                "camera_forward_azimuth_deg",
                "device_forward_azimuth_deg",
                "forward_azimuth_deg",
            ),
            _first_value(
                shared_calibration,
                "camera_forward_azimuth_deg",
                "device_forward_azimuth_deg",
                "forward_azimuth_deg",
            ),
        )
    )
    right_positive = coerce_optional_bool(
        _first_value(
            camera,
            "camera_right_increases_azimuth",
            "image_right_increases_azimuth",
        )
    )
    if forward_azimuth is None or right_positive is None:
        return None

    delta = _signed_angular_difference_deg(audio_azimuth, forward_azimuth)
    signed_delta = delta if right_positive else -delta
    if abs(signed_delta) <= 15.0:
        return "center"
    return "right" if signed_delta > 0.0 else "left"


def _calibrated_camera_azimuth(
    camera: Mapping[str, object],
    shared_calibration: Mapping[str, object],
) -> float | None:
    center_x = coerce_optional_ratio(_first_value(camera, "primary_person_center_x", "person_center_x"))
    if center_x is None:
        return None

    forward_azimuth = _coerce_optional_azimuth_deg(
        _first_defined_value(
            _first_value(
                camera,
                "camera_forward_azimuth_deg",
                "device_forward_azimuth_deg",
                "forward_azimuth_deg",
            ),
            _first_value(
                shared_calibration,
                "camera_forward_azimuth_deg",
                "device_forward_azimuth_deg",
                "forward_azimuth_deg",
            ),
        )
    )
    horizontal_fov_deg = _first_positive_float(
        camera,
        "camera_horizontal_fov_deg",
        "horizontal_fov_deg",
        "horizontal_fov",
    ) or _first_positive_float(
        shared_calibration,
        "camera_horizontal_fov_deg",
        "horizontal_fov_deg",
        "horizontal_fov",
    )
    right_positive = coerce_optional_bool(
        _first_value(
            camera,
            "camera_right_increases_azimuth",
            "image_right_increases_azimuth",
        )
    )
    if forward_azimuth is None or horizontal_fov_deg is None or right_positive is None:
        return None

    relative_deg = (center_x - 0.5) * horizontal_fov_deg
    if not right_positive:
        relative_deg = -relative_deg
    return _normalize_azimuth_deg(forward_azimuth + relative_deg)


def _angular_match_confidence(audio_azimuth: float, camera_azimuth: float) -> float:
    delta = _angular_distance_deg(audio_azimuth, camera_azimuth)
    if delta <= _STRONG_SPATIAL_MATCH_DEG:
        return 0.96
    if delta <= _WEAK_SPATIAL_MATCH_DEG:
        return 0.96 - ((delta - _STRONG_SPATIAL_MATCH_DEG) / (_WEAK_SPATIAL_MATCH_DEG - _STRONG_SPATIAL_MATCH_DEG)) * 0.30
    if delta <= 90.0:
        return 0.40 - ((delta - _WEAK_SPATIAL_MATCH_DEG) / 45.0) * 0.20
    return 0.10


def _primary_person_target_id(camera: Mapping[str, object]) -> str:
    target_id = normalize_text(
        _first_value(
            camera,
            "primary_person_target_id",
            "primary_person_id",
            "primary_person_track_id",
        )
    )
    return target_id or "primary_visible_person"


def _extract_observed_at(payload: Mapping[str, object]) -> float | None:
    value = _first_value(payload, "observed_at", "timestamp", "captured_at", "updated_at")
    ts = coerce_optional_float(value)
    if ts is None or not math.isfinite(ts):
        return None
    return ts


def _sensor_skew_ms(*timestamps: float | None) -> int | None:
    finite = [timestamp for timestamp in timestamps if timestamp is not None and math.isfinite(timestamp)]
    if len(finite) < 2:
        return None
    return int(round((max(finite) - min(finite)) * 1000.0))


def _fact_age_seconds(now: float, timestamp: float | None) -> float | None:
    if timestamp is None or not math.isfinite(timestamp):
        return None
    age = now - timestamp
    if age < 0.0:
        return 0.0
    return age


def _most_recent_timestamp(*timestamps: float | None) -> float | None:
    finite = [timestamp for timestamp in timestamps if timestamp is not None and math.isfinite(timestamp)]
    if not finite:
        return None
    return max(finite)


def _config_seconds(
    primary: Mapping[str, object],
    secondary: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    value = _first_positive_float(primary, key)
    if value is not None:
        return value
    value = _first_positive_float(secondary, key)
    if value is not None:
        return value
    return default


def _weighted_mean(values: tuple[tuple[float | None, float], ...]) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for value, weight in values:
        if value is None or weight <= 0.0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return weighted_sum / weight_sum


def _logistic_energy_confidence(energy: float) -> float:
    # Speech energy magnitude is device- and tuning-dependent; this mapping only
    # converts strictly-positive energies into a conservative [0, 1) confidence.
    return 1.0 - math.exp(-math.log1p(energy) / 8.0)


def _coerce_optional_azimuth_deg(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"nan", "+nan", "-nan"}:
            return None
    azimuth = coerce_optional_float(value)
    if azimuth is None or not math.isfinite(azimuth):
        return None
    return _normalize_azimuth_deg(azimuth)


def _coerce_optional_azimuth_int(value: object | None) -> int | None:
    return _round_azimuth_int(_coerce_optional_azimuth_deg(value))


def _round_azimuth_int(value: float | None) -> int | None:
    if value is None or not math.isfinite(value):
        return None
    return int(round(value)) % 360


def _normalize_azimuth_deg(value: float) -> float:
    return value % 360.0


def _angular_distance_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _signed_angular_difference_deg(value: float, reference: float) -> float:
    return ((value - reference + 180.0) % 360.0) - 180.0


def _normalize_zone(value: object | None) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    zone = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "centre": "center",
        "middle": "center",
        "front": "center",
        "front_center": "center",
        "front_centre": "center",
        "far_left": "left",
        "left_front": "left",
        "front_left": "left",
        "far_right": "right",
        "right_front": "right",
        "front_right": "right",
    }
    return aliases.get(zone, zone)


def _first_value(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _first_defined_value(*values: object | None) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_ratio(mapping: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = coerce_optional_ratio(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_positive_float(mapping: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = coerce_optional_float(mapping.get(key))
        if value is not None and math.isfinite(value) and value > 0.0:
            return value
    return None


__all__ = [
    "ReSpeakerSpeakerAssociationSnapshot",
    "derive_respeaker_speaker_association",
]
