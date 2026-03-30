"""Derive conservative multimodal initiative state from camera and ReSpeaker.

This layer does not replace the social trigger engine or the proactive
delivery policy. It provides one explicit, confidence-bearing gate that later
proactive behaviors can consult before speaking into an ambiguous room.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, log

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


# CHANGELOG: 2026-03-29
# BUG-1: Fixed false-positive readiness caused by flooring low visual_attention_score up to 0.60.
# BUG-2: Fixed blocked snapshots exporting speaker-association confidence as multimodal initiative confidence.
# BUG-3: Fixed ignored upstream initiative_block_reason values; any normalized upstream block now blocks conservatively.
# BUG-4: Fixed stale-sensor blind spot by rejecting stale camera/audio/association facts when timestamps are available.
# SEC-1: Canonicalized externally sourced block reasons to a bounded vocabulary before emitting automation/event payloads.
# IMP-1: Added freshness-aware, reliability-weighted multimodal fusion instead of a flat arithmetic mean over optimistic heuristics.
# IMP-2: Added optional temporal hysteresis via previous_snapshot/live_facts multimodal_initiative payload to reduce readiness flapping.
# IMP-3: Added support for richer 2026 timing/addressee signals when present (turn_yield_confidence, interruption_risk, track stability, audio-visual alignment).

_INITIATIVE_READY_MIN_CONFIDENCE = 0.72
_INITIATIVE_READY_HYSTERESIS_MIN_CONFIDENCE = 0.66

_CAMERA_SOFT_STALE_SECONDS = 1.5
_CAMERA_HARD_STALE_SECONDS = 3.0
_AUDIO_SOFT_STALE_SECONDS = 1.0
_AUDIO_HARD_STALE_SECONDS = 2.5
_ASSOCIATION_SOFT_STALE_SECONDS = 1.5
_ASSOCIATION_HARD_STALE_SECONDS = 3.0

_HIGH_INTERRUPTION_RISK_THRESHOLD = 0.78

_ALLOWED_BLOCK_REASONS = frozenset(
    {
        "audio_policy_blocked",
        "background_media_active",
        "camera_unavailable",
        "high_interrupt_risk",
        "low_confidence_speaker_association",
        "low_multimodal_initiative_confidence",
        "missing_camera_facts",
        "multi_person_context",
        "mute_blocks_voice_capture",
        "no_visible_person",
        "non_speech_audio_active",
        "person_visibility_unknown",
        "respeaker_unavailable",
        "respeaker_unready",
        "room_busy_or_overlapping",
        "speech_delivery_deferred",
        "stale_audio_policy_facts",
        "stale_camera_facts",
        "stale_speaker_association",
        "upstream_policy_block",
    }
)

_BLOCK_REASON_ALIASES = {
    "audio_policy_block": "audio_policy_blocked",
    "audio_policy_blocked": "audio_policy_blocked",
    "background_media_active": "background_media_active",
    "defer": "speech_delivery_deferred",
    "deferred": "speech_delivery_deferred",
    "display_only": "speech_delivery_deferred",
    "initiative_blocked": "upstream_policy_block",
    "microphone_muted": "mute_blocks_voice_capture",
    "multi_person_context": "multi_person_context",
    "multiple_people_visible": "multi_person_context",
    "mute_blocks_voice_capture": "mute_blocks_voice_capture",
    "muted": "mute_blocks_voice_capture",
    "no_visible_person": "no_visible_person",
    "non_speech_audio_active": "non_speech_audio_active",
    "overlapping_speech": "room_busy_or_overlapping",
    "person_visibility_unknown": "person_visibility_unknown",
    "respeaker_unavailable": "respeaker_unavailable",
    "respeaker_unready": "respeaker_unready",
    "room_busy": "room_busy_or_overlapping",
    "room_busy_or_overlapping": "room_busy_or_overlapping",
    "speaker_association_low_confidence": "low_confidence_speaker_association",
    "speech_delivery_defer_reason": "speech_delivery_deferred",
    "speech_delivery_deferred": "speech_delivery_deferred",
    "upstream_policy_block": "upstream_policy_block",
}


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
            block_reason=_normalize_block_reason(payload.get("block_reason")),
            recommended_channel=_normalize_channel(payload.get("recommended_channel")),
            speaker_association_state=normalize_text(payload.get("speaker_association_state")) or None,
            speaker_association_confidence=coerce_optional_ratio(payload.get("speaker_association_confidence")),
        )


def derive_respeaker_multimodal_initiative(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object],
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
    previous_snapshot: ReSpeakerMultimodalInitiativeSnapshot | None = None,
) -> ReSpeakerMultimodalInitiativeSnapshot:
    """Return one conservative multimodal initiative snapshot.

    The function remains drop-in compatible with the older call pattern.
    When available, it additionally consumes richer timing and freshness
    signals from ``live_facts`` to behave more like a 2026 uncertainty-aware,
    mixed-initiative gate.
    """

    camera = coerce_mapping(live_facts.get("camera"))
    audio_policy = coerce_mapping(live_facts.get("audio_policy"))
    association = speaker_association or derive_respeaker_speaker_association(
        observed_at=observed_at,
        live_facts=live_facts,
    )
    previous = previous_snapshot or _coerce_previous_snapshot(live_facts)

    if not camera:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="missing_camera_facts",
            recommended_channel="display",
            association=association,
        )

    camera_age = _sensor_age_seconds(
        observed_at=observed_at,
        sensor_mapping=camera,
        fallback_timestamp=None,
    )
    if _is_stale(camera_age, _CAMERA_HARD_STALE_SECONDS):
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="stale_camera_facts",
            recommended_channel="display",
            association=association,
        )

    audio_age = _sensor_age_seconds(
        observed_at=observed_at,
        sensor_mapping=audio_policy,
        fallback_timestamp=None,
    )
    if _is_stale(audio_age, _AUDIO_HARD_STALE_SECONDS):
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="stale_audio_policy_facts",
            recommended_channel="display",
            association=association,
        )

    association_age = _sensor_age_seconds(
        observed_at=observed_at,
        sensor_mapping=None,
        fallback_timestamp=coerce_optional_float(getattr(association, "observed_at", None)),
    )
    if _is_stale(association_age, _ASSOCIATION_HARD_STALE_SECONDS):
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="stale_speaker_association",
            recommended_channel="display",
            association=association,
        )

    person_visible_value = _infer_person_visible(camera)
    person_visible = person_visible_value is True
    person_count = coerce_optional_int(camera.get("person_count"))
    single_addressee_visible = _coerce_single_addressee_visible(camera)

    room_busy = coerce_optional_bool(audio_policy.get("room_busy_or_overlapping")) is True
    defer_reason = _normalize_policy_reason(
        audio_policy.get("speech_delivery_defer_reason"),
        default_reason="speech_delivery_deferred",
    )
    initiative_block_reason = _normalize_policy_reason(
        audio_policy.get("initiative_block_reason"),
        default_reason="upstream_policy_block",
    )
    interruption_risk = _max_ratio(
        audio_policy.get("interruption_risk"),
        audio_policy.get("speech_overlap_risk"),
        audio_policy.get("barge_in_risk"),
    )

    if room_busy:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="room_busy_or_overlapping",
            recommended_channel="display",
            association=association,
        )
    if person_count is not None and person_count > 1 and not single_addressee_visible:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="multi_person_context",
            recommended_channel="display",
            association=association,
        )
    if person_visible_value is None:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="person_visibility_unknown",
            recommended_channel="display",
            association=association,
        )
    if not person_visible:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="no_visible_person",
            recommended_channel="display",
            association=association,
        )
    if defer_reason is not None:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason=defer_reason,
            recommended_channel="display",
            association=association,
        )
    if initiative_block_reason is not None:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason=initiative_block_reason,
            recommended_channel="display",
            association=association,
        )
    if interruption_risk is not None and interruption_risk >= _HIGH_INTERRUPTION_RISK_THRESHOLD:
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="high_interrupt_risk",
            recommended_channel="display",
            association=association,
        )
    if not getattr(association, "associated", False):
        return _blocked_snapshot(
            observed_at=observed_at,
            block_reason="low_confidence_speaker_association",
            recommended_channel="display",
            association=association,
        )

    confidence = _initiative_confidence(
        camera=camera,
        audio_policy=audio_policy,
        association=association,
        camera_age=camera_age,
        audio_age=audio_age,
        association_age=association_age,
    )
    ready = _resolve_ready(confidence=confidence, previous_snapshot=previous)
    return ReSpeakerMultimodalInitiativeSnapshot(
        observed_at=observed_at,
        ready=ready,
        confidence=confidence,
        block_reason=(None if ready else "low_multimodal_initiative_confidence"),
        recommended_channel="speech" if ready else "display",
        speaker_association_state=normalize_text(getattr(association, "state", None)) or None,
        speaker_association_confidence=coerce_optional_ratio(getattr(association, "confidence", None)),
    )


def _initiative_confidence(
    *,
    camera: Mapping[str, object],
    audio_policy: Mapping[str, object],
    association: ReSpeakerSpeakerAssociationSnapshot,
    camera_age: float | None,
    audio_age: float | None,
    association_age: float | None,
) -> float | None:
    """Return one conservative multimodal initiative confidence score."""

    weighted_values: list[tuple[float, float]] = []

    association_score = coerce_optional_ratio(getattr(association, "confidence", None))
    if association_score is not None:
        weighted_values.append(
            (
                association_score,
                0.46 * _freshness_weight(
                    association_age,
                    soft_limit=_ASSOCIATION_SOFT_STALE_SECONDS,
                    hard_limit=_ASSOCIATION_HARD_STALE_SECONDS,
                ),
            )
        )

    visual_score = _visual_confidence(camera)
    if visual_score is not None:
        weighted_values.append(
            (
                visual_score,
                0.34 * _freshness_weight(
                    camera_age,
                    soft_limit=_CAMERA_SOFT_STALE_SECONDS,
                    hard_limit=_CAMERA_HARD_STALE_SECONDS,
                ),
            )
        )

    audio_score = _audio_timing_confidence(audio_policy)
    if audio_score is not None:
        weighted_values.append(
            (
                audio_score,
                0.20 * _freshness_weight(
                    audio_age,
                    soft_limit=_AUDIO_SOFT_STALE_SECONDS,
                    hard_limit=_AUDIO_HARD_STALE_SECONDS,
                ),
            )
        )

    if not weighted_values:
        return None

    confidence = _weighted_geometric_confidence(weighted_values)
    if confidence is None:
        return None

    modality_scores = tuple(value for value, _weight in weighted_values)
    max_modality_score = max(modality_scores)
    min_modality_score = min(modality_scores)
    # Prevent one strong modality from completely masking a weak one while also
    # avoiding multimodal confidence collapse far below the available evidence.
    confidence = min(confidence, max_modality_score)
    confidence = min(confidence, min_modality_score + 0.24)

    interruption_risk = _max_ratio(
        audio_policy.get("interruption_risk"),
        audio_policy.get("speech_overlap_risk"),
        audio_policy.get("barge_in_risk"),
    )
    if interruption_risk is not None:
        confidence *= max(0.0, 1.0 - (0.45 * interruption_risk))

    return _clamp_ratio(confidence)


def _visual_confidence(camera: Mapping[str, object]) -> float | None:
    """Return conservative visual engagement confidence."""

    explicit_attention = coerce_optional_ratio(camera.get("visual_attention_score"))
    gaze_alignment = _max_ratio(
        camera.get("gaze_alignment_score"),
        camera.get("active_speaker_face_alignment_confidence"),
        camera.get("audio_visual_alignment_confidence"),
    )
    track_stability = _max_ratio(
        camera.get("face_track_stability"),
        camera.get("person_track_confidence"),
        camera.get("primary_person_track_confidence"),
    )

    evidence: list[float] = []
    if explicit_attention is not None:
        evidence.append(explicit_attention)
    if gaze_alignment is not None:
        evidence.append(0.45 + (0.45 * gaze_alignment))
    if track_stability is not None:
        evidence.append(0.40 + (0.40 * track_stability))
    if coerce_optional_bool(camera.get("engaged_with_device")) is True:
        evidence.append(0.84)
    if coerce_optional_bool(camera.get("looking_toward_device")) is True:
        evidence.append(0.78)
    if coerce_optional_bool(camera.get("single_addressee_visible")) is True:
        evidence.append(0.80)

    if evidence:
        return _clamp_ratio(mean_confidence(tuple(evidence)))

    # Conservative fallback: person is visible but there is no richer engagement evidence.
    return 0.58


def _audio_timing_confidence(audio_policy: Mapping[str, object]) -> float | None:
    """Return timing/availability confidence for speaking now."""

    direct_timing = _max_ratio(
        audio_policy.get("turn_yield_confidence"),
        audio_policy.get("initiative_timing_confidence"),
        audio_policy.get("speech_interjection_window_confidence"),
        audio_policy.get("response_timing_confidence"),
    )
    quiet_window = coerce_optional_bool(audio_policy.get("quiet_window_open")) is True
    follow_up = coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True
    presence_audio = coerce_optional_bool(audio_policy.get("presence_audio_active")) is True

    evidence: list[float] = []
    if direct_timing is not None:
        evidence.append(direct_timing)
    if quiet_window:
        evidence.append(0.74)
    if follow_up:
        evidence.append(0.80)
    if presence_audio:
        evidence.append(0.62)

    if not evidence:
        return None

    confidence = _clamp_ratio(mean_confidence(tuple(evidence)))
    interruption_risk = _max_ratio(
        audio_policy.get("interruption_risk"),
        audio_policy.get("speech_overlap_risk"),
        audio_policy.get("barge_in_risk"),
    )
    if interruption_risk is not None:
        confidence *= max(0.0, 1.0 - (0.55 * interruption_risk))
    return _clamp_ratio(confidence)


def _blocked_snapshot(
    *,
    observed_at: float | None,
    block_reason: str,
    recommended_channel: str,
    association: ReSpeakerSpeakerAssociationSnapshot | None,
) -> ReSpeakerMultimodalInitiativeSnapshot:
    """Build one normalized blocked snapshot."""

    return ReSpeakerMultimodalInitiativeSnapshot(
        observed_at=observed_at,
        ready=False,
        confidence=None,
        block_reason=_normalize_block_reason(block_reason),
        recommended_channel=_normalize_channel(recommended_channel),
        speaker_association_state=normalize_text(getattr(association, "state", None)) or None,
        speaker_association_confidence=coerce_optional_ratio(getattr(association, "confidence", None)),
    )


def _coerce_previous_snapshot(
    live_facts: Mapping[str, object],
) -> ReSpeakerMultimodalInitiativeSnapshot | None:
    """Best-effort parse of a previous snapshot from live facts."""

    previous_candidates = (
        live_facts.get("multimodal_initiative"),
        live_facts.get("respeaker_multimodal_initiative"),
    )
    for candidate in previous_candidates:
        snapshot = ReSpeakerMultimodalInitiativeSnapshot.from_fact_map(candidate)
        if snapshot is not None:
            return snapshot
    return None


def _resolve_ready(
    *,
    confidence: float | None,
    previous_snapshot: ReSpeakerMultimodalInitiativeSnapshot | None,
) -> bool:
    """Apply hysteresis so readiness does not flap around one threshold."""

    if confidence is None:
        return False
    if previous_snapshot is not None and previous_snapshot.ready:
        return confidence >= _INITIATIVE_READY_HYSTERESIS_MIN_CONFIDENCE
    return confidence >= _INITIATIVE_READY_MIN_CONFIDENCE


def _infer_person_visible(camera: Mapping[str, object]) -> bool | None:
    """Infer whether at least one person is visible from explicit or richer cues."""

    explicit = coerce_optional_bool(camera.get("person_visible"))
    if explicit is not None:
        return explicit

    person_count = coerce_optional_int(camera.get("person_count"))
    if person_count is not None:
        return person_count > 0

    if _max_ratio(
        camera.get("visual_attention_score"),
        camera.get("gaze_alignment_score"),
        camera.get("active_speaker_face_alignment_confidence"),
        camera.get("audio_visual_alignment_confidence"),
        camera.get("face_track_stability"),
        camera.get("person_track_confidence"),
        camera.get("primary_person_track_confidence"),
    ) is not None:
        return True

    if (
        coerce_optional_bool(camera.get("engaged_with_device")) is True
        or coerce_optional_bool(camera.get("looking_toward_device")) is True
        or coerce_optional_bool(camera.get("single_addressee_visible")) is True
    ):
        return True

    return None


def _coerce_single_addressee_visible(camera: Mapping[str, object]) -> bool:
    """Return whether richer addressee cues resolve a multi-person scene."""

    if coerce_optional_bool(camera.get("single_addressee_visible")) is True:
        return True
    addressee_count = coerce_optional_int(camera.get("addressee_count"))
    if addressee_count == 1:
        return True
    alignment = _max_ratio(
        camera.get("active_speaker_face_alignment_confidence"),
        camera.get("audio_visual_alignment_confidence"),
    )
    return alignment is not None and alignment >= 0.85


def _sensor_age_seconds(
    *,
    observed_at: float | None,
    sensor_mapping: Mapping[str, object] | None,
    fallback_timestamp: float | None,
) -> float | None:
    """Return non-negative sensor age in seconds when timestamps are available."""

    if observed_at is None:
        return None
    sensor_timestamp = fallback_timestamp
    if sensor_timestamp is None and sensor_mapping:
        sensor_timestamp = _extract_timestamp(sensor_mapping)
    if sensor_timestamp is None:
        return None
    age = observed_at - sensor_timestamp
    if age < 0.0:
        return 0.0
    return age


def _extract_timestamp(mapping: Mapping[str, object]) -> float | None:
    """Extract a likely observation timestamp from one mapping."""

    for key in ("observed_at", "updated_at", "timestamp", "captured_at", "emitted_at", "ts"):
        value = coerce_optional_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _freshness_weight(age: float | None, *, soft_limit: float, hard_limit: float) -> float:
    """Return a reliability weight based on staleness."""

    if age is None or age <= soft_limit:
        return 1.0
    if age >= hard_limit:
        return 0.0
    window = hard_limit - soft_limit
    if window <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((age - soft_limit) / window)))


def _is_stale(age: float | None, hard_limit: float) -> bool:
    """Return whether one sensor stream is decisively stale."""

    return age is not None and age >= hard_limit


def _weighted_geometric_confidence(
    weighted_values: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> float | None:
    """Return a conservative weighted geometric mean over confidence values."""

    filtered = [(value, weight) for value, weight in weighted_values if weight > 0.0]
    if not filtered:
        return None
    total_weight = sum(weight for _value, weight in filtered)
    if total_weight <= 0.0:
        return None
    weighted_log_sum = 0.0
    for value, weight in filtered:
        weighted_log_sum += weight * log(max(1e-6, _clamp_ratio(value)))
    return _clamp_ratio(exp(weighted_log_sum / total_weight))


def _normalize_policy_reason(value: object | None, *, default_reason: str) -> str | None:
    """Normalize one upstream reason into a bounded blocking vocabulary."""

    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    mapped = _BLOCK_REASON_ALIASES.get(normalized)
    if mapped is not None:
        return mapped
    if default_reason in _ALLOWED_BLOCK_REASONS:
        return default_reason
    return "upstream_policy_block"


def _normalize_block_reason(value: object | None) -> str | None:
    """Normalize one emitted block reason into a bounded vocabulary."""

    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    mapped = _BLOCK_REASON_ALIASES.get(normalized, normalized)
    if mapped in _ALLOWED_BLOCK_REASONS:
        return mapped
    return "upstream_policy_block"


def _normalize_channel(value: object | None) -> str:
    """Normalize one channel token into the small proactive vocabulary."""

    normalized = normalize_text(value).lower()
    if normalized not in {"speech", "display", "print"}:
        return "display"
    return normalized


def _max_ratio(*values: object | None) -> float | None:
    """Return the maximum valid ratio across multiple optional values."""

    ratios = [ratio for ratio in (coerce_optional_ratio(value) for value in values) if ratio is not None]
    if not ratios:
        return None
    return max(ratios)


def _clamp_ratio(value: float) -> float:
    """Clamp one numeric value into [0.0, 1.0]."""

    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


__all__ = [
    "ReSpeakerMultimodalInitiativeSnapshot",
    "derive_respeaker_multimodal_initiative",
]
