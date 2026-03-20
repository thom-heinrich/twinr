"""Derive bounded audio micro-events for short-window fusion.

This module intentionally stays on the event-proposal layer. It accepts the
normalized social audio observation contract plus optional future classifier
scores and emits small inspectable audio event records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from twinr.proactive.social.engine import SocialAudioObservation


def _clamp_ratio(value: float | None, *, default: float = 0.0) -> float:
    """Clamp one optional ratio into ``[0.0, 1.0]``."""

    if value is None:
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


class AudioEventKind(StrEnum):
    """Describe the V1 audio micro-event vocabulary."""

    SPEECH_ACTIVITY = "speech_activity"
    SHOUT_LIKE_AUDIO = "shout_like_audio"
    LAUGH_LIKE_AUDIO = "laugh_like_audio"
    CRY_LIKE_AUDIO = "cry_like_audio"
    COUGH_LIKE_AUDIO = "cough_like_audio"
    BACKGROUND_MEDIA_LIKELY = "background_media_likely"
    SPEECH_OVERLAP_LIKELY = "speech_overlap_likely"


@dataclass(frozen=True, slots=True)
class AudioClassifierHints:
    """Carry optional future classifier scores into the micro-event layer."""

    shout_like_confidence: float | None = None
    laugh_like_confidence: float | None = None
    cry_like_confidence: float | None = None
    cough_like_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AudioEventConfig:
    """Store bounded thresholds for audio micro-event proposals."""

    external_hint_threshold: float = 0.64
    speech_activity_confidence: float = 0.72
    distress_shout_confidence: float = 0.84
    fallback_shout_confidence: float = 0.67
    background_media_confidence: float = 0.9
    overlap_confidence: float = 0.86


@dataclass(frozen=True, slots=True)
class AudioMicroEvent:
    """Describe one short-lived audio event proposal."""

    kind: AudioEventKind
    observed_at: float
    confidence: float
    source: str
    active: bool = True
    supporting_fields: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        """Serialize one audio micro-event into plain facts."""

        return {
            "kind": self.kind.value,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "source": self.source,
            "active": self.active,
            "supporting_fields": list(self.supporting_fields),
        }


def derive_audio_micro_events(
    *,
    observed_at: float,
    observation: SocialAudioObservation,
    hints: AudioClassifierHints | None = None,
    config: AudioEventConfig | None = None,
) -> tuple[AudioMicroEvent, ...]:
    """Return active V1 audio micro-events for one normalized observation tick."""

    cfg = config or AudioEventConfig()
    hint_scores = hints or AudioClassifierHints()
    events: list[AudioMicroEvent] = []

    if observation.speech_detected is True:
        events.append(
            AudioMicroEvent(
                kind=AudioEventKind.SPEECH_ACTIVITY,
                observed_at=observed_at,
                confidence=cfg.speech_activity_confidence,
                source="normalized_audio_observation",
                supporting_fields=("speech_detected",),
            )
        )

    if observation.background_media_likely is True:
        events.append(
            AudioMicroEvent(
                kind=AudioEventKind.BACKGROUND_MEDIA_LIKELY,
                observed_at=observed_at,
                confidence=cfg.background_media_confidence,
                source="respeaker_ambient_classification",
                supporting_fields=("background_media_likely",),
            )
        )

    if observation.speech_overlap_likely is True:
        events.append(
            AudioMicroEvent(
                kind=AudioEventKind.SPEECH_OVERLAP_LIKELY,
                observed_at=observed_at,
                confidence=cfg.overlap_confidence,
                source="respeaker_overlap_detection",
                supporting_fields=("speech_overlap_likely",),
            )
        )

    shout_confidence = _clamp_ratio(hint_scores.shout_like_confidence, default=0.0)
    if observation.distress_detected is True or shout_confidence >= cfg.external_hint_threshold:
        events.append(
            AudioMicroEvent(
                kind=AudioEventKind.SHOUT_LIKE_AUDIO,
                observed_at=observed_at,
                confidence=(
                    max(cfg.distress_shout_confidence, shout_confidence)
                    if observation.distress_detected is True
                    else max(cfg.fallback_shout_confidence, shout_confidence)
                ),
                source=(
                    "distress_flag_plus_audio_activity"
                    if observation.distress_detected is True
                    else "external_audio_classifier_hint"
                ),
                supporting_fields=(
                    ("distress_detected", "speech_detected")
                    if observation.distress_detected is True
                    else ("shout_like_confidence",)
                ),
            )
        )

    events.extend(
        _events_from_classifier_hints(
            observed_at=observed_at,
            hints=hint_scores,
            config=cfg,
        )
    )
    return tuple(events)


def _events_from_classifier_hints(
    *,
    observed_at: float,
    hints: AudioClassifierHints,
    config: AudioEventConfig,
) -> tuple[AudioMicroEvent, ...]:
    """Convert optional classifier scores into bounded event proposals."""

    candidates = (
        (AudioEventKind.LAUGH_LIKE_AUDIO, "laugh_like_confidence", hints.laugh_like_confidence),
        (AudioEventKind.CRY_LIKE_AUDIO, "cry_like_confidence", hints.cry_like_confidence),
        (AudioEventKind.COUGH_LIKE_AUDIO, "cough_like_confidence", hints.cough_like_confidence),
    )
    events: list[AudioMicroEvent] = []
    for kind, field_name, score in candidates:
        confidence = _clamp_ratio(score, default=0.0)
        if confidence < config.external_hint_threshold:
            continue
        events.append(
            AudioMicroEvent(
                kind=kind,
                observed_at=observed_at,
                confidence=confidence,
                source="external_audio_classifier_hint",
                supporting_fields=(field_name,),
            )
        )
    return tuple(events)


__all__ = [
    "AudioClassifierHints",
    "AudioEventConfig",
    "AudioEventKind",
    "AudioMicroEvent",
    "derive_audio_micro_events",
]
