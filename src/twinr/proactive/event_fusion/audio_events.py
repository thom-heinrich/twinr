# CHANGELOG: 2026-03-29
# BUG-1: Non-finite classifier scores (NaN/inf) could leak through `_clamp_ratio()`
# and produce invalid confidences/payloads; scores and timestamps are now sanitized.
# BUG-2: Shout events driven only by an external classifier were previously assigned
# a confidence floor (`fallback_shout_confidence`) that could exceed the true model
# score, skewing downstream fusion/ranking; classifier-only events now preserve the
# raw classifier confidence and expose the threshold separately.
# SEC-1: Hardened the module against malformed numeric inputs that could poison JSON /
# IPC payloads or destabilize edge pipelines through NaN/inf propagation.
# IMP-1: Added bounded event windows, schema-versioned payloads, richer provenance
# (`raw_confidence`, `threshold`, `calibrated`, `evidence_sources`) and deterministic
# same-kind evidence merging.
# IMP-2: Added per-kind thresholds plus optional stateful cooldown / monotonic-time
# handling for streaming deployments on resource-constrained edge devices.

"""Derive bounded audio micro-events for short-window fusion.

This module intentionally stays on the event-proposal layer. It accepts the
normalized social audio observation contract plus optional future classifier
scores and emits small inspectable audio event records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math

from twinr.proactive.social.engine import SocialAudioObservation


def _clamp_ratio(value: float | None, *, default: float = 0.0) -> float:
    """Clamp one optional ratio into ``[0.0, 1.0]`` and drop non-finite values."""

    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _sanitize_timestamp(value: float, *, field_name: str) -> float:
    """Return one finite timestamp or raise a precise error."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a real timestamp, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return numeric


def _merge_tuple_values(*parts: tuple[str, ...]) -> tuple[str, ...]:
    """Merge ordered string tuples while preserving first appearance."""

    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for item in part:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


def _resolve_window(
    *,
    observed_at: float,
    config: "AudioEventConfig",
    hints: "AudioClassifierHints",
) -> tuple[float, float]:
    """Return a bounded trailing window for the current observation."""

    window_seconds = config.window_duration_seconds
    if hints.window_duration_seconds is not None:
        hinted_window = float(hints.window_duration_seconds)
        if math.isfinite(hinted_window) and hinted_window > 0.0:
            window_seconds = hinted_window
    start = observed_at - window_seconds
    if start < 0.0:
        start = 0.0
    return (start, observed_at)


class AudioEventKind(StrEnum):
    """Describe the V1/V2 audio micro-event vocabulary."""

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
    scores_are_calibrated: bool | None = None
    window_duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AudioEventConfig:
    """Store bounded thresholds and streaming controls for audio micro-events."""

    external_hint_threshold: float = 0.64
    speech_activity_confidence: float = 0.72
    distress_shout_confidence: float = 0.84
    fallback_shout_confidence: float = 0.67  # retained for config compatibility
    background_media_confidence: float = 0.9
    overlap_confidence: float = 0.86
    shout_hint_threshold: float | None = None
    laugh_hint_threshold: float | None = None
    cry_hint_threshold: float | None = None
    cough_hint_threshold: float | None = None
    window_duration_seconds: float = 1.0
    emission_cooldown_seconds: float = 0.0
    min_confidence_delta_for_cooldown_bypass: float = 0.12
    require_monotonic_observed_at: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_hint_threshold",
            _clamp_ratio(self.external_hint_threshold, default=0.64),
        )
        object.__setattr__(
            self,
            "speech_activity_confidence",
            _clamp_ratio(self.speech_activity_confidence, default=0.72),
        )
        object.__setattr__(
            self,
            "distress_shout_confidence",
            _clamp_ratio(self.distress_shout_confidence, default=0.84),
        )
        object.__setattr__(
            self,
            "fallback_shout_confidence",
            _clamp_ratio(self.fallback_shout_confidence, default=0.67),
        )
        object.__setattr__(
            self,
            "background_media_confidence",
            _clamp_ratio(self.background_media_confidence, default=0.9),
        )
        object.__setattr__(
            self,
            "overlap_confidence",
            _clamp_ratio(self.overlap_confidence, default=0.86),
        )
        for field_name, default in (
            ("shout_hint_threshold", self.external_hint_threshold),
            ("laugh_hint_threshold", self.external_hint_threshold),
            ("cry_hint_threshold", self.external_hint_threshold),
            ("cough_hint_threshold", self.external_hint_threshold),
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _clamp_ratio(value, default=default) if value is not None else None,
            )

        try:
            window_duration_seconds = float(self.window_duration_seconds)
        except (TypeError, ValueError):
            window_duration_seconds = 1.0
        if not math.isfinite(window_duration_seconds) or window_duration_seconds <= 0.0:
            window_duration_seconds = 1.0
        object.__setattr__(self, "window_duration_seconds", window_duration_seconds)

        try:
            emission_cooldown_seconds = float(self.emission_cooldown_seconds)
        except (TypeError, ValueError):
            emission_cooldown_seconds = 0.0
        if not math.isfinite(emission_cooldown_seconds) or emission_cooldown_seconds < 0.0:
            emission_cooldown_seconds = 0.0
        object.__setattr__(self, "emission_cooldown_seconds", emission_cooldown_seconds)

        object.__setattr__(
            self,
            "min_confidence_delta_for_cooldown_bypass",
            _clamp_ratio(
                self.min_confidence_delta_for_cooldown_bypass,
                default=0.12,
            ),
        )

    def threshold_for(self, kind: AudioEventKind) -> float:
        """Return the configured threshold for one hint-driven event class."""

        if kind is AudioEventKind.SHOUT_LIKE_AUDIO:
            return (
                self.shout_hint_threshold
                if self.shout_hint_threshold is not None
                else self.external_hint_threshold
            )
        if kind is AudioEventKind.LAUGH_LIKE_AUDIO:
            return (
                self.laugh_hint_threshold
                if self.laugh_hint_threshold is not None
                else self.external_hint_threshold
            )
        if kind is AudioEventKind.CRY_LIKE_AUDIO:
            return (
                self.cry_hint_threshold
                if self.cry_hint_threshold is not None
                else self.external_hint_threshold
            )
        if kind is AudioEventKind.COUGH_LIKE_AUDIO:
            return (
                self.cough_hint_threshold
                if self.cough_hint_threshold is not None
                else self.external_hint_threshold
            )
        return self.external_hint_threshold


@dataclass(slots=True)
class AudioEventTrackerState:
    """Track emission time/confidence per event kind for streaming cooldowns."""

    last_observed_at: float | None = None
    last_emitted_at_by_kind: dict[AudioEventKind, float] = field(default_factory=dict)
    last_emitted_confidence_by_kind: dict[AudioEventKind, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioMicroEvent:
    """Describe one short-lived, bounded audio event proposal."""

    kind: AudioEventKind
    observed_at: float
    confidence: float
    source: str
    active: bool = True
    supporting_fields: tuple[str, ...] = field(default_factory=tuple)
    observed_from: float | None = None
    observed_until: float | None = None
    threshold: float | None = None
    raw_confidence: float | None = None
    calibrated: bool | None = None
    evidence_sources: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            _sanitize_timestamp(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "confidence",
            _clamp_ratio(self.confidence, default=0.0),
        )
        if self.observed_from is not None:
            object.__setattr__(
                self,
                "observed_from",
                _sanitize_timestamp(self.observed_from, field_name="observed_from"),
            )
        if self.observed_until is not None:
            object.__setattr__(
                self,
                "observed_until",
                _sanitize_timestamp(self.observed_until, field_name="observed_until"),
            )
        if self.observed_from is not None and self.observed_until is not None:
            if self.observed_from > self.observed_until:
                original_from = self.observed_from
                object.__setattr__(self, "observed_from", self.observed_until)
                object.__setattr__(self, "observed_until", original_from)
        if self.threshold is not None:
            object.__setattr__(
                self,
                "threshold",
                _clamp_ratio(self.threshold, default=0.0),
            )
        if self.raw_confidence is not None:
            object.__setattr__(
                self,
                "raw_confidence",
                _clamp_ratio(self.raw_confidence, default=0.0),
            )
        object.__setattr__(self, "supporting_fields", tuple(self.supporting_fields))
        object.__setattr__(
            self,
            "evidence_sources",
            tuple(self.evidence_sources) if self.evidence_sources else (self.source,),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize one audio micro-event into plain JSON-safe facts."""

        payload: dict[str, object] = {
            "kind": self.kind.value,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "source": self.source,
            "active": self.active,
            "supporting_fields": list(self.supporting_fields),
            "schema_version": self.schema_version,
        }
        if self.observed_from is not None:
            payload["observed_from"] = self.observed_from
        if self.observed_until is not None:
            payload["observed_until"] = self.observed_until
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.raw_confidence is not None:
            payload["raw_confidence"] = self.raw_confidence
        if self.calibrated is not None:
            payload["calibrated"] = self.calibrated
        if self.evidence_sources:
            payload["evidence_sources"] = list(self.evidence_sources)
        return payload


def _merge_event(
    events: dict[AudioEventKind, AudioMicroEvent],
    event: AudioMicroEvent,
) -> None:
    """Insert or merge one event proposal by kind."""

    existing = events.get(event.kind)
    if existing is None:
        events[event.kind] = event
        return

    primary = event if event.confidence > existing.confidence else existing
    secondary = existing if primary is event else event

    observed_from_candidates = tuple(
        value
        for value in (existing.observed_from, event.observed_from)
        if value is not None
    )
    observed_until_candidates = tuple(
        value
        for value in (existing.observed_until, event.observed_until)
        if value is not None
    )

    events[event.kind] = AudioMicroEvent(
        kind=primary.kind,
        observed_at=max(existing.observed_at, event.observed_at),
        confidence=max(existing.confidence, event.confidence),
        source=primary.source,
        active=existing.active or event.active,
        supporting_fields=_merge_tuple_values(
            existing.supporting_fields,
            event.supporting_fields,
        ),
        observed_from=min(observed_from_candidates) if observed_from_candidates else None,
        observed_until=max(observed_until_candidates) if observed_until_candidates else None,
        threshold=primary.threshold if primary.threshold is not None else secondary.threshold,
        raw_confidence=max(
            (
                value
                for value in (existing.raw_confidence, event.raw_confidence)
                if value is not None
            ),
            default=None,
        ),
        calibrated=(
            primary.calibrated
            if primary.calibrated is not None
            else secondary.calibrated
        ),
        evidence_sources=_merge_tuple_values(
            existing.evidence_sources,
            event.evidence_sources,
        ),
        schema_version=max(existing.schema_version, event.schema_version),
    )


def _apply_tracker_state(
    *,
    observed_at: float,
    events: tuple[AudioMicroEvent, ...],
    tracker_state: AudioEventTrackerState | None,
    config: AudioEventConfig,
) -> tuple[AudioMicroEvent, ...]:
    """Optionally suppress re-emission bursts for streaming deployments."""

    if tracker_state is None:
        return events

    if (
        config.require_monotonic_observed_at
        and tracker_state.last_observed_at is not None
        and observed_at < tracker_state.last_observed_at
    ):
        raise ValueError(
            "observed_at must be monotonically increasing when "
            "require_monotonic_observed_at=True"
        )
    tracker_state.last_observed_at = observed_at

    if config.emission_cooldown_seconds <= 0.0:
        for event in events:
            tracker_state.last_emitted_at_by_kind[event.kind] = observed_at
            tracker_state.last_emitted_confidence_by_kind[event.kind] = event.confidence
        return events

    filtered: list[AudioMicroEvent] = []
    for event in events:
        last_emitted_at = tracker_state.last_emitted_at_by_kind.get(event.kind)
        last_emitted_confidence = tracker_state.last_emitted_confidence_by_kind.get(
            event.kind,
            0.0,
        )

        should_emit = True
        if last_emitted_at is not None:
            delta_t = observed_at - last_emitted_at
            if delta_t >= 0.0 and delta_t < config.emission_cooldown_seconds:
                should_emit = (
                    event.confidence
                    >= last_emitted_confidence
                    + config.min_confidence_delta_for_cooldown_bypass
                )

        if not should_emit:
            continue

        filtered.append(event)
        tracker_state.last_emitted_at_by_kind[event.kind] = observed_at
        tracker_state.last_emitted_confidence_by_kind[event.kind] = event.confidence

    return tuple(filtered)


def derive_audio_micro_events(
    *,
    observed_at: float,
    observation: SocialAudioObservation,
    hints: AudioClassifierHints | None = None,
    config: AudioEventConfig | None = None,
    tracker_state: AudioEventTrackerState | None = None,
) -> tuple[AudioMicroEvent, ...]:
    """Return active bounded audio micro-events for one normalized observation tick."""

    cfg = config or AudioEventConfig()
    hint_scores = hints or AudioClassifierHints()
    observed_at = _sanitize_timestamp(observed_at, field_name="observed_at")
    observed_from, observed_until = _resolve_window(
        observed_at=observed_at,
        config=cfg,
        hints=hint_scores,
    )

    events: dict[AudioEventKind, AudioMicroEvent] = {}

    if observation.speech_detected is True:
        _merge_event(
            events,
            AudioMicroEvent(
                kind=AudioEventKind.SPEECH_ACTIVITY,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=cfg.speech_activity_confidence,
                source="normalized_audio_observation",
                supporting_fields=("speech_detected",),
            ),
        )

    if observation.background_media_likely is True:
        _merge_event(
            events,
            AudioMicroEvent(
                kind=AudioEventKind.BACKGROUND_MEDIA_LIKELY,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=cfg.background_media_confidence,
                source="respeaker_ambient_classification",
                supporting_fields=("background_media_likely",),
            ),
        )

    if observation.speech_overlap_likely is True:
        _merge_event(
            events,
            AudioMicroEvent(
                kind=AudioEventKind.SPEECH_OVERLAP_LIKELY,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=cfg.overlap_confidence,
                source="respeaker_overlap_detection",
                supporting_fields=("speech_overlap_likely",),
            ),
        )

    if observation.distress_detected is True:
        _merge_event(
            events,
            AudioMicroEvent(
                kind=AudioEventKind.SHOUT_LIKE_AUDIO,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=cfg.distress_shout_confidence,
                source="distress_signal_fusion",
                supporting_fields=(
                    ("distress_detected", "speech_detected")
                    if observation.speech_detected is True
                    else ("distress_detected",)
                ),
            ),
        )

    shout_confidence = _clamp_ratio(hint_scores.shout_like_confidence, default=0.0)
    shout_threshold = cfg.threshold_for(AudioEventKind.SHOUT_LIKE_AUDIO)
    if shout_confidence >= shout_threshold:
        # BREAKING: classifier-only shout events now preserve the raw classifier
        # confidence instead of flooring it to `fallback_shout_confidence`.
        _merge_event(
            events,
            AudioMicroEvent(
                kind=AudioEventKind.SHOUT_LIKE_AUDIO,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=shout_confidence,
                raw_confidence=shout_confidence,
                threshold=shout_threshold,
                calibrated=hint_scores.scores_are_calibrated,
                source="external_audio_classifier_hint",
                supporting_fields=("shout_like_confidence",),
            ),
        )

    for kind, field_name, score in (
        (
            AudioEventKind.LAUGH_LIKE_AUDIO,
            "laugh_like_confidence",
            hint_scores.laugh_like_confidence,
        ),
        (
            AudioEventKind.CRY_LIKE_AUDIO,
            "cry_like_confidence",
            hint_scores.cry_like_confidence,
        ),
        (
            AudioEventKind.COUGH_LIKE_AUDIO,
            "cough_like_confidence",
            hint_scores.cough_like_confidence,
        ),
    ):
        confidence = _clamp_ratio(score, default=0.0)
        threshold = cfg.threshold_for(kind)
        if confidence < threshold:
            continue
        _merge_event(
            events,
            AudioMicroEvent(
                kind=kind,
                observed_at=observed_at,
                observed_from=observed_from,
                observed_until=observed_until,
                confidence=confidence,
                raw_confidence=confidence,
                threshold=threshold,
                calibrated=hint_scores.scores_are_calibrated,
                source="external_audio_classifier_hint",
                supporting_fields=(field_name,),
            ),
        )

    ordered_events = tuple(events.values())
    return _apply_tracker_state(
        observed_at=observed_at,
        events=ordered_events,
        tracker_state=tracker_state,
        config=cfg,
    )


__all__ = [
    "AudioClassifierHints",
    "AudioEventConfig",
    "AudioEventKind",
    "AudioEventTrackerState",
    "AudioMicroEvent",
    "derive_audio_micro_events",
]