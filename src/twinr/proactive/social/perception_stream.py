"""Explicit temporal contract for Twinr's productive camera perception stream."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import SocialFineHandGesture, SocialGestureEvent


@dataclass(frozen=True, slots=True)
class PerceptionAttentionStreamObservation:
    """Describe one authoritative attention stream state."""

    authoritative: bool = False
    stable_looking_toward_device: bool | None = None
    stable_visual_attention_score: float | None = None
    stable_signal_state: str | None = None
    stable_signal_source: str | None = None
    instant_looking_toward_device: bool | None = None
    instant_visual_attention_score: float | None = None
    instant_signal_state: str | None = None
    instant_signal_source: str | None = None
    candidate_signal_state: str | None = None
    candidate_signal_source: str | None = None
    changed: bool = False


@dataclass(frozen=True, slots=True)
class PerceptionGestureStreamObservation:
    """Describe one authoritative gesture stream state."""

    authoritative: bool = False
    activation_key: str | None = None
    activation_token: int | None = None
    activation_started_at: float | None = None
    activation_changed_at: float | None = None
    activation_source: str | None = None
    activation_rising: bool = False
    stable_gesture_event: SocialGestureEvent = SocialGestureEvent.NONE
    stable_gesture_confidence: float | None = None
    stable_fine_hand_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE
    stable_fine_hand_gesture_confidence: float | None = None
    instant_gesture_event: SocialGestureEvent = SocialGestureEvent.NONE
    instant_gesture_confidence: float | None = None
    instant_fine_hand_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE
    instant_fine_hand_gesture_confidence: float | None = None
    hand_or_object_near_camera: bool = False
    temporal_reason: str | None = None
    resolved_source: str | None = None
    changed: bool = False


@dataclass(frozen=True, slots=True)
class PerceptionStreamObservation:
    """Bundle the authoritative productive camera stream state for one frame."""

    mode: str = "stream"
    source: str | None = None
    captured_at: float | None = None
    attention: PerceptionAttentionStreamObservation | None = None
    gesture: PerceptionGestureStreamObservation | None = None


def attention_stream(observation: object) -> PerceptionAttentionStreamObservation | None:
    """Return the attached attention stream contract when present."""

    stream = getattr(observation, "perception_stream", None)
    if not isinstance(stream, PerceptionStreamObservation):
        return None
    if not isinstance(stream.attention, PerceptionAttentionStreamObservation):
        return None
    return stream.attention


def gesture_stream(observation: object) -> PerceptionGestureStreamObservation | None:
    """Return the attached gesture stream contract when present."""

    stream = getattr(observation, "perception_stream", None)
    if not isinstance(stream, PerceptionStreamObservation):
        return None
    if not isinstance(stream.gesture, PerceptionGestureStreamObservation):
        return None
    return stream.gesture


def attention_stream_authoritative(observation: object) -> bool:
    """Return whether the observation carries authoritative attention stream truth."""

    stream = attention_stream(observation)
    return bool(stream is not None and stream.authoritative)


def gesture_stream_authoritative(observation: object) -> bool:
    """Return whether the observation carries authoritative gesture stream truth."""

    stream = gesture_stream(observation)
    return bool(stream is not None and stream.authoritative)


__all__ = [
    "PerceptionAttentionStreamObservation",
    "PerceptionGestureStreamObservation",
    "PerceptionStreamObservation",
    "attention_stream",
    "attention_stream_authoritative",
    "gesture_stream",
    "gesture_stream_authoritative",
]
