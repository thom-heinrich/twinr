"""Block sensitive proactive behavior on weak ReSpeaker privacy context.

This gate is intentionally narrow: it does not decide whether a proactive
candidate is useful, only whether current room context is too ambiguous for
sensitive/private proactive speech.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from twinr.memory.longterm.core.ontology import normalize_memory_sensitivity


_SENSITIVE_MEMORY_LEVELS = frozenset({"private", "sensitive", "critical"})
_MIN_DIRECTION_CONFIDENCE = 0.75


@dataclass(frozen=True, slots=True)
class ReSpeakerSensitiveBehaviorGateDecision:
    """Describe whether sensitive proactive behavior is allowed right now."""

    allowed: bool
    reason: str | None
    candidate_sensitivity: str
    multi_person_likely: bool = False
    low_confidence_audio: bool = False
    audio_context_active: bool = False
    camera_person_count: int | None = None
    direction_confidence: float | None = None
    speaker_direction_stable: bool | None = None

    def event_data(self) -> dict[str, object]:
        """Render one JSON-safe event payload for runtime diagnostics."""

        return {
            "candidate_sensitivity": self.candidate_sensitivity,
            "sensitive_behavior_allowed": self.allowed,
            "sensitive_behavior_block_reason": self.reason,
            "sensitive_multi_person_likely": self.multi_person_likely,
            "sensitive_low_confidence_audio": self.low_confidence_audio,
            "sensitive_audio_context_active": self.audio_context_active,
            "camera_person_count": self.camera_person_count,
            "audio_direction_confidence": self.direction_confidence,
            "audio_speaker_direction_stable": self.speaker_direction_stable,
        }


def evaluate_respeaker_sensitive_behavior_gate(
    *,
    candidate_sensitivity: object,
    live_facts: Mapping[str, object] | None,
) -> ReSpeakerSensitiveBehaviorGateDecision:
    """Return whether current room context is safe enough for sensitive speech."""

    sensitivity = _normalize_sensitivity(candidate_sensitivity)
    if sensitivity not in _SENSITIVE_MEMORY_LEVELS:
        return ReSpeakerSensitiveBehaviorGateDecision(
            allowed=True,
            reason=None,
            candidate_sensitivity=sensitivity,
        )

    facts = live_facts if isinstance(live_facts, Mapping) else {}
    camera = facts.get("camera")
    respeaker = facts.get("respeaker")
    audio_policy = facts.get("audio_policy")
    sensor = facts.get("sensor")

    camera_map = camera if isinstance(camera, Mapping) else {}
    respeaker_map = respeaker if isinstance(respeaker, Mapping) else {}
    audio_policy_map = audio_policy if isinstance(audio_policy, Mapping) else {}
    sensor_map = sensor if isinstance(sensor, Mapping) else {}

    camera_person_count = _coerce_optional_int(camera_map.get("person_count"))
    overlap_likely = _coerce_optional_bool(audio_policy_map.get("room_busy_or_overlapping")) is True
    multi_person_likely = (camera_person_count is not None and camera_person_count > 1) or overlap_likely

    direction_confidence = _coerce_optional_float(respeaker_map.get("direction_confidence"))
    speaker_direction_stable = _coerce_optional_bool(audio_policy_map.get("speaker_direction_stable"))
    audio_context_active = any(
        (
            _coerce_optional_bool(audio_policy_map.get("presence_audio_active")) is True,
            _coerce_optional_bool(audio_policy_map.get("recent_follow_up_speech")) is True,
            _coerce_optional_bool(audio_policy_map.get("resume_window_open")) is True,
            _coerce_optional_bool(facts.get("vad", {}).get("speech_detected") if isinstance(facts.get("vad"), Mapping) else None) is True,
            _coerce_optional_int(sensor_map.get("presence_session_id")) is not None,
        )
    )
    low_confidence_audio = (
        audio_context_active
        and (
            speaker_direction_stable is not True
            or direction_confidence is None
            or direction_confidence < _MIN_DIRECTION_CONFIDENCE
        )
    )

    if multi_person_likely:
        return ReSpeakerSensitiveBehaviorGateDecision(
            allowed=False,
            reason="sensitive_multi_person_context",
            candidate_sensitivity=sensitivity,
            multi_person_likely=True,
            low_confidence_audio=low_confidence_audio,
            audio_context_active=audio_context_active,
            camera_person_count=camera_person_count,
            direction_confidence=direction_confidence,
            speaker_direction_stable=speaker_direction_stable,
        )
    if low_confidence_audio:
        return ReSpeakerSensitiveBehaviorGateDecision(
            allowed=False,
            reason="sensitive_low_confidence_audio_context",
            candidate_sensitivity=sensitivity,
            multi_person_likely=False,
            low_confidence_audio=True,
            audio_context_active=True,
            camera_person_count=camera_person_count,
            direction_confidence=direction_confidence,
            speaker_direction_stable=speaker_direction_stable,
        )
    return ReSpeakerSensitiveBehaviorGateDecision(
        allowed=True,
        reason=None,
        candidate_sensitivity=sensitivity,
        multi_person_likely=False,
        low_confidence_audio=False,
        audio_context_active=audio_context_active,
        camera_person_count=camera_person_count,
        direction_confidence=direction_confidence,
        speaker_direction_stable=speaker_direction_stable,
    )


def _normalize_sensitivity(value: object) -> str:
    try:
        return normalize_memory_sensitivity(str(value or "normal"))
    except Exception:
        return "sensitive"


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = [
    "ReSpeakerSensitiveBehaviorGateDecision",
    "evaluate_respeaker_sensitive_behavior_gate",
]
