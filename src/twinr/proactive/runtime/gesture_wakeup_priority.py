"""Arbitrate visual wakeups against higher-priority Twinr interaction paths.

Twinr exposes multiple turn-entry paths. Button-triggered and speech-driven
turns outrank gesture-triggered turns. This module keeps that precedence policy
out of the proactive service orchestration loop so gesture wake decisions can be
suppressed before they pause or steal shared audio capture.
"""

from __future__ import annotations

from dataclasses import dataclass

from .presence import PresenceSessionSnapshot


@dataclass(frozen=True, slots=True)
class GestureWakeupPriorityDecision:
    """Describe whether one visual wakeup may continue to workflow dispatch."""

    allow: bool
    reason: str


def decide_gesture_wakeup_priority(
    *,
    runtime_status_value: object,
    voice_path_enabled: bool,
    presence_snapshot: PresenceSessionSnapshot | None,
    recent_speech_guard_s: float,
) -> GestureWakeupPriorityDecision:
    """Return whether one accepted visual wakeup should still be dispatched."""

    runtime_status = str(runtime_status_value or "").strip().lower()
    if runtime_status and runtime_status != "waiting":
        return GestureWakeupPriorityDecision(
            allow=False,
            reason=f"gesture_wakeup_suppressed_runtime_{runtime_status}",
        )
    if not voice_path_enabled or presence_snapshot is None:
        return GestureWakeupPriorityDecision(allow=True, reason="gesture_wakeup_allowed")
    if presence_snapshot.presence_audio_active is True:
        return GestureWakeupPriorityDecision(
            allow=False,
            reason="gesture_wakeup_suppressed_presence_audio_active",
        )
    if presence_snapshot.recent_follow_up_speech is True:
        return GestureWakeupPriorityDecision(
            allow=False,
            reason="gesture_wakeup_suppressed_recent_follow_up_speech",
        )
    if presence_snapshot.resume_window_open is True:
        return GestureWakeupPriorityDecision(
            allow=False,
            reason="gesture_wakeup_suppressed_resume_window_open",
        )
    last_speech_age_s = _coerce_non_negative_optional_float(presence_snapshot.last_speech_age_s)
    if last_speech_age_s is None:
        return GestureWakeupPriorityDecision(allow=True, reason="gesture_wakeup_allowed")
    if last_speech_age_s <= max(0.0, float(recent_speech_guard_s)):
        return GestureWakeupPriorityDecision(
            allow=False,
            reason="gesture_wakeup_suppressed_recent_speech",
        )
    return GestureWakeupPriorityDecision(allow=True, reason="gesture_wakeup_allowed")


def _coerce_non_negative_optional_float(value: object) -> float | None:
    """Return one finite non-negative float, or ``None`` when unavailable."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric < 0.0:
        return None
    return numeric


__all__ = ["GestureWakeupPriorityDecision", "decide_gesture_wakeup_priority"]
