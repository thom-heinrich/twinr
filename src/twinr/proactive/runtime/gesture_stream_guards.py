"""Guard user-facing gesture side effects behind current live hand evidence.

The dedicated gesture stream may expose bounded rescue outputs for wider runtime
semantics and diagnostics, but HDMI acknowledgements and visual wakeups should
only fire when the current frame still looks like deliberate live user intent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..social.perception_stream import gesture_stream

_UNSAFE_USER_INTENT_SOURCES = frozenset(
    {
        "recent_person_roi",
        "recent_visible_person_roi",
        "recent_live_hand_roi",
        "full_frame_hand_roi",
    }
)


@dataclass(frozen=True, slots=True)
class GestureStreamUserIntentGuard:
    """Describe whether the current gesture stream is safe for user-facing side effects."""

    allowed: bool = False
    reason: str = "missing_stream"
    resolved_source: str | None = None
    hand_or_object_near_camera: bool = False


def evaluate_gesture_stream_user_intent(observation: object) -> GestureStreamUserIntentGuard:
    """Return whether HDMI/wakeup consumers may treat the frame as live user intent."""

    stream = gesture_stream(observation)
    if stream is None:
        return GestureStreamUserIntentGuard()

    resolved_source = _normalize_optional_text(
        getattr(stream, "resolved_source", None)
    ) or _normalize_optional_text(getattr(stream, "activation_source", None))
    hand_or_object_near_camera = bool(
        getattr(stream, "hand_or_object_near_camera", False)
        or getattr(observation, "hand_or_object_near_camera", False)
    )

    if not hand_or_object_near_camera:
        return GestureStreamUserIntentGuard(
            reason="missing_hand_evidence",
            resolved_source=resolved_source,
            hand_or_object_near_camera=False,
        )

    if resolved_source in _UNSAFE_USER_INTENT_SOURCES:
        return GestureStreamUserIntentGuard(
            reason="unsafe_resolved_source",
            resolved_source=resolved_source,
            hand_or_object_near_camera=hand_or_object_near_camera,
        )

    return GestureStreamUserIntentGuard(
        allowed=True,
        reason="allowed",
        resolved_source=resolved_source,
        hand_or_object_near_camera=hand_or_object_near_camera,
    )


def _normalize_optional_text(value: object) -> str | None:
    """Return one small trimmed telemetry token."""

    text = str(value or "").strip()
    return text or None


__all__ = [
    "GestureStreamUserIntentGuard",
    "evaluate_gesture_stream_user_intent",
]
