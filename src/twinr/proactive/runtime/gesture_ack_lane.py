"""Stabilize the dedicated live gesture lane into HDMI emoji acknowledgements.

This module exists so user-facing gesture acknowledgement can stay independent
from the broader social camera surface. It accepts only the explicit hand
symbols Twinr should mirror quickly on the display, applies bounded
per-gesture confirmation/cooldown rules, and returns ready-to-publish emoji
decisions without touching eye-follow state.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig

from ..social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation
from ..social.gesture_calibration import GestureCalibrationProfile
from .display_gesture_emoji import (
    DisplayGestureEmojiDecision,
    decision_for_coarse_gesture,
    decision_for_fine_hand_gesture,
)


_SUPPORTED_FINE_GESTURES = frozenset(
    {
        SocialFineHandGesture.THUMBS_UP,
        SocialFineHandGesture.THUMBS_DOWN,
        SocialFineHandGesture.POINTING,
        SocialFineHandGesture.PEACE_SIGN,
        SocialFineHandGesture.OK_SIGN,
        SocialFineHandGesture.MIDDLE_FINGER,
    }
)
_SUPPORTED_COARSE_GESTURES = frozenset({SocialGestureEvent.WAVE})
_PENDING_RESET_AFTER_S = 0.9


@dataclass(frozen=True, slots=True)
class _GestureCandidate:
    """Describe one accepted raw gesture candidate before publish."""

    key: str
    decision: DisplayGestureEmojiDecision
    confirm_samples: int
    hold_s: float


class GestureAckLane:
    """Keep user-facing HDMI gesture acknowledgement fast and self-contained."""

    def __init__(
        self,
        *,
        calibration: GestureCalibrationProfile,
    ) -> None:
        self.calibration = calibration
        self._pending_key: str | None = None
        self._pending_count = 0
        self._pending_seen_at: float | None = None
        self._last_emitted_key: str | None = None
        self._last_emitted_at: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "GestureAckLane":
        """Build one gesture lane from runtime calibration config."""

        return cls(
            calibration=GestureCalibrationProfile.from_runtime_config(config),
        )

    def observe(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> DisplayGestureEmojiDecision:
        """Return one publish-ready emoji decision or one inactive reason."""

        candidate = self._candidate_from_observation(observation)
        if candidate is None:
            self._reset_pending_if_stale(observed_at)
            return DisplayGestureEmojiDecision(reason="no_supported_live_gesture")

        if self._pending_key == candidate.key:
            self._pending_count += 1
        else:
            self._pending_key = candidate.key
            self._pending_count = 1
        self._pending_seen_at = observed_at

        if self._pending_count < candidate.confirm_samples:
            return DisplayGestureEmojiDecision(reason="awaiting_live_gesture_confirmation")

        if (
            self._last_emitted_key == candidate.key
            and self._last_emitted_at is not None
            and (observed_at - self._last_emitted_at) < candidate.hold_s
        ):
            return DisplayGestureEmojiDecision(reason="live_gesture_cooldown")

        self._last_emitted_key = candidate.key
        self._last_emitted_at = observed_at
        return candidate.decision

    def _candidate_from_observation(
        self,
        observation: SocialVisionObservation,
    ) -> _GestureCandidate | None:
        if observation.gesture_event in _SUPPORTED_COARSE_GESTURES:
            confidence = _coerce_confidence(observation.gesture_confidence)
            if confidence >= 0.68:
                decision = decision_for_coarse_gesture(
                    observation.gesture_event,
                    motion_priority=True,
                )
                if decision.active:
                    return _GestureCandidate(
                        key=f"coarse:{observation.gesture_event.value}",
                        decision=decision,
                        confirm_samples=1,
                        hold_s=0.35,
                    )

        fine_gesture = observation.fine_hand_gesture
        if fine_gesture not in _SUPPORTED_FINE_GESTURES:
            return None
        confidence = _coerce_confidence(observation.fine_hand_gesture_confidence)
        policy = self.calibration.fine_hand_policy(
            fine_gesture,
            fallback_min_confidence=0.72,
            fallback_confirm_samples=1,
            fallback_hold_s=0.35,
        )
        if confidence < policy.min_confidence:
            return None
        decision = decision_for_fine_hand_gesture(fine_gesture)
        if not decision.active:
            return None
        return _GestureCandidate(
            key=f"fine:{fine_gesture.value}",
            decision=decision,
            confirm_samples=policy.confirm_samples,
            hold_s=policy.hold_s,
        )

    def _reset_pending_if_stale(self, observed_at: float) -> None:
        seen_at = self._pending_seen_at
        if seen_at is None or (observed_at - seen_at) < _PENDING_RESET_AFTER_S:
            return
        self._pending_key = None
        self._pending_count = 0
        self._pending_seen_at = None


def _coerce_confidence(value: object) -> float:
    """Clamp one optional gesture confidence into a bounded float."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return numeric


__all__ = ["GestureAckLane"]
