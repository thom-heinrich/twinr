"""Stabilize the dedicated live gesture lane into HDMI emoji acknowledgements.

This module exists so user-facing gesture acknowledgement can stay independent
from the broader social camera surface. It accepts only the explicit hand
symbols Twinr should mirror quickly on the display, applies bounded
per-gesture confirmation/cooldown rules, and returns ready-to-publish emoji
decisions without touching eye-follow state.

The current Pi-focused live HCI contract is intentionally limited to
`thumbs_up`, `thumbs_down`, and `peace_sign`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation
from ..social.gesture_calibration import FineHandGesturePolicy, GestureCalibrationProfile
from .display_gesture_emoji import (
    DisplayGestureEmojiDecision,
    decision_for_coarse_gesture,
    decision_for_fine_hand_gesture,
)


_SUPPORTED_FINE_GESTURES = frozenset(
    {
        SocialFineHandGesture.THUMBS_UP,
        SocialFineHandGesture.THUMBS_DOWN,
        SocialFineHandGesture.PEACE_SIGN,
    }
)
_SUPPORTED_COARSE_GESTURES: frozenset[SocialGestureEvent] = frozenset()
_PENDING_RESET_AFTER_S = 0.9
_DISPLAY_ACK_FALLBACK_POLICIES = {
    # Pi-side live/ROI gesture scores are materially lower than the broader
    # social-path defaults. Keep the HDMI ack lane tuned to real device traces
    # so clear user gestures publish, while still blocking the weakest false
    # positives seen in candidate-frame QA.
    SocialFineHandGesture.THUMBS_UP: FineHandGesturePolicy(0.56, 2, 0.35),
    SocialFineHandGesture.THUMBS_DOWN: FineHandGesturePolicy(0.44, 2, 0.35),
    SocialFineHandGesture.PEACE_SIGN: FineHandGesturePolicy(0.60, 1, 0.40),
}


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
            decision = DisplayGestureEmojiDecision(reason="no_supported_live_gesture")
            self._trace_ack_decision(observation=observation, candidate=None, decision=decision)
            return decision

        if self._pending_key == candidate.key:
            self._pending_count += 1
        else:
            self._pending_key = candidate.key
            self._pending_count = 1
        self._pending_seen_at = observed_at

        if self._pending_count < candidate.confirm_samples:
            decision = DisplayGestureEmojiDecision(reason="awaiting_live_gesture_confirmation")
            self._trace_ack_decision(observation=observation, candidate=candidate, decision=decision)
            return decision

        if (
            self._last_emitted_key == candidate.key
            and self._last_emitted_at is not None
            and (observed_at - self._last_emitted_at) < candidate.hold_s
        ):
            decision = DisplayGestureEmojiDecision(reason="live_gesture_cooldown")
            self._trace_ack_decision(observation=observation, candidate=candidate, decision=decision)
            return decision

        self._last_emitted_key = candidate.key
        self._last_emitted_at = observed_at
        self._trace_ack_decision(observation=observation, candidate=candidate, decision=candidate.decision)
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
        fallback_policy = _DISPLAY_ACK_FALLBACK_POLICIES.get(
            fine_gesture,
            FineHandGesturePolicy(0.72, 1, 0.35),
        )
        policy = self.calibration.fine_hand_policy(
            fine_gesture,
            fallback_min_confidence=fallback_policy.min_confidence,
            fallback_confirm_samples=fallback_policy.confirm_samples,
            fallback_hold_s=fallback_policy.hold_s,
        )
        effective_policy = FineHandGesturePolicy(
            min_confidence=min(policy.min_confidence, fallback_policy.min_confidence),
            confirm_samples=max(policy.confirm_samples, fallback_policy.confirm_samples),
            hold_s=policy.hold_s,
        )
        if confidence < effective_policy.min_confidence:
            return None
        decision = decision_for_fine_hand_gesture(fine_gesture)
        if not decision.active:
            return None
        return _GestureCandidate(
            key=f"fine:{fine_gesture.value}",
            decision=decision,
            confirm_samples=effective_policy.confirm_samples,
            hold_s=effective_policy.hold_s,
        )

    def _reset_pending_if_stale(self, observed_at: float) -> None:
        seen_at = self._pending_seen_at
        if seen_at is None or (observed_at - seen_at) < _PENDING_RESET_AFTER_S:
            return
        self._pending_key = None
        self._pending_count = 0
        self._pending_seen_at = None

    def _trace_ack_decision(
        self,
        *,
        observation: SocialVisionObservation,
        candidate: _GestureCandidate | None,
        decision: DisplayGestureEmojiDecision,
    ) -> None:
        """Emit one bounded decision ledger entry for the ack lane."""

        workflow_decision(
            msg="gesture_ack_lane_observe",
            question="Should the HDMI ack lane emit a user-facing gesture acknowledgement now?",
            selected={
                "id": decision.reason,
                "summary": (
                    f"Emit {decision.symbol.value}."
                    if decision.active
                    else "Do not emit a user-facing gesture acknowledgement."
                ),
            },
            options=[
                {"id": "emit", "summary": "Emit the current gesture acknowledgement immediately."},
                {"id": "awaiting_live_gesture_confirmation", "summary": "Keep the current gesture pending until confirmation."},
                {"id": "live_gesture_cooldown", "summary": "Suppress a repeated gesture during cooldown."},
                {"id": "no_supported_live_gesture", "summary": "Ignore the frame because no supported gesture survived gating."},
            ],
            context={
                "observed_fine_hand_gesture": observation.fine_hand_gesture.value,
                "observed_fine_hand_confidence": _coerce_confidence(observation.fine_hand_gesture_confidence),
                "observed_gesture_event": observation.gesture_event.value,
                "observed_gesture_confidence": _coerce_confidence(observation.gesture_confidence),
                "candidate_key": None if candidate is None else candidate.key,
                "pending_key": self._pending_key,
                "pending_count": self._pending_count,
                "last_emitted_key": self._last_emitted_key,
            },
            confidence=_coerce_confidence(observation.fine_hand_gesture_confidence or observation.gesture_confidence),
            guardrails=["gesture_ack_lane"],
            kpi_impact_estimate={"latency": "low", "user_feedback": "high"},
        )


def _coerce_confidence(value: object) -> float:
    """Clamp one optional gesture confidence into a bounded float."""

    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
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
