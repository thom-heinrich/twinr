"""Consume authoritative gesture-stream activations for HDMI emoji acknowledgement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative
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
_REPEAT_HOLD_S = {
    SocialFineHandGesture.THUMBS_UP: 0.35,
    SocialFineHandGesture.THUMBS_DOWN: 0.35,
    SocialFineHandGesture.PEACE_SIGN: 0.40,
}


@dataclass(frozen=True, slots=True)
class _GestureCandidate:
    """Describe one accepted authoritative activation before publish."""

    key: str
    activation_token: int
    decision: DisplayGestureEmojiDecision
    repeat_hold_s: float


class GestureAckLane:
    """Keep HDMI acknowledgement tied to the single authoritative gesture lane."""

    def __init__(self) -> None:
        self._last_seen_activation_token: int | None = None
        self._last_emitted_key: str | None = None
        self._last_emitted_at: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "GestureAckLane":
        """Build one gesture lane from runtime config."""

        del config
        return cls()

    def observe(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> DisplayGestureEmojiDecision:
        """Return one publish-ready acknowledgement decision."""

        candidate = self._candidate_from_observation(observation)
        if candidate is None:
            decision = DisplayGestureEmojiDecision(
                reason=(
                    "gesture_stream_not_authoritative"
                    if not gesture_stream_authoritative(observation)
                    else "no_supported_live_gesture"
                )
            )
            self._trace_ack_decision(
                observation=observation,
                candidate=None,
                decision=decision,
            )
            return decision

        if candidate.activation_token == self._last_seen_activation_token:
            decision = DisplayGestureEmojiDecision(reason="live_gesture_already_active")
            self._trace_ack_decision(
                observation=observation,
                candidate=candidate,
                decision=decision,
            )
            return decision

        self._last_seen_activation_token = candidate.activation_token
        if (
            self._last_emitted_key == candidate.key
            and self._last_emitted_at is not None
            and (observed_at - self._last_emitted_at) < candidate.repeat_hold_s
        ):
            decision = DisplayGestureEmojiDecision(reason="live_gesture_cooldown")
            self._trace_ack_decision(
                observation=observation,
                candidate=candidate,
                decision=decision,
            )
            return decision

        self._last_emitted_key = candidate.key
        self._last_emitted_at = observed_at
        self._trace_ack_decision(
            observation=observation,
            candidate=candidate,
            decision=candidate.decision,
        )
        return candidate.decision

    def _candidate_from_observation(
        self,
        observation: SocialVisionObservation,
    ) -> _GestureCandidate | None:
        if not gesture_stream_authoritative(observation):
            return None
        stream = gesture_stream(observation)
        if stream is None:
            return None
        activation_token = _coerce_optional_non_negative_int(stream.activation_token)
        if activation_token is None:
            return None

        fine_gesture = observation.fine_hand_gesture
        if fine_gesture in _SUPPORTED_FINE_GESTURES:
            decision = decision_for_fine_hand_gesture(fine_gesture)
            if decision.active:
                return _GestureCandidate(
                    key=f"fine:{fine_gesture.value}",
                    activation_token=activation_token,
                    decision=decision,
                    repeat_hold_s=_REPEAT_HOLD_S.get(fine_gesture, 0.35),
                )

        coarse_gesture = observation.gesture_event
        if coarse_gesture in _SUPPORTED_COARSE_GESTURES:
            decision = decision_for_coarse_gesture(
                coarse_gesture,
                motion_priority=True,
            )
            if decision.active:
                return _GestureCandidate(
                    key=f"coarse:{coarse_gesture.value}",
                    activation_token=activation_token,
                    decision=decision,
                    repeat_hold_s=0.35,
                )
        return None

    def _trace_ack_decision(
        self,
        *,
        observation: SocialVisionObservation,
        candidate: _GestureCandidate | None,
        decision: DisplayGestureEmojiDecision,
    ) -> None:
        """Emit one bounded decision ledger entry for the ack lane."""

        stream = gesture_stream(observation)
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
                {"id": "gesture_stream_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
                {"id": "live_gesture_already_active", "summary": "Do not re-emit while the same authoritative activation token remains active."},
                {"id": "live_gesture_cooldown", "summary": "Suppress a repeated gesture during the HDMI repeat hold window."},
                {"id": "no_supported_live_gesture", "summary": "Ignore the frame because the authoritative gesture is unsupported by the HDMI path."},
            ],
            context={
                "gesture_stream_authoritative": gesture_stream_authoritative(observation),
                "observed_fine_hand_gesture": observation.fine_hand_gesture.value,
                "observed_fine_hand_confidence": _coerce_confidence(observation.fine_hand_gesture_confidence),
                "observed_gesture_event": observation.gesture_event.value,
                "observed_gesture_confidence": _coerce_confidence(observation.gesture_confidence),
                "stream_activation_key": None if stream is None else stream.activation_key,
                "stream_activation_token": None if stream is None else stream.activation_token,
                "stream_activation_rising": None if stream is None else stream.activation_rising,
                "candidate_key": None if candidate is None else candidate.key,
                "last_seen_activation_token": self._last_seen_activation_token,
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


def _coerce_optional_non_negative_int(value: object) -> int | None:
    """Return one optional activation token as a non-negative integer."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


__all__ = ["GestureAckLane"]
