"""Consume authoritative gesture-stream activations for visual wakeup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialVisionObservation
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative


_DEFAULT_TRIGGER_GESTURE = SocialFineHandGesture.PEACE_SIGN
_DEFAULT_REQUEST_SOURCE = "gesture"


@dataclass(frozen=True, slots=True)
class GestureWakeupDecision:
    """Describe one optional visual listen-start trigger."""

    active: bool = False
    reason: str = "inactive"
    trigger_gesture: SocialFineHandGesture = _DEFAULT_TRIGGER_GESTURE
    observed_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE
    confidence: float = 0.0
    request_source: str = _DEFAULT_REQUEST_SOURCE


class GestureWakeupLane:
    """Keep visual wake tied to the authoritative gesture stream only."""

    def __init__(
        self,
        *,
        enabled: bool,
        trigger_gesture: SocialFineHandGesture,
        cooldown_s: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.trigger_gesture = trigger_gesture
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._last_seen_activation_token: int | None = None
        self._last_triggered_at: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "GestureWakeupLane":
        """Build one visual wake lane from runtime config."""

        return cls(
            enabled=bool(getattr(config, "gesture_wakeup_enabled", True)),
            trigger_gesture=_coerce_trigger_gesture(
                getattr(config, "gesture_wakeup_trigger", _DEFAULT_TRIGGER_GESTURE.value)
            ),
            cooldown_s=_coerce_non_negative_float(
                getattr(config, "gesture_wakeup_cooldown_s", 3.0),
                default=3.0,
            ),
        )

    def observe(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> GestureWakeupDecision:
        """Return one visual wake decision for the current live gesture frame."""

        if not self.enabled:
            decision = GestureWakeupDecision(
                reason="gesture_wakeup_disabled",
                trigger_gesture=self.trigger_gesture,
            )
            self._trace_wakeup_decision(
                observation=observation,
                decision=decision,
            )
            return decision

        activation_token = self._candidate_activation_token(observation)
        observed_gesture = observation.fine_hand_gesture
        confidence = _coerce_confidence(observation.fine_hand_gesture_confidence)
        if activation_token is None:
            decision = GestureWakeupDecision(
                reason=(
                    "gesture_wakeup_not_authoritative"
                    if not gesture_stream_authoritative(observation)
                    else "no_gesture_wakeup_candidate"
                ),
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(
                observation=observation,
                decision=decision,
            )
            return decision

        if activation_token == self._last_seen_activation_token:
            decision = GestureWakeupDecision(
                reason="gesture_wakeup_already_active",
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(
                observation=observation,
                decision=decision,
            )
            return decision

        self._last_seen_activation_token = activation_token
        if (
            self._last_triggered_at is not None
            and (observed_at - self._last_triggered_at) < self.cooldown_s
        ):
            decision = GestureWakeupDecision(
                reason="gesture_wakeup_cooldown",
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(
                observation=observation,
                decision=decision,
            )
            return decision

        self._last_triggered_at = observed_at
        decision = GestureWakeupDecision(
            active=True,
            reason=f"gesture_wakeup:{self.trigger_gesture.value}",
            trigger_gesture=self.trigger_gesture,
            observed_gesture=observed_gesture,
            confidence=confidence,
        )
        self._trace_wakeup_decision(
            observation=observation,
            decision=decision,
        )
        return decision

    def _candidate_activation_token(self, observation: SocialVisionObservation) -> int | None:
        if not gesture_stream_authoritative(observation):
            return None
        if observation.fine_hand_gesture != self.trigger_gesture:
            return None
        stream = gesture_stream(observation)
        if stream is None:
            return None
        return _coerce_optional_non_negative_int(stream.activation_token)

    def _trace_wakeup_decision(
        self,
        *,
        observation: SocialVisionObservation,
        decision: GestureWakeupDecision,
    ) -> None:
        """Emit one bounded decision ledger entry for the wake lane."""

        stream = gesture_stream(observation)
        workflow_decision(
            msg="gesture_wakeup_lane_observe",
            question="Should the visual wake lane request a new voice turn now?",
            selected={
                "id": decision.reason,
                "summary": (
                    "Dispatch a wake request now."
                    if decision.active
                    else "Do not dispatch a wake request from this frame."
                ),
            },
            options=[
                {"id": "dispatch", "summary": "Dispatch the configured wake gesture immediately."},
                {"id": "gesture_wakeup_disabled", "summary": "Skip wakeup because the feature is disabled."},
                {"id": "gesture_wakeup_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
                {"id": "gesture_wakeup_already_active", "summary": "Do not re-dispatch while the same authoritative activation token remains active."},
                {"id": "gesture_wakeup_cooldown", "summary": "Suppress a repeated wake gesture during cooldown."},
                {"id": "no_gesture_wakeup_candidate", "summary": "Ignore the frame because the configured trigger gesture is absent."},
            ],
            context={
                "gesture_stream_authoritative": gesture_stream_authoritative(observation),
                "trigger_gesture": self.trigger_gesture.value,
                "observed_gesture": observation.fine_hand_gesture.value,
                "observed_confidence": _coerce_confidence(observation.fine_hand_gesture_confidence),
                "stream_activation_key": None if stream is None else stream.activation_key,
                "stream_activation_token": None if stream is None else stream.activation_token,
                "stream_activation_rising": None if stream is None else stream.activation_rising,
                "last_seen_activation_token": self._last_seen_activation_token,
                "cooldown_s": self.cooldown_s,
            },
            confidence=decision.confidence,
            guardrails=["gesture_wakeup_lane"],
            kpi_impact_estimate={"latency": "low", "voice_turn": "high"},
        )


def _coerce_trigger_gesture(value: object) -> SocialFineHandGesture:
    """Normalize the configured trigger token into one supported hand gesture."""

    text = " ".join(str(value or "").strip().lower().replace("-", " ").split())
    if not text:
        return _DEFAULT_TRIGGER_GESTURE
    for member in SocialFineHandGesture:
        if member.value == text.replace(" ", "_"):
            return member
    return _DEFAULT_TRIGGER_GESTURE


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


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    """Clamp one optional cooldown to a finite non-negative float."""

    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric < 0.0:
        return default
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


__all__ = ["GestureWakeupDecision", "GestureWakeupLane"]
