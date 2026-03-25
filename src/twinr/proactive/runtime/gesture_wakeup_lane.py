"""Translate one configured hand gesture into a bounded listen-start request.

This module keeps visual hands-free turn entry separate from both HDMI emoji
acknowledgement and the broader social camera surface. It accepts only one
configured fine-hand symbol, applies bounded confirmation and cooldown rules,
and returns a ready-to-dispatch wake decision without touching eye-follow,
emoji publish policy, or workflow orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialVisionObservation
from ..social.gesture_calibration import GestureCalibrationProfile


_DEFAULT_TRIGGER_GESTURE = SocialFineHandGesture.PEACE_SIGN
_DEFAULT_REQUEST_SOURCE = "gesture"
_PENDING_RESET_AFTER_S = 0.9


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
    """Keep the visual wake trigger independent from display and social policy."""

    def __init__(
        self,
        *,
        enabled: bool,
        trigger_gesture: SocialFineHandGesture,
        cooldown_s: float,
        calibration: GestureCalibrationProfile,
    ) -> None:
        self.enabled = bool(enabled)
        self.trigger_gesture = trigger_gesture
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.calibration = calibration
        self._pending_count = 0
        self._pending_seen_at: float | None = None
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
            calibration=GestureCalibrationProfile.from_runtime_config(config),
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
            self._trace_wakeup_decision(observation=observation, decision=decision)
            return decision

        observed_gesture = observation.fine_hand_gesture
        confidence = _coerce_confidence(observation.fine_hand_gesture_confidence)
        policy = self.calibration.fine_hand_policy(
            self.trigger_gesture,
            fallback_min_confidence=0.78,
            fallback_confirm_samples=1,
            fallback_hold_s=0.40,
        )
        if observed_gesture != self.trigger_gesture:
            self._reset_pending_if_stale(observed_at)
            decision = GestureWakeupDecision(
                reason="no_gesture_wakeup_candidate",
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(observation=observation, decision=decision)
            return decision
        if confidence < policy.min_confidence:
            self._reset_pending_if_stale(observed_at)
            decision = GestureWakeupDecision(
                reason="gesture_wakeup_low_confidence",
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(observation=observation, decision=decision)
            return decision

        self._pending_seen_at = observed_at
        self._pending_count += 1
        if self._pending_count < policy.confirm_samples:
            decision = GestureWakeupDecision(
                reason="awaiting_gesture_wakeup_confirmation",
                trigger_gesture=self.trigger_gesture,
                observed_gesture=observed_gesture,
                confidence=confidence,
            )
            self._trace_wakeup_decision(observation=observation, decision=decision)
            return decision

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
            self._trace_wakeup_decision(observation=observation, decision=decision)
            return decision

        self._last_triggered_at = observed_at
        self._pending_count = 0
        self._pending_seen_at = observed_at
        decision = GestureWakeupDecision(
            active=True,
            reason=f"gesture_wakeup:{self.trigger_gesture.value}",
            trigger_gesture=self.trigger_gesture,
            observed_gesture=observed_gesture,
            confidence=confidence,
        )
        self._trace_wakeup_decision(observation=observation, decision=decision)
        return decision

    def _reset_pending_if_stale(self, observed_at: float) -> None:
        """Clear partial confirmation state after a bounded gap."""

        seen_at = self._pending_seen_at
        if seen_at is None or (observed_at - seen_at) < _PENDING_RESET_AFTER_S:
            return
        self._pending_count = 0
        self._pending_seen_at = None

    def _trace_wakeup_decision(
        self,
        *,
        observation: SocialVisionObservation,
        decision: GestureWakeupDecision,
    ) -> None:
        """Emit one bounded decision ledger entry for the wake lane."""

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
                {"id": "awaiting_gesture_wakeup_confirmation", "summary": "Keep the wake gesture pending until confirmation."},
                {"id": "gesture_wakeup_low_confidence", "summary": "Reject the wake gesture because confidence is too low."},
                {"id": "gesture_wakeup_cooldown", "summary": "Suppress a repeated wake gesture during cooldown."},
                {"id": "no_gesture_wakeup_candidate", "summary": "Ignore the frame because the trigger gesture is absent."},
            ],
            context={
                "trigger_gesture": self.trigger_gesture.value,
                "observed_gesture": observation.fine_hand_gesture.value,
                "observed_confidence": _coerce_confidence(observation.fine_hand_gesture_confidence),
                "pending_count": self._pending_count,
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
    normalized = text.replace(" ", "_")
    aliases = {
        "peace": SocialFineHandGesture.PEACE_SIGN,
        "victory": SocialFineHandGesture.PEACE_SIGN,
    }
    gesture = aliases.get(normalized)
    if gesture is not None:
        return gesture
    try:
        parsed = SocialFineHandGesture(normalized)
    except ValueError:
        return _DEFAULT_TRIGGER_GESTURE
    if parsed in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
        return _DEFAULT_TRIGGER_GESTURE
    return parsed


def _coerce_confidence(value: object) -> float:
    """Clamp one optional confidence score into a bounded ratio."""

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


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    """Return one finite non-negative float with fallback."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric < 0.0:
        return default
    return numeric


__all__ = ["GestureWakeupDecision", "GestureWakeupLane"]
