# CHANGELOG: 2026-03-29
# BUG-1: Do not consume a new activation token before the repeat-hold window expires; otherwise a fresh gesture that starts during cooldown is dropped forever.
# BUG-2: Use an internal monotonic timeline for cooldown math so wall-clock jumps or out-of-order frames cannot suppress acknowledgements incorrectly.
# BUG-3: Deduplicate on (stream key, activation token, gesture key) instead of token alone, avoiding false suppression when the same token is reused or a gesture is refined mid-activation.
# BUG-4: Make observe() state updates thread-safe and keep telemetry failures from breaking the HDMI acknowledgement path.
# BUG-5: Restore Pi-tuned per-symbol HDMI ack confidence floors so authoritative live thumbs-up/down gestures are not re-rejected by one generic threshold.
# SEC-1: Bound and sanitize forensics trace fields so malformed upstream metadata cannot turn tracing into a log/availability problem on the Pi.
# IMP-1: Add confidence-gated acknowledgement aligned with current edge gesture-recognition practice (temporal tracking + confidence thresholds).
# IMP-2: Prefer authoritative rising-edge metadata when available to reduce duplicate emits on tracker/token churn.
# IMP-3: Rate-limit repeated forensics ledger entries and sanitize trace fields for edge-device robustness.
# BUG-6: Reject user-facing HDMI emoji emits when the authoritative gesture stream lacks current hand evidence or only resolves through stale/slow rescue sources.

"""Consume authoritative gesture-stream activations for HDMI emoji acknowledgement."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation
from ..social.gesture_calibration import (
    DEFAULT_HDMI_ACK_FINE_HAND_POLICIES,
    FineHandGesturePolicy,
    GestureCalibrationProfile,
)
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative
from .display_gesture_emoji import (
    DisplayGestureEmojiDecision,
    decision_for_coarse_gesture,
    decision_for_fine_hand_gesture,
)
from .gesture_stream_guards import evaluate_gesture_stream_user_intent


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
# BREAKING: Acks now default to a 0.50 confidence gate. Set the config override to 0.0 to restore the legacy 'emit on any authoritative label' behaviour.
_DEFAULT_MIN_FINE_CONFIDENCE = 0.50
_DEFAULT_MIN_COARSE_CONFIDENCE = 0.50
_DEFAULT_STALE_ACTIVATION_AFTER_S = 2.00
_DEFAULT_TRACE_REPEAT_INTERVAL_S = 0.75
_DEFAULT_TRACE_STRING_MAX_LEN = 96


@dataclass(frozen=True, slots=True)
class _ActivationIdentity:
    """Uniquely describe one live authoritative activation for ack dedupe."""

    stream_key: object | None
    activation_token: int
    gesture_key: str


@dataclass(frozen=True, slots=True)
class _GestureCandidate:
    """Describe one accepted authoritative activation before publish."""

    key: str
    activation_identity: _ActivationIdentity
    decision: DisplayGestureEmojiDecision
    repeat_hold_s: float
    confidence: float
    min_confidence: float
    activation_rising: bool | None


@dataclass(frozen=True, slots=True)
class _LaneSettings:
    """Runtime policy for one gesture ack lane."""

    min_fine_confidence: float = _DEFAULT_MIN_FINE_CONFIDENCE
    min_coarse_confidence: float = _DEFAULT_MIN_COARSE_CONFIDENCE
    stale_activation_after_s: float = _DEFAULT_STALE_ACTIVATION_AFTER_S
    trace_repeat_interval_s: float = _DEFAULT_TRACE_REPEAT_INTERVAL_S
    max_trace_string_len: int = _DEFAULT_TRACE_STRING_MAX_LEN
    prefer_activation_rising: bool = True
    fine_hand_min_confidence_by_gesture: dict[SocialFineHandGesture, float] | None = None
    supported_fine_gestures: frozenset[SocialFineHandGesture] = _SUPPORTED_FINE_GESTURES
    supported_coarse_gestures: frozenset[SocialGestureEvent] = _SUPPORTED_COARSE_GESTURES


class GestureAckLane:
    """Keep HDMI acknowledgement tied to the single authoritative gesture lane."""

    def __init__(self, *, settings: _LaneSettings | None = None) -> None:
        self._settings = settings or _LaneSettings()
        self._lock = threading.RLock()
        self._last_emitted_activation: _ActivationIdentity | None = None
        self._last_emitted_at_by_key_s: dict[str, float] = {}
        self._last_authoritative_seen_at_s: float | None = None
        self._last_internal_observed_at_s: float | None = None
        self._last_trace_signature: tuple[str, str | None, int | None] | None = None
        self._last_trace_at_s: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "GestureAckLane":
        """Build one gesture lane from runtime config."""

        return cls(settings=_settings_from_config(config))

    def observe(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> DisplayGestureEmojiDecision:
        """Return one publish-ready acknowledgement decision."""

        with self._lock:
            observed_at_s = self._next_internal_now_s(observed_at)
            self._expire_stale_state(observed_at_s)

            authoritative = gesture_stream_authoritative(observation)
            if authoritative:
                self._last_authoritative_seen_at_s = observed_at_s

            candidate, candidate_reason = self._candidate_from_observation(observation)

            if candidate is None:
                decision = DisplayGestureEmojiDecision(reason=candidate_reason)
                trace_payload = self._build_trace_payload(
                    observed_at_s=observed_at_s,
                    observation=observation,
                    candidate=None,
                    candidate_reason=candidate_reason,
                    decision=decision,
                )
            elif candidate.activation_identity == self._last_emitted_activation:
                decision = DisplayGestureEmojiDecision(reason="live_gesture_already_active")
                trace_payload = self._build_trace_payload(
                    observed_at_s=observed_at_s,
                    observation=observation,
                    candidate=candidate,
                    candidate_reason=candidate_reason,
                    decision=decision,
                )
            elif (
                # BREAKING: observe() can now return reason="live_gesture_not_rising"
                # when the authoritative stream explicitly marks the frame as a
                # non-rising continuation and the same gesture was just emitted.
                self._settings.prefer_activation_rising
                and candidate.activation_rising is False
                and self._recently_emitted_same_key(
                    candidate.key,
                    observed_at_s,
                    max(candidate.repeat_hold_s, self._settings.stale_activation_after_s),
                )
            ):
                decision = DisplayGestureEmojiDecision(reason="live_gesture_not_rising")
                trace_payload = self._build_trace_payload(
                    observed_at_s=observed_at_s,
                    observation=observation,
                    candidate=candidate,
                    candidate_reason=candidate_reason,
                    decision=decision,
                )
            elif self._recently_emitted_same_key(
                candidate.key,
                observed_at_s,
                candidate.repeat_hold_s,
            ):
                decision = DisplayGestureEmojiDecision(reason="live_gesture_cooldown")
                trace_payload = self._build_trace_payload(
                    observed_at_s=observed_at_s,
                    observation=observation,
                    candidate=candidate,
                    candidate_reason=candidate_reason,
                    decision=decision,
                )
            else:
                self._last_emitted_activation = candidate.activation_identity
                self._last_emitted_at_by_key_s[candidate.key] = observed_at_s
                decision = candidate.decision
                trace_payload = self._build_trace_payload(
                    observed_at_s=observed_at_s,
                    observation=observation,
                    candidate=candidate,
                    candidate_reason=candidate_reason,
                    decision=decision,
                )

        return self._emit_trace(decision, trace_payload)

    def _candidate_from_observation(
        self,
        observation: SocialVisionObservation,
    ) -> tuple[_GestureCandidate | None, str]:
        if not gesture_stream_authoritative(observation):
            return None, "gesture_stream_not_authoritative"

        stream = gesture_stream(observation)
        if stream is None:
            return None, "no_supported_live_gesture"

        activation_token = _coerce_optional_non_negative_int(getattr(stream, "activation_token", None))
        if activation_token is None:
            return None, "no_supported_live_gesture"

        activation_identity_prefix = _ActivationIdentity(
            stream_key=_identity_component(getattr(stream, "activation_key", None)),
            activation_token=activation_token,
            gesture_key="",
        )
        activation_rising = _coerce_optional_bool(getattr(stream, "activation_rising", None))

        fine_gesture = getattr(observation, "fine_hand_gesture", None)
        fine_confidence = _coerce_confidence(getattr(observation, "fine_hand_gesture_confidence", None))
        if fine_gesture in self._settings.supported_fine_gestures:
            decision = decision_for_fine_hand_gesture(fine_gesture)
            fine_min_confidence = self._fine_hand_min_confidence(fine_gesture)
            if decision.active and fine_confidence >= fine_min_confidence:
                guard_reason = _ack_guard_block_reason(observation)
                if guard_reason is not None:
                    return None, guard_reason
                gesture_key = f"fine:{fine_gesture.value}"
                return _GestureCandidate(
                    key=gesture_key,
                    activation_identity=_ActivationIdentity(
                        stream_key=activation_identity_prefix.stream_key,
                        activation_token=activation_identity_prefix.activation_token,
                        gesture_key=gesture_key,
                    ),
                    decision=decision,
                    repeat_hold_s=_REPEAT_HOLD_S.get(fine_gesture, 0.35),
                    confidence=fine_confidence,
                    min_confidence=fine_min_confidence,
                    activation_rising=activation_rising,
                ), "emit"

        coarse_gesture = getattr(observation, "gesture_event", None)
        coarse_confidence = _coerce_confidence(getattr(observation, "gesture_confidence", None))
        if coarse_gesture in self._settings.supported_coarse_gestures:
            decision = decision_for_coarse_gesture(
                coarse_gesture,
                motion_priority=True,
            )
            if decision.active and coarse_confidence >= self._settings.min_coarse_confidence:
                guard_reason = _ack_guard_block_reason(observation)
                if guard_reason is not None:
                    return None, guard_reason
                gesture_key = f"coarse:{coarse_gesture.value}"
                return _GestureCandidate(
                    key=gesture_key,
                    activation_identity=_ActivationIdentity(
                        stream_key=activation_identity_prefix.stream_key,
                        activation_token=activation_identity_prefix.activation_token,
                        gesture_key=gesture_key,
                    ),
                    decision=decision,
                    repeat_hold_s=0.35,
                    confidence=coarse_confidence,
                    min_confidence=self._settings.min_coarse_confidence,
                    activation_rising=activation_rising,
                ), "emit"

        return None, "no_supported_live_gesture"

    def _fine_hand_min_confidence(self, gesture: SocialFineHandGesture) -> float:
        fine_hand_min_confidence_by_gesture = self._settings.fine_hand_min_confidence_by_gesture
        if fine_hand_min_confidence_by_gesture is None:
            return self._settings.min_fine_confidence
        return fine_hand_min_confidence_by_gesture.get(gesture, self._settings.min_fine_confidence)

    def _recently_emitted_same_key(
        self,
        key: str,
        observed_at_s: float,
        hold_s: float,
    ) -> bool:
        last_emitted_at_s = self._last_emitted_at_by_key_s.get(key)
        if last_emitted_at_s is None:
            return False
        delta_s = observed_at_s - last_emitted_at_s
        if delta_s < 0.0:
            delta_s = 0.0
        return delta_s < hold_s

    def _expire_stale_state(self, observed_at_s: float) -> None:
        """Forget stale activation identity after idle gaps or clock repair."""

        if (
            self._last_authoritative_seen_at_s is not None
            and (observed_at_s - self._last_authoritative_seen_at_s) >= self._settings.stale_activation_after_s
        ):
            self._last_emitted_activation = None

        prune_before_s = observed_at_s - max(
            self._settings.stale_activation_after_s,
            max(_REPEAT_HOLD_S.values(), default=0.35),
        )
        if self._last_emitted_at_by_key_s:
            self._last_emitted_at_by_key_s = {
                key: emitted_at_s
                for key, emitted_at_s in self._last_emitted_at_by_key_s.items()
                if emitted_at_s >= prune_before_s
            }

    def _next_internal_now_s(self, observed_at: object) -> float:
        """Return one internal monotonic timestamp for elapsed-time logic."""

        del observed_at
        now_s = time.monotonic()
        if self._last_internal_observed_at_s is not None and now_s < self._last_internal_observed_at_s:
            now_s = self._last_internal_observed_at_s
        self._last_internal_observed_at_s = now_s
        return now_s

    def _emit_trace(
        self,
        decision: DisplayGestureEmojiDecision,
        trace_payload: dict[str, Any] | None,
    ) -> DisplayGestureEmojiDecision:
        if trace_payload is None:
            return decision
        try:
            workflow_decision(**trace_payload)
        except Exception:
            # Forensics must not take down the user-facing HDMI acknowledgement path.
            pass
        return decision

    def _build_trace_payload(
        self,
        *,
        observed_at_s: float,
        observation: SocialVisionObservation,
        candidate: _GestureCandidate | None,
        candidate_reason: str,
        decision: DisplayGestureEmojiDecision,
    ) -> dict[str, Any] | None:
        """Build one bounded decision ledger entry for the ack lane."""

        stream = gesture_stream(observation)
        user_intent_guard = evaluate_gesture_stream_user_intent(observation)
        signature = (
            decision.reason,
            None if candidate is None else candidate.key,
            None if candidate is None else candidate.activation_identity.activation_token,
        )
        if (
            not decision.active
            and self._last_trace_signature == signature
            and self._last_trace_at_s is not None
            and (observed_at_s - self._last_trace_at_s) < self._settings.trace_repeat_interval_s
        ):
            return None

        self._last_trace_signature = signature
        self._last_trace_at_s = observed_at_s

        observed_fine_confidence = _coerce_confidence(getattr(observation, "fine_hand_gesture_confidence", None))
        observed_coarse_confidence = _coerce_confidence(getattr(observation, "gesture_confidence", None))
        candidate_confidence = None if candidate is None else candidate.confidence
        candidate_min_confidence = None if candidate is None else candidate.min_confidence
        fine_gesture = getattr(observation, "fine_hand_gesture", None)
        effective_fine_threshold = (
            self._fine_hand_min_confidence(fine_gesture)
            if fine_gesture in self._settings.supported_fine_gestures
            else None
        )

        return {
            "msg": "gesture_ack_lane_observe",
            "question": "Should the HDMI ack lane emit a user-facing gesture acknowledgement now?",
            "selected": {
                "id": decision.reason,
                "summary": (
                    f"Emit {decision.symbol.value}."
                    if decision.active
                    else "Do not emit a user-facing gesture acknowledgement."
                ),
            },
            "options": [
                {"id": "emit", "summary": "Emit the current gesture acknowledgement immediately."},
                {"id": "gesture_stream_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
                {"id": "live_gesture_already_active", "summary": "Do not re-emit while the same authoritative activation is still live."},
                {"id": "live_gesture_not_rising", "summary": "Suppress a tracker/token churn duplicate because the authoritative stream did not mark a rising edge."},
                {"id": "live_gesture_cooldown", "summary": "Suppress a repeated gesture during the HDMI repeat hold window."},
                {"id": "live_gesture_missing_hand_evidence", "summary": "Ignore the gesture because the fast lane has no current hand-near-camera evidence."},
                {"id": "live_gesture_unsafe_source", "summary": "Ignore the gesture because it only survived through a stale or slow rescue source."},
                {"id": "no_supported_live_gesture", "summary": "Ignore the frame because the authoritative gesture is unsupported or below the ack confidence gate."},
            ],
            "context": {
                "gesture_stream_authoritative": gesture_stream_authoritative(observation),
                "observed_fine_hand_gesture": _enum_value(getattr(observation, "fine_hand_gesture", None)),
                "observed_fine_hand_confidence": observed_fine_confidence,
                "observed_fine_hand_min_confidence": effective_fine_threshold,
                "observed_gesture_event": _enum_value(getattr(observation, "gesture_event", None)),
                "observed_gesture_confidence": observed_coarse_confidence,
                "min_fine_confidence": self._settings.min_fine_confidence,
                "min_coarse_confidence": self._settings.min_coarse_confidence,
                "stream_activation_key": _sanitize_for_trace(
                    None if stream is None else getattr(stream, "activation_key", None),
                    max_len=self._settings.max_trace_string_len,
                ),
                "stream_activation_token": None if stream is None else _coerce_optional_non_negative_int(getattr(stream, "activation_token", None)),
                "stream_activation_rising": None if stream is None else _coerce_optional_bool(getattr(stream, "activation_rising", None)),
                "stream_resolved_source": _sanitize_for_trace(
                    None if stream is None else getattr(stream, "resolved_source", None),
                    max_len=self._settings.max_trace_string_len,
                ),
                "stream_hand_or_object_near_camera": (
                    None if stream is None else _coerce_optional_bool(getattr(stream, "hand_or_object_near_camera", None))
                ),
                "user_intent_guard_reason": user_intent_guard.reason,
                "candidate_reason": candidate_reason,
                "candidate_key": None if candidate is None else candidate.key,
                "candidate_confidence": candidate_confidence,
                "candidate_min_confidence": candidate_min_confidence,
                "last_emitted_stream_key": None if self._last_emitted_activation is None else _sanitize_for_trace(
                    self._last_emitted_activation.stream_key,
                    max_len=self._settings.max_trace_string_len,
                ),
                "last_emitted_activation_token": None if self._last_emitted_activation is None else self._last_emitted_activation.activation_token,
                "last_emitted_gesture_key": None if self._last_emitted_activation is None else self._last_emitted_activation.gesture_key,
            },
            "confidence": _trace_confidence(candidate_confidence, observed_fine_confidence, observed_coarse_confidence),
            "guardrails": ["gesture_ack_lane"],
            "kpi_impact_estimate": {"latency": "low", "user_feedback": "high"},
        }


def _ack_guard_block_reason(observation: SocialVisionObservation) -> str | None:
    """Map gesture-stream user-intent guards into ack-lane decision reasons."""

    guard = evaluate_gesture_stream_user_intent(observation)
    if guard.allowed:
        return None
    if guard.reason == "missing_hand_evidence":
        return "live_gesture_missing_hand_evidence"
    if guard.reason == "unsafe_resolved_source":
        return "live_gesture_unsafe_source"
    return "no_supported_live_gesture"


def _settings_from_config(config: TwinrConfig | object) -> _LaneSettings:
    """Resolve best-effort lane overrides from runtime config without assuming schema."""

    raw_min_fine_confidence = _config_lookup(
        config,
        "gesture_ack_lane.min_fine_confidence",
        "gesture_ack.min_fine_confidence",
        "social.gesture_ack_lane.min_fine_confidence",
        "gesture_ack_lane_min_fine_confidence",
    )
    min_fine_confidence = _coerce_probability(
        raw_min_fine_confidence,
        default=_DEFAULT_MIN_FINE_CONFIDENCE,
    )
    min_coarse_confidence = _coerce_probability(
        _config_lookup(
            config,
            "gesture_ack_lane.min_coarse_confidence",
            "gesture_ack.min_coarse_confidence",
            "social.gesture_ack_lane.min_coarse_confidence",
            "gesture_ack_lane_min_coarse_confidence",
        ),
        default=_DEFAULT_MIN_COARSE_CONFIDENCE,
    )
    stale_activation_after_s = _coerce_positive_float(
        _config_lookup(
            config,
            "gesture_ack_lane.stale_activation_after_s",
            "gesture_ack.stale_activation_after_s",
            "social.gesture_ack_lane.stale_activation_after_s",
            "gesture_ack_lane_stale_activation_after_s",
        ),
        default=_DEFAULT_STALE_ACTIVATION_AFTER_S,
    )
    trace_repeat_interval_s = _coerce_non_negative_float(
        _config_lookup(
            config,
            "gesture_ack_lane.trace_repeat_interval_s",
            "gesture_ack.trace_repeat_interval_s",
            "social.gesture_ack_lane.trace_repeat_interval_s",
            "gesture_ack_lane_trace_repeat_interval_s",
        ),
        default=_DEFAULT_TRACE_REPEAT_INTERVAL_S,
    )
    max_trace_string_len = _coerce_positive_int(
        _config_lookup(
            config,
            "gesture_ack_lane.max_trace_string_len",
            "gesture_ack.max_trace_string_len",
            "social.gesture_ack_lane.max_trace_string_len",
            "gesture_ack_lane_max_trace_string_len",
        ),
        default=_DEFAULT_TRACE_STRING_MAX_LEN,
    )
    prefer_activation_rising = _coerce_bool(
        _config_lookup(
            config,
            "gesture_ack_lane.prefer_activation_rising",
            "gesture_ack.prefer_activation_rising",
            "social.gesture_ack_lane.prefer_activation_rising",
            "gesture_ack_lane_prefer_activation_rising",
        ),
        default=True,
    )

    supported_fine_gestures = _coerce_supported_fine_gestures(
        _config_lookup(
            config,
            "gesture_ack_lane.supported_fine_gestures",
            "gesture_ack.supported_fine_gestures",
            "social.gesture_ack_lane.supported_fine_gestures",
            "gesture_ack_lane_supported_fine_gestures",
        ),
        default=_SUPPORTED_FINE_GESTURES,
    )
    supported_coarse_gestures = _coerce_supported_coarse_gestures(
        _config_lookup(
            config,
            "gesture_ack_lane.supported_coarse_gestures",
            "gesture_ack.supported_coarse_gestures",
            "social.gesture_ack_lane.supported_coarse_gestures",
            "gesture_ack_lane_supported_coarse_gestures",
        ),
        default=_SUPPORTED_COARSE_GESTURES,
    )
    fine_hand_min_confidence_by_gesture = _resolve_fine_hand_min_confidence_by_gesture(
        config,
        supported_fine_gestures=supported_fine_gestures,
        raw_min_fine_confidence=raw_min_fine_confidence,
        min_fine_confidence=min_fine_confidence,
    )

    return _LaneSettings(
        min_fine_confidence=min_fine_confidence,
        min_coarse_confidence=min_coarse_confidence,
        stale_activation_after_s=stale_activation_after_s,
        trace_repeat_interval_s=trace_repeat_interval_s,
        max_trace_string_len=max_trace_string_len,
        prefer_activation_rising=prefer_activation_rising,
        fine_hand_min_confidence_by_gesture=fine_hand_min_confidence_by_gesture,
        supported_fine_gestures=supported_fine_gestures,
        supported_coarse_gestures=supported_coarse_gestures,
    )


def _resolve_fine_hand_min_confidence_by_gesture(
    config: TwinrConfig | object,
    *,
    supported_fine_gestures: frozenset[SocialFineHandGesture],
    raw_min_fine_confidence: object,
    min_fine_confidence: float,
) -> dict[SocialFineHandGesture, float]:
    """Resolve effective per-symbol HDMI ack floors for the supported fine gestures."""

    if raw_min_fine_confidence is not None:
        return {
            gesture: min_fine_confidence
            for gesture in supported_fine_gestures
        }

    calibration = GestureCalibrationProfile.from_runtime_config(config)
    resolved: dict[SocialFineHandGesture, float] = {}
    for gesture in supported_fine_gestures:
        fallback_policy = DEFAULT_HDMI_ACK_FINE_HAND_POLICIES.get(
            gesture,
            FineHandGesturePolicy(
                min_fine_confidence,
                1,
                _REPEAT_HOLD_S.get(gesture, 0.35),
            ),
        )
        calibrated_policy = calibration.fine_hand_policy(
            gesture,
            fallback_min_confidence=fallback_policy.min_confidence,
            fallback_confirm_samples=fallback_policy.confirm_samples,
            fallback_hold_s=fallback_policy.hold_s,
            fallback_min_visible_s=fallback_policy.min_visible_s,
        )
        resolved[gesture] = min(calibrated_policy.min_confidence, fallback_policy.min_confidence)
    return resolved


def _config_lookup(config: object, *paths: str) -> object | None:
    """Resolve the first matching config path across attr-style or dict-style objects."""

    for path in paths:
        current: object = config
        matched = True
        for segment in path.split("."):
            if current is None:
                matched = False
                break
            if isinstance(current, dict):
                if segment not in current:
                    matched = False
                    break
                current = current[segment]
                continue
            if not hasattr(current, segment):
                matched = False
                break
            current = getattr(current, segment)
        if matched:
            return current
    return None


def _coerce_supported_fine_gestures(
    value: object,
    *,
    default: frozenset[SocialFineHandGesture],
) -> frozenset[SocialFineHandGesture]:
    gestures = _coerce_supported_gestures(value, enum_type=SocialFineHandGesture)
    if gestures is None:
        return default
    return frozenset(
        gesture
        for gesture in gestures
        if decision_for_fine_hand_gesture(gesture).active
    )


def _coerce_supported_coarse_gestures(
    value: object,
    *,
    default: frozenset[SocialGestureEvent],
) -> frozenset[SocialGestureEvent]:
    gestures = _coerce_supported_gestures(value, enum_type=SocialGestureEvent)
    if gestures is None:
        return default
    return frozenset(
        gesture
        for gesture in gestures
        if decision_for_coarse_gesture(gesture, motion_priority=True).active
    )


def _coerce_supported_gestures(
    value: object,
    *,
    enum_type: type,
) -> frozenset[object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = list(value)
    else:
        return None

    resolved: list[object] = []
    for raw_item in raw_items:
        gesture = _coerce_enum_member(enum_type, raw_item)
        if gesture is not None:
            resolved.append(gesture)
    return frozenset(resolved)


def _coerce_enum_member(enum_type: type, value: object) -> object | None:
    if value is None:
        return None
    try:
        if isinstance(value, enum_type):
            return value
    except TypeError:
        return None

    normalized = _normalize_enum_token(value)
    if normalized is None:
        return None

    for member in enum_type:
        if normalized in {
            _normalize_enum_token(getattr(member, "name", None)),
            _normalize_enum_token(getattr(member, "value", None)),
        }:
            return member
    return None


def _normalize_enum_token(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _trace_confidence(
    candidate_confidence: float | None,
    observed_fine_confidence: float,
    observed_coarse_confidence: float,
) -> float:
    if candidate_confidence is not None:
        return candidate_confidence
    return max(observed_fine_confidence, observed_coarse_confidence, 0.0)


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _sanitize_for_trace(value: object, *, max_len: int) -> object:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    text = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else " " for ch in text).strip()
    if len(text) > max_len:
        return f"{text[: max_len - 1]}…"
    return text


def _identity_component(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, bytes, bytearray)):
        return _sanitize_for_trace(value, max_len=256)
    try:
        hash(value)
    except TypeError:
        return _sanitize_for_trace(repr(value), max_len=256)
    return value


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


def _coerce_optional_finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_probability(value: object, *, default: float) -> float:
    numeric = _coerce_optional_finite_float(value)
    if numeric is None:
        return default
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return numeric


def _coerce_positive_float(value: object, *, default: float) -> float:
    numeric = _coerce_optional_finite_float(value)
    if numeric is None or numeric <= 0.0:
        return default
    return numeric


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    numeric = _coerce_optional_finite_float(value)
    if numeric is None or numeric < 0.0:
        return default
    return numeric


def _coerce_positive_int(value: object, *, default: int) -> int:
    numeric = _coerce_optional_non_negative_int(value)
    if numeric is None or numeric <= 0:
        return default
    return numeric


def _coerce_bool(value: object, *, default: bool) -> bool:
    coerced = _coerce_optional_bool(value)
    if coerced is None:
        return default
    return coerced


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8", "ignore")
        except Exception:
            return None
    else:
        text = str(value)
    normalized = text.strip().casefold()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _coerce_optional_non_negative_int(value: object) -> int | None:
    """Return one optional activation token as a non-negative integer."""

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 0 else None

    if isinstance(value, float):
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            return None
        return int(value)

    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8", "strict")
        except Exception:
            return None
    else:
        text = str(value)

    text = text.strip()
    if not text:
        return None

    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        return None
    return int(numeric)


__all__ = ["GestureAckLane"]
