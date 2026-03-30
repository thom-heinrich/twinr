# CHANGELOG: 2026-03-29
# BUG-1: Fixed workflow_decision selected.id for active wakeups; the emitted option id is now the stable value "dispatch" instead of a free-form reason string.
# BUG-2: Fixed false visual wakeups from replayed or post-restart frames by honoring authoritative activation_rising when the stream exposes it.
# BUG-3: Fixed duplicate suppression errors across upstream stream resets by deduplicating on (activation_key, activation_token), not token alone.
# BUG-4: Fixed cooldown instability under wall-clock jumps or invalid observed_at values by using a local monotonic clock for elapsed-time checks.
# BUG-5: Fixed async callback race conditions by protecting mutable lane state with a lock.
# SEC-1: Fixed practical privacy/availability risks from stringly typed config (for example "false" no longer enables wakeup) and per-frame trace flooding on Raspberry Pi deployments.
# IMP-1: Added configurable confidence gating for the authoritative gesture stream to match 2026 live-stream gesture-recognition practice.
# IMP-2: Upgraded forensics emission to low-cardinality, rate-limited structured events with bounded context fields.
# IMP-3: Hardened trigger-gesture parsing to accept enum members, names, values, and common config spellings without silently falling back.
# BUG-6: Reject visual wake dispatch when the gesture stream lacks current hand evidence or only resolves through stale/slow rescue sources.

"""Consume authoritative gesture-stream activations for visual wakeup."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision

from ..social.engine import SocialFineHandGesture, SocialVisionObservation
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative
from .gesture_stream_guards import evaluate_gesture_stream_user_intent


_DEFAULT_TRIGGER_GESTURE = SocialFineHandGesture.PEACE_SIGN
_DEFAULT_REQUEST_SOURCE = "gesture"
_DEFAULT_COOLDOWN_S = 3.0
_DEFAULT_MIN_CONFIDENCE = 0.5
_DEFAULT_TRACE_MIN_INTERVAL_S = 2.0
_MAX_TRACE_TEXT_LEN = 128
_WORKFLOW_DECISION_OPTIONS = [
    {"id": "dispatch", "summary": "Dispatch the configured wake gesture immediately."},
    {"id": "gesture_wakeup_disabled", "summary": "Skip wakeup because the feature is disabled."},
    {"id": "gesture_wakeup_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
    {"id": "gesture_wakeup_not_rising", "summary": "Ignore the frame because the authoritative activation was not on a rising edge."},
    {"id": "gesture_wakeup_low_confidence", "summary": "Ignore the frame because the authoritative gesture confidence is below the configured minimum."},
    {"id": "gesture_wakeup_missing_hand_evidence", "summary": "Ignore the frame because the fast lane has no current hand-near-camera evidence."},
    {"id": "gesture_wakeup_unsafe_source", "summary": "Ignore the frame because the gesture only survived through a stale or slow rescue source."},
    {"id": "gesture_wakeup_already_active", "summary": "Do not re-dispatch while the same authoritative activation remains active."},
    {"id": "gesture_wakeup_cooldown", "summary": "Suppress a repeated wake gesture during cooldown."},
    {"id": "no_gesture_wakeup_candidate", "summary": "Ignore the frame because the configured trigger gesture is absent or incomplete."},
]


@dataclass(frozen=True, slots=True)
class GestureWakeupDecision:
    """Describe one optional visual listen-start trigger."""

    active: bool = False
    reason: str = "inactive"
    trigger_gesture: SocialFineHandGesture = _DEFAULT_TRIGGER_GESTURE
    observed_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE
    confidence: float = 0.0
    request_source: str = _DEFAULT_REQUEST_SOURCE


@dataclass(frozen=True, slots=True)
class _ActivationFingerprint:
    """Stable identity for one authoritative gesture activation."""

    activation_key: str | None
    activation_token: int


class GestureWakeupLane:
    """Keep visual wake tied to the authoritative gesture stream only."""

    def __init__(
        self,
        *,
        enabled: bool,
        trigger_gesture: SocialFineHandGesture,
        cooldown_s: float,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        require_rising: bool = True,
        trace_state_changes_only: bool = True,
        trace_min_interval_s: float = _DEFAULT_TRACE_MIN_INTERVAL_S,
    ) -> None:
        self.enabled = _coerce_bool(enabled, default=True)
        self.trigger_gesture = _coerce_trigger_gesture(trigger_gesture)
        self.cooldown_s = _coerce_non_negative_float(cooldown_s, default=_DEFAULT_COOLDOWN_S)
        self.min_confidence = _coerce_probability(min_confidence, default=_DEFAULT_MIN_CONFIDENCE)
        self.require_rising = _coerce_bool(require_rising, default=True)
        self.trace_state_changes_only = _coerce_bool(trace_state_changes_only, default=True)
        self.trace_min_interval_s = _coerce_non_negative_float(
            trace_min_interval_s,
            default=_DEFAULT_TRACE_MIN_INTERVAL_S,
        )
        self._cooldown_ns = _seconds_to_ns(self.cooldown_s)
        self._trace_min_interval_ns = _seconds_to_ns(self.trace_min_interval_s)
        self._last_seen_activation: _ActivationFingerprint | None = None
        self._last_triggered_clock_ns: int | None = None
        self._last_trace_signature: tuple[object, ...] | None = None
        self._last_trace_clock_ns: int | None = None
        self._suppressed_trace_count = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "GestureWakeupLane":
        """Build one visual wake lane from runtime config."""

        min_confidence_raw = getattr(config, "gesture_wakeup_min_confidence", _DEFAULT_MIN_CONFIDENCE)
        # BREAKING: the default minimum confidence is now 0.5. Set
        # gesture_wakeup_min_confidence=0.0 to restore the legacy behavior.
        return cls(
            enabled=_coerce_bool(getattr(config, "gesture_wakeup_enabled", True), default=True),
            trigger_gesture=_coerce_trigger_gesture(
                getattr(config, "gesture_wakeup_trigger", _DEFAULT_TRIGGER_GESTURE)
            ),
            cooldown_s=_coerce_non_negative_float(
                getattr(config, "gesture_wakeup_cooldown_s", _DEFAULT_COOLDOWN_S),
                default=_DEFAULT_COOLDOWN_S,
            ),
            min_confidence=_coerce_probability(
                min_confidence_raw,
                default=_DEFAULT_MIN_CONFIDENCE,
            ),
            require_rising=_coerce_bool(
                getattr(config, "gesture_wakeup_require_rising", True),
                default=True,
            ),
            trace_state_changes_only=_coerce_bool(
                getattr(config, "gesture_wakeup_trace_state_changes_only", True),
                default=True,
            ),
            trace_min_interval_s=_coerce_non_negative_float(
                getattr(config, "gesture_wakeup_trace_min_interval_s", _DEFAULT_TRACE_MIN_INTERVAL_S),
                default=_DEFAULT_TRACE_MIN_INTERVAL_S,
            ),
        )

    def observe(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> GestureWakeupDecision:
        """Return one visual wake decision for the current live gesture frame."""

        now_ns = time.monotonic_ns()
        observed_gesture = observation.fine_hand_gesture
        confidence = _coerce_confidence(observation.fine_hand_gesture_confidence)
        fingerprint, candidate_reason = self._candidate_activation(
            observation=observation,
            confidence=confidence,
        )

        with self._lock:
            if not self.enabled:
                decision = GestureWakeupDecision(
                    reason="gesture_wakeup_disabled",
                    trigger_gesture=self.trigger_gesture,
                    observed_gesture=observed_gesture,
                    confidence=confidence,
                )
            elif fingerprint is None:
                decision = GestureWakeupDecision(
                    reason=candidate_reason,
                    trigger_gesture=self.trigger_gesture,
                    observed_gesture=observed_gesture,
                    confidence=confidence,
                )
            elif fingerprint == self._last_seen_activation:
                decision = GestureWakeupDecision(
                    reason="gesture_wakeup_already_active",
                    trigger_gesture=self.trigger_gesture,
                    observed_gesture=observed_gesture,
                    confidence=confidence,
                )
            else:
                self._last_seen_activation = fingerprint
                if self._is_in_cooldown(now_ns):
                    decision = GestureWakeupDecision(
                        reason="gesture_wakeup_cooldown",
                        trigger_gesture=self.trigger_gesture,
                        observed_gesture=observed_gesture,
                        confidence=confidence,
                    )
                else:
                    self._last_triggered_clock_ns = now_ns
                    decision = GestureWakeupDecision(
                        active=True,
                        reason=f"gesture_wakeup:{self.trigger_gesture.value}",
                        trigger_gesture=self.trigger_gesture,
                        observed_gesture=observed_gesture,
                        confidence=confidence,
                    )

            trace_payload = self._prepare_trace_payload_locked(
                observation=observation,
                decision=decision,
                observed_at=observed_at,
                now_ns=now_ns,
                fingerprint=fingerprint,
            )

        if trace_payload is not None:
            workflow_decision(**trace_payload)
        return decision

    def _candidate_activation(
        self,
        *,
        observation: SocialVisionObservation,
        confidence: float,
    ) -> tuple[_ActivationFingerprint | None, str]:
        authoritative = gesture_stream_authoritative(observation)
        if not authoritative:
            return None, "gesture_wakeup_not_authoritative"
        if observation.fine_hand_gesture != self.trigger_gesture:
            return None, "no_gesture_wakeup_candidate"
        if confidence < self.min_confidence:
            return None, "gesture_wakeup_low_confidence"

        stream = gesture_stream(observation)
        if stream is None:
            return None, "no_gesture_wakeup_candidate"

        activation_token = _coerce_optional_non_negative_int(getattr(stream, "activation_token", None))
        if activation_token is None:
            return None, "no_gesture_wakeup_candidate"

        activation_rising = _coerce_optional_bool(getattr(stream, "activation_rising", None))
        if self.require_rising and activation_rising is False:
            return None, "gesture_wakeup_not_rising"

        guard_reason = _wakeup_guard_block_reason(observation)
        if guard_reason is not None:
            return None, guard_reason

        return (
            _ActivationFingerprint(
                activation_key=_sanitize_small_text(getattr(stream, "activation_key", None)),
                activation_token=activation_token,
            ),
            "dispatch",
        )

    def _is_in_cooldown(self, now_ns: int) -> bool:
        if self._last_triggered_clock_ns is None:
            return False
        return (now_ns - self._last_triggered_clock_ns) < self._cooldown_ns

    def _prepare_trace_payload_locked(
        self,
        *,
        observation: SocialVisionObservation,
        decision: GestureWakeupDecision,
        observed_at: float,
        now_ns: int,
        fingerprint: _ActivationFingerprint | None,
    ) -> dict[str, object] | None:
        """Prepare one bounded decision ledger entry for the wake lane."""

        stream = gesture_stream(observation)
        user_intent_guard = evaluate_gesture_stream_user_intent(observation)
        signature = (
            decision.active,
            decision.reason,
            decision.observed_gesture,
            decision.trigger_gesture,
            None if fingerprint is None else fingerprint.activation_key,
            None if fingerprint is None else fingerprint.activation_token,
            _coerce_optional_bool(getattr(stream, "activation_rising", None)) if stream is not None else None,
            gesture_stream_authoritative(observation),
        )
        if not self._should_trace_locked(signature=signature, decision=decision, now_ns=now_ns):
            self._suppressed_trace_count += 1
            return None

        suppressed_trace_count = self._suppressed_trace_count
        self._suppressed_trace_count = 0
        self._last_trace_signature = signature
        self._last_trace_clock_ns = now_ns

        return {
            "msg": "gesture_wakeup_lane_observe",
            "question": "Should the visual wake lane request a new voice turn now?",
            "selected": {
                "id": "dispatch" if decision.active else decision.reason,
                "summary": (
                    "Dispatch a wake request now."
                    if decision.active
                    else "Do not dispatch a wake request from this frame."
                ),
            },
            "options": _WORKFLOW_DECISION_OPTIONS,
            "context": {
                "gesture_stream_authoritative": gesture_stream_authoritative(observation),
                "trigger_gesture": decision.trigger_gesture.value,
                "observed_gesture": decision.observed_gesture.value,
                "observed_confidence": decision.confidence,
                "min_confidence": self.min_confidence,
                "request_source": decision.request_source,
                "selected_reason": decision.reason,
                "stream_activation_key": _sanitize_small_text(
                    None if stream is None else getattr(stream, "activation_key", None)
                ),
                "stream_activation_token": _coerce_optional_non_negative_int(
                    None if stream is None else getattr(stream, "activation_token", None)
                ),
                "stream_activation_rising": _coerce_optional_bool(
                    None if stream is None else getattr(stream, "activation_rising", None)
                ),
                "stream_resolved_source": _sanitize_small_text(
                    None if stream is None else getattr(stream, "resolved_source", None)
                ),
                "stream_hand_or_object_near_camera": _coerce_optional_bool(
                    None if stream is None else getattr(stream, "hand_or_object_near_camera", None)
                ),
                "user_intent_guard_reason": user_intent_guard.reason,
                "last_seen_activation_key": (
                    None if self._last_seen_activation is None else self._last_seen_activation.activation_key
                ),
                "last_seen_activation_token": (
                    None if self._last_seen_activation is None else self._last_seen_activation.activation_token
                ),
                "cooldown_s": self.cooldown_s,
                "cooldown_remaining_s": self._cooldown_remaining_s(now_ns),
                "observed_at": _coerce_optional_finite_float(observed_at),
                "suppressed_trace_count_since_last_emit": suppressed_trace_count,
            },
            "confidence": decision.confidence,
            "guardrails": ["gesture_wakeup_lane"],
            "kpi_impact_estimate": {"latency": "low", "voice_turn": "high"},
        }

    def _cooldown_remaining_s(self, now_ns: int) -> float:
        if self._last_triggered_clock_ns is None:
            return 0.0
        remaining_ns = self._cooldown_ns - (now_ns - self._last_triggered_clock_ns)
        if remaining_ns <= 0:
            return 0.0
        return remaining_ns / 1_000_000_000.0

    def _should_trace_locked(
        self,
        *,
        signature: tuple[object, ...],
        decision: GestureWakeupDecision,
        now_ns: int,
    ) -> bool:
        if decision.active:
            return True
        if self._last_trace_clock_ns is None:
            return True
        if not self.trace_state_changes_only:
            return (now_ns - self._last_trace_clock_ns) >= self._trace_min_interval_ns
        if signature != self._last_trace_signature:
            return True
        return (now_ns - self._last_trace_clock_ns) >= self._trace_min_interval_ns


def _wakeup_guard_block_reason(observation: SocialVisionObservation) -> str | None:
    """Map gesture-stream user-intent guards into wake-lane decision reasons."""

    guard = evaluate_gesture_stream_user_intent(observation)
    if guard.allowed:
        return None
    if guard.reason == "missing_hand_evidence":
        return "gesture_wakeup_missing_hand_evidence"
    if guard.reason == "unsafe_resolved_source":
        return "gesture_wakeup_unsafe_source"
    return "no_gesture_wakeup_candidate"


def _coerce_trigger_gesture(value: object) -> SocialFineHandGesture:
    """Normalize the configured trigger token into one supported hand gesture."""

    if isinstance(value, SocialFineHandGesture):
        return value

    text = _normalize_token(value)
    if not text:
        return _DEFAULT_TRIGGER_GESTURE

    for member in SocialFineHandGesture:
        if text in {
            _normalize_token(member),
            _normalize_token(member.name),
            _normalize_token(member.value),
        }:
            return member
    return _DEFAULT_TRIGGER_GESTURE


def _coerce_confidence(value: object) -> float:
    """Clamp one optional gesture confidence into a bounded float."""

    return _coerce_probability(value, default=0.0)


def _coerce_probability(value: object, *, default: float) -> float:
    """Clamp one optional probability-like value into [0.0, 1.0]."""

    numeric = _coerce_optional_finite_float(value)
    if numeric is None:
        return default
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return numeric


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    """Clamp one optional cooldown to a finite non-negative float."""

    numeric = _coerce_optional_finite_float(value)
    if numeric is None or numeric < 0.0:
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


def _coerce_optional_finite_float(value: object) -> float | None:
    """Return one optional finite float."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Normalize one optional bool-like configuration value."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)

    text = _normalize_token(value)
    if text in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "disabled", "disable", "none", "null", ""}:
        return False
    return default


def _coerce_optional_bool(value: object) -> bool | None:
    """Normalize one optional bool-like field without inventing a value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    text = _normalize_token(value)
    if text in {"1", "true", "t", "yes", "y", "on", "rising", "rise"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "falling", "steady"}:
        return False
    return None


def _normalize_token(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_").replace(".", "_")
    return "_".join(part for part in text.split("_") if part)


def _sanitize_small_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= _MAX_TRACE_TEXT_LEN:
        return text
    return f"{text[:_MAX_TRACE_TEXT_LEN - 1]}…"


def _seconds_to_ns(value: float) -> int:
    numeric = _coerce_non_negative_float(value, default=0.0)
    if numeric <= 0.0:
        return 0
    return int(numeric * 1_000_000_000)


__all__ = ["GestureWakeupDecision", "GestureWakeupLane"]
