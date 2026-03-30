"""Gesture stabilization helpers for the camera surface."""

# CHANGELOG: 2026-03-29
# BUG-1: Prevented synthetic held fine-hand gestures from retriggering
#        rising-edge emits every cooldown interval during brief dropouts.
# BUG-2: Sanitized non-monotonic/poisoned timestamps so cooldown and hold logic
#        no longer sticks after wall-clock rollback, jitter, or one bad frame.
# SEC-1: Hardened authoritative token parsing to accept only primitive integral
#        inputs and fail closed on malformed upstream payloads.
# IMP-1: Made gesture state transitions atomic across async callback threads.
# IMP-2: Upgraded explicit fine-hand confirmation to adaptive time-aware
#        confirmation with confidence-aware hold decay for Pi live streams.

from __future__ import annotations

import math
import threading
import time
from numbers import Integral
from typing import Any

from ..engine import SocialFineHandGesture, SocialGestureEvent
from ..perception_stream import gesture_stream
from .coercion import (
    coerce_fine_hand_gesture,
    coerce_gesture_event,
    coerce_optional_ratio,
    coerce_timestamp,
)

EXPLICIT_FINE_HAND_GESTURES = frozenset(
    {
        SocialFineHandGesture.THUMBS_UP,
        SocialFineHandGesture.THUMBS_DOWN,
        SocialFineHandGesture.POINTING,
        SocialFineHandGesture.PEACE_SIGN,
        SocialFineHandGesture.OK_SIGN,
        SocialFineHandGesture.MIDDLE_FINGER,
    }
)

_DEFAULT_GESTURE_STREAM_INTERVAL_S = 1.0 / 15.0
_MIN_TIME_AWARE_CONFIRM_S = 0.04
_MAX_TIME_AWARE_CONFIRM_S = 0.20
_MAX_REASONABLE_STREAM_GAP_S = 1.50
_SMALL_TIMESTAMP_ROLLBACK_S = 0.05
_HELD_CONFIDENCE_DECAY = 0.50
_WEAK_COLLAPSE_RATIO = 0.75


class ProactiveCameraGestureMixin:
    """Resolve coarse and fine gesture state without changing semantics."""

    def _resolve_gesture(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        gesture_event: object,
        gesture_confidence: object,
        temporal_authoritative: bool = False,
        activation_token: object = None,
    ) -> tuple[SocialGestureEvent, bool, float | None, bool, bool]:
        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()

            now = self._sanitize_observed_at(observed_at)
            self._update_observed_interval(now)

            if inspected:
                event = coerce_gesture_event(gesture_event)
                confidence = self._sanitize_optional_ratio(
                    coerce_optional_ratio(gesture_confidence)
                )
                self._last_gesture_event = event
                self._last_gesture_event_at = now
                self._last_gesture_confidence = confidence
                self._last_gesture_confidence_at = now
                if temporal_authoritative:
                    rising = self._authoritative_gesture_ready_to_emit(
                        event=event,
                        none_event=SocialGestureEvent.NONE,
                        unknown_event=SocialGestureEvent.UNKNOWN,
                        last_emitted_attr="_last_gesture_emitted_event",
                        last_emitted_at_attr="_last_gesture_emitted_at",
                        last_emitted_token_attr="_last_gesture_emitted_token",
                        activation_token=activation_token,
                        now=now,
                    )
                    return event, False, confidence, False, rising
                rising = False
                if event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
                    if self._gesture_ready_to_emit(
                        event=event,
                        last_emitted=self._last_gesture_emitted_event,
                        last_emitted_at=self._last_gesture_emitted_at,
                        now=now,
                    ):
                        self._last_gesture_emitted_at = now
                        self._last_gesture_emitted_event = event
                        rising = True
                return event, False, confidence, False, rising

            event, event_unknown = self._hold_secondary(
                value=self._last_gesture_event,
                last_seen_at=self._last_gesture_event_at,
                fallback=SocialGestureEvent.UNKNOWN,
                observed_at=now,
            )
            confidence, confidence_unknown = self._hold_secondary(
                value=self._last_gesture_confidence,
                last_seen_at=self._last_gesture_confidence_at,
                fallback=None,
                observed_at=now,
            )
            return event, event_unknown, confidence, confidence_unknown, False

    def _resolve_fine_hand_gesture(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        fine_hand_gesture: object,
        fine_hand_gesture_confidence: object,
        temporal_authoritative: bool = False,
        activation_token: object = None,
    ) -> tuple[SocialFineHandGesture, bool, float | None, bool, bool]:
        """Stabilize fine-hand gesture output with the same bounded cadence rules."""

        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()

            now = self._sanitize_observed_at(observed_at)
            self._update_observed_interval(now)

            if inspected:
                raw_event = coerce_fine_hand_gesture(fine_hand_gesture)
                raw_confidence = self._sanitize_optional_ratio(
                    coerce_optional_ratio(fine_hand_gesture_confidence)
                )
                if temporal_authoritative:
                    event, confidence = raw_event, raw_confidence
                    synthetic_hold = False
                else:
                    event, confidence = self._stabilize_fine_hand_gesture(
                        event=raw_event,
                        confidence=raw_confidence,
                        now=now,
                    )
                    synthetic_hold = bool(
                        getattr(self, "_fine_hand_gesture_held_synthetic", False)
                    )
                self._last_fine_hand_gesture = event
                self._last_fine_hand_gesture_at = now
                self._last_fine_hand_gesture_confidence = confidence
                self._last_fine_hand_gesture_confidence_at = now
                if temporal_authoritative:
                    rising = self._authoritative_gesture_ready_to_emit(
                        event=event,
                        none_event=SocialFineHandGesture.NONE,
                        unknown_event=SocialFineHandGesture.UNKNOWN,
                        last_emitted_attr="_last_fine_hand_gesture_emitted_event",
                        last_emitted_at_attr="_last_fine_hand_gesture_emitted_at",
                        last_emitted_token_attr="_last_fine_hand_gesture_emitted_token",
                        activation_token=activation_token,
                        now=now,
                    )
                    return event, False, confidence, False, rising
                rising = False
                if (
                    event
                    not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}
                    and not synthetic_hold
                ):
                    if self._gesture_ready_to_emit(
                        event=event,
                        last_emitted=self._last_fine_hand_gesture_emitted_event,
                        last_emitted_at=self._last_fine_hand_gesture_emitted_at,
                        now=now,
                    ):
                        self._last_fine_hand_gesture_emitted_at = now
                        self._last_fine_hand_gesture_emitted_event = event
                        rising = True
                return event, False, confidence, False, rising

            event, event_unknown = self._hold_secondary(
                value=self._last_fine_hand_gesture,
                last_seen_at=self._last_fine_hand_gesture_at,
                fallback=SocialFineHandGesture.UNKNOWN,
                observed_at=now,
            )
            confidence, confidence_unknown = self._hold_secondary(
                value=self._last_fine_hand_gesture_confidence,
                last_seen_at=self._last_fine_hand_gesture_confidence_at,
                fallback=None,
                observed_at=now,
            )
            return event, event_unknown, confidence, confidence_unknown, False

    def _stabilize_fine_hand_gesture(
        self: Any,
        *,
        event: SocialFineHandGesture,
        confidence: float | None,
        now: float,
    ) -> tuple[SocialFineHandGesture, float | None]:
        """Preserve explicit hand symbols across brief motion/dropout jitter.

        Explicit symbols like thumbs-up or OK-sign are harder for the live Pi
        path than a generic open palm. Keep the last explicit symbol briefly
        when the current frame drops to ``none`` or weakly collapses to
        ``open_palm`` so users do not need to freeze unnaturally.
        """

        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()
            self._fine_hand_gesture_held_synthetic = False

            policy = self.config.gesture_calibration.fine_hand_policy(
                event,
                fallback_min_confidence=self.config.fine_hand_explicit_min_confidence,
                fallback_confirm_samples=self.config.fine_hand_explicit_confirm_samples,
                fallback_hold_s=self.config.fine_hand_explicit_hold_s,
            )
            confidence = self._sanitize_optional_ratio(confidence)

            if event in EXPLICIT_FINE_HAND_GESTURES:
                current_confidence = 0.0 if confidence is None else confidence
                if current_confidence < policy.min_confidence:
                    self._clear_pending_explicit_fine_hand_gesture()
                    event = SocialFineHandGesture.NONE
                    confidence = None
                else:
                    if event == self._pending_explicit_fine_hand_gesture:
                        self._pending_explicit_fine_hand_gesture_count += 1
                        pending_confidence = (
                            self._pending_explicit_fine_hand_gesture_confidence
                        )
                        if (
                            pending_confidence is None
                            or current_confidence > pending_confidence
                        ):
                            self._pending_explicit_fine_hand_gesture_confidence = (
                                current_confidence
                            )
                    else:
                        self._pending_explicit_fine_hand_gesture = event
                        self._pending_explicit_fine_hand_gesture_count = 1
                        self._pending_explicit_fine_hand_gesture_confidence = (
                            current_confidence
                        )
                        self._pending_explicit_fine_hand_gesture_started_at = now

                    if not self._pending_explicit_fine_hand_gesture_ready(
                        policy=policy, now=now
                    ):
                        return SocialFineHandGesture.NONE, None

                    confidence = self._pending_explicit_fine_hand_gesture_confidence
                    self._clear_pending_explicit_fine_hand_gesture()
            else:
                self._clear_pending_explicit_fine_hand_gesture()

            if event in EXPLICIT_FINE_HAND_GESTURES:
                self._last_explicit_fine_hand_gesture = event
                self._last_explicit_fine_hand_gesture_at = now
                self._last_explicit_fine_hand_gesture_confidence = confidence
                return event, confidence

            held_event = self._last_explicit_fine_hand_gesture
            held_at = self._last_explicit_fine_hand_gesture_at
            if held_event not in EXPLICIT_FINE_HAND_GESTURES or held_at is None:
                return event, confidence

            held_policy = self.config.gesture_calibration.fine_hand_policy(
                held_event,
                fallback_min_confidence=self.config.fine_hand_explicit_min_confidence,
                fallback_confirm_samples=self.config.fine_hand_explicit_confirm_samples,
                fallback_hold_s=self.config.fine_hand_explicit_hold_s,
            )
            held_confidence = self._sanitize_optional_ratio(
                self._last_explicit_fine_hand_gesture_confidence
            )
            effective_hold_s = self._effective_explicit_hold_s(base_hold_s=held_policy.hold_s)
            age_s = self._elapsed_s(now, held_at)
            if age_s > effective_hold_s:
                return event, confidence

            held_confidence_out = self._decay_held_confidence(
                confidence=held_confidence,
                age_s=age_s,
                hold_s=effective_hold_s,
            )
            current_confidence = 0.0 if confidence is None else confidence

            if event in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
                self._fine_hand_gesture_held_synthetic = True
                return held_event, held_confidence_out
            if (
                event == SocialFineHandGesture.OPEN_PALM
                and (held_confidence or 0.0) >= max(0.55, current_confidence + 0.1)
            ):
                self._fine_hand_gesture_held_synthetic = True
                return held_event, held_confidence_out
            if (
                confidence is not None
                and confidence < (held_policy.min_confidence * _WEAK_COLLAPSE_RATIO)
                and age_s <= min(effective_hold_s, held_policy.hold_s * 0.5)
            ):
                self._fine_hand_gesture_held_synthetic = True
                return held_event, held_confidence_out
            return event, confidence

    def _clear_pending_explicit_fine_hand_gesture(self: Any) -> None:
        """Forget one unconfirmed explicit fine-hand candidate."""

        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()
            self._pending_explicit_fine_hand_gesture = SocialFineHandGesture.NONE
            self._pending_explicit_fine_hand_gesture_count = 0
            self._pending_explicit_fine_hand_gesture_confidence = None
            self._pending_explicit_fine_hand_gesture_started_at = None

    def _gesture_ready_to_emit(
        self: Any,
        *,
        event: SocialGestureEvent | SocialFineHandGesture,
        last_emitted: SocialGestureEvent | SocialFineHandGesture,
        last_emitted_at: float | None,
        now: float,
    ) -> bool:
        """Allow changed gestures immediately while damping repeated identical ones.

        The local HDMI acknowledgement path refreshes much faster than the slow
        proactive full-inspect cadence. Gesture acknowledgements therefore must
        not inherit multi-second cooldowns between *different* user symbols like
        ``✋`` followed by ``👍``. We still keep a short repeat gate for the same
        gesture so jittery recognizer output does not flash the same emoji on
        every frame.
        """

        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()
            if last_emitted_at is None:
                return True
            if event != last_emitted:
                return True
            return self._elapsed_s(now, last_emitted_at) >= self.config.gesture_event_cooldown_s

    def _authoritative_gesture_ready_to_emit(
        self: Any,
        *,
        event: SocialGestureEvent | SocialFineHandGesture,
        none_event: SocialGestureEvent | SocialFineHandGesture,
        unknown_event: SocialGestureEvent | SocialFineHandGesture,
        last_emitted_attr: str,
        last_emitted_at_attr: str,
        last_emitted_token_attr: str,
        activation_token: object,
        now: float,
    ) -> bool:
        """Emit only on stable upstream state transitions for authoritative streams."""

        with self._gesture_state_lock():
            self._initialize_gesture_state_if_missing()

            if event in {none_event, unknown_event}:
                setattr(self, last_emitted_attr, none_event)
                setattr(self, last_emitted_at_attr, None)
                setattr(self, last_emitted_token_attr, None)
                return False

            token = _coerce_authoritative_token(activation_token)
            if token is not None:
                previous_token = getattr(self, last_emitted_token_attr, None)
                if token == previous_token:
                    return False
                setattr(self, last_emitted_token_attr, token)
                setattr(self, last_emitted_attr, event)
                setattr(self, last_emitted_at_attr, now)
                return True

            previous = getattr(self, last_emitted_attr, none_event)
            setattr(self, last_emitted_token_attr, None)
            if event == previous:
                return False
            setattr(self, last_emitted_attr, event)
            setattr(self, last_emitted_at_attr, now)
            return True

    def _gesture_state_lock(self: Any) -> threading.RLock:
        lock = getattr(self, "_gesture_runtime_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_gesture_runtime_lock", lock)
        return lock

    def _initialize_gesture_state_if_missing(self: Any) -> None:
        defaults: tuple[tuple[str, object], ...] = (
            ("_last_gesture_event", SocialGestureEvent.UNKNOWN),
            ("_last_gesture_event_at", None),
            ("_last_gesture_confidence", None),
            ("_last_gesture_confidence_at", None),
            ("_last_gesture_emitted_event", SocialGestureEvent.NONE),
            ("_last_gesture_emitted_at", None),
            ("_last_gesture_emitted_token", None),
            ("_last_fine_hand_gesture", SocialFineHandGesture.UNKNOWN),
            ("_last_fine_hand_gesture_at", None),
            ("_last_fine_hand_gesture_confidence", None),
            ("_last_fine_hand_gesture_confidence_at", None),
            ("_last_fine_hand_gesture_emitted_event", SocialFineHandGesture.NONE),
            ("_last_fine_hand_gesture_emitted_at", None),
            ("_last_fine_hand_gesture_emitted_token", None),
            ("_pending_explicit_fine_hand_gesture", SocialFineHandGesture.NONE),
            ("_pending_explicit_fine_hand_gesture_count", 0),
            ("_pending_explicit_fine_hand_gesture_confidence", None),
            ("_pending_explicit_fine_hand_gesture_started_at", None),
            ("_last_explicit_fine_hand_gesture", SocialFineHandGesture.NONE),
            ("_last_explicit_fine_hand_gesture_at", None),
            ("_last_explicit_fine_hand_gesture_confidence", None),
            ("_fine_hand_gesture_held_synthetic", False),
            ("_gesture_clock_last", None),
            ("_gesture_monotonic_shadow_epoch_source", None),
            ("_gesture_monotonic_shadow_epoch_local", None),
            ("_gesture_observed_interval_ema_s", None),
            ("_gesture_last_observed_at", None),
        )
        for attr, default in defaults:
            if not hasattr(self, attr):
                setattr(self, attr, default)

    def _sanitize_observed_at(self: Any, observed_at: float) -> float:
        raw = coerce_timestamp(observed_at)
        if not math.isfinite(raw):
            last = getattr(self, "_gesture_clock_last", None)
            if last is not None:
                return last
            return 0.0

        last = getattr(self, "_gesture_clock_last", None)
        local_now = time.monotonic()
        epoch_source = getattr(self, "_gesture_monotonic_shadow_epoch_source", None)
        epoch_local = getattr(self, "_gesture_monotonic_shadow_epoch_local", None)

        if epoch_source is None or epoch_local is None:
            self._gesture_monotonic_shadow_epoch_source = raw
            self._gesture_monotonic_shadow_epoch_local = local_now
            shadow_now = raw
        else:
            shadow_now = epoch_source + (local_now - epoch_local)

        if last is None:
            sanitized = raw
        else:
            interval = getattr(self, "_gesture_observed_interval_ema_s", None)
            rollback = last - raw
            forward_jump = raw - max(last, shadow_now)

            if rollback > _SMALL_TIMESTAMP_ROLLBACK_S:
                sanitized = max(last, shadow_now)
            elif rollback > 0.0:
                sanitized = last
            elif forward_jump > self._max_reasonable_stream_gap_s(interval):
                sanitized = max(last, shadow_now)
            else:
                sanitized = raw

        self._gesture_clock_last = sanitized
        return sanitized

    def _update_observed_interval(self: Any, now: float) -> None:
        last = getattr(self, "_gesture_last_observed_at", None)
        if last is None:
            self._gesture_last_observed_at = now
            return
        if now <= last:
            return

        delta = now - last
        if 0.0 < delta <= _MAX_REASONABLE_STREAM_GAP_S:
            ema = getattr(self, "_gesture_observed_interval_ema_s", None)
            self._gesture_observed_interval_ema_s = (
                delta if ema is None else ((0.25 * delta) + (0.75 * ema))
            )
        self._gesture_last_observed_at = now

    def _pending_explicit_fine_hand_gesture_ready(
        self: Any, *, policy: object, now: float
    ) -> bool:
        count = int(getattr(self, "_pending_explicit_fine_hand_gesture_count", 0) or 0)
        if count >= int(policy.confirm_samples):
            return True

        started_at = getattr(self, "_pending_explicit_fine_hand_gesture_started_at", None)
        if count < 2 or started_at is None:
            return False

        elapsed = self._elapsed_s(now, started_at)
        return elapsed >= self._time_aware_confirm_s(int(policy.confirm_samples))

    def _time_aware_confirm_s(self: Any, confirm_samples: int) -> float:
        frame_interval = getattr(self, "_gesture_observed_interval_ema_s", None)
        if frame_interval is None or frame_interval <= 0.0:
            frame_interval = _DEFAULT_GESTURE_STREAM_INTERVAL_S
        target = frame_interval * max(confirm_samples - 1, 1)
        return min(_MAX_TIME_AWARE_CONFIRM_S, max(_MIN_TIME_AWARE_CONFIRM_S, target))

    def _effective_explicit_hold_s(
        self: Any,
        *,
        base_hold_s: float,
    ) -> float:
        return max(0.0, float(base_hold_s))

    def _decay_held_confidence(
        self: Any,
        *,
        confidence: float | None,
        age_s: float,
        hold_s: float,
    ) -> float | None:
        if confidence is None:
            return None
        if hold_s <= 0.0:
            return confidence
        ratio = max(0.0, min(1.0, age_s / hold_s))
        return max(0.0, confidence * (1.0 - (_HELD_CONFIDENCE_DECAY * ratio)))

    def _elapsed_s(self: Any, now: float, then: float | None) -> float:
        if then is None:
            return math.inf
        elapsed = now - then
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            return 0.0
        return elapsed

    def _max_reasonable_stream_gap_s(self: Any, interval: float | None) -> float:
        if interval is None or interval <= 0.0:
            return _MAX_REASONABLE_STREAM_GAP_S
        return max(_MAX_REASONABLE_STREAM_GAP_S, interval * 20.0)

    def _sanitize_optional_ratio(self: Any, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            return None
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return float(value)


def authoritative_gesture_activation_token(observation: object) -> int | None:
    """Return the current authoritative gesture activation token when available."""

    stream = gesture_stream(observation)
    if stream is None:
        return None
    return _coerce_authoritative_token(getattr(stream, "activation_token", None))


def _coerce_authoritative_token(value: object) -> int | None:
    """Normalize one optional activation token to a non-negative integer."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        number = int(value)
        return number if number >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        number = int(value)
        return number if number >= 0 else None
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text or len(text) > 64:
            return None
        if text.startswith("+"):
            text = text[1:]
        if not text.isdigit():
            return None
        number = int(text)
        return number if number >= 0 else None
    return None
