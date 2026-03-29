"""Gesture stabilization helpers for the camera surface."""

from __future__ import annotations

from typing import Any

from ..engine import SocialFineHandGesture, SocialGestureEvent
from ..perception_stream import gesture_stream
from .coercion import coerce_fine_hand_gesture, coerce_gesture_event, coerce_optional_ratio, coerce_timestamp

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
        now = coerce_timestamp(observed_at)
        if inspected:
            event = coerce_gesture_event(gesture_event)
            confidence = coerce_optional_ratio(gesture_confidence)
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
            observed_at=observed_at,
        )
        confidence, confidence_unknown = self._hold_secondary(
            value=self._last_gesture_confidence,
            last_seen_at=self._last_gesture_confidence_at,
            fallback=None,
            observed_at=observed_at,
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

        now = coerce_timestamp(observed_at)
        if inspected:
            raw_event = coerce_fine_hand_gesture(fine_hand_gesture)
            raw_confidence = coerce_optional_ratio(fine_hand_gesture_confidence)
            if temporal_authoritative:
                event, confidence = raw_event, raw_confidence
            else:
                event, confidence = self._stabilize_fine_hand_gesture(
                    event=raw_event,
                    confidence=raw_confidence,
                    now=now,
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
            if event not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
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
            observed_at=observed_at,
        )
        confidence, confidence_unknown = self._hold_secondary(
            value=self._last_fine_hand_gesture_confidence,
            last_seen_at=self._last_fine_hand_gesture_confidence_at,
            fallback=None,
            observed_at=observed_at,
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

        policy = self.config.gesture_calibration.fine_hand_policy(
            event,
            fallback_min_confidence=self.config.fine_hand_explicit_min_confidence,
            fallback_confirm_samples=self.config.fine_hand_explicit_confirm_samples,
            fallback_hold_s=self.config.fine_hand_explicit_hold_s,
        )
        if event in EXPLICIT_FINE_HAND_GESTURES:
            current_confidence = 0.0 if confidence is None else confidence
            if current_confidence < policy.min_confidence:
                self._clear_pending_explicit_fine_hand_gesture()
                event = SocialFineHandGesture.NONE
                confidence = None
            else:
                if event == self._pending_explicit_fine_hand_gesture:
                    self._pending_explicit_fine_hand_gesture_count += 1
                    pending_confidence = self._pending_explicit_fine_hand_gesture_confidence
                    if pending_confidence is None or current_confidence > pending_confidence:
                        self._pending_explicit_fine_hand_gesture_confidence = current_confidence
                else:
                    self._pending_explicit_fine_hand_gesture = event
                    self._pending_explicit_fine_hand_gesture_count = 1
                    self._pending_explicit_fine_hand_gesture_confidence = current_confidence
                if self._pending_explicit_fine_hand_gesture_count < policy.confirm_samples:
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
        if (now - held_at) > held_policy.hold_s:
            return event, confidence
        held_confidence = self._last_explicit_fine_hand_gesture_confidence
        current_confidence = 0.0 if confidence is None else confidence
        if event in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
            return held_event, held_confidence
        if (
            event == SocialFineHandGesture.OPEN_PALM
            and (held_confidence or 0.0) >= max(0.55, current_confidence + 0.1)
        ):
            return held_event, held_confidence
        return event, confidence

    def _clear_pending_explicit_fine_hand_gesture(self: Any) -> None:
        """Forget one unconfirmed explicit fine-hand candidate."""

        self._pending_explicit_fine_hand_gesture = SocialFineHandGesture.NONE
        self._pending_explicit_fine_hand_gesture_count = 0
        self._pending_explicit_fine_hand_gesture_confidence = None

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

        if last_emitted_at is None:
            return True
        if event != last_emitted:
            return True
        return (now - last_emitted_at) >= self.config.gesture_event_cooldown_s

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
        if event == previous:
            return False
        setattr(self, last_emitted_attr, event)
        setattr(self, last_emitted_at_attr, now)
        return True


def authoritative_gesture_activation_token(observation: object) -> int | None:
    """Return the current authoritative gesture activation token when available."""

    stream = gesture_stream(observation)
    if stream is None:
        return None
    return _coerce_authoritative_token(stream.activation_token)


def _coerce_authoritative_token(value: object) -> int | None:
    """Normalize one optional activation token to a non-negative integer."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number
