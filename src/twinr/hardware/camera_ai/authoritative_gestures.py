"""Authoritative activation state for the dedicated live gesture lane.

This module converts already-resolved live-gesture outputs into one stable
activation identity that downstream consumers can trust. It deliberately does
not re-score confidence or re-run temporal voting; that work stays in the live
gesture pipeline. Its responsibility is narrower:

- decide whether the current resolved gesture output is active
- assign one monotonic activation token per rising activation
- preserve one authoritative activation identity while the same gesture remains active
- expose compact debug metadata for downstream forensics
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AICameraFineHandGesture, AICameraGestureEvent


@dataclass(frozen=True, slots=True)
class AuthoritativeGestureActivation:
    """Describe one authoritative activation snapshot for the current frame."""

    active: bool = False
    activation_key: str | None = None
    activation_token: int | None = None
    activation_started_at: float | None = None
    activation_changed_at: float | None = None
    activation_source: str | None = None
    activation_rising: bool = False
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    gesture_event: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_confidence: float | None = None


class AuthoritativeGestureLane:
    """Assign one stable activation identity to the resolved live gesture output."""

    def __init__(self) -> None:
        self._next_token = 1
        self._active_key: str | None = None
        self._active_token: int | None = None
        self._active_started_at: float | None = None
        self._active_changed_at: float | None = None
        self._active_source: str | None = None

    def reset(self) -> None:
        """Forget the currently active gesture identity."""

        self._active_key = None
        self._active_token = None
        self._active_started_at = None
        self._active_changed_at = None
        self._active_source = None

    def observe(
        self,
        *,
        observed_at: float,
        fine_hand_gesture: AICameraFineHandGesture,
        fine_hand_gesture_confidence: float | None,
        gesture_event: AICameraGestureEvent,
        gesture_confidence: float | None,
        resolved_source: str,
    ) -> tuple[AuthoritativeGestureActivation, dict[str, object]]:
        """Return the authoritative activation snapshot for one resolved frame."""

        activation_key = _activation_key(
            fine_hand_gesture=fine_hand_gesture,
            gesture_event=gesture_event,
        )
        activation_rising = False

        if activation_key is None:
            if self._active_key is not None:
                self._active_changed_at = observed_at
            self.reset()
            activation = AuthoritativeGestureActivation(
                active=False,
                fine_hand_gesture=fine_hand_gesture,
                fine_hand_gesture_confidence=fine_hand_gesture_confidence,
                gesture_event=gesture_event,
                gesture_confidence=gesture_confidence,
            )
            return activation, {
                "authoritative_gesture_active": False,
                "authoritative_gesture_key": None,
                "authoritative_gesture_token": None,
                "authoritative_gesture_rising": False,
                "authoritative_gesture_started_at": None,
                "authoritative_gesture_changed_at": None,
                "authoritative_gesture_source": None,
            }

        if activation_key != self._active_key or self._active_token is None:
            activation_rising = True
            self._active_key = activation_key
            self._active_token = self._next_token
            self._next_token += 1
            self._active_started_at = observed_at
            self._active_changed_at = observed_at
            self._active_source = resolved_source or "none"
        elif self._active_source != (resolved_source or "none"):
            self._active_source = resolved_source or "none"

        activation = AuthoritativeGestureActivation(
            active=True,
            activation_key=self._active_key,
            activation_token=self._active_token,
            activation_started_at=self._active_started_at,
            activation_changed_at=self._active_changed_at,
            activation_source=self._active_source,
            activation_rising=activation_rising,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
        )
        return activation, {
            "authoritative_gesture_active": True,
            "authoritative_gesture_key": self._active_key,
            "authoritative_gesture_token": self._active_token,
            "authoritative_gesture_rising": activation_rising,
            "authoritative_gesture_started_at": _round_optional_timestamp(self._active_started_at),
            "authoritative_gesture_changed_at": _round_optional_timestamp(self._active_changed_at),
            "authoritative_gesture_source": self._active_source,
        }


def _activation_key(
    *,
    fine_hand_gesture: AICameraFineHandGesture,
    gesture_event: AICameraGestureEvent,
) -> str | None:
    """Return the canonical activation key for the current resolved gesture."""

    if fine_hand_gesture not in {AICameraFineHandGesture.NONE, AICameraFineHandGesture.UNKNOWN}:
        return f"fine:{fine_hand_gesture.value}"
    if gesture_event not in {AICameraGestureEvent.NONE, AICameraGestureEvent.UNKNOWN}:
        return f"coarse:{gesture_event.value}"
    return None


def _round_optional_timestamp(value: float | None) -> float | None:
    """Round one optional timestamp for bounded debug payloads."""

    if value is None:
        return None
    return round(float(value), 3)


__all__ = ["AuthoritativeGestureActivation", "AuthoritativeGestureLane"]
