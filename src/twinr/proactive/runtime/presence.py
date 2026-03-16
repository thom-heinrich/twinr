"""Track wakeword presence sessions from recent sensor activity.

This module owns the bounded presence-session controller used by the proactive
runtime to decide whether wakeword listening should remain armed after recent
visual, PIR, or qualifying speech activity.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from threading import RLock


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PresenceSessionSnapshot:
    """Describe the current armed state of the presence session."""

    armed: bool
    reason: str | None = None
    person_visible: bool = False
    session_id: int | None = None
    last_person_seen_age_s: float | None = None
    last_motion_age_s: float | None = None
    last_speech_age_s: float | None = None


class PresenceSessionController:
    # AUDIT-FIX(#5): Make the monotonic-time contract explicit to reduce accidental wall-clock usage.
    """Tracks whether a presence session should remain armed.

    `now` should come from a monotonic second counter. The controller also
    hardens itself against malformed input and backward-moving wall clocks so a
    stale activity signal cannot be made fresh again by a time-sync jump.
    """

    def __init__(
        self,
        *,
        presence_grace_s: float,
        motion_grace_s: float,
        speech_grace_s: float,
    ) -> None:
        """Initialize one controller with per-signal grace windows."""

        # AUDIT-FIX(#1): Reject non-finite/boolean grace values so misconfig cannot create broken or never-ending sessions.
        self.presence_grace_s = _validate_non_negative_seconds(
            presence_grace_s,
            field_name="presence_grace_s",
        )
        # AUDIT-FIX(#1): Reject non-finite/boolean grace values so misconfig cannot create broken or never-ending sessions.
        self.motion_grace_s = _validate_non_negative_seconds(
            motion_grace_s,
            field_name="motion_grace_s",
        )
        # AUDIT-FIX(#1): Reject non-finite/boolean grace values so misconfig cannot create broken or never-ending sessions.
        self.speech_grace_s = _validate_non_negative_seconds(
            speech_grace_s,
            field_name="speech_grace_s",
        )
        self._last_person_seen_at: float | None = None
        self._last_motion_at: float | None = None
        self._last_speech_at: float | None = None
        # AUDIT-FIX(#2): Track qualifying speech separately so speech grace only extends sessions after real presence context.
        self._last_speech_while_present_at: float | None = None
        self._armed: bool = False
        self._current_session_id: int = 0
        # AUDIT-FIX(#1): Preserve a monotonic internal timeline even if callers pass wall-clock timestamps that move backwards.
        self._time_offset_s: float = 0.0
        self._last_observed_now: float | None = None
        # AUDIT-FIX(#3): Serialize state mutation for mixed callback/thread sensor ingestion.
        self._lock = RLock()

    def observe(
        self,
        *,
        now: float,
        person_visible: bool | None,
        motion_active: bool,
        speech_detected: bool,
    ) -> PresenceSessionSnapshot:
        """Update session state from one sensor tick and return a snapshot.

        Args:
            now: Monotonic-like seconds for the current observation tick.
            person_visible: Latest camera presence flag, or None when unknown.
            motion_active: Latest PIR motion state for this tick.
            speech_detected: Whether ambient audio detected speech this tick.

        Returns:
            One snapshot describing whether wakeword listening should remain
            armed and why.
        """

        # AUDIT-FIX(#3): Guard the entire read-update-decide sequence so session ids and timestamps cannot race.
        with self._lock:
            # AUDIT-FIX(#1): Normalize time into a finite, non-regressing internal timeline.
            safe_now, self._time_offset_s = _normalize_now(
                now,
                previous=self._last_observed_now,
                offset=self._time_offset_s,
            )
            self._last_observed_now = safe_now

            # AUDIT-FIX(#4): Normalize sensor flags explicitly and fail closed on malformed values.
            safe_person_visible = _normalize_optional_bool(
                person_visible,
                field_name="person_visible",
            )
            # AUDIT-FIX(#4): Normalize sensor flags explicitly and fail closed on malformed values.
            safe_motion_active = _normalize_bool(
                motion_active,
                field_name="motion_active",
            )
            # AUDIT-FIX(#4): Normalize sensor flags explicitly and fail closed on malformed values.
            safe_speech_detected = _normalize_bool(
                speech_detected,
                field_name="speech_detected",
            )

            if safe_person_visible is True:
                self._last_person_seen_at = safe_now
            if safe_motion_active:
                self._last_motion_at = safe_now
            if safe_speech_detected:
                self._last_speech_at = safe_now

            person_age = _age(safe_now, self._last_person_seen_at)
            motion_age = _age(safe_now, self._last_motion_at)
            speech_age = _age(safe_now, self._last_speech_at)

            recently_present = (
                (person_age is not None and person_age <= self.presence_grace_s)
                or (motion_age is not None and motion_age <= self.motion_grace_s)
            )
            if safe_speech_detected and recently_present:
                # AUDIT-FIX(#2): Remember only speech that happened during a valid presence window.
                self._last_speech_while_present_at = safe_now
            qualifying_speech_age = _age(safe_now, self._last_speech_while_present_at)

            if safe_person_visible is True:
                return self._build_snapshot(
                    armed=True,
                    reason="person_visible",
                    person_visible=True,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            if safe_motion_active:
                return self._build_snapshot(
                    armed=True,
                    reason="pir_motion",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            if safe_speech_detected and recently_present:
                # AUDIT-FIX(#2): Make the speech-triggered branch reachable before stale recency branches short-circuit it.
                return self._build_snapshot(
                    armed=True,
                    reason="speech_while_recently_present",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            if person_age is not None and person_age <= self.presence_grace_s:
                return self._build_snapshot(
                    armed=True,
                    reason="recent_person_visible",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            if motion_age is not None and motion_age <= self.motion_grace_s:
                return self._build_snapshot(
                    armed=True,
                    reason="recent_pir_motion",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            if qualifying_speech_age is not None and qualifying_speech_age <= self.speech_grace_s:
                # AUDIT-FIX(#2): Allow qualifying speech to keep the session alive after visual/PIR grace has expired.
                return self._build_snapshot(
                    armed=True,
                    reason="recent_speech_while_present",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                )
            return self._build_snapshot(
                armed=False,
                reason="idle",
                person_visible=False,
                last_person_seen_age_s=person_age,
                last_motion_age_s=motion_age,
                last_speech_age_s=speech_age,
            )

    def _build_snapshot(
        self,
        *,
        armed: bool,
        reason: str | None,
        person_visible: bool,
        last_person_seen_age_s: float | None,
        last_motion_age_s: float | None,
        last_speech_age_s: float | None,
    ) -> PresenceSessionSnapshot:
        """Build one snapshot and advance the session counter on re-arming."""

        if armed and not self._armed:
            self._current_session_id += 1
        self._armed = armed
        return PresenceSessionSnapshot(
            armed=armed,
            reason=reason,
            person_visible=person_visible,
            session_id=self._current_session_id if armed else None,
            last_person_seen_age_s=last_person_seen_age_s,
            last_motion_age_s=last_motion_age_s,
            last_speech_age_s=last_speech_age_s,
        )


# AUDIT-FIX(#1): Centralize strict seconds validation for constructor inputs.
def _validate_non_negative_seconds(value: float, *, field_name: str) -> float:
    """Validate one non-negative finite seconds value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be a finite int or float number of seconds, got {value!r}",
        )
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return max(0.0, seconds)


# AUDIT-FIX(#1): Repair backward-moving wall-clock inputs into a monotonic internal timeline.
def _normalize_now(
    value: float,
    *,
    previous: float | None,
    offset: float,
) -> tuple[float, float]:
    """Repair regressing or malformed timestamps into a monotonic timeline."""

    fallback_now = previous if previous is not None else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        LOGGER.warning(
            "PresenceSessionController.observe received non-numeric now=%r; reusing %.6f",
            value,
            fallback_now,
        )
        return fallback_now, offset

    raw_now = float(value)
    if not math.isfinite(raw_now):
        LOGGER.warning(
            "PresenceSessionController.observe received non-finite now=%r; reusing %.6f",
            value,
            fallback_now,
        )
        return fallback_now, offset

    adjusted_now = raw_now + offset
    if previous is not None and adjusted_now < previous:
        offset = previous - raw_now
        adjusted_now = previous
        LOGGER.warning(
            "PresenceSessionController.observe received regressing time %.6f after %.6f; applying offset %.6f",
            raw_now,
            previous,
            offset,
        )

    return adjusted_now, offset


# AUDIT-FIX(#4): Narrow runtime normalization to explicit booleans and legacy 0/1 ints.
def _normalize_bool(value: bool, *, field_name: str) -> bool:
    """Normalize one required runtime flag to a fail-closed boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    LOGGER.warning("%s received non-bool %r; failing closed", field_name, value)
    return False


# AUDIT-FIX(#4): Treat malformed optional booleans as unknown instead of letting truthiness leak through.
def _normalize_optional_bool(value: bool | None, *, field_name: str) -> bool | None:
    """Normalize one optional runtime flag to bool-or-unknown."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    LOGGER.warning("%s received invalid optional bool %r; treating as unknown", field_name, value)
    return None


# AUDIT-FIX(#1): Clamp impossible negative ages defensively and surface the anomaly in logs.
def _age(now: float, since: float | None) -> float | None:
    """Return the non-negative age of one timestamp in seconds."""

    if since is None:
        return None
    age = now - since
    if age < 0.0:
        LOGGER.warning("PresenceSessionController computed negative age %.6f; clamping to 0.0", age)
        return 0.0
    return age


__all__ = [
    "PresenceSessionController",
    "PresenceSessionSnapshot",
]
