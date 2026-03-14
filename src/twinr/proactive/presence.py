from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresenceSessionSnapshot:
    armed: bool
    reason: str | None = None
    person_visible: bool = False
    session_id: int | None = None
    last_person_seen_age_s: float | None = None
    last_motion_age_s: float | None = None
    last_speech_age_s: float | None = None


class PresenceSessionController:
    def __init__(
        self,
        *,
        presence_grace_s: float,
        motion_grace_s: float,
        speech_grace_s: float,
    ) -> None:
        self.presence_grace_s = max(0.0, presence_grace_s)
        self.motion_grace_s = max(0.0, motion_grace_s)
        self.speech_grace_s = max(0.0, speech_grace_s)
        self._last_person_seen_at: float | None = None
        self._last_motion_at: float | None = None
        self._last_speech_at: float | None = None
        self._armed: bool = False
        self._current_session_id: int = 0

    def observe(
        self,
        *,
        now: float,
        person_visible: bool | None,
        motion_active: bool,
        speech_detected: bool,
    ) -> PresenceSessionSnapshot:
        if person_visible is True:
            self._last_person_seen_at = now
        if motion_active:
            self._last_motion_at = now
        if speech_detected:
            self._last_speech_at = now

        person_age = _age(now, self._last_person_seen_at)
        motion_age = _age(now, self._last_motion_at)
        speech_age = _age(now, self._last_speech_at)

        if person_visible is True:
            return self._build_snapshot(
                armed=True,
                reason="person_visible",
                person_visible=True,
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
        if motion_active:
            return self._build_snapshot(
                armed=True,
                reason="pir_motion",
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
        recently_present = (
            (person_age is not None and person_age <= self.presence_grace_s)
            or (motion_age is not None and motion_age <= self.motion_grace_s)
        )
        if speech_detected and recently_present:
            return self._build_snapshot(
                armed=True,
                reason="speech_while_recently_present",
                person_visible=False,
                last_person_seen_age_s=person_age,
                last_motion_age_s=motion_age,
                last_speech_age_s=speech_age,
            )
        if speech_age is not None and speech_age <= self.speech_grace_s and recently_present:
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


def _age(now: float, since: float | None) -> float | None:
    if since is None:
        return None
    return max(0.0, now - since)


__all__ = [
    "PresenceSessionController",
    "PresenceSessionSnapshot",
]
