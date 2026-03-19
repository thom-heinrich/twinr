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
    presence_audio_active: bool | None = None
    recent_follow_up_speech: bool | None = None
    room_busy_or_overlapping: bool | None = None
    quiet_window_open: bool | None = None
    barge_in_recent: bool | None = None
    speaker_direction_stable: bool | None = None
    mute_blocks_voice_capture: bool | None = None
    resume_window_open: bool | None = None
    device_runtime_mode: str | None = None
    transport_reason: str | None = None


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
        recent_speech_age_s: float | None = None,
        presence_audio_active: bool | None = None,
        recent_follow_up_speech: bool | None = None,
        room_busy_or_overlapping: bool | None = None,
        quiet_window_open: bool | None = None,
        barge_in_recent: bool | None = None,
        speaker_direction_stable: bool | None = None,
        mute_blocks_voice_capture: bool | None = None,
        resume_window_open: bool | None = None,
        device_runtime_mode: str | None = None,
        transport_reason: str | None = None,
    ) -> PresenceSessionSnapshot:
        """Update session state from one sensor tick and return a snapshot.

        Args:
            now: Monotonic-like seconds for the current observation tick.
            person_visible: Latest camera presence flag, or None when unknown.
            motion_active: Latest PIR motion state for this tick.
            speech_detected: Whether ambient audio detected speech this tick.
            recent_speech_age_s: Age of the latest recent-speech fact, when available.
            presence_audio_active: Whether calm single-speaker speech is active.
            recent_follow_up_speech: Whether recent speech happened in a short follow-up window.
            room_busy_or_overlapping: Whether overlap or multi-speaker audio is likely.
            quiet_window_open: Whether the room currently looks quiet enough for initiative.
            barge_in_recent: Whether a recent interruption-like event happened.
            speaker_direction_stable: Whether current direction confidence is policy-usable.
            mute_blocks_voice_capture: Whether mute or runtime mode blocks voice capture.
            resume_window_open: Whether the room is in a short resume/follow-up window.
            device_runtime_mode: Current operator-facing ReSpeaker runtime mode.
            transport_reason: Optional transport degradation reason.

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
            safe_recent_speech_age_s = _normalize_optional_seconds(
                recent_speech_age_s,
                field_name="recent_speech_age_s",
            )
            safe_presence_audio_active = _normalize_optional_bool(
                presence_audio_active,
                field_name="presence_audio_active",
            )
            safe_recent_follow_up_speech = _normalize_optional_bool(
                recent_follow_up_speech,
                field_name="recent_follow_up_speech",
            )
            safe_room_busy_or_overlapping = _normalize_optional_bool(
                room_busy_or_overlapping,
                field_name="room_busy_or_overlapping",
            )
            safe_quiet_window_open = _normalize_optional_bool(
                quiet_window_open,
                field_name="quiet_window_open",
            )
            safe_barge_in_recent = _normalize_optional_bool(
                barge_in_recent,
                field_name="barge_in_recent",
            )
            safe_speaker_direction_stable = _normalize_optional_bool(
                speaker_direction_stable,
                field_name="speaker_direction_stable",
            )
            safe_mute_blocks_voice_capture = _normalize_optional_bool(
                mute_blocks_voice_capture,
                field_name="mute_blocks_voice_capture",
            )
            safe_resume_window_open = _normalize_optional_bool(
                resume_window_open,
                field_name="resume_window_open",
            )
            normalized_device_runtime_mode = _normalize_optional_text(
                device_runtime_mode,
                field_name="device_runtime_mode",
            )
            normalized_transport_reason = _normalize_optional_text(
                transport_reason,
                field_name="transport_reason",
            )

            if safe_person_visible is True:
                self._last_person_seen_at = safe_now
            if safe_motion_active:
                self._last_motion_at = safe_now
            if safe_speech_detected:
                self._last_speech_at = safe_now
            elif safe_recent_speech_age_s is not None:
                reconstructed_last_speech_at = max(0.0, safe_now - safe_recent_speech_age_s)
                if self._last_speech_at is None or reconstructed_last_speech_at > self._last_speech_at:
                    self._last_speech_at = reconstructed_last_speech_at

            person_age = _age(safe_now, self._last_person_seen_at)
            motion_age = _age(safe_now, self._last_motion_at)
            speech_age = _age(safe_now, self._last_speech_at)

            recently_present = (
                (person_age is not None and person_age <= self.presence_grace_s)
                or (motion_age is not None and motion_age <= self.motion_grace_s)
            )
            qualifying_audio_allowed = (
                safe_room_busy_or_overlapping is not True
                and safe_mute_blocks_voice_capture is not True
            )
            effective_presence_audio = (
                safe_presence_audio_active
                if safe_presence_audio_active is not None
                else safe_speech_detected
            )
            if effective_presence_audio and recently_present and qualifying_audio_allowed:
                # AUDIT-FIX(#2): Remember only speech that happened during a valid presence window.
                if self._last_speech_at is not None:
                    self._last_speech_while_present_at = self._last_speech_at
            elif safe_recent_follow_up_speech and recently_present and qualifying_audio_allowed:
                if self._last_speech_at is not None:
                    if (
                        self._last_speech_while_present_at is None
                        or self._last_speech_at > self._last_speech_while_present_at
                    ):
                        self._last_speech_while_present_at = self._last_speech_at
            qualifying_speech_age = _age(safe_now, self._last_speech_while_present_at)

            if safe_person_visible is True:
                return self._build_snapshot(
                    armed=True,
                    reason="person_visible",
                    person_visible=True,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            if safe_motion_active:
                return self._build_snapshot(
                    armed=True,
                    reason="pir_motion",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            if effective_presence_audio and recently_present and qualifying_audio_allowed:
                # AUDIT-FIX(#2): Make the speech-triggered branch reachable before stale recency branches short-circuit it.
                return self._build_snapshot(
                    armed=True,
                    reason="speech_while_recently_present",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            if safe_recent_follow_up_speech and recently_present and qualifying_audio_allowed:
                return self._build_snapshot(
                    armed=True,
                    reason="follow_up_speech_while_recently_present",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            if person_age is not None and person_age <= self.presence_grace_s:
                return self._build_snapshot(
                    armed=True,
                    reason="recent_person_visible",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            if motion_age is not None and motion_age <= self.motion_grace_s:
                return self._build_snapshot(
                    armed=True,
                    reason="recent_pir_motion",
                    person_visible=False,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
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
                    presence_audio_active=safe_presence_audio_active,
                    recent_follow_up_speech=safe_recent_follow_up_speech,
                    room_busy_or_overlapping=safe_room_busy_or_overlapping,
                    quiet_window_open=safe_quiet_window_open,
                    barge_in_recent=safe_barge_in_recent,
                    speaker_direction_stable=safe_speaker_direction_stable,
                    mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                    resume_window_open=safe_resume_window_open,
                    device_runtime_mode=normalized_device_runtime_mode,
                    transport_reason=normalized_transport_reason,
                )
            return self._build_snapshot(
                armed=False,
                reason="idle",
                person_visible=False,
                last_person_seen_age_s=person_age,
                last_motion_age_s=motion_age,
                last_speech_age_s=speech_age,
                presence_audio_active=safe_presence_audio_active,
                recent_follow_up_speech=safe_recent_follow_up_speech,
                room_busy_or_overlapping=safe_room_busy_or_overlapping,
                quiet_window_open=safe_quiet_window_open,
                barge_in_recent=safe_barge_in_recent,
                speaker_direction_stable=safe_speaker_direction_stable,
                mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                resume_window_open=safe_resume_window_open,
                device_runtime_mode=normalized_device_runtime_mode,
                transport_reason=normalized_transport_reason,
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
        presence_audio_active: bool | None,
        recent_follow_up_speech: bool | None,
        room_busy_or_overlapping: bool | None,
        quiet_window_open: bool | None,
        barge_in_recent: bool | None,
        speaker_direction_stable: bool | None,
        mute_blocks_voice_capture: bool | None,
        resume_window_open: bool | None,
        device_runtime_mode: str | None,
        transport_reason: str | None,
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
            presence_audio_active=presence_audio_active,
            recent_follow_up_speech=recent_follow_up_speech,
            room_busy_or_overlapping=room_busy_or_overlapping,
            quiet_window_open=quiet_window_open,
            barge_in_recent=barge_in_recent,
            speaker_direction_stable=speaker_direction_stable,
            mute_blocks_voice_capture=mute_blocks_voice_capture,
            resume_window_open=resume_window_open,
            device_runtime_mode=device_runtime_mode,
            transport_reason=transport_reason,
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


def _normalize_optional_seconds(value: float | None, *, field_name: str) -> float | None:
    """Normalize one optional non-negative duration to seconds-or-unknown."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        LOGGER.warning("%s received invalid optional seconds %r; treating as unknown", field_name, value)
        return None
    seconds = float(value)
    if not math.isfinite(seconds):
        LOGGER.warning("%s received non-finite optional seconds %r; treating as unknown", field_name, value)
        return None
    return max(0.0, seconds)


def _normalize_optional_text(value: str | None, *, field_name: str) -> str | None:
    """Normalize one optional text field into a trimmed single line."""

    if value is None:
        return None
    if not isinstance(value, str):
        LOGGER.warning("%s received invalid optional text %r; treating as unknown", field_name, value)
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


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
