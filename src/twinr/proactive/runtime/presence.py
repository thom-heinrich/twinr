# CHANGELOG: 2026-03-29
# BUG-1: Invalid, non-finite, backward, or stalled `now` inputs no longer freeze ages and keep sessions armed forever.
# BUG-2: Raw `speech_detected` no longer counts as qualifying presence speech unless directedness/context signals support it.
# BUG-3: `mute_blocks_voice_capture` and explicit blocking runtime modes now hard-disarm the session instead of being pass-through only.
# SEC-1: Warning logs are rate-limited to prevent malformed high-rate sensor input from filling logs on Raspberry Pi SD cards.
# SEC-2: Optional text fields are length-capped and single-lined to reduce memory/log amplification from untrusted transports.
# IMP-1: Replaced age-only branching with deadline-based multimodal fusion and explicit arm expiry bookkeeping.
# IMP-2: Added follow-up/resume/barge-in context windows and debounce/hysteresis for frontier-grade session stability.
# IMP-3: Snapshot now includes blocker_reason, qualifying_audio_active, arm_deadline_s, arm_expires_in_s, last_qualifying_speech_age_s, and policy_version.

"""Track voice-activation presence sessions from recent sensor activity.

This module owns the bounded presence-session controller used by the proactive
runtime to decide whether voice-activation listening should remain armed after recent
visual, PIR, or qualifying speech activity.

The 2026 policy favors:
- monotonic internal time even when upstream timestamps are malformed or stall
- hard privacy blocks for mute / capture-disabled runtime modes
- qualifying speech only when directionality or upstream device-directed signals exist
- short hysteresis to avoid rapid arm/disarm chatter
- explicit deadline bookkeeping so downstream systems can reason about expiry
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from threading import RLock
import time
from typing import Callable


LOGGER = logging.getLogger(__name__)

_POLICY_VERSION = "2026.03-frontier"

_BLOCKING_RUNTIME_MODES = frozenset(
    {
        "disabled",
        "mute",
        "muted",
        "off",
        "privacy",
        "privacy_mode",
        "push_to_talk",
        "ptt",
        "sleep",
        "standby",
    }
)


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
    blocker_reason: str | None = None
    qualifying_audio_active: bool | None = None
    arm_deadline_s: float | None = None
    arm_expires_in_s: float | None = None
    last_qualifying_speech_age_s: float | None = None
    last_follow_up_age_s: float | None = None
    policy_version: str | None = None


class PresenceSessionController:
    """Tracks whether a presence session should remain armed.

    `now` should ideally come from a monotonic second counter. For robustness,
    the controller repairs malformed, regressing, and stalled timestamps using a
    local monotonic clock so stale activity cannot be made fresh again by a
    time-sync jump or by a broken caller that stops advancing `now`.
    """

    def __init__(
        self,
        *,
        presence_grace_s: float,
        motion_grace_s: float,
        speech_grace_s: float,
        follow_up_grace_s: float | None = None,
        resume_grace_s: float | None = None,
        barge_in_grace_s: float | None = None,
        arm_hysteresis_s: float = 0.35,
        log_throttle_s: float = 30.0,
        max_text_len: int = 160,
        blocking_runtime_modes: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Initialize one controller with per-signal grace windows."""

        self.presence_grace_s = _validate_non_negative_seconds(
            presence_grace_s,
            field_name="presence_grace_s",
        )
        self.motion_grace_s = _validate_non_negative_seconds(
            motion_grace_s,
            field_name="motion_grace_s",
        )
        self.speech_grace_s = _validate_non_negative_seconds(
            speech_grace_s,
            field_name="speech_grace_s",
        )
        self.follow_up_grace_s = _validate_non_negative_seconds(
            speech_grace_s if follow_up_grace_s is None else follow_up_grace_s,
            field_name="follow_up_grace_s",
        )
        self.resume_grace_s = _validate_non_negative_seconds(
            speech_grace_s if resume_grace_s is None else resume_grace_s,
            field_name="resume_grace_s",
        )
        self.barge_in_grace_s = _validate_non_negative_seconds(
            min(self.resume_grace_s, 2.0)
            if barge_in_grace_s is None
            else barge_in_grace_s,
            field_name="barge_in_grace_s",
        )
        self.arm_hysteresis_s = _validate_non_negative_seconds(
            arm_hysteresis_s,
            field_name="arm_hysteresis_s",
        )
        self.log_throttle_s = _validate_non_negative_seconds(
            log_throttle_s,
            field_name="log_throttle_s",
        )
        self.max_text_len = _validate_positive_int(max_text_len, field_name="max_text_len")

        runtime_modes = blocking_runtime_modes or _BLOCKING_RUNTIME_MODES
        self._blocking_runtime_modes = frozenset(
            mode.strip().casefold()
            for mode in runtime_modes
            if isinstance(mode, str) and mode.strip()
        )

        self._last_person_seen_at: float | None = None
        self._last_motion_at: float | None = None
        self._last_speech_at: float | None = None
        self._last_qualifying_speech_at: float | None = None
        self._last_follow_up_at: float | None = None
        self._last_resume_window_at: float | None = None
        self._last_barge_in_at: float | None = None

        self._armed: bool = False
        self._current_session_id: int = 0
        self._arm_until_s: float | None = None

        self._time_offset_s: float = 0.0
        self._last_observed_now: float | None = None
        self._last_local_monotonic_ns: int = time.monotonic_ns()

        self._last_warning_by_key: dict[str, float] = {}
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
        """Update session state from one sensor tick and return a snapshot."""

        with self._lock:
            safe_now, self._time_offset_s, _local_elapsed_s = self._normalize_now(now)
            self._last_observed_now = safe_now

            warn = self._warn

            safe_person_visible = _normalize_optional_bool(
                person_visible,
                field_name="person_visible",
                warn=warn,
            )
            safe_motion_active = _normalize_bool(
                motion_active,
                field_name="motion_active",
                warn=warn,
            )
            safe_speech_detected = _normalize_bool(
                speech_detected,
                field_name="speech_detected",
                warn=warn,
            )
            safe_recent_speech_age_s = _normalize_optional_seconds(
                recent_speech_age_s,
                field_name="recent_speech_age_s",
                warn=warn,
            )
            safe_presence_audio_active = _normalize_optional_bool(
                presence_audio_active,
                field_name="presence_audio_active",
                warn=warn,
            )
            safe_recent_follow_up_speech = _normalize_optional_bool(
                recent_follow_up_speech,
                field_name="recent_follow_up_speech",
                warn=warn,
            )
            safe_room_busy_or_overlapping = _normalize_optional_bool(
                room_busy_or_overlapping,
                field_name="room_busy_or_overlapping",
                warn=warn,
            )
            safe_quiet_window_open = _normalize_optional_bool(
                quiet_window_open,
                field_name="quiet_window_open",
                warn=warn,
            )
            safe_barge_in_recent = _normalize_optional_bool(
                barge_in_recent,
                field_name="barge_in_recent",
                warn=warn,
            )
            safe_speaker_direction_stable = _normalize_optional_bool(
                speaker_direction_stable,
                field_name="speaker_direction_stable",
                warn=warn,
            )
            safe_mute_blocks_voice_capture = _normalize_optional_bool(
                mute_blocks_voice_capture,
                field_name="mute_blocks_voice_capture",
                warn=warn,
            )
            safe_resume_window_open = _normalize_optional_bool(
                resume_window_open,
                field_name="resume_window_open",
                warn=warn,
            )
            normalized_device_runtime_mode = _normalize_optional_text(
                device_runtime_mode,
                field_name="device_runtime_mode",
                max_len=self.max_text_len,
                warn=warn,
            )
            normalized_transport_reason = _normalize_optional_text(
                transport_reason,
                field_name="transport_reason",
                max_len=self.max_text_len,
                warn=warn,
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

            if safe_resume_window_open is True:
                self._last_resume_window_at = safe_now
            if safe_barge_in_recent is True:
                self._last_barge_in_at = safe_now

            person_age = _age(safe_now, self._last_person_seen_at)
            motion_age = _age(safe_now, self._last_motion_at)
            speech_age = _age(safe_now, self._last_speech_at)

            recently_present = (
                (person_age is not None and person_age <= self.presence_grace_s)
                or (motion_age is not None and motion_age <= self.motion_grace_s)
            )

            runtime_blocks_capture = _runtime_mode_blocks_capture(
                normalized_device_runtime_mode,
                blocking_modes=self._blocking_runtime_modes,
            )
            hard_blocker_reason = _select_hard_blocker_reason(
                mute_blocks_voice_capture=safe_mute_blocks_voice_capture,
                runtime_blocks_capture=runtime_blocks_capture,
                device_runtime_mode=normalized_device_runtime_mode,
            )

            qualifying_audio_allowed = (
                safe_room_busy_or_overlapping is not True
                and hard_blocker_reason is None
            )

            qualifying_audio_active, qualifying_audio_reason = _is_qualifying_audio(
                speech_detected=safe_speech_detected,
                presence_audio_active=safe_presence_audio_active,
                recent_follow_up_speech=safe_recent_follow_up_speech,
                speaker_direction_stable=safe_speaker_direction_stable,
                quiet_window_open=safe_quiet_window_open,
                room_busy_or_overlapping=safe_room_busy_or_overlapping,
                barge_in_recent=safe_barge_in_recent,
                resume_window_open=safe_resume_window_open,
                qualifying_audio_allowed=qualifying_audio_allowed,
            )

            qualifying_speech_anchor_active = (
                recently_present
                or self._armed
                or safe_resume_window_open is True
                or safe_barge_in_recent is True
            )
            if qualifying_audio_active and qualifying_speech_anchor_active:
                self._last_qualifying_speech_at = safe_now

            follow_up_anchor_active = qualifying_speech_anchor_active
            if safe_recent_follow_up_speech is True and follow_up_anchor_active:
                # BREAKING: follow-up context now gets its own deadline even when the raw speech timestamp is unavailable.
                self._last_follow_up_at = safe_now
            elif (
                qualifying_audio_active
                and qualifying_audio_reason in {"recent_follow_up_speech", "barge_in_recent", "resume_window_open"}
                and follow_up_anchor_active
            ):
                self._last_follow_up_at = safe_now

            qualifying_speech_age = _age(safe_now, self._last_qualifying_speech_at)
            follow_up_age = _age(safe_now, self._last_follow_up_at)
            _resume_age = _age(safe_now, self._last_resume_window_at)
            _barge_in_age = _age(safe_now, self._last_barge_in_at)

            person_deadline = _deadline(self._last_person_seen_at, self.presence_grace_s)
            motion_deadline = _deadline(self._last_motion_at, self.motion_grace_s)
            speech_deadline = _deadline(self._last_qualifying_speech_at, self.speech_grace_s)
            follow_up_deadline = _deadline(self._last_follow_up_at, self.follow_up_grace_s)
            resume_deadline = _deadline(self._last_resume_window_at, self.resume_grace_s)
            barge_in_deadline = _deadline(self._last_barge_in_at, self.barge_in_grace_s)

            active_reasons: list[tuple[int, str, float]] = []

            if safe_person_visible is True and person_deadline is not None and person_deadline >= safe_now:
                active_reasons.append((100, "person_visible", person_deadline))
            if safe_motion_active and motion_deadline is not None and motion_deadline >= safe_now:
                active_reasons.append((95, "pir_motion", motion_deadline))
            if (
                qualifying_audio_active
                and recently_present
                and speech_deadline is not None
                and speech_deadline >= safe_now
            ):
                # BREAKING: `speech_detected` by itself no longer qualifies unless directedness/context signals support it.
                active_reasons.append((90, "speech_while_recently_present", speech_deadline))
            if (
                safe_recent_follow_up_speech is True
                and (recently_present or safe_resume_window_open is True or safe_barge_in_recent is True)
                and follow_up_deadline is not None
                and follow_up_deadline >= safe_now
            ):
                active_reasons.append((92, "follow_up_speech_while_recently_present", follow_up_deadline))
            if person_deadline is not None and person_deadline >= safe_now:
                active_reasons.append((70, "recent_person_visible", person_deadline))
            if motion_deadline is not None and motion_deadline >= safe_now:
                active_reasons.append((65, "recent_pir_motion", motion_deadline))
            if speech_deadline is not None and speech_deadline >= safe_now:
                active_reasons.append((60, "recent_speech_while_present", speech_deadline))
            if follow_up_deadline is not None and follow_up_deadline >= safe_now:
                active_reasons.append((55, "recent_follow_up_context", follow_up_deadline))
            if resume_deadline is not None and resume_deadline >= safe_now:
                active_reasons.append((50, "recent_resume_window", resume_deadline))
            if barge_in_deadline is not None and barge_in_deadline >= safe_now:
                active_reasons.append((45, "recent_barge_in", barge_in_deadline))

            candidate_reason = None
            candidate_deadline = None
            if active_reasons:
                candidate_deadline = max(deadline for _, _, deadline in active_reasons)
                _, candidate_reason, _ = max(active_reasons, key=lambda item: (item[0], item[2]))

            if hard_blocker_reason is not None:
                # BREAKING: a hard privacy block now wins over presence/motion evidence to avoid reporting armed=True while capture is disabled.
                self._arm_until_s = None
                return self._build_snapshot(
                    safe_now=safe_now,
                    armed=False,
                    reason=hard_blocker_reason,
                    blocker_reason=hard_blocker_reason,
                    person_visible=safe_person_visible is True,
                    qualifying_audio_active=qualifying_audio_active,
                    arm_deadline_s=None,
                    last_person_seen_age_s=person_age,
                    last_motion_age_s=motion_age,
                    last_speech_age_s=speech_age,
                    last_qualifying_speech_age_s=qualifying_speech_age,
                    last_follow_up_age_s=follow_up_age,
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

            if candidate_deadline is not None:
                candidate_deadline = max(candidate_deadline, safe_now + self.arm_hysteresis_s)
                if self._arm_until_s is None:
                    self._arm_until_s = candidate_deadline
                else:
                    self._arm_until_s = max(self._arm_until_s, candidate_deadline)

            armed = self._arm_until_s is not None and safe_now <= self._arm_until_s
            if not armed:
                self._arm_until_s = None
                candidate_reason = "idle"
            elif candidate_reason is None:
                candidate_reason = "debounce_hold"

            return self._build_snapshot(
                safe_now=safe_now,
                armed=armed,
                reason=candidate_reason,
                blocker_reason=None,
                person_visible=safe_person_visible is True,
                qualifying_audio_active=qualifying_audio_active,
                arm_deadline_s=self._arm_until_s if armed else None,
                last_person_seen_age_s=person_age,
                last_motion_age_s=motion_age,
                last_speech_age_s=speech_age,
                last_qualifying_speech_age_s=qualifying_speech_age,
                last_follow_up_age_s=follow_up_age,
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
        safe_now: float,
        armed: bool,
        reason: str | None,
        blocker_reason: str | None,
        person_visible: bool,
        qualifying_audio_active: bool | None,
        arm_deadline_s: float | None,
        last_person_seen_age_s: float | None,
        last_motion_age_s: float | None,
        last_speech_age_s: float | None,
        last_qualifying_speech_age_s: float | None,
        last_follow_up_age_s: float | None,
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

        arm_expires_in_s = None
        if armed and arm_deadline_s is not None:
            arm_expires_in_s = max(0.0, arm_deadline_s - safe_now)

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
            blocker_reason=blocker_reason,
            qualifying_audio_active=qualifying_audio_active,
            arm_deadline_s=arm_deadline_s,
            arm_expires_in_s=arm_expires_in_s,
            last_qualifying_speech_age_s=last_qualifying_speech_age_s,
            last_follow_up_age_s=last_follow_up_age_s,
            policy_version=_POLICY_VERSION,
        )

    def _normalize_now(self, value: float) -> tuple[float, float, float]:
        """Repair regressing or stalled timestamps into a monotonic internal timeline."""

        local_monotonic_ns = time.monotonic_ns()
        local_elapsed_s = max(0, local_monotonic_ns - self._last_local_monotonic_ns) / 1_000_000_000.0
        self._last_local_monotonic_ns = local_monotonic_ns

        previous = self._last_observed_now
        fallback_now = (previous if previous is not None else 0.0) + local_elapsed_s

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._warn(
                "now.invalid_type",
                "PresenceSessionController.observe received non-numeric now=%r; using local monotonic fallback %.6f",
                value,
                fallback_now,
            )
            return fallback_now, self._time_offset_s, local_elapsed_s

        raw_now = float(value)
        if not math.isfinite(raw_now):
            self._warn(
                "now.non_finite",
                "PresenceSessionController.observe received non-finite now=%r; using local monotonic fallback %.6f",
                value,
                fallback_now,
            )
            return fallback_now, self._time_offset_s, local_elapsed_s

        adjusted_now = raw_now + self._time_offset_s
        if previous is not None and adjusted_now <= previous:
            if adjusted_now < previous:
                self._warn(
                    "now.regressing",
                    "PresenceSessionController.observe received regressing time %.6f after %.6f; repairing with local monotonic delta %.6f",
                    raw_now,
                    previous,
                    local_elapsed_s,
                )
            else:
                self._warn(
                    "now.stalled",
                    "PresenceSessionController.observe received stalled time %.6f; repairing with local monotonic delta %.6f",
                    raw_now,
                    local_elapsed_s,
                )
            adjusted_now = previous + local_elapsed_s
            self._time_offset_s = adjusted_now - raw_now

        return adjusted_now, self._time_offset_s, local_elapsed_s

    def _warn(self, key: str, message: str, *args: object) -> None:
        """Emit one warning per key per throttle window."""

        now = self._last_observed_now
        if now is None:
            now = time.monotonic_ns() / 1_000_000_000.0
        last_warning_at = self._last_warning_by_key.get(key)
        if last_warning_at is not None and now - last_warning_at < self.log_throttle_s:
            return
        self._last_warning_by_key[key] = now
        LOGGER.warning(message, *args)


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


def _validate_positive_int(value: int, *, field_name: str) -> int:
    """Validate one strictly positive integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a positive int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value!r}")
    return value


def _normalize_bool(
    value: bool,
    *,
    field_name: str,
    warn: Callable[..., None],
) -> bool:
    """Normalize one required runtime flag to a fail-closed boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    warn(
        f"{field_name}.invalid",
        "%s received non-bool %r; failing closed",
        field_name,
        value,
    )
    return False


def _normalize_optional_bool(
    value: bool | None,
    *,
    field_name: str,
    warn: Callable[..., None],
) -> bool | None:
    """Normalize one optional runtime flag to bool-or-unknown."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    warn(
        f"{field_name}.invalid",
        "%s received invalid optional bool %r; treating as unknown",
        field_name,
        value,
    )
    return None


def _normalize_optional_seconds(
    value: float | None,
    *,
    field_name: str,
    warn: Callable[..., None],
) -> float | None:
    """Normalize one optional non-negative duration to seconds-or-unknown."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warn(
            f"{field_name}.invalid",
            "%s received invalid optional seconds %r; treating as unknown",
            field_name,
            value,
        )
        return None
    seconds = float(value)
    if not math.isfinite(seconds):
        warn(
            f"{field_name}.non_finite",
            "%s received non-finite optional seconds %r; treating as unknown",
            field_name,
            value,
        )
        return None
    return max(0.0, seconds)


def _normalize_optional_text(
    value: str | None,
    *,
    field_name: str,
    max_len: int,
    warn: Callable[..., None],
) -> str | None:
    """Normalize one optional text field into a trimmed single line with a hard cap."""

    if value is None:
        return None
    if not isinstance(value, str):
        warn(
            f"{field_name}.invalid",
            "%s received invalid optional text %r; treating as unknown",
            field_name,
            value,
        )
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return None
    if len(normalized) > max_len:
        warn(
            f"{field_name}.truncated",
            "%s exceeded %d characters; truncating for safety",
            field_name,
            max_len,
        )
        return normalized[: max_len - 1].rstrip() + "…"
    return normalized


def _runtime_mode_blocks_capture(
    device_runtime_mode: str | None,
    *,
    blocking_modes: frozenset[str],
) -> bool:
    """Return whether the runtime mode explicitly blocks voice capture."""

    if device_runtime_mode is None:
        return False
    return device_runtime_mode.casefold() in blocking_modes


def _select_hard_blocker_reason(
    *,
    mute_blocks_voice_capture: bool | None,
    runtime_blocks_capture: bool,
    device_runtime_mode: str | None,
) -> str | None:
    """Return the highest-priority hard blocker reason, if any."""

    if mute_blocks_voice_capture is True:
        return "mute_blocks_voice_capture"
    if runtime_blocks_capture:
        suffix = device_runtime_mode.casefold() if device_runtime_mode is not None else "unknown"
        return f"runtime_mode_blocks_voice_capture:{suffix}"
    return None


def _is_qualifying_audio(
    *,
    speech_detected: bool,
    presence_audio_active: bool | None,
    recent_follow_up_speech: bool | None,
    speaker_direction_stable: bool | None,
    quiet_window_open: bool | None,
    room_busy_or_overlapping: bool | None,
    barge_in_recent: bool | None,
    resume_window_open: bool | None,
    qualifying_audio_allowed: bool,
) -> tuple[bool, str | None]:
    """Return whether current audio should count as qualifying presence speech."""

    if not qualifying_audio_allowed:
        return False, None

    if presence_audio_active is True:
        return True, "presence_audio_active"

    if recent_follow_up_speech is True and (
        resume_window_open is True
        or quiet_window_open is not False
        or barge_in_recent is True
    ):
        return True, "recent_follow_up_speech"

    if barge_in_recent is True:
        return True, "barge_in_recent"

    if resume_window_open is True and speech_detected:
        return True, "resume_window_open"

    if speech_detected and speaker_direction_stable is True and (
        quiet_window_open is not False
        and room_busy_or_overlapping is not True
    ):
        return True, "directed_speech"

    return False, None


def _deadline(since: float | None, grace_s: float) -> float | None:
    """Return the expiry timestamp for one observation."""

    if since is None:
        return None
    return since + grace_s


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