# CHANGELOG: 2026-03-29
# BUG-1: Fixed startup crash in from_config when proactive_capture_interval_s is None, non-numeric, or non-finite.
# BUG-2: Fixed permissive initiative/resume behavior when ReSpeaker runtime is degraded, unavailable, or host-control is untrusted.
# BUG-3: Fixed quiet-window false opens when recent_speech_age_s is missing by requiring explicit quiet evidence before opening initiative.
# BUG-4: Fixed elapsed-time logic against backward wall-clock jumps by normalizing observe() time with a local monotonic fallback.
# SEC-1: Sanitized and length-bounded operator-facing transport details to reduce terminal/log injection and log-spam risk.
# IMP-1: Added explicit capture-readiness, signal-trust, staleness, and recovery-guard outputs for 2026 streaming/duplex audio pipelines.
# IMP-2: Added runtime alert severity while preserving the existing public API and drop-in observe()/describe_respeaker_runtime_alert() entry points.
# BREAKING: ReSpeakerAudioPolicySnapshot now exposes additional optional fields and initiative_block_reason may return the new conservative
#            values "audio_observation_stale", "audio_signals_untrusted", "device_recovery_guard", and
#            "respeaker_host_control_unavailable". Existing fields and function signatures are preserved.

"""Derive conservative ReSpeaker policy hooks from normalized audio facts.

This module keeps ReSpeaker-specific policy interpretation out of the proactive
coordinator and presence-session controller. It turns low-level audio facts
into bounded runtime signals such as quiet windows, resume windows, overlap
guards, and operator-readable device alerts.

The 2026 upgrade in this file keeps the module dependency-light for Raspberry Pi
deployments while adding the policy hooks that newer duplex/streaming front-ends
need: signal trust, observation staleness, and short recovery guards after
runtime degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.social.engine import SocialAudioObservation


_DEFAULT_DIRECTION_STABLE_MIN_CONFIDENCE = 0.75
_DEFAULT_FOLLOW_UP_WINDOW_S = 4.0
_MIN_FOLLOW_UP_WINDOW_S = 2.0
_MAX_FOLLOW_UP_WINDOW_S = 8.0
_DEFAULT_QUIET_WINDOW_MIN_S = 1.0
_DEFAULT_OBSERVATION_STALE_AFTER_S = 3.0
_MIN_OBSERVATION_STALE_AFTER_S = 0.5
_MAX_OBSERVATION_STALE_AFTER_S = 30.0
_DEFAULT_RUNTIME_RECOVERY_GUARD_S = 0.75
_MIN_RUNTIME_RECOVERY_GUARD_S = 0.0
_MAX_RUNTIME_RECOVERY_GUARD_S = 5.0
_MAX_REASONABLE_OBSERVATION_AGE_S = 300.0
_MAX_OPERATOR_DETAIL_CHARS = 160

_RUNTIME_CAPTURE_BLOCKED_MODES = frozenset(
    {
        "not_detected",
        "usb_visible_no_capture",
        "usb_visible_capture_unknown",
    }
)
_RUNTIME_SIGNAL_UNTRUSTED_MODES = frozenset(
    _RUNTIME_CAPTURE_BLOCKED_MODES
    | {
        "probe_unavailable",
        "provider_lock_timeout",
        "signal_provider_error",
    }
)


@dataclass(frozen=True, slots=True)
class ReSpeakerAudioPolicySnapshot:
    """Store conservative policy-facing facts derived from one audio tick."""

    observed_at: float
    presence_audio_active: bool | None = None
    recent_follow_up_speech: bool | None = None
    room_busy_or_overlapping: bool | None = None
    quiet_window_open: bool | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    barge_in_recent: bool | None = None
    speaker_direction_stable: bool | None = None
    mute_blocks_voice_capture: bool | None = None
    device_capture_ready: bool | None = None
    audio_signals_trusted: bool | None = None
    observation_age_s: float | None = None
    audio_observation_stale: bool | None = None
    recovery_guard_active: bool | None = None
    resume_window_open: bool | None = None
    initiative_block_reason: str | None = None
    speech_delivery_defer_reason: str | None = None
    runtime_alert_code: str | None = None
    runtime_alert_severity: str | None = None
    runtime_alert_message: str | None = None


class ReSpeakerAudioPolicyTracker:
    """Track bounded ReSpeaker policy state across audio observations.

    The tracker remembers only short-lived runtime state needed to turn raw
    audio facts into operator-safe policy inputs. It does not persist memory or
    product behavior by itself.
    """

    def __init__(
        self,
        *,
        follow_up_window_s: float = _DEFAULT_FOLLOW_UP_WINDOW_S,
        quiet_window_min_s: float = _DEFAULT_QUIET_WINDOW_MIN_S,
        direction_stable_min_confidence: float = _DEFAULT_DIRECTION_STABLE_MIN_CONFIDENCE,
        observation_stale_after_s: float = _DEFAULT_OBSERVATION_STALE_AFTER_S,
        runtime_recovery_guard_s: float = _DEFAULT_RUNTIME_RECOVERY_GUARD_S,
    ) -> None:
        """Initialize one tracker with bounded conservative policy windows."""

        self.follow_up_window_s = _coerce_seconds(
            follow_up_window_s,
            default=_DEFAULT_FOLLOW_UP_WINDOW_S,
            minimum=_MIN_FOLLOW_UP_WINDOW_S,
            maximum=_MAX_FOLLOW_UP_WINDOW_S,
        )
        self.quiet_window_min_s = _coerce_seconds(
            quiet_window_min_s,
            default=_DEFAULT_QUIET_WINDOW_MIN_S,
            minimum=0.5,
            maximum=10.0,
        )
        self.direction_stable_min_confidence = _coerce_ratio(
            direction_stable_min_confidence,
            default=_DEFAULT_DIRECTION_STABLE_MIN_CONFIDENCE,
        )
        self.observation_stale_after_s = _coerce_seconds(
            observation_stale_after_s,
            default=_DEFAULT_OBSERVATION_STALE_AFTER_S,
            minimum=_MIN_OBSERVATION_STALE_AFTER_S,
            maximum=_MAX_OBSERVATION_STALE_AFTER_S,
        )
        self.runtime_recovery_guard_s = _coerce_seconds(
            runtime_recovery_guard_s,
            default=_DEFAULT_RUNTIME_RECOVERY_GUARD_S,
            minimum=_MIN_RUNTIME_RECOVERY_GUARD_S,
            maximum=_MAX_RUNTIME_RECOVERY_GUARD_S,
        )
        self._last_assistant_output_at: float | None = None
        self._last_assistant_output_finished_at: float | None = None
        self._last_follow_up_speech_at: float | None = None
        self._last_barge_in_at: float | None = None
        self._last_untrusted_runtime_at: float | None = None
        self._previous_assistant_output_active: bool | None = None
        self._last_observed_at: float | None = None
        self._last_local_monotonic_at: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ReSpeakerAudioPolicyTracker":
        """Build one tracker from existing Twinr config without new env keys."""

        capture_interval_s = _coerce_seconds(
            getattr(config, "proactive_capture_interval_s", _DEFAULT_QUIET_WINDOW_MIN_S),
            default=_DEFAULT_QUIET_WINDOW_MIN_S,
            minimum=0.5,
            maximum=10.0,
        )
        follow_up_window_s = _coerce_seconds(
            getattr(
                config,
                "conversation_follow_up_timeout_s",
                _DEFAULT_FOLLOW_UP_WINDOW_S,
            ),
            default=_DEFAULT_FOLLOW_UP_WINDOW_S,
            minimum=_MIN_FOLLOW_UP_WINDOW_S,
            maximum=_MAX_FOLLOW_UP_WINDOW_S,
        )
        quiet_window_min_s = max(_DEFAULT_QUIET_WINDOW_MIN_S, capture_interval_s)
        observation_stale_after_s = max(
            _DEFAULT_OBSERVATION_STALE_AFTER_S,
            capture_interval_s * 2.0,
        )
        runtime_recovery_guard_s = min(
            _MAX_RUNTIME_RECOVERY_GUARD_S,
            max(_DEFAULT_RUNTIME_RECOVERY_GUARD_S, capture_interval_s),
        )
        return cls(
            follow_up_window_s=follow_up_window_s,
            quiet_window_min_s=quiet_window_min_s,
            direction_stable_min_confidence=getattr(
                config,
                "audio_direction_stable_min_confidence",
                _DEFAULT_DIRECTION_STABLE_MIN_CONFIDENCE,
            ),
            observation_stale_after_s=observation_stale_after_s,
            runtime_recovery_guard_s=runtime_recovery_guard_s,
        )

    def observe(
        self,
        *,
        now: float,
        audio: SocialAudioObservation,
    ) -> ReSpeakerAudioPolicySnapshot:
        """Return conservative policy facts for one normalized audio tick."""

        observed_at = self._normalize_observed_at(now)
        assistant_output_active = audio.assistant_output_active is True
        if assistant_output_active:
            self._last_assistant_output_at = observed_at
        elif self._previous_assistant_output_active is True:
            self._last_assistant_output_finished_at = observed_at

        runtime_alert_code, runtime_alert_message, runtime_alert_severity = (
            describe_respeaker_runtime_alert_details(audio)
        )
        device_capture_ready = _capture_ready(audio.device_runtime_mode)

        observation_age_s = _observation_age_s(audio=audio, now=observed_at)
        audio_observation_stale = (
            observation_age_s > self.observation_stale_after_s
            if observation_age_s is not None
            else None
        )
        audio_signals_trusted = _audio_signals_trusted(
            mode=audio.device_runtime_mode,
            host_control_ready=audio.host_control_ready,
            audio_observation_stale=audio_observation_stale,
            transport_reason=audio.transport_reason,
        )
        if audio_signals_trusted is not True:
            self._last_untrusted_runtime_at = observed_at
        recovery_guard_active = (
            self.runtime_recovery_guard_s > 0.0
            and self._recent(
                self._last_untrusted_runtime_at,
                observed_at,
                self.runtime_recovery_guard_s,
            )
        )

        if audio.mute_active is True:
            mute_blocks_voice_capture: bool | None = True
        elif device_capture_ready is not None:
            mute_blocks_voice_capture = not device_capture_ready
        else:
            mute_blocks_voice_capture = None

        room_busy_or_overlapping = audio.speech_overlap_likely
        recent_speech_age_s = _coerce_optional_seconds(audio.recent_speech_age_s, minimum=0.0)
        recent_output_context = self._recent(
            self._last_assistant_output_at,
            observed_at,
            self.follow_up_window_s,
        ) or self._recent(
            self._last_assistant_output_finished_at,
            observed_at,
            self.follow_up_window_s,
        )
        if (
            audio_signals_trusted is True
            and recent_output_context
            and recent_speech_age_s is not None
            and recent_speech_age_s <= self.follow_up_window_s
        ):
            reconstructed_speech_at = max(0.0, observed_at - recent_speech_age_s)
            if (
                self._last_follow_up_speech_at is None
                or reconstructed_speech_at > self._last_follow_up_speech_at
            ):
                self._last_follow_up_speech_at = reconstructed_speech_at

        if audio_signals_trusted is True and audio.barge_in_detected is True:
            self._last_barge_in_at = observed_at

        recent_follow_up_speech = self._recent(
            self._last_follow_up_speech_at,
            observed_at,
            self.follow_up_window_s,
        )
        barge_in_recent = self._recent(
            self._last_barge_in_at,
            observed_at,
            self.follow_up_window_s,
        )
        presence_audio_active = _presence_audio_active(
            audio=audio,
            room_busy_or_overlapping=room_busy_or_overlapping,
            mute_blocks_voice_capture=mute_blocks_voice_capture,
            audio_signals_trusted=audio_signals_trusted,
            recovery_guard_active=recovery_guard_active,
        )
        quiet_window_open = _quiet_window_open(
            audio=audio,
            recent_speech_age_s=recent_speech_age_s,
            quiet_window_min_s=self.quiet_window_min_s,
            room_busy_or_overlapping=room_busy_or_overlapping,
            audio_signals_trusted=audio_signals_trusted,
            recovery_guard_active=recovery_guard_active,
        )
        non_speech_audio_likely = audio.non_speech_audio_likely
        background_media_likely = audio.background_media_likely
        speaker_direction_stable = _speaker_direction_stable(
            direction_confidence=audio.direction_confidence,
            minimum=self.direction_stable_min_confidence,
            room_busy_or_overlapping=room_busy_or_overlapping,
            audio_signals_trusted=audio_signals_trusted,
            speech_detected=audio.speech_detected,
            recovery_guard_active=recovery_guard_active,
        )
        resume_window_open = (
            audio_signals_trusted is True
            and recovery_guard_active is not True
            and (barge_in_recent is True or recent_follow_up_speech is True)
            and mute_blocks_voice_capture is not True
        )
        initiative_block_reason = _initiative_block_reason(
            presence_audio_active=presence_audio_active,
            room_busy_or_overlapping=room_busy_or_overlapping,
            quiet_window_open=quiet_window_open,
            mute_blocks_voice_capture=mute_blocks_voice_capture,
            resume_window_open=resume_window_open,
            runtime_alert_code=runtime_alert_code,
            non_speech_audio_likely=non_speech_audio_likely,
            background_media_likely=background_media_likely,
            audio_signals_trusted=audio_signals_trusted,
            audio_observation_stale=audio_observation_stale,
            recovery_guard_active=recovery_guard_active,
        )
        speech_delivery_defer_reason = _speech_delivery_defer_reason(
            non_speech_audio_likely=non_speech_audio_likely,
            background_media_likely=background_media_likely,
        )
        self._previous_assistant_output_active = assistant_output_active
        return ReSpeakerAudioPolicySnapshot(
            observed_at=observed_at,
            presence_audio_active=presence_audio_active,
            recent_follow_up_speech=recent_follow_up_speech,
            room_busy_or_overlapping=room_busy_or_overlapping,
            quiet_window_open=quiet_window_open,
            non_speech_audio_likely=non_speech_audio_likely,
            background_media_likely=background_media_likely,
            barge_in_recent=barge_in_recent,
            speaker_direction_stable=speaker_direction_stable,
            mute_blocks_voice_capture=mute_blocks_voice_capture,
            device_capture_ready=device_capture_ready,
            audio_signals_trusted=audio_signals_trusted,
            observation_age_s=observation_age_s,
            audio_observation_stale=audio_observation_stale,
            recovery_guard_active=recovery_guard_active,
            resume_window_open=resume_window_open,
            initiative_block_reason=initiative_block_reason,
            speech_delivery_defer_reason=speech_delivery_defer_reason,
            runtime_alert_code=runtime_alert_code,
            runtime_alert_severity=runtime_alert_severity,
            runtime_alert_message=runtime_alert_message,
        )

    def _normalize_observed_at(self, now: object) -> float:
        """Return one non-decreasing observation timestamp.

        Upstream often passes wall-clock seconds. On Raspberry Pi deployments,
        NTP or RTC corrections can move wall time backward. This method preserves
        elapsed-time semantics by falling back to a local monotonic delta when
        that happens.
        """

        observed_at = _coerce_seconds(now, default=0.0, minimum=0.0)
        local_monotonic_at = time.monotonic()
        if self._last_observed_at is None:
            self._last_observed_at = observed_at
            self._last_local_monotonic_at = local_monotonic_at
            return observed_at
        if observed_at >= self._last_observed_at:
            self._last_observed_at = observed_at
            self._last_local_monotonic_at = local_monotonic_at
            return observed_at
        previous_local_monotonic_at = (
            self._last_local_monotonic_at
            if self._last_local_monotonic_at is not None
            else local_monotonic_at
        )
        observed_at = self._last_observed_at + max(
            0.0,
            local_monotonic_at - previous_local_monotonic_at,
        )
        self._last_observed_at = observed_at
        self._last_local_monotonic_at = local_monotonic_at
        return observed_at

    def _recent(
        self,
        since: float | None,
        now: float,
        window_s: float,
    ) -> bool:
        """Return whether one optional timestamp still falls into a window."""

        if since is None:
            return False
        return max(0.0, now - since) <= window_s


def describe_respeaker_runtime_alert(
    audio: SocialAudioObservation,
) -> tuple[str | None, str | None]:
    """Return one operator-readable runtime alert code and message."""

    code, message, _severity = describe_respeaker_runtime_alert_details(audio)
    return code, message


def describe_respeaker_runtime_alert_details(
    audio: SocialAudioObservation,
) -> tuple[str | None, str | None, str | None]:
    """Return one operator-readable runtime alert code, message, and severity."""

    mode = _normalize_mode(audio.device_runtime_mode)
    transport_reason = _sanitize_operator_text(audio.transport_reason)
    if mode == "audio_ready":
        if audio.mute_active is True:
            return (
                "mic_muted",
                "ReSpeaker microphone is muted. Voice capture is blocked until the mic is unmuted.",
                "warning",
            )
        if audio.host_control_ready is False:
            detail = transport_reason or "host_control_blocked"
            return (
                "host_control_unavailable",
                f"ReSpeaker capture is present, but host-control reads are blocked ({detail}).",
                "warning",
            )
        return (
            "ready",
            "ReSpeaker capture and host-control are ready.",
            "ok",
        )
    if mode == "usb_visible_no_capture":
        return (
            "dfu_mode",
            "ReSpeaker is visible on USB but has no ALSA capture device. The board is likely in DFU/safe mode or its audio runtime did not boot.",
            "error",
        )
    if mode == "usb_visible_capture_unknown":
        return (
            "capture_unknown",
            "ReSpeaker is visible on USB, but ALSA capture readiness could not be confirmed.",
            "warning",
        )
    if mode == "not_detected":
        return (
            "disconnected",
            "ReSpeaker is not visible to the Pi. Check cable, board port, or USB data path.",
            "error",
        )
    if mode == "probe_unavailable":
        return (
            "probe_unavailable",
            "ReSpeaker runtime tools are unavailable, so device state could not be verified.",
            "error",
        )
    if mode == "provider_lock_timeout":
        return (
            "provider_lock_timeout",
            "ReSpeaker signal reads timed out because the provider lock stayed busy.",
            "error",
        )
    if mode == "signal_provider_error":
        detail = transport_reason or "unknown_error"
        return (
            "signal_provider_error",
            f"ReSpeaker signal reads failed inside the runtime provider ({detail}).",
            "error",
        )
    if transport_reason:
        return (
            "transport_blocked",
            f"ReSpeaker host-control transport is blocked ({transport_reason}).",
            "warning",
        )
    return None, None, None


def _presence_audio_active(
    *,
    audio: SocialAudioObservation,
    room_busy_or_overlapping: bool | None,
    mute_blocks_voice_capture: bool | None,
    audio_signals_trusted: bool | None,
    recovery_guard_active: bool,
) -> bool | None:
    """Return whether current speech should count as calm presence audio."""

    if audio_signals_trusted is not True:
        return None
    if recovery_guard_active:
        return None
    if audio.speech_detected is None:
        return None
    return (
        audio.speech_detected is True
        and audio.non_speech_audio_likely is not True
        and audio.background_media_likely is not True
        and room_busy_or_overlapping is not True
        and mute_blocks_voice_capture is not True
    )


def _quiet_window_open(
    *,
    audio: SocialAudioObservation,
    recent_speech_age_s: float | None,
    quiet_window_min_s: float,
    room_busy_or_overlapping: bool | None,
    audio_signals_trusted: bool | None,
    recovery_guard_active: bool,
) -> bool | None:
    """Return whether the room currently looks quiet enough for initiative."""

    if audio_signals_trusted is not True:
        return None
    if recovery_guard_active:
        return False
    if audio.room_quiet is None:
        return False
    if audio.non_speech_audio_likely is True or audio.background_media_likely is True:
        return False
    if audio.room_quiet is not True:
        return False
    if audio.assistant_output_active is True:
        return False
    if room_busy_or_overlapping is True:
        return False
    if recent_speech_age_s is not None:
        return recent_speech_age_s >= quiet_window_min_s
    return audio.speech_detected is False


def _speaker_direction_stable(
    *,
    direction_confidence: float | None,
    minimum: float,
    room_busy_or_overlapping: bool | None,
    audio_signals_trusted: bool | None,
    speech_detected: bool | None,
    recovery_guard_active: bool,
) -> bool | None:
    """Return whether direction confidence is strong enough for policy use."""

    if audio_signals_trusted is not True:
        return None
    if recovery_guard_active:
        return None
    if speech_detected is not True:
        return None
    if direction_confidence is None:
        return None
    return direction_confidence >= minimum and room_busy_or_overlapping is not True


def _initiative_block_reason(
    *,
    presence_audio_active: bool | None,
    room_busy_or_overlapping: bool | None,
    quiet_window_open: bool | None,
    mute_blocks_voice_capture: bool | None,
    resume_window_open: bool | None,
    runtime_alert_code: str | None,
    non_speech_audio_likely: bool | None,
    background_media_likely: bool | None,
    audio_signals_trusted: bool | None,
    audio_observation_stale: bool | None,
    recovery_guard_active: bool | None,
) -> str | None:
    """Return one conservative non-safety suppression reason when needed."""

    if audio_observation_stale is True:
        return "audio_observation_stale"
    if mute_blocks_voice_capture is True:
        if runtime_alert_code == "dfu_mode":
            return "respeaker_dfu_mode"
        if runtime_alert_code in {
            "disconnected",
            "capture_unknown",
            "probe_unavailable",
        }:
            return "respeaker_unavailable"
        return "mute_blocks_voice_capture"
    if audio_signals_trusted is not True:
        if runtime_alert_code == "host_control_unavailable":
            return "respeaker_host_control_unavailable"
        return "audio_signals_untrusted"
    if recovery_guard_active is True:
        return "device_recovery_guard"
    if room_busy_or_overlapping is True:
        return "room_busy_or_overlapping"
    if resume_window_open is True:
        return "resume_window_open"
    if presence_audio_active is True:
        return "presence_audio_active"
    if background_media_likely is True or non_speech_audio_likely is True:
        return None
    if quiet_window_open is False:
        return "quiet_window_not_open"
    return None


def _speech_delivery_defer_reason(
    *,
    non_speech_audio_likely: bool | None,
    background_media_likely: bool | None,
) -> str | None:
    """Return whether speech should downgrade to a visual-first delivery."""

    if background_media_likely is True:
        return "background_media_active"
    if non_speech_audio_likely is True:
        return "non_speech_audio_active"
    return None


def _audio_signals_trusted(
    *,
    mode: str | None,
    host_control_ready: bool | None,
    audio_observation_stale: bool | None,
    transport_reason: str | None,
) -> bool | None:
    """Return whether policy should trust the current runtime signals."""

    if audio_observation_stale is True:
        return False
    normalized_mode = _normalize_mode(mode)
    if host_control_ready is False:
        return False
    if normalized_mode == "audio_ready":
        return True
    if normalized_mode in _RUNTIME_SIGNAL_UNTRUSTED_MODES:
        return False
    if _sanitize_operator_text(transport_reason) is not None:
        return False
    return None


def _capture_ready(mode: str | None) -> bool | None:
    """Return whether the current runtime mode implies working capture."""

    normalized_mode = _normalize_mode(mode)
    if not normalized_mode:
        return None
    if normalized_mode == "audio_ready":
        return True
    if normalized_mode in _RUNTIME_CAPTURE_BLOCKED_MODES:
        return False
    return None


def _normalize_mode(value: object) -> str | None:
    """Normalize one optional runtime mode string."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    normalized = "_".join(normalized.split())
    return normalized or None


def _normalize_optional_text(value: object) -> str | None:
    """Normalize one optional text field to a trimmed single line."""

    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _sanitize_operator_text(
    value: object,
    *,
    maximum_chars: int = _MAX_OPERATOR_DETAIL_CHARS,
) -> str | None:
    """Return bounded printable operator text.

    This keeps terminal escape/control characters and oversized transport
    details out of operator-visible alerts and logs.
    """

    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    printable_characters: list[str] = []
    for character in str(normalized):
        if character.isprintable():
            printable_characters.append(character)
    sanitized = "".join(printable_characters)
    sanitized = sanitized.strip()
    if not sanitized:
        return None
    if len(sanitized) <= maximum_chars:
        return sanitized
    return f"{sanitized[: max(1, maximum_chars - 1)].rstrip()}…"


def _observation_age_s(
    *,
    audio: SocialAudioObservation,
    now: float,
) -> float | None:
    """Return optional age of the current normalized observation.

    This method is deliberately tolerant of older upstream schemas. It uses an
    explicit age when provided and otherwise falls back to a same-clock observed
    timestamp only when the computed delta looks reasonable.
    """

    explicit_age = _coerce_optional_seconds(
        getattr(audio, "observation_age_s", None),
        minimum=0.0,
    )
    if explicit_age is not None:
        return explicit_age

    observed_at = _coerce_optional_seconds(
        getattr(audio, "observed_at", None),
        minimum=0.0,
    )
    if observed_at is None or observed_at > now:
        return None

    computed_age_s = now - observed_at
    if computed_age_s > _MAX_REASONABLE_OBSERVATION_AGE_S:
        return None
    return computed_age_s


def _coerce_seconds(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Coerce one config-like number into finite bounded seconds."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return float(number)


def _coerce_optional_seconds(value: object, *, minimum: float) -> float | None:
    """Coerce one optional duration into a non-negative finite number."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(minimum, number)


def _coerce_ratio(value: object, *, default: float) -> float:
    """Coerce one optional ratio into the unit interval."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(1.0, number))


__all__ = [
    "ReSpeakerAudioPolicySnapshot",
    "ReSpeakerAudioPolicyTracker",
    "describe_respeaker_runtime_alert",
    "describe_respeaker_runtime_alert_details",
]
