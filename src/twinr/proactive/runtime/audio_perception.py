"""Build bounded ReSpeaker audio-perception snapshots for diagnostics.

This module reuses the same normalized audio observation and ReSpeaker policy
surfaces that the proactive monitor already trusts. It exists so operator-facing
CLI checks and post-recovery hardware smokes can validate real runtime facts
such as speech, non-speech activity, background-media suspicion, and
device-directed speech candidacy without duplicating monitor orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler, AudioCaptureReadinessProbe
from twinr.hardware.respeaker import ReSpeakerSignalProvider, config_targets_respeaker
from twinr.proactive.social import SocialAudioObservation
from twinr.proactive.social.observers import (
    AmbientAudioObservationProvider,
    ProactiveAudioSnapshot,
    ReSpeakerAudioObservationProvider,
)

from .audio_policy import ReSpeakerAudioPolicySnapshot, ReSpeakerAudioPolicyTracker


@dataclass(frozen=True, slots=True)
class ReSpeakerAudioPerceptionGuardSnapshot:
    """Summarize conservative room-context and device-directedness hints."""

    room_context: str
    device_directed_speech_candidate: bool
    guard_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReSpeakerAudioPerceptionSnapshot:
    """Bundle one runtime-faithful audio observation with guard metadata."""

    audio_snapshot: ProactiveAudioSnapshot
    audio_policy_snapshot: ReSpeakerAudioPolicySnapshot
    perception_guard: ReSpeakerAudioPerceptionGuardSnapshot
    respeaker_targeted: bool
    capture_probe: AudioCaptureReadinessProbe | None = None


def observe_audio_perception_once(config: TwinrConfig) -> ReSpeakerAudioPerceptionSnapshot:
    """Capture one bounded audio-perception snapshot from current config."""

    sampler = AmbientAudioSampler.from_config(config)
    fallback_observer = AmbientAudioObservationProvider(
        sampler=sampler,
        sample_ms=config.proactive_audio_sample_ms,
        distress_enabled=bool(config.proactive_enabled and config.proactive_audio_distress_enabled),
    )
    observer: object = fallback_observer
    capture_probe: AudioCaptureReadinessProbe | None = None
    respeaker_targeted = config_targets_respeaker(
        getattr(config, "audio_input_device", None),
        getattr(config, "proactive_audio_input_device", None),
    )
    if respeaker_targeted:
        capture_probe = sampler.require_readable_frames(
            duration_ms=min(max(sampler.chunk_ms, 250), max(sampler.chunk_ms, config.proactive_audio_sample_ms)),
        )
        observer = ReSpeakerAudioObservationProvider(
            signal_provider=ReSpeakerSignalProvider(
                sensor_window_ms=config.proactive_audio_sample_ms,
                assistant_output_active_predicate=lambda: False,
            ),
            fallback_observer=fallback_observer,
        )
    try:
        audio_snapshot = observer.observe()
    finally:
        close = getattr(observer, "close", None)
        if callable(close):
            close()
    audio_policy_snapshot = ReSpeakerAudioPolicyTracker.from_config(config).observe(
        now=time.monotonic(),
        audio=audio_snapshot.observation,
    )
    return ReSpeakerAudioPerceptionSnapshot(
        audio_snapshot=audio_snapshot,
        audio_policy_snapshot=audio_policy_snapshot,
        perception_guard=derive_respeaker_audio_perception_guard(
            audio=audio_snapshot.observation,
            policy_snapshot=audio_policy_snapshot,
        ),
        respeaker_targeted=respeaker_targeted,
        capture_probe=capture_probe,
    )


def derive_respeaker_audio_perception_guard(
    *,
    audio: SocialAudioObservation,
    policy_snapshot: ReSpeakerAudioPolicySnapshot,
) -> ReSpeakerAudioPerceptionGuardSnapshot:
    """Derive one conservative room-context guard from normalized audio facts."""

    runtime_alert_code = _normalize_optional_text(policy_snapshot.runtime_alert_code)
    if runtime_alert_code not in {None, "ready"}:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="capture_blocked",
            device_directed_speech_candidate=False,
            guard_reason=runtime_alert_code,
        )
    if policy_snapshot.mute_blocks_voice_capture is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="muted",
            device_directed_speech_candidate=False,
            guard_reason="mute_blocks_voice_capture",
        )
    if audio.assistant_output_active is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="assistant_output",
            device_directed_speech_candidate=False,
            guard_reason="assistant_output_active",
        )
    if policy_snapshot.background_media_likely is True or audio.background_media_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="background_media",
            device_directed_speech_candidate=False,
            guard_reason="background_media_active",
        )
    if policy_snapshot.non_speech_audio_likely is True or audio.non_speech_audio_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="non_speech_activity",
            device_directed_speech_candidate=False,
            guard_reason="non_speech_audio_active",
        )
    if policy_snapshot.room_busy_or_overlapping is True or audio.speech_overlap_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="overlapping_speech",
            device_directed_speech_candidate=False,
            guard_reason="room_busy_or_overlapping",
        )
    if audio.speech_detected is True:
        if policy_snapshot.presence_audio_active is not True:
            return ReSpeakerAudioPerceptionGuardSnapshot(
                room_context="speech",
                device_directed_speech_candidate=False,
                guard_reason="presence_audio_inactive",
            )
        if policy_snapshot.speaker_direction_stable is not True:
            return ReSpeakerAudioPerceptionGuardSnapshot(
                room_context="speech",
                device_directed_speech_candidate=False,
                guard_reason="speaker_direction_unstable",
            )
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="speech",
            device_directed_speech_candidate=True,
            guard_reason=None,
        )
    if policy_snapshot.quiet_window_open is True or audio.room_quiet is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="quiet",
            device_directed_speech_candidate=False,
            guard_reason="quiet_room",
        )
    return ReSpeakerAudioPerceptionGuardSnapshot(
        room_context="unknown",
        device_directed_speech_candidate=False,
        guard_reason="insufficient_audio_evidence",
    )


def render_audio_perception_snapshot_lines(
    snapshot: ReSpeakerAudioPerceptionSnapshot,
) -> tuple[str, ...]:
    """Render one operator-readable line set for CLI and shell diagnostics."""

    audio_snapshot = snapshot.audio_snapshot
    observation = audio_snapshot.observation
    policy_snapshot = snapshot.audio_policy_snapshot
    signal_snapshot = audio_snapshot.signal_snapshot
    sample = audio_snapshot.sample
    capture_probe = snapshot.capture_probe
    lines = [
        f"proactive_audio_target={'respeaker' if snapshot.respeaker_targeted else 'ambient'}",
        f"proactive_audio_capture_probe_ready={_format_optional_bool(None if capture_probe is None else capture_probe.ready)}",
        "proactive_audio_capture_probe_chunks="
        f"{'unknown' if capture_probe is None else f'{capture_probe.captured_chunk_count}/{capture_probe.target_chunk_count}'}",
        f"proactive_audio_capture_probe_bytes={_format_optional_int(None if capture_probe is None else capture_probe.captured_bytes)}",
        f"proactive_speech_detected={_format_optional_bool(observation.speech_detected)}",
        f"proactive_distress_detected={_format_optional_bool(observation.distress_detected)}",
        f"proactive_room_quiet={_format_optional_bool(observation.room_quiet)}",
        f"proactive_recent_speech_age_s={_format_optional_float(observation.recent_speech_age_s)}",
        f"proactive_assistant_output_active={_format_optional_bool(observation.assistant_output_active)}",
        f"proactive_audio_azimuth_deg={_format_optional_int(observation.azimuth_deg)}",
        f"proactive_audio_direction_confidence={_format_optional_float(observation.direction_confidence)}",
        f"proactive_audio_device_runtime_mode={_format_optional_text(observation.device_runtime_mode)}",
        f"proactive_audio_host_control_ready={_format_optional_bool(observation.host_control_ready)}",
        f"proactive_audio_transport_reason={_format_optional_text(observation.transport_reason)}",
        f"proactive_non_speech_audio_likely={_format_optional_bool(observation.non_speech_audio_likely)}",
        f"proactive_background_media_likely={_format_optional_bool(observation.background_media_likely)}",
        f"proactive_speech_overlap_likely={_format_optional_bool(observation.speech_overlap_likely)}",
        f"proactive_barge_in_detected={_format_optional_bool(observation.barge_in_detected)}",
        f"proactive_mute_active={_format_optional_bool(observation.mute_active)}",
        f"proactive_audio_peak_rms={_format_optional_int(None if sample is None else sample.peak_rms)}",
        f"proactive_audio_average_rms={_format_optional_int(None if sample is None else sample.average_rms)}",
        f"proactive_audio_active_ratio={_format_optional_float(None if sample is None else sample.active_ratio)}",
        f"proactive_audio_policy_presence_audio_active={_format_optional_bool(policy_snapshot.presence_audio_active)}",
        f"proactive_audio_policy_recent_follow_up_speech={_format_optional_bool(policy_snapshot.recent_follow_up_speech)}",
        f"proactive_audio_policy_room_busy_or_overlapping={_format_optional_bool(policy_snapshot.room_busy_or_overlapping)}",
        f"proactive_audio_policy_quiet_window_open={_format_optional_bool(policy_snapshot.quiet_window_open)}",
        f"proactive_audio_policy_non_speech_audio_likely={_format_optional_bool(policy_snapshot.non_speech_audio_likely)}",
        f"proactive_audio_policy_background_media_likely={_format_optional_bool(policy_snapshot.background_media_likely)}",
        f"proactive_audio_policy_barge_in_recent={_format_optional_bool(policy_snapshot.barge_in_recent)}",
        f"proactive_audio_policy_speaker_direction_stable={_format_optional_bool(policy_snapshot.speaker_direction_stable)}",
        f"proactive_audio_policy_mute_blocks_voice_capture={_format_optional_bool(policy_snapshot.mute_blocks_voice_capture)}",
        f"proactive_audio_policy_resume_window_open={_format_optional_bool(policy_snapshot.resume_window_open)}",
        f"proactive_audio_policy_initiative_block_reason={_format_optional_text(policy_snapshot.initiative_block_reason)}",
        f"proactive_audio_policy_speech_delivery_defer_reason={_format_optional_text(policy_snapshot.speech_delivery_defer_reason)}",
        f"proactive_audio_policy_runtime_alert_code={_format_optional_text(policy_snapshot.runtime_alert_code)}",
        f"proactive_audio_policy_runtime_alert_message={_format_optional_text(policy_snapshot.runtime_alert_message)}",
        f"proactive_audio_room_context={snapshot.perception_guard.room_context}",
        "proactive_device_directed_speech_candidate="
        f"{_format_optional_bool(snapshot.perception_guard.device_directed_speech_candidate)}",
        f"proactive_audio_guard_reason={_format_optional_text(snapshot.perception_guard.guard_reason)}",
    ]
    if signal_snapshot is not None:
        lines.extend(
            (
                f"proactive_audio_signal_source={_format_optional_text(signal_snapshot.source)}",
                "proactive_audio_signal_requires_elevated_permissions="
                f"{_format_optional_bool(signal_snapshot.requires_elevated_permissions)}",
            )
        )
    return tuple(lines)


def _format_optional_bool(value: bool | None) -> str:
    """Render one optional boolean as CLI-safe text."""

    if value is None:
        return "unknown"
    return "true" if value else "false"


def _format_optional_float(value: float | None) -> str:
    """Render one optional float as CLI-safe text."""

    if value is None:
        return "unknown"
    return f"{float(value):.3f}"


def _format_optional_int(value: int | None) -> str:
    """Render one optional integer as CLI-safe text."""

    if value is None:
        return "unknown"
    return str(int(value))


def _format_optional_text(value: object) -> str:
    """Render one optional scalar as one trimmed single-line string."""

    normalized = _normalize_optional_text(value)
    return normalized or "unknown"


def _normalize_optional_text(value: object) -> str | None:
    """Normalize one optional text field to a trimmed single line."""

    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


__all__ = [
    "ReSpeakerAudioPerceptionGuardSnapshot",
    "ReSpeakerAudioPerceptionSnapshot",
    "derive_respeaker_audio_perception_guard",
    "observe_audio_perception_once",
    "render_audio_perception_snapshot_lines",
]
