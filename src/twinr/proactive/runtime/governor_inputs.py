"""Package ReSpeaker-derived policy facts for proactive governor callers.

This module keeps governor-facing context assembly out of the realtime
background mixin. The governor itself still owns cooldown and reservation
policy; this layer only normalizes the current presence/audio facts that the
workflow wants to attach to one governed delivery attempt.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.multimodal_initiative import ReSpeakerMultimodalInitiativeSnapshot
from twinr.proactive.runtime.presence import PresenceSessionSnapshot


@dataclass(frozen=True, slots=True)
class ReSpeakerGovernorInputs:
    """Describe the current ReSpeaker context attached to one candidate."""

    channel: str
    presence_session_id: int | None = None
    runtime_alert_code: str | None = None
    initiative_block_reason: str | None = None
    speech_delivery_defer_reason: str | None = None
    room_busy_or_overlapping: bool | None = None
    quiet_window_open: bool | None = None
    resume_window_open: bool | None = None
    mute_blocks_voice_capture: bool | None = None
    multimodal_initiative_ready: bool | None = None
    multimodal_initiative_confidence: float | None = None
    multimodal_initiative_block_reason: str | None = None

    def event_data(self) -> dict[str, object]:
        """Render one JSON-safe event payload for operator telemetry."""

        return {
            "governor_channel": self.channel,
            "audio_runtime_alert_code": self.runtime_alert_code,
            "audio_initiative_block_reason": self.initiative_block_reason,
            "audio_speech_delivery_defer_reason": self.speech_delivery_defer_reason,
            "audio_room_busy_or_overlapping": self.room_busy_or_overlapping,
            "audio_quiet_window_open": self.quiet_window_open,
            "audio_resume_window_open": self.resume_window_open,
            "audio_mute_blocks_voice_capture": self.mute_blocks_voice_capture,
            "multimodal_initiative_ready": self.multimodal_initiative_ready,
            "multimodal_initiative_confidence": self.multimodal_initiative_confidence,
            "multimodal_initiative_block_reason": self.multimodal_initiative_block_reason,
        }


def build_respeaker_governor_inputs(
    *,
    requested_channel: str,
    presence_snapshot: PresenceSessionSnapshot | None,
    audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    multimodal_initiative_snapshot: ReSpeakerMultimodalInitiativeSnapshot | None = None,
) -> ReSpeakerGovernorInputs:
    """Return one bounded governor-input bundle from current runtime facts."""

    normalized_channel = _normalize_channel(requested_channel)
    presence_session_id = _presence_session_id(presence_snapshot)
    if audio_policy_snapshot is None:
        return ReSpeakerGovernorInputs(
            channel=normalized_channel,
            presence_session_id=presence_session_id,
            multimodal_initiative_ready=(
                None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.ready
            ),
            multimodal_initiative_confidence=(
                None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.confidence
            ),
            multimodal_initiative_block_reason=_normalize_optional_text(
                None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.block_reason
            ),
        )
    return ReSpeakerGovernorInputs(
        channel=normalized_channel,
        presence_session_id=presence_session_id,
        runtime_alert_code=_normalize_optional_text(audio_policy_snapshot.runtime_alert_code),
        initiative_block_reason=_normalize_optional_text(audio_policy_snapshot.initiative_block_reason),
        speech_delivery_defer_reason=_normalize_optional_text(audio_policy_snapshot.speech_delivery_defer_reason),
        room_busy_or_overlapping=audio_policy_snapshot.room_busy_or_overlapping,
        quiet_window_open=audio_policy_snapshot.quiet_window_open,
        resume_window_open=audio_policy_snapshot.resume_window_open,
        mute_blocks_voice_capture=audio_policy_snapshot.mute_blocks_voice_capture,
        multimodal_initiative_ready=(
            None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.ready
        ),
        multimodal_initiative_confidence=(
            None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.confidence
        ),
        multimodal_initiative_block_reason=_normalize_optional_text(
            None if multimodal_initiative_snapshot is None else multimodal_initiative_snapshot.block_reason
        ),
    )


def _presence_session_id(snapshot: PresenceSessionSnapshot | None) -> int | None:
    """Return the active presence-session id when one session is armed."""

    if snapshot is None or snapshot.armed is not True:
        return None
    value = snapshot.session_id
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_channel(value: object) -> str:
    """Normalize one candidate channel into the governor vocabulary."""

    text = _normalize_optional_text(value) or "speech"
    if text not in {"speech", "display", "print"}:
        return "speech"
    return text


def _normalize_optional_text(value: object) -> str | None:
    """Return one compact optional text value."""

    text = " ".join(str(value or "").split()).strip()
    return text or None


__all__ = [
    "ReSpeakerGovernorInputs",
    "build_respeaker_governor_inputs",
]
