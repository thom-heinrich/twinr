from __future__ import annotations

from typing import Any

from twinr.agent.tools.support import require_current_turn_audio, require_sensitive_voice_confirmation


def handle_enroll_voice_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    summary = owner.voice_profile_monitor.summary()
    if summary.enrolled:
        require_sensitive_voice_confirmation(owner, arguments, action_label="replace the saved voice profile")
    audio_pcm = require_current_turn_audio(owner)
    template = owner.voice_profile_monitor.enroll_pcm16(
        audio_pcm,
        sample_rate=owner.config.openai_realtime_input_sample_rate,
        channels=owner.config.audio_channels,
    )
    assessment = owner.voice_profile_monitor.assess_pcm16(
        audio_pcm,
        sample_rate=owner.config.openai_realtime_input_sample_rate,
        channels=owner.config.audio_channels,
    )
    if assessment.should_persist:
        owner.runtime.update_user_voice_assessment(
            status=assessment.status,
            confidence=assessment.confidence,
            checked_at=assessment.checked_at,
        )
    owner.emit("voice_profile_tool_call=true")
    owner.emit(f"voice_profile_samples={template.sample_count}")
    owner._record_event(
        "voice_profile_enrolled",
        "Realtime tool stored or refreshed the local voice profile.",
        sample_count=template.sample_count,
        average_duration_ms=template.average_duration_ms,
    )
    return {
        "status": "enrolled",
        "sample_count": template.sample_count,
        "average_duration_ms": template.average_duration_ms,
        "detail": "Local voice profile stored from the current spoken turn.",
    }


def handle_get_voice_profile_status(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    del arguments
    summary = owner.voice_profile_monitor.summary()
    owner.emit("voice_profile_tool_call=true")
    owner._record_event(
        "voice_profile_status_read",
        "Realtime tool read the local voice-profile status.",
        enrolled=summary.enrolled,
        sample_count=summary.sample_count,
    )
    return {
        "status": "ok",
        "enrolled": summary.enrolled,
        "sample_count": summary.sample_count,
        "updated_at": summary.updated_at,
        "average_duration_ms": summary.average_duration_ms,
        "current_signal": owner.runtime.user_voice_status,
        "current_confidence": owner.runtime.user_voice_confidence,
    }


def handle_reset_voice_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="delete the saved voice profile")
    summary = owner.voice_profile_monitor.reset()
    owner.runtime.update_user_voice_assessment(
        status=None,
        confidence=None,
        checked_at=None,
    )
    owner.emit("voice_profile_tool_call=true")
    owner._record_event(
        "voice_profile_reset",
        "Realtime tool deleted the local voice profile.",
        enrolled=summary.enrolled,
    )
    return {
        "status": "reset",
        "enrolled": summary.enrolled,
        "sample_count": summary.sample_count,
    }
