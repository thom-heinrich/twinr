from __future__ import annotations

from twinr.agent.base_agent import RuntimeSnapshot, TwinrConfig
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.hardware.voice_profile import VoiceAssessment, VoiceProfileMonitor
from twinr.ops import loop_lock_owner


def _voice_profile_page_context(
    config: TwinrConfig,
    snapshot: RuntimeSnapshot,
    *,
    action_result: dict[str, str] | None = None,
    action_error: str | None = None,
) -> dict[str, object]:
    monitor = VoiceProfileMonitor.from_config(config)
    return {
        "profile_summary": monitor.summary(),
        "snapshot": snapshot,
        "voice_snapshot_label": _voice_snapshot_label(snapshot),
        "capture_block_reason": _voice_profile_capture_block_reason(config),
        "action_result": action_result,
        "action_error": action_error,
    }


def _voice_snapshot_label(snapshot: RuntimeSnapshot) -> str:
    status = (snapshot.user_voice_status or "").strip()
    if not status:
        return "No recent live voice check."
    label = status.replace("_", " ")
    if snapshot.user_voice_confidence is None:
        return label
    return f"{label} ({snapshot.user_voice_confidence * 100:.0f}%)"


def _voice_profile_capture_block_reason(config: TwinrConfig) -> str | None:
    busy: list[str] = []
    for loop_name, label in (("hardware-loop", "hardware loop"), ("realtime-loop", "realtime loop")):
        owner = loop_lock_owner(config, loop_name)
        if owner is not None:
            busy.append(f"{label} pid {owner}")
    if not busy:
        return None
    joined = ", ".join(busy)
    return f"Stop the running {joined} before capturing a voice profile sample."


def _capture_voice_profile_sample(config: TwinrConfig) -> bytes:
    blocked_reason = _voice_profile_capture_block_reason(config)
    if blocked_reason:
        raise RuntimeError(blocked_reason)
    recorder = SilenceDetectedRecorder.from_config(config)
    return recorder.record_until_pause(pause_ms=config.speech_pause_ms)


def _voice_action_result(assessment: VoiceAssessment) -> dict[str, str]:
    status = "warn"
    if assessment.status == "likely_user":
        status = "ok"
    elif assessment.status in {"disabled", "not_enrolled"}:
        status = "muted"
    detail = assessment.detail
    if assessment.confidence is not None:
        detail = f"{detail} Confidence {assessment.confidence_percent()}."
    return {
        "status": status,
        "title": assessment.label,
        "detail": detail,
    }
