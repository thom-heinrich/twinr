"""Focused builders and format helpers for the orchestrator voice session."""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.orchestrator.remote_asr import RemoteAsrBackendAdapter
from twinr.orchestrator.voice_activation import VoiceActivationPhraseMatcher

from .types import _TranscriptBackend


def pcm_capture_to_wav_bytes(capture: AmbientAudioCaptureWindow) -> bytes:
    """Convert one recent PCM capture window into a WAV upload payload."""

    return pcm16_to_wav_bytes(
        capture.pcm_bytes,
        sample_rate=capture.sample_rate,
        channels=capture.channels,
    )


def build_remote_asr_backend(
    config: TwinrConfig,
    *,
    timeout_s: float | None = None,
) -> RemoteAsrBackendAdapter:
    """Build the thh1986 remote-ASR backend shared by transcript-first paths."""

    remote_asr_url = str(getattr(config, "voice_orchestrator_remote_asr_url", "") or "").strip()
    if not remote_asr_url:
        raise ValueError(
            "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL is required for the "
            "transcript-first remote voice gateway."
        )
    if timeout_s is None:
        timeout_s = float(getattr(config, "voice_orchestrator_remote_asr_timeout_s", 3.0) or 3.0)
    return RemoteAsrBackendAdapter(
        base_url=remote_asr_url,
        bearer_token=getattr(config, "voice_orchestrator_remote_asr_bearer_token", None),
        language=(
            getattr(config, "voice_orchestrator_remote_asr_language", None)
            or config.openai_realtime_language
        ),
        mode=getattr(config, "voice_orchestrator_remote_asr_mode", "active_listening"),
        timeout_s=float(timeout_s),
        retry_attempts=int(getattr(config, "voice_orchestrator_remote_asr_retry_attempts", 1) or 0),
        retry_backoff_s=float(
            getattr(config, "voice_orchestrator_remote_asr_retry_backoff_s", 0.35) or 0.0
        ),
    )


def build_transcript_backend(config: TwinrConfig) -> _TranscriptBackend:
    """Build the transcript backend used for follow-up and barge-in scans."""

    return build_remote_asr_backend(config)


def build_wake_phrase_spotter(
    config: TwinrConfig,
    *,
    backend: _TranscriptBackend,
    phrases: tuple[str, ...] | None = None,
) -> VoiceActivationPhraseMatcher:
    """Build the bounded transcript matcher used for wake confirmation."""

    return VoiceActivationPhraseMatcher(
        backend=backend,
        phrases=(
            getattr(config, "voice_activation_phrases", ())
            if phrases is None
            else phrases
        ),
        language=config.openai_realtime_language,
        suppress_transcription_errors=False,
    )


__all__ = [
    "build_remote_asr_backend",
    "build_transcript_backend",
    "build_wake_phrase_spotter",
    "pcm_capture_to_wav_bytes",
]
