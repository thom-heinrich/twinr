from __future__ import annotations

from types import SimpleNamespace

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.orchestrator.voice_activation import VoiceActivationPhraseMatcher
from twinr.orchestrator.voice_session_impl.builders import build_wake_phrase_spotter


class _PromptSensitiveBackend:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        del audio_bytes, filename, content_type, language
        self.prompts.append(prompt)
        if prompt is not None:
            return "Das war's. Bis zum nächsten Mal."
        return "Twinr wie spät ist es"


def _capture_window() -> AmbientAudioCaptureWindow:
    pcm_bytes = b"\x00\x00" * 16_000
    return AmbientAudioCaptureWindow(
        sample=AmbientAudioLevelSample(
            duration_ms=1000,
            chunk_count=10,
            active_chunk_count=10,
            average_rms=0,
            peak_rms=0,
            active_ratio=1.0,
        ),
        pcm_bytes=pcm_bytes,
        sample_rate=16_000,
        channels=1,
    )


def test_voice_activation_phrase_matcher_uses_promptless_transcript_path() -> None:
    backend = _PromptSensitiveBackend()
    matcher = VoiceActivationPhraseMatcher(
        backend=backend,
        phrases=("twinr",),
        suppress_transcription_errors=False,
    )

    match = matcher.detect(_capture_window())

    assert backend.prompts == [None]
    assert match.detected is True
    assert match.matched_phrase == "twinr"
    assert match.remaining_text == "wie spät ist es"
    assert match.transcript == "Twinr wie spät ist es"


def test_build_wake_phrase_spotter_disables_heuristic_matching() -> None:
    matcher = build_wake_phrase_spotter(
        SimpleNamespace(
            voice_activation_phrases=("twinr", "twinner"),
            openai_realtime_language="de",
        ),
        backend=_PromptSensitiveBackend(),
    )

    assert matcher.enable_fuzzy_prefix_match is False
    assert matcher.allow_twi_head_variant_recovery is False
    assert matcher.suppress_transcription_errors is False
