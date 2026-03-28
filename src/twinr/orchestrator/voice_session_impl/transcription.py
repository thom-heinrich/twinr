# mypy: disable-error-code="attr-defined,has-type,assignment"
"""Backend-facing wake detection and transcription helpers."""

from __future__ import annotations

from contextlib import nullcontext

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_signal_profile
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from twinr.orchestrator.voice_familiarity import (
    FamiliarSpeakerWakeAssessment,
    assess_familiar_speaker_pcm16,
)
from twinr.orchestrator.voice_forensics import prefixed_signal_profile_details

from .builders import pcm_capture_to_wav_bytes


class VoiceSessionTranscriptionMixin:
    """Own the remote-ASR request surface for the voice session."""

    def _wake_phrase_spotter_supports_transcript_matching(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None,
    ) -> bool:
        """Return whether the selected wake spotter can match a provided transcript."""

        familiar_speaker_assessment = self._assess_familiar_speaker_capture(
            capture,
            origin_state=origin_state,
        )
        wake_phrase_spotter = self._wake_phrase_spotter_for_origin_state(
            origin_state=origin_state,
            allow_contextual_aliases=self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state=origin_state,
            ),
        )
        return callable(getattr(wake_phrase_spotter, "match_transcript", None))

    def _assess_familiar_speaker_capture(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        origin_state: str | None,
    ) -> FamiliarSpeakerWakeAssessment | None:
        """Assess whether this wake candidate sounds like an enrolled speaker."""

        resolved_origin_state = str(origin_state or "").strip().lower() or self._state
        if resolved_origin_state != "waiting":
            return None
        if not self._voice_identity_profiles:
            return FamiliarSpeakerWakeAssessment(
                assessment=None,
                familiar=False,
                revision=self._voice_identity_profiles_revision,
                profile_count=0,
            )
        return assess_familiar_speaker_pcm16(
            self.config,
            pcm_bytes=bytes(capture.pcm_bytes or b""),
            sample_rate=capture.sample_rate,
            channels=capture.channels,
            profiles=self._voice_identity_profiles,
            revision=self._voice_identity_profiles_revision,
        )

    def _detect_wake_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> VoiceActivationMatch | None:
        """Run one activation-detector pass and surface backend errors as evidence."""

        origin_state_value = None if details is None else details.get("origin_state")
        origin_state = str(origin_state_value).strip() or None
        familiar_speaker_assessment = self._assess_familiar_speaker_capture(
            capture,
            origin_state=origin_state,
        )
        wake_phrase_spotter = self._wake_phrase_spotter_for_origin_state(
            origin_state=origin_state,
            allow_contextual_aliases=self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state=origin_state,
            ),
        )
        try:
            with self._backend_request_context(
                stage=stage,
                capture=capture,
                origin_state=origin_state,
            ):
                return wake_phrase_spotter.detect(capture)
        except Exception as exc:
            error_message = str(exc).strip()
            resolved_details: dict[str, object] = {
                "error_type": type(exc).__name__,
            }
            if error_message:
                resolved_details["error_message"] = error_message[:240]
            if details:
                resolved_details.update(details)
            resolved_details.update(
                self._wake_bias_details(
                    origin_state=origin_state,
                    familiar_speaker_assessment=familiar_speaker_assessment,
                )
            )
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_backend_error",
                kind="warning",
                level="WARN",
                details={
                    "stage": stage,
                    **resolved_details,
                },
            )
            return None

    def _transcribe_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
        prompt: str | None = None,
    ) -> str | None:
        """Transcribe one bounded capture without routing it through wake detection."""

        origin_state_value = None if details is None else details.get("origin_state")
        origin_state = str(origin_state_value).strip() or None
        resolved_prompt = str(prompt or "").strip() or None
        try:
            with self._backend_request_context(
                stage=stage,
                capture=capture,
                origin_state=origin_state,
            ):
                return self.backend.transcribe(
                    pcm_capture_to_wav_bytes(capture),
                    filename="voice-window.wav",
                    content_type="audio/wav",
                    language=self.config.openai_realtime_language,
                    prompt=resolved_prompt,
                ).strip()
        except Exception as exc:
            error_message = str(exc).strip()
            resolved_details: dict[str, object] = {
                "error_type": type(exc).__name__,
            }
            if error_message:
                resolved_details["error_message"] = error_message[:240]
            if resolved_prompt is not None:
                resolved_details["prompt_chars"] = len(resolved_prompt)
            if details:
                resolved_details.update(details)
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_transcription_backend_error",
                kind="warning",
                level="WARN",
                details={
                    "stage": stage,
                    **resolved_details,
                },
            )
            return None

    def _match_transcribed_wake(
        self,
        *,
        transcript: str,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> VoiceActivationMatch | None:
        """Match one already-transcribed utterance against wake phrases."""

        origin_state_value = None if details is None else details.get("origin_state")
        origin_state = str(origin_state_value).strip() or None
        familiar_speaker_assessment = self._assess_familiar_speaker_capture(
            capture,
            origin_state=origin_state,
        )
        wake_phrase_spotter = self._wake_phrase_spotter_for_origin_state(
            origin_state=origin_state,
            allow_contextual_aliases=self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state=origin_state,
            ),
        )
        match_transcript = getattr(wake_phrase_spotter, "match_transcript", None)
        if callable(match_transcript):
            try:
                return match_transcript(transcript)
            except Exception as exc:
                error_message = str(exc).strip()
                resolved_details: dict[str, object] = {
                    "error_type": type(exc).__name__,
                }
                if error_message:
                    resolved_details["error_message"] = error_message[:240]
                if details:
                    resolved_details.update(details)
                resolved_details.update(
                    self._wake_bias_details(
                        origin_state=origin_state,
                        familiar_speaker_assessment=familiar_speaker_assessment,
                    )
                )
                self._record_transcript_debug(
                    stage=stage,
                    outcome="matcher_error",
                    transcript=transcript,
                    capture=capture,
                    details=resolved_details,
                )
                self._trace_event(
                    "voice_activation_matcher_error",
                    kind="warning",
                    level="WARN",
                    details={
                        "stage": stage,
                        "transcript_chars": len(transcript),
                        **resolved_details,
                    },
                )
                return None
        return self._detect_wake_capture(
            capture=capture,
            stage=stage,
            details=details,
        )

    def _backend_request_context(
        self,
        *,
        stage: str,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None = None,
    ):
        """Expose compact capture metadata to the remote-ASR client adapter."""

        bind_context = getattr(self.backend, "bind_request_context", None)
        if not callable(bind_context):
            return nullcontext()
        signal_profile = pcm16_signal_profile(capture.pcm_bytes)
        return bind_context(
            {
                "session_id": self._session_id,
                "trace_id": self._trace_id,
                "stage": stage,
                "state": self._state,
                "origin_state": origin_state,
                "capture_duration_ms": int(capture.sample.duration_ms),
                "capture_average_rms": int(capture.sample.average_rms),
                "capture_peak_rms": int(capture.sample.peak_rms),
                "capture_active_ratio": round(float(capture.sample.active_ratio), 6),
                **prefixed_signal_profile_details(signal_profile, prefix="capture_signal"),
            }
        )
