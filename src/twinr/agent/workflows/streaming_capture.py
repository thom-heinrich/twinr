"""Own microphone capture and streaming-STT handoff for the streaming loop.

This controller keeps `streaming_runner.py` focused on loop orchestration by
moving the capture lifecycle, streaming callback fan-out, batch-STT fallback,
and no-speech timeout handling into one focused runtime component.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from twinr.agent.base_agent.contracts import StreamingSpeechToTextProvider
from twinr.agent.workflows.listen_timeout_diagnostics import (
    diagnostics_as_details,
    diagnostics_from_exception,
    emit_listen_timeout_diagnostics,
)
from twinr.hardware.audio import pcm16_to_wav_bytes


@dataclass(frozen=True, slots=True)
class StreamingAudioTurnRequest:
    """Describe one audio turn that should be captured and transcribed."""

    initial_source: str
    follow_up: bool
    listening_window: object
    listen_source: str
    proactive_trigger: str | None
    speech_start_chunks: int | None
    ignore_initial_ms: int
    timeout_emit_key: str
    timeout_message: str
    play_initial_beep: bool


class StreamingCaptureController:
    """Coordinate capture, STT fallback, and timeout handling for one loop."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def capture_and_transcribe_streaming(
        self,
        *,
        listening_window,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ):
        """Reset speculative routing state before one streaming-STT capture."""

        loop = self._loop
        loop._reset_speculative_supervisor_decision()
        loop._trace_event(
            "streaming_capture_cycle_reset_speculation",
            kind="branch",
            details={
                "speech_start_chunks": speech_start_chunks,
                "ignore_initial_ms": ignore_initial_ms,
                "pause_ms": listening_window.speech_pause_ms,
            },
        )
        return loop._capture_and_transcribe_with_turn_controller(
            stt_provider=loop.stt_provider,
            listening_window=listening_window,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
        )

    def handle_stt_interim(self, text: str) -> None:
        """Feed interim streaming transcripts into the speculative warmups."""

        loop = self._loop
        loop._trace_event(
            "streaming_stt_interim_received",
            kind="io",
            details={"text_len": len(text.strip()), "preview": text.strip()[:80]},
        )
        loop._maybe_start_speculative_first_word(text)
        loop._maybe_start_speculative_supervisor_decision(text)

    def handle_stt_endpoint(self, event) -> None:
        """Feed final endpoint transcripts into the speculative warmups."""

        loop = self._loop
        transcript = str(getattr(event, "transcript", "") or "").strip()
        loop._trace_event(
            "streaming_stt_endpoint_received",
            kind="io",
            details={
                "text_len": len(transcript),
                "speech_final": bool(getattr(event, "speech_final", False)),
                "utterance_end": bool(getattr(event, "utterance_end", False)),
            },
        )
        if transcript:
            loop._maybe_start_speculative_first_word(transcript)
            loop._maybe_start_speculative_supervisor_decision(transcript)

    def run_audio_turn(self, request: StreamingAudioTurnRequest) -> bool:
        """Capture one spoken turn and hand the finished transcript to execution."""

        loop = self._loop
        turn_started = time.monotonic()
        loop._trace_event(
            "streaming_audio_turn_started",
            kind="span_start",
            details={
                "initial_source": request.initial_source,
                "follow_up": request.follow_up,
                "listen_source": request.listen_source,
                "play_initial_beep": request.play_initial_beep,
            },
        )
        self._begin_listening(request)
        capture_started = time.monotonic()
        loop._notify_voice_orchestrator_state(
            "listening",
            detail=request.listen_source,
            follow_up_allowed=request.follow_up,
        )
        loop._pause_voice_orchestrator_capture(reason=request.listen_source)
        try:
            capture_result, transcript, capture_ms, stt_ms = self._capture_audio(
                request=request,
                capture_started=capture_started,
            )
        except RuntimeError as exc:
            if loop._active_turn_stop_requested():
                loop._cancel_interrupted_turn()
                loop._trace_event("streaming_audio_turn_interrupted_during_capture", kind="branch", details={})
                return False
            if not loop._is_no_speech_timeout(exc):
                raise
            self._handle_no_speech_timeout(request=request, exc=exc)
            return False
        finally:
            loop._resume_voice_orchestrator_capture(reason=request.listen_source)

        audio_pcm = capture_result.pcm_bytes
        loop.runtime.remember_listen_capture(
            initial_source=request.initial_source,
            follow_up=request.follow_up,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        loop._update_voice_assessment_from_pcm(audio_pcm)

        recorder_sample_rate = loop._recorder_sample_rate()
        loop._current_turn_audio_pcm = audio_pcm
        loop._current_turn_audio_sample_rate = recorder_sample_rate
        try:
            transcript, stt_ms = self._transcribe_if_needed(
                audio_pcm=audio_pcm,
                transcript=transcript,
                stt_ms=stt_ms,
                recorder_sample_rate=recorder_sample_rate,
            )
            if not transcript:
                loop._trace_event(
                    "streaming_transcript_empty_after_stt",
                    kind="warning",
                    level="WARN",
                    details={"capture_ms": capture_ms, "stt_ms": stt_ms},
                )
                raise RuntimeError("Speech-to-text returned an empty transcript")
            loop.emit(f"transcript={transcript}")
            loop._trace_event(
                "streaming_transcript_ready",
                kind="observation",
                details={"transcript_len": len(transcript), "capture_ms": capture_ms, "stt_ms": stt_ms},
            )
            return loop._complete_streaming_turn(
                transcript=transcript,
                listen_source=request.listen_source,
                proactive_trigger=request.proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
                allow_follow_up_rearm=loop._follow_up_allowed_for_source(initial_source=request.initial_source),
            )
        finally:
            loop._current_turn_audio_pcm = None
            loop._trace_event(
                "streaming_audio_turn_finished",
                kind="span_end",
                kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
            )

    def _begin_listening(self, request: StreamingAudioTurnRequest) -> None:
        loop = self._loop
        if request.listen_source == "button":
            loop.runtime.press_green_button()
        elif request.follow_up and getattr(loop.runtime.status, "value", None) == "listening":
            loop._trace_event(
                "streaming_follow_up_reuse_listening_state",
                kind="branch",
                details={"listen_source": request.listen_source},
            )
        else:
            loop.runtime.begin_listening(
                request_source=request.listen_source,
                proactive_trigger=request.proactive_trigger,
            )
        loop._emit_status(force=True)
        if request.play_initial_beep:
            loop._play_listen_beep()

    def _capture_audio(
        self,
        *,
        request: StreamingAudioTurnRequest,
        capture_started: float,
    ):
        loop = self._loop
        with loop._audio_lock:
            loop._trace_event(
                "streaming_audio_capture_started",
                kind="span_start",
                details={
                    "streaming_stt": isinstance(loop.stt_provider, StreamingSpeechToTextProvider),
                    "speech_start_chunks": request.speech_start_chunks,
                    "ignore_initial_ms": request.ignore_initial_ms,
                },
            )
            if isinstance(loop.stt_provider, StreamingSpeechToTextProvider):
                capture_result, transcript, capture_ms, stt_ms = self._capture_with_streaming_stt(
                    request=request,
                    capture_started=capture_started,
                )
            else:
                loop._trace_event("streaming_audio_capture_path", kind="decision", details={"path": "recorder_only"})
                capture_result, capture_ms = self._capture_with_recorder(
                    request=request,
                    capture_started=capture_started,
                )
                transcript = ""
                stt_ms = -1
            loop._trace_event(
                "streaming_audio_capture_completed",
                kind="span_end",
                details={
                    "pcm_bytes": len(capture_result.pcm_bytes),
                    "speech_started_after_ms": capture_result.speech_started_after_ms,
                    "resumed_after_pause_count": capture_result.resumed_after_pause_count,
                },
                kpi={"duration_ms": capture_ms},
            )
        return capture_result, transcript, capture_ms, stt_ms

    def _capture_with_streaming_stt(
        self,
        *,
        request: StreamingAudioTurnRequest,
        capture_started: float,
    ):
        loop = self._loop
        try:
            loop._trace_event("streaming_audio_capture_path", kind="decision", details={"path": "streaming_stt"})
            capture_result, transcript, capture_ms, stt_ms, _turn_label = loop._capture_and_transcribe_streaming(
                listening_window=request.listening_window,
                speech_start_chunks=request.speech_start_chunks,
                ignore_initial_ms=request.ignore_initial_ms,
            )
        except RuntimeError as exc:
            if loop._is_no_speech_timeout(exc):
                raise
            loop.emit(f"turn_controller_fallback={type(exc).__name__}")
            loop._trace_event(
                "streaming_audio_capture_fallback",
                kind="branch",
                level="WARN",
                details={"error_type": type(exc).__name__, "path": "streaming_stt"},
            )
            capture_result, capture_ms = self._capture_with_recorder(
                request=request,
                capture_started=capture_started,
            )
            transcript = ""
            stt_ms = -1
        except Exception as exc:
            loop.emit(f"turn_controller_fallback={type(exc).__name__}")
            loop._trace_event(
                "streaming_audio_capture_fallback",
                kind="branch",
                level="WARN",
                details={"error_type": type(exc).__name__, "path": "streaming_stt"},
            )
            capture_result, capture_ms = self._capture_with_recorder(
                request=request,
                capture_started=capture_started,
            )
            transcript = ""
            stt_ms = -1
        return capture_result, transcript, capture_ms, stt_ms

    def _capture_with_recorder(
        self,
        *,
        request: StreamingAudioTurnRequest,
        capture_started: float,
    ):
        loop = self._loop
        capture_result = loop.recorder.capture_pcm_until_pause_with_options(
            pause_ms=request.listening_window.speech_pause_ms,
            start_timeout_s=request.listening_window.start_timeout_s,
            speech_start_chunks=request.speech_start_chunks,
            ignore_initial_ms=request.ignore_initial_ms,
            pause_grace_ms=request.listening_window.pause_grace_ms,
            should_stop=loop._active_turn_stop_requested,
        )
        capture_ms = int((time.monotonic() - capture_started) * 1000)
        return capture_result, capture_ms

    def _handle_no_speech_timeout(self, *, request: StreamingAudioTurnRequest, exc: RuntimeError) -> None:
        loop = self._loop
        timeout_diagnostics = diagnostics_from_exception(exc)
        emit_listen_timeout_diagnostics(loop.emit, timeout_diagnostics)
        loop.runtime.remember_listen_timeout(
            initial_source=request.initial_source,
            follow_up=request.follow_up,
        )
        loop.runtime.cancel_listening()
        loop._emit_status(force=True)
        loop._notify_voice_orchestrator_state("waiting", detail="listen_timeout")
        loop.emit(f"{request.timeout_emit_key}=true")
        loop._record_event("listen_timeout", request.timeout_message, request_source=request.listen_source)
        loop._trace_event(
            "streaming_audio_turn_timeout_no_speech",
            kind="warning",
            level="WARN",
            details={
                "timeout_emit_key": request.timeout_emit_key,
                "listen_source": request.listen_source,
                **diagnostics_as_details(timeout_diagnostics),
            },
        )

    def _transcribe_if_needed(
        self,
        *,
        audio_pcm: bytes,
        transcript: str,
        stt_ms: int,
        recorder_sample_rate: int,
    ) -> tuple[str, int]:
        loop = self._loop
        if stt_ms >= 0:
            return transcript, stt_ms
        audio_bytes = pcm16_to_wav_bytes(
            audio_pcm,
            sample_rate=recorder_sample_rate,
            channels=loop.config.audio_channels,
        )
        stt_started = time.monotonic()
        loop._trace_event(
            "streaming_batch_stt_started",
            kind="llm_call",
            details={"audio_bytes": len(audio_bytes)},
        )
        transcript = loop.stt_provider.transcribe(
            audio_bytes,
            filename="twinr-streaming-listen.wav",
            content_type="audio/wav",
        ).strip()
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        loop._trace_event(
            "streaming_batch_stt_completed",
            kind="llm_call",
            details={"transcript_len": len(transcript)},
            kpi={"duration_ms": stt_ms},
        )
        return transcript, stt_ms
