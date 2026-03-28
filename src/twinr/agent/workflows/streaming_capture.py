"""Own microphone capture and streaming-STT handoff for the streaming loop.

This controller keeps `streaming_runner.py` focused on loop orchestration by
moving the capture lifecycle, streaming callback fan-out, batch-STT fallback,
and no-speech timeout handling into one focused runtime component.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Empty remote transcripts could complete a turn and execute downstream logic with no words.
# BUG-2: Empty batch-STT results raised an unhandled RuntimeError and could crash the loop instead of failing the turn cleanly.
# BUG-3: Turn cancellation requested after capture could still execute STT / action completion, causing stale commands to run.
# BUG-4: `_current_turn_audio_sample_rate` was left stale after a turn finished, leaking invalid per-turn state.
# SEC-1: Raw interim/transcript text was logged and emitted without sanitization, enabling privacy leaks and log/control-plane injection via CR/LF.
# SEC-2: Recorder fallback had no hard utterance budget, allowing practical memory / CPU denial-of-service on Raspberry Pi via continuous audio.
# IMP-1: Added provider-aware 2026 hooks for VAD-enabled batch STT, adaptation hints, and streaming circuit breaking / graceful degradation.
# IMP-2: Added unified transcript normalization, safe previews, bounded capture enforcement, and interruption-aware turn finalization.

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import os
import time
from typing import Any, Callable

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


@dataclass(frozen=True, slots=True)
class _CaptureBudget:
    """Operational bounds for one captured utterance."""

    max_turn_s: float | None
    max_pcm_bytes: int | None
    source: str


class StreamingCaptureController:
    """Coordinate capture, STT fallback, and timeout handling for one loop."""

    # BREAKING: turns are now bounded by default to reduce practical Pi-4 DoS
    # risk from continuous audio. Override via listening_window, config or env.
    _DEFAULT_MAX_TURN_SECONDS = 30.0

    _DEFAULT_STREAMING_FAILURE_THRESHOLD = 3
    _DEFAULT_STREAMING_COOLDOWN_S = 15.0
    _TRACE_RAW_TRANSCRIPTS_ENV = "TWINR_TRACE_RAW_TRANSCRIPTS"

    def __init__(self, loop) -> None:
        self._loop = loop
        self._streaming_failure_count = 0
        self._streaming_circuit_open_until = 0.0

    # ---------- Public hooks -------------------------------------------------

    def capture_and_transcribe_streaming(
        self,
        *,
        listening_window,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ):
        """Reset speculative routing state before one streaming-STT capture."""

        loop = self._loop
        budget = self._resolve_capture_budget(listening_window=listening_window)

        loop._reset_speculative_supervisor_decision()
        loop._trace_event(
            "streaming_capture_cycle_reset_speculation",
            kind="branch",
            details={
                "speech_start_chunks": speech_start_chunks,
                "ignore_initial_ms": ignore_initial_ms,
                "pause_ms": getattr(listening_window, "speech_pause_ms", None),
                "max_turn_s": budget.max_turn_s,
                "budget_source": budget.source,
            },
        )

        kwargs = {
            "stt_provider": loop.stt_provider,
            "listening_window": listening_window,
            "speech_start_chunks": speech_start_chunks,
            "ignore_initial_ms": ignore_initial_ms,
        }
        self._add_supported_kwarg(
            loop._capture_and_transcribe_with_turn_controller,
            kwargs,
            "max_turn_s",
            budget.max_turn_s,
        )
        self._add_supported_kwarg(
            loop._capture_and_transcribe_with_turn_controller,
            kwargs,
            "max_capture_s",
            budget.max_turn_s,
        )
        self._add_supported_kwarg(
            loop._capture_and_transcribe_with_turn_controller,
            kwargs,
            "max_speech_duration_s",
            budget.max_turn_s,
        )
        return loop._capture_and_transcribe_with_turn_controller(**kwargs)

    def handle_stt_interim(self, text: str) -> None:
        """Feed interim streaming transcripts into the speculative warmups."""

        loop = self._loop
        normalized = self._normalize_transcript(text)
        loop._trace_event(
            "streaming_stt_interim_received",
            kind="io",
            details=self._transcript_trace_details(normalized),
        )
        if normalized:
            loop._maybe_start_speculative_first_word(normalized)
            loop._maybe_start_speculative_supervisor_decision(normalized)

    def handle_stt_endpoint(self, event) -> None:
        """Feed final endpoint transcripts into the speculative warmups."""

        loop = self._loop
        transcript = self._normalize_transcript(str(getattr(event, "transcript", "") or ""))
        loop._trace_event(
            "streaming_stt_endpoint_received",
            kind="io",
            details={
                "text_len": len(transcript),
                "speech_final": bool(getattr(event, "speech_final", False)),
                "utterance_end": bool(getattr(event, "utterance_end", False)),
                **self._transcript_trace_details(transcript, include_len=False),
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

        if loop._voice_orchestrator_owns_live_listening():
            return self._run_remote_audio_turn(
                request=request,
                turn_started=turn_started,
                capture_started=capture_started,
            )

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
                loop._trace_event(
                    "streaming_audio_turn_interrupted_during_capture",
                    kind="branch",
                    details={},
                )
                return False
            if not loop._is_no_speech_timeout(exc):
                raise
            self._handle_no_speech_timeout(request=request, exc=exc)
            return False
        finally:
            loop._resume_voice_orchestrator_capture(reason=request.listen_source)

        if self._abort_if_stop_requested(phase="after_capture", request=request):
            return False

        audio_pcm = capture_result.pcm_bytes
        loop.runtime.remember_listen_capture(
            initial_source=request.initial_source,
            follow_up=request.follow_up,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )

        recorder_sample_rate = loop._recorder_sample_rate()
        audio_pcm = self._clamp_audio_to_budget(
            audio_pcm=audio_pcm,
            listening_window=request.listening_window,
            recorder_sample_rate=recorder_sample_rate,
            channels=loop.config.audio_channels,
        )
        loop._update_voice_assessment_from_pcm(audio_pcm)

        loop._current_turn_audio_pcm = audio_pcm
        loop._current_turn_audio_sample_rate = recorder_sample_rate
        try:
            transcript, stt_ms = self._transcribe_if_needed(
                audio_pcm=audio_pcm,
                transcript=transcript,
                stt_ms=stt_ms,
                recorder_sample_rate=recorder_sample_rate,
                channels=loop.config.audio_channels,
                listening_window=request.listening_window,
            )
            transcript = self._normalize_transcript(transcript)

            if self._abort_if_stop_requested(phase="after_stt", request=request):
                return False

            if not transcript:
                self._handle_empty_transcript(
                    request=request,
                    capture_ms=capture_ms,
                    stt_ms=stt_ms,
                    source="local",
                )
                return False

            self._emit_transcript(transcript)
            loop._trace_event(
                "streaming_transcript_ready",
                kind="observation",
                details={
                    "transcript_len": len(transcript),
                    "capture_ms": capture_ms,
                    "stt_ms": stt_ms,
                    **self._transcript_trace_details(transcript, include_len=False),
                },
            )
            return loop._complete_streaming_turn(
                transcript=transcript,
                listen_source=request.listen_source,
                proactive_trigger=request.proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
                allow_follow_up_rearm=loop._follow_up_allowed_for_source(
                    initial_source=request.initial_source
                ),
            )
        finally:
            loop._current_turn_audio_pcm = None
            loop._current_turn_audio_sample_rate = None
            loop._trace_event(
                "streaming_audio_turn_finished",
                kind="span_end",
                kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
            )

    # ---------- Turn execution -----------------------------------------------

    def _run_remote_audio_turn(
        self,
        *,
        request: StreamingAudioTurnRequest,
        turn_started: float,
        capture_started: float,
    ) -> bool:
        """Wait for the same-stream remote transcript instead of reopening capture."""

        loop = self._loop
        wait_handle = loop._begin_remote_transcript_wait(source="listening")
        if wait_handle is None:
            loop.runtime.cancel_listening()
            loop._emit_status(force=True)
            loop._notify_voice_orchestrator_state("waiting", detail="remote_wait_failed")
            return False

        try:
            loop._notify_voice_orchestrator_state(
                "listening",
                detail=request.listen_source,
                follow_up_allowed=request.follow_up,
            )
            committed = loop._wait_for_remote_transcript_commit(
                wait_handle=wait_handle,
                timeout_s=request.listening_window.start_timeout_s,
                initial_source=request.initial_source,
                follow_up=request.follow_up,
                listen_source=request.listen_source,
                timeout_emit_key=request.timeout_emit_key,
                timeout_message=request.timeout_message,
            )
        finally:
            loop._remote_transcript_commits.clear_wait(wait_handle)

        if committed is None:
            loop._trace_event(
                "streaming_remote_audio_turn_closed_without_commit",
                kind="branch",
                details={"listen_source": request.listen_source},
            )
            return False

        if self._abort_if_stop_requested(phase="remote_commit", request=request):
            return False

        capture_ms = int((time.monotonic() - capture_started) * 1000)
        transcript = self._normalize_transcript(getattr(committed, "transcript", ""))

        loop._trace_event(
            "streaming_remote_transcript_ready",
            kind="observation",
            details={
                "listen_source": request.listen_source,
                "transcript_len": len(transcript),
                "capture_ms": capture_ms,
                **self._transcript_trace_details(transcript, include_len=False),
            },
        )

        if not transcript:
            self._handle_empty_transcript(
                request=request,
                capture_ms=capture_ms,
                stt_ms=-1,
                source="remote",
            )
            return False

        self._emit_transcript(transcript)
        return loop._complete_streaming_turn(
            transcript=transcript,
            listen_source=request.listen_source,
            proactive_trigger=request.proactive_trigger,
            turn_started=turn_started,
            capture_ms=capture_ms,
            stt_ms=-1,
            allow_follow_up_rearm=loop._follow_up_allowed_for_source(
                initial_source=request.initial_source
            ),
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
            using_streaming_stt = isinstance(loop.stt_provider, StreamingSpeechToTextProvider)
            circuit_open = self._streaming_circuit_open()

            loop._trace_event(
                "streaming_audio_capture_started",
                kind="span_start",
                details={
                    "streaming_stt": using_streaming_stt,
                    "speech_start_chunks": request.speech_start_chunks,
                    "ignore_initial_ms": request.ignore_initial_ms,
                    "streaming_circuit_open": circuit_open,
                },
            )

            if using_streaming_stt and not circuit_open:
                capture_result, transcript, capture_ms, stt_ms = self._capture_with_streaming_stt(
                    request=request,
                    capture_started=capture_started,
                )
            else:
                path = "streaming_circuit_open" if using_streaming_stt else "recorder_only"
                loop._trace_event("streaming_audio_capture_path", kind="decision", details={"path": path})
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
            self._mark_streaming_success()
        except RuntimeError as exc:
            if loop._is_no_speech_timeout(exc):
                self._mark_streaming_success()
                raise
            self._mark_streaming_failure(exc)
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
            self._mark_streaming_failure(exc)
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
        budget = self._resolve_capture_budget(listening_window=request.listening_window)
        deadline = time.monotonic() + budget.max_turn_s if budget.max_turn_s else None

        def should_stop() -> bool:
            if loop._active_turn_stop_requested():
                return True
            return bool(deadline is not None and time.monotonic() >= deadline)

        capture_result = loop.recorder.capture_pcm_until_pause_with_options(
            pause_ms=request.listening_window.speech_pause_ms,
            start_timeout_s=request.listening_window.start_timeout_s,
            speech_start_chunks=request.speech_start_chunks,
            ignore_initial_ms=request.ignore_initial_ms,
            pause_grace_ms=request.listening_window.pause_grace_ms,
            should_stop=should_stop,
        )
        capture_ms = int((time.monotonic() - capture_started) * 1000)

        if deadline is not None and capture_ms >= int(budget.max_turn_s * 1000):
            loop._trace_event(
                "streaming_audio_capture_budget_reached",
                kind="warning",
                level="WARN",
                details={
                    "capture_ms": capture_ms,
                    "max_turn_s": budget.max_turn_s,
                    "budget_source": budget.source,
                },
            )

        return capture_result, capture_ms

    def _transcribe_if_needed(
        self,
        *,
        audio_pcm: bytes,
        transcript: str,
        stt_ms: int,
        recorder_sample_rate: int,
        channels: int,
        listening_window,
    ) -> tuple[str, int]:
        loop = self._loop

        if stt_ms >= 0:
            return self._normalize_transcript(transcript), stt_ms

        if self._abort_if_stop_requested(phase="before_batch_stt", request=None):
            return "", stt_ms

        audio_bytes = pcm16_to_wav_bytes(
            audio_pcm,
            sample_rate=recorder_sample_rate,
            channels=channels,
        )
        optional_kwargs = self._build_transcribe_optional_kwargs(
            listening_window=listening_window,
        )

        loop._trace_event(
            "streaming_batch_stt_started",
            kind="llm_call",
            details={
                "audio_bytes": len(audio_bytes),
                "optional_kwargs": sorted(optional_kwargs.keys()),
            },
        )

        started = time.monotonic()
        transcript = self._invoke_transcribe(
            audio_bytes=audio_bytes,
            optional_kwargs=optional_kwargs,
        )
        stt_ms = int((time.monotonic() - started) * 1000)
        transcript = self._normalize_transcript(transcript)

        loop._trace_event(
            "streaming_batch_stt_completed",
            kind="llm_call",
            details={"transcript_len": len(transcript)},
            kpi={"duration_ms": stt_ms},
        )
        return transcript, stt_ms

    # ---------- Timeout / empty-turn handling --------------------------------

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

    def _handle_empty_transcript(
        self,
        *,
        request: StreamingAudioTurnRequest,
        capture_ms: int,
        stt_ms: int,
        source: str,
    ) -> None:
        loop = self._loop
        loop.runtime.cancel_listening()
        loop._emit_status(force=True)
        loop._notify_voice_orchestrator_state("waiting", detail="empty_transcript")
        loop.emit("transcript_empty=true")
        loop._record_event(
            "empty_transcript",
            "I could not derive a usable transcript from the captured audio.",
            request_source=request.listen_source,
        )
        loop._trace_event(
            "streaming_transcript_empty_after_stt",
            kind="warning",
            level="WARN",
            details={
                "capture_ms": capture_ms,
                "stt_ms": stt_ms,
                "listen_source": request.listen_source,
                "source": source,
            },
        )

    def _abort_if_stop_requested(
        self,
        *,
        phase: str,
        request: StreamingAudioTurnRequest | None,
    ) -> bool:
        loop = self._loop
        if not loop._active_turn_stop_requested():
            return False

        loop._cancel_interrupted_turn()
        loop._trace_event(
            "streaming_audio_turn_interrupted",
            kind="branch",
            details={
                "phase": phase,
                "listen_source": getattr(request, "listen_source", None),
            },
        )
        return True

    # ---------- Frontier hooks: provider-aware STT ---------------------------

    def _build_transcribe_optional_kwargs(self, *, listening_window) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        speech_pause_ms = getattr(listening_window, "speech_pause_ms", None)

        if self._batch_vad_enabled():
            kwargs["vad_filter"] = True
            if speech_pause_ms and speech_pause_ms > 0:
                kwargs["vad_parameters"] = {
                    "min_silence_duration_ms": int(max(250, min(2000, speech_pause_ms))),
                }

        kwargs.update(self._collect_stt_adaptation_hints())
        return kwargs

    def _collect_stt_adaptation_hints(self) -> dict[str, Any]:
        loop = self._loop
        hints: dict[str, Any] = {}

        for getter_name in ("get_stt_adaptation_hints", "_get_stt_adaptation_hints"):
            getter = getattr(loop, getter_name, None)
            if callable(getter):
                provided = getter()
                if isinstance(provided, dict):
                    hints.update({k: v for k, v in provided.items() if v is not None})
                break

        config = getattr(loop, "config", None)
        runtime = getattr(loop, "runtime", None)
        attr_map = {
            "language": ("stt_language", "speech_language", "preferred_language"),
            "hotwords": ("stt_hotwords", "speech_hotwords", "hotwords"),
            "initial_prompt": ("stt_initial_prompt", "speech_initial_prompt"),
            "prompt": ("stt_prompt", "speech_prompt"),
        }
        for key, names in attr_map.items():
            if key in hints:
                continue
            value = self._first_attr(config, *names)
            if value is None:
                value = self._first_attr(runtime, *names)
            if value is not None:
                hints[key] = value

        return hints

    def _invoke_transcribe(self, *, audio_bytes: bytes, optional_kwargs: dict[str, Any]) -> str:
        loop = self._loop
        provider = loop.stt_provider
        base_kwargs = {
            "filename": "twinr-streaming-listen.wav",
            "content_type": "audio/wav",
        }
        supported_optional = self._filter_supported_kwargs(provider.transcribe, optional_kwargs)

        try:
            result = provider.transcribe(audio_bytes, **base_kwargs, **supported_optional)
        except TypeError as exc:
            if not supported_optional or "keyword" not in str(exc).lower():
                raise
            loop._trace_event(
                "streaming_batch_stt_optional_kwargs_rejected",
                kind="branch",
                level="WARN",
                details={"optional_kwargs": sorted(supported_optional.keys())},
            )
            result = provider.transcribe(audio_bytes, **base_kwargs)

        return str(result or "")

    # ---------- Security / transcript hygiene --------------------------------

    # BREAKING: transcript control characters are normalized before being
    # emitted on the controller event channel so voice input cannot inject extra
    # frames or malformed log records.
    def _emit_transcript(self, transcript: str) -> None:
        self._loop.emit(f"transcript={self._normalize_transcript(transcript)}")

    def _normalize_transcript(self, text: str) -> str:
        if not text:
            return ""

        out: list[str] = []
        previous_was_space = False
        for ch in str(text):
            if ch.isspace() or ord(ch) < 32:
                if not previous_was_space:
                    out.append(" ")
                previous_was_space = True
                continue
            out.append(ch)
            previous_was_space = False
        return "".join(out).strip()

    def _transcript_trace_details(self, text: str, *, include_len: bool = True) -> dict[str, Any]:
        normalized = self._normalize_transcript(text)
        raw = self._trace_raw_transcripts()
        details: dict[str, Any] = {
            "preview_mode": "raw" if raw else "redacted",
            "preview": normalized[:80] if raw else ("<redacted>" if normalized else ""),
            "preview_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16] if normalized else "0" * 16,
            "word_count": len(normalized.split()) if normalized else 0,
        }
        if include_len:
            details["text_len"] = len(normalized)
        return details

    def _trace_raw_transcripts(self) -> bool:
        raw = os.getenv(self._TRACE_RAW_TRANSCRIPTS_ENV)
        return bool(raw and raw.strip().lower() not in {"0", "false", "no", "off"})

    # ---------- Capture budget / Pi safety -----------------------------------

    def _resolve_capture_budget(self, *, listening_window) -> _CaptureBudget:
        config = getattr(self._loop, "config", None)

        max_turn_s, source = self._resolve_positive_float(
            attr_names=("max_turn_s", "max_capture_s", "max_speech_duration_s"),
            objs=(listening_window, config),
            env_names=("TWINR_MAX_TURN_SECONDS",),
            default=self._DEFAULT_MAX_TURN_SECONDS,
        )
        max_pcm_bytes, pcm_source = self._resolve_positive_int(
            attr_names=("max_turn_pcm_bytes", "max_capture_pcm_bytes"),
            objs=(listening_window, config),
            env_names=("TWINR_MAX_PCM_BYTES",),
            default=None,
        )
        if max_pcm_bytes is not None:
            source = pcm_source if source == "default" else f"{source}+{pcm_source}"

        return _CaptureBudget(
            max_turn_s=max_turn_s,
            max_pcm_bytes=max_pcm_bytes,
            source=source,
        )

    def _clamp_audio_to_budget(
        self,
        *,
        audio_pcm: bytes,
        listening_window,
        recorder_sample_rate: int,
        channels: int,
    ) -> bytes:
        budget = self._resolve_capture_budget(listening_window=listening_window)
        max_pcm_bytes = budget.max_pcm_bytes
        if max_pcm_bytes is None and budget.max_turn_s:
            max_pcm_bytes = int(budget.max_turn_s * recorder_sample_rate * channels * 2)

        if max_pcm_bytes is None or len(audio_pcm) <= max_pcm_bytes:
            return audio_pcm

        clamped = audio_pcm[:max_pcm_bytes]
        self._loop._trace_event(
            "streaming_audio_capture_clamped",
            kind="warning",
            level="WARN",
            details={
                "pcm_bytes_before": len(audio_pcm),
                "pcm_bytes_after": len(clamped),
                "max_turn_s": budget.max_turn_s,
                "budget_source": budget.source,
            },
        )
        return clamped

    # ---------- Streaming STT resilience -------------------------------------

    def _streaming_circuit_open(self) -> bool:
        return time.monotonic() < self._streaming_circuit_open_until

    def _mark_streaming_success(self) -> None:
        if self._streaming_failure_count or self._streaming_circuit_open_until:
            self._loop._trace_event(
                "streaming_stt_circuit_closed",
                kind="branch",
                details={"previous_failures": self._streaming_failure_count},
            )
        self._streaming_failure_count = 0
        self._streaming_circuit_open_until = 0.0

    def _mark_streaming_failure(self, exc: Exception) -> None:
        loop = self._loop
        self._streaming_failure_count += 1
        threshold = self._streaming_failure_threshold()
        details = {
            "failure_count": self._streaming_failure_count,
            "threshold": threshold,
            "error_type": type(exc).__name__,
        }

        if self._streaming_failure_count >= threshold:
            cooldown_s = self._streaming_cooldown_s()
            self._streaming_circuit_open_until = time.monotonic() + cooldown_s
            details["cooldown_s"] = cooldown_s
            loop._trace_event(
                "streaming_stt_circuit_opened",
                kind="warning",
                level="WARN",
                details=details,
            )
            return

        loop._trace_event(
            "streaming_stt_failure_recorded",
            kind="warning",
            level="WARN",
            details=details,
        )

    def _streaming_failure_threshold(self) -> int:
        config_value = self._first_attr(getattr(self._loop, "config", None), "streaming_failure_threshold")
        if config_value is not None:
            try:
                return max(1, int(config_value))
            except (TypeError, ValueError):
                pass

        env_value = os.getenv("TWINR_STREAMING_FAILURE_THRESHOLD")
        if env_value:
            try:
                return max(1, int(env_value))
            except (TypeError, ValueError):
                pass

        return self._DEFAULT_STREAMING_FAILURE_THRESHOLD

    def _streaming_cooldown_s(self) -> float:
        config_value = self._first_attr(getattr(self._loop, "config", None), "streaming_failure_cooldown_s")
        if config_value is not None:
            try:
                return max(1.0, float(config_value))
            except (TypeError, ValueError):
                pass

        env_value = os.getenv("TWINR_STREAMING_FAILURE_COOLDOWN_S")
        if env_value:
            try:
                return max(1.0, float(env_value))
            except (TypeError, ValueError):
                pass

        return self._DEFAULT_STREAMING_COOLDOWN_S

    def _batch_vad_enabled(self) -> bool:
        config_value = self._first_attr(getattr(self._loop, "config", None), "batch_stt_vad_filter")
        if config_value is not None:
            return bool(config_value)

        env_value = os.getenv("TWINR_BATCH_STT_VAD_FILTER")
        if env_value is None:
            return True
        return env_value.strip().lower() not in {"0", "false", "no", "off"}

    # ---------- Introspection / config helpers -------------------------------

    def _filter_supported_kwargs(self, fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs:
            return {}

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return {}

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return {k: v for k, v in kwargs.items() if v is not None}

        return {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters and value is not None
        }

    def _add_supported_kwarg(
        self,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if value is None:
            return
        filtered = self._filter_supported_kwargs(fn, {key: value})
        if key in filtered:
            kwargs[key] = filtered[key]

    @staticmethod
    def _first_attr(obj: object | None, *names: str) -> Any:
        if obj is None:
            return None
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return None

    @classmethod
    def _resolve_positive_float(
        cls,
        *,
        attr_names: tuple[str, ...],
        objs: tuple[object | None, ...],
        env_names: tuple[str, ...],
        default: float | None,
    ) -> tuple[float | None, str]:
        for obj in objs:
            if obj is None:
                continue
            for name in attr_names:
                value = getattr(obj, name, None)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value, f"{type(obj).__name__}.{name}"

        for env_name in env_names:
            raw = os.getenv(env_name)
            if not raw:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value, f"env:{env_name}"

        return default, "default"

    @classmethod
    def _resolve_positive_int(
        cls,
        *,
        attr_names: tuple[str, ...],
        objs: tuple[object | None, ...],
        env_names: tuple[str, ...],
        default: int | None,
    ) -> tuple[int | None, str]:
        for obj in objs:
            if obj is None:
                continue
            for name in attr_names:
                value = getattr(obj, name, None)
                if value is None:
                    continue
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value, f"{type(obj).__name__}.{name}"

        for env_name in env_names:
            raw = os.getenv(env_name)
            if not raw:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value, f"env:{env_name}"

        return default, "default"