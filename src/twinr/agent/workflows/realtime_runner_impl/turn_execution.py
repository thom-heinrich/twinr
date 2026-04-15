"""Per-turn execution, playback, ack, and print helpers for the realtime loop."""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed false-positive user interruptions caused by counting non-consecutive
#        interrupt windows after failed/too-short STT results.
# BUG-2: Fixed realtime interruption drift by requesting immediate response cancel and
#        best-effort server-side audio truncation/sync when the user barges in.
# BUG-3: Fixed stale runtime/orchestrator state after provider/playback failures by
#        performing explicit failure cleanup before re-raising.
# BUG-4: Fixed unbounded streamed-audio buffering that could grow until RAM pressure
#        on Raspberry Pi 4; playback now uses bounded backpressure-aware queues.
# BUG-5: Seeded text turns now reuse an already-open `listening` state so remote
#        follow-up transcript commits cannot reopen listening and trip the runtime
#        state machine into a spurious live `error`.
# SEC-1: Fixed raw transcript/response log leakage and log-injection risk by sanitizing
#        emitted text and making raw text observability opt-in.
# IMP-1: Added pluggable interrupt-front-end hooks so AEC/NS/pVAD/EOT front-ends can be
#        injected without changing call sites.
# IMP-2: Added played-audio estimation, interruption telemetry, and safer follow-up
#        finalization when transport-level truncation is unavailable.


from __future__ import annotations

import hashlib
import inspect
import time
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Callable, cast

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveListeningWindow
from twinr.agent.base_agent.conversation.decision_core import normalize_turn_text
from twinr.agent.base_agent.conversation.follow_up_context import (
    pending_conversation_follow_up_hint_scope,
)
from twinr.agent.base_agent.contracts import StreamingSpeechToTextProvider
from twinr.agent.workflows.forensics import current_workflow_trace_id
from twinr.agent.workflows.listen_timeout_diagnostics import (
    diagnostics_from_exception,
    emit_listen_timeout_diagnostics,
)
from twinr.agent.workflows.playback_coordinator import PlaybackPriority
from twinr.agent.workflows.print_lane import PrintLaneRequest
from twinr.agent.workflows.voice_turn_latency import (
    emit_voice_turn_latency_breakdown,
    mark_voice_turn_supervisor_ready,
    mark_voice_turn_tts_started,
)
from twinr.hardware.audio import normalize_wav_playback_level, pcm16_to_wav_bytes


_OPTIONAL_HANDLER_MISSING = object()


class TwinrRealtimeTurnExecutionMixin:
    """Execute one realtime turn once capture/transcript inputs are ready."""

    def _config_value(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    def _sanitize_emitted_text(self, value: str | None, *, limit: int | None = None) -> str:
        raw = "" if value is None else str(value)
        sanitized_chars: list[str] = []
        for ch in raw:
            codepoint = ord(ch)
            if ch in "\r\n\t":
                sanitized_chars.append(" ")
            elif codepoint < 32 or codepoint == 127:
                sanitized_chars.append("?")
            else:
                sanitized_chars.append(ch)
        sanitized = " ".join("".join(sanitized_chars).split())
        if limit is not None and limit >= 0 and len(sanitized) > limit:
            if limit <= 1:
                return "…"
            return sanitized[: limit - 1].rstrip() + "…"
        return sanitized

    def _text_observation_digest(self, value: str | None) -> tuple[int, str]:
        raw = "" if value is None else str(value)
        digest = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16]
        return len(raw), digest

    def _emit_text_observation(self, key: str, value: str | None) -> None:
        raw = "" if value is None else str(value)
        sanitized = self._sanitize_emitted_text(raw)
        preview_limit = max(0, int(self._config_value("safe_emit_text_preview_chars", 0)))
        text_len, digest = self._text_observation_digest(raw)
        # BREAKING: Raw transcript/response emission is now opt-in via
        # config.unsafe_emit_raw_turn_text to avoid leaking PII and allowing
        # log-forging/control-character injection from untrusted text inputs.
        if bool(self._config_value("unsafe_emit_raw_turn_text", False)):
            self.emit(f"{key}={sanitized}")
            return
        self.emit(f"{key}=[redacted len={text_len} sha256={digest}]")
        if preview_limit > 0:
            self.emit(f"{key}_preview={self._sanitize_emitted_text(raw, limit=preview_limit)}")

    def _usage_transcript_for_recording(self, transcript: str | None) -> str | None:
        # BREAKING: Usage telemetry stops storing raw transcripts by default.
        # Restore the old behavior only by setting config.store_usage_transcript = True.
        if bool(self._config_value("store_usage_transcript", False)):
            return transcript
        return None

    def _first_attr(self, obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if obj is None:
                break
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return default

    def _call_optional_handler(
        self,
        target: Any,
        method_names: tuple[str, ...],
        /,
        **kwargs: Any,
    ) -> Any:
        if target is None:
            return _OPTIONAL_HANDLER_MISSING
        for method_name in method_names:
            method = getattr(target, method_name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                return method(**kwargs)
            accepts_var_kwargs = False
            supported_kwargs: dict[str, Any] = {}
            for parameter in signature.parameters.values():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    accepts_var_kwargs = True
                    break
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ) and parameter.name in kwargs:
                    supported_kwargs[parameter.name] = kwargs[parameter.name]
            if accepts_var_kwargs:
                supported_kwargs = dict(kwargs)
            return method(**supported_kwargs)
        return _OPTIONAL_HANDLER_MISSING

    def _interrupt_frontend(self) -> Any:
        return getattr(self, "interrupt_frontend", None)

    def _prepare_interrupt_window(self, window: Any) -> Any:
        frontend = self._interrupt_frontend()
        method = getattr(frontend, "prepare_window", None)
        if not callable(method):
            return window
        prepare_window = cast(Callable[[Any], Any], method)
        prepared = prepare_window(window)  # pylint: disable=not-callable
        return window if prepared is None else prepared

    def _confirm_interrupt_candidate(self, *, window: Any, transcript: str, sample: Any) -> bool:
        frontend = self._interrupt_frontend()
        method = getattr(frontend, "confirm_interrupt", None)
        if not callable(method):
            return True
        confirm_interrupt = cast(Callable[..., Any], method)
        return bool(
            confirm_interrupt(  # pylint: disable=not-callable
                window=window,
                transcript=transcript,
                sample=sample,
            )
        )

    def _realtime_interrupt_targets(self) -> tuple[Any, ...]:
        targets: list[Any] = []
        for candidate in (
            getattr(self, "realtime_session", None),
            getattr(self, "voice_orchestrator", None),
            self,
        ):
            if candidate is None:
                continue
            if candidate not in targets:
                targets.append(candidate)
        return tuple(targets)

    def _request_realtime_response_cancel(self, *, turn: Any = None) -> bool:
        response_id = self._first_attr(turn, "response_id", "id")
        invoked = False
        for target in self._realtime_interrupt_targets():
            result = self._call_optional_handler(
                target,
                (
                    "cancel_response",
                    "response_cancel",
                    "cancel_current_response",
                    "cancel_active_response",
                    "interrupt_response",
                    "interrupt",
                    "cancel",
                ),
                response_id=response_id,
                turn=turn,
            )
            if result is not _OPTIONAL_HANDLER_MISSING:
                invoked = True
        self.emit(f"realtime_interrupt_cancel_requested={str(invoked).lower()}")
        return invoked

    def _sync_realtime_interrupt_with_provider(
        self,
        *,
        turn: Any,
        played_audio_ms: int | None,
    ) -> tuple[bool, str | None]:
        if turn is None:
            return False, None
        response_id = self._first_attr(turn, "response_id", "id")
        item_id = self._first_attr(
            turn,
            "item_id",
            "assistant_item_id",
            "output_item_id",
            "conversation_item_id",
        )
        synced = False
        truncated_text: str | None = None
        for target in self._realtime_interrupt_targets():
            cancel_result = self._call_optional_handler(
                target,
                (
                    "cancel_response",
                    "response_cancel",
                    "cancel_current_response",
                    "cancel_active_response",
                    "interrupt_response",
                    "interrupt",
                    "cancel",
                ),
                response_id=response_id,
                turn=turn,
            )
            if cancel_result is not _OPTIONAL_HANDLER_MISSING:
                synced = True
            truncate_result = self._call_optional_handler(
                target,
                (
                    "sync_interrupted_playback",
                    "truncate_response_audio",
                    "truncate_current_response",
                    "truncate_response",
                    "truncate_audio",
                    "conversation_item_truncate",
                    "truncate",
                ),
                response_id=response_id,
                item_id=item_id,
                content_index=0,
                audio_end_ms=played_audio_ms,
                turn=turn,
            )
            if truncate_result is _OPTIONAL_HANDLER_MISSING:
                continue
            synced = True
            if isinstance(truncate_result, str):
                truncated_text = truncate_result
            elif isinstance(truncate_result, dict):
                text_candidate = truncate_result.get("response_text_truncated") or truncate_result.get(
                    "truncated_text"
                )
                if isinstance(text_candidate, str):
                    truncated_text = text_candidate
        self.emit(f"realtime_interrupt_provider_synced={str(synced).lower()}")
        return synced, truncated_text

    def _assistant_text_after_interrupt(
        self,
        *,
        response_text: str,
        turn: Any,
        played_audio_ms: int | None,
        provider_truncated_text: str | None,
    ) -> str:
        if provider_truncated_text is not None:
            return provider_truncated_text
        custom_result = self._call_optional_handler(
            self,
            (
                "assistant_text_after_interrupt",
                "truncate_assistant_text_after_interrupt",
            ),
            response_text=response_text,
            turn=turn,
            played_audio_ms=played_audio_ms,
        )
        if isinstance(custom_result, str):
            return custom_result
        if not response_text:
            return ""
        total_audio_ms = self._first_attr(
            turn,
            "audio_duration_ms",
            "output_audio_duration_ms",
            "response_audio_duration_ms",
            default=None,
        )
        if played_audio_ms is None:
            self.emit("realtime_interrupt_text_dropped=true")
            return ""
        if total_audio_ms and total_audio_ms > 0:
            played_ratio = max(0.0, min(1.0, float(played_audio_ms) / float(total_audio_ms)))
            if played_ratio >= 0.98:
                return response_text
        self.emit("realtime_interrupt_text_dropped=true")
        return ""

    def _cleanup_failed_realtime_turn(
        self,
        *,
        listen_source: str,
        error: Exception,
        turn_label: str | None,
    ) -> None:
        self.emit(f"realtime_turn_failed={type(error).__name__}")
        try:
            self._record_event(
                "realtime_turn_failed",
                "Realtime turn aborted before finalization.",
                error_type=type(error).__name__,
                request_source=listen_source,
                turn_label=turn_label,
            )
        except Exception:
            pass
        try:
            self.runtime.finish_speaking()
        finally:
            self._emit_status(force=True)
            self._notify_voice_orchestrator_state("waiting", detail=listen_source)
            self._stop_working_feedback()

    def _run_single_audio_turn(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        listening_window: AdaptiveListeningWindow,
        listen_source: str,
        proactive_trigger: str | None,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
        timeout_emit_key: str,
        timeout_message: str,
        play_initial_beep: bool,
    ) -> bool:
        turn_started = time.monotonic()
        self._ensure_live_audio_turn_supported(
            initial_source=initial_source,
            listen_source=listen_source,
        )
        self._begin_audio_turn_listening(
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        if play_initial_beep:
            self._play_listen_beep()

        capture_started = time.monotonic()
        if self._voice_orchestrator_owns_live_listening():
            wait_handle = self._begin_remote_transcript_wait(source="listening")
            if wait_handle is None:
                self.runtime.cancel_listening()
                self._emit_status(force=True)
                self._notify_voice_orchestrator_state("waiting", detail="remote_wait_failed")
                return False
            try:
                self._notify_voice_orchestrator_state(
                    "listening",
                    detail=listen_source,
                    follow_up_allowed=follow_up,
                    wait_id=wait_handle.wait_id,
                )
                committed = self._wait_for_remote_transcript_commit(
                    wait_handle=wait_handle,
                    timeout_s=listening_window.start_timeout_s,
                    initial_source=initial_source,
                    follow_up=follow_up,
                    listen_source=listen_source,
                    timeout_emit_key=timeout_emit_key,
                    timeout_message=timeout_message,
                )
            finally:
                self._remote_transcript_commits.clear_wait(wait_handle)
            if committed is None:
                return False
            capture_ms = int((time.monotonic() - capture_started) * 1000)
            realtime_started = time.monotonic()
            with pending_conversation_follow_up_hint_scope(
                self.runtime,
                active=listen_source == "follow_up",
            ):
                return self._complete_realtime_turn(
                    transcript_seed=committed.transcript,
                    turn_label=None,
                    initial_source=initial_source,
                    listen_source=listen_source,
                    proactive_trigger=proactive_trigger,
                    turn_started=turn_started,
                    capture_ms=capture_ms,
                    stt_ms=-1,
                    turn_runner=lambda on_audio_chunk, on_output_text_delta: self.realtime_session.run_text_turn(
                        committed.transcript,
                        conversation=self._conversation_context_for_turn_label(None),
                        on_audio_chunk=on_audio_chunk,
                        on_output_text_delta=on_output_text_delta,
                    ),
                    realtime_started=realtime_started,
                )
        self._notify_voice_orchestrator_state(
            "listening",
            detail=listen_source,
            follow_up_allowed=follow_up,
        )
        try:
            with self._audio_lock:
                transcript_seed = ""
                stt_ms = -1
                turn_label = None
                if isinstance(self.turn_stt_provider, StreamingSpeechToTextProvider):
                    try:
                        capture_result, transcript_seed, capture_ms, stt_ms, turn_label = (
                            self._capture_and_transcribe_with_turn_controller(
                                stt_provider=self.turn_stt_provider,
                                listening_window=listening_window,
                                speech_start_chunks=speech_start_chunks,
                                ignore_initial_ms=ignore_initial_ms,
                            )
                        )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        self.emit(f"turn_controller_capture_failed={type(exc).__name__}")
                        raise RuntimeError(
                            f"turn_controller_capture_failed:{type(exc).__name__}"
                        ) from exc
                else:
                    capture_result = self.recorder.capture_pcm_until_pause_with_options(
                        pause_ms=listening_window.speech_pause_ms,
                        start_timeout_s=listening_window.start_timeout_s,
                        speech_start_chunks=speech_start_chunks,
                        ignore_initial_ms=ignore_initial_ms,
                        pause_grace_ms=listening_window.pause_grace_ms,
                        should_stop=self._active_turn_stop_requested,
                    )
                    capture_ms = int((time.monotonic() - capture_started) * 1000)
        except RuntimeError as exc:
            if self._active_turn_stop_requested():
                self._cancel_interrupted_turn()
                return False
            if not self._is_no_speech_timeout(exc):
                raise
            emit_listen_timeout_diagnostics(
                self.emit,
                diagnostics_from_exception(exc),
            )
            self.runtime.remember_listen_timeout(
                initial_source=initial_source,
                follow_up=follow_up,
            )
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self._notify_voice_orchestrator_state("waiting", detail="listen_timeout")
            self.emit(f"{timeout_emit_key}=true")
            self._record_event("listen_timeout", timeout_message, request_source=listen_source)
            return False
        audio_pcm = capture_result.pcm_bytes
        if self._active_turn_stop_requested():
            self._cancel_interrupted_turn()
            return False
        self.runtime.remember_listen_capture(
            initial_source=initial_source,
            follow_up=follow_up,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        try:
            self._update_voice_assessment_from_pcm(audio_pcm)
        except Exception as exc:
            self.emit(f"voice_assessment_failed={type(exc).__name__}")
        realtime_started = time.monotonic()
        self._current_turn_audio_pcm = audio_pcm
        self._current_turn_audio_sample_rate = self._recorder_sample_rate()
        if turn_label:
            self.emit(f"turn_controller_selected_label={turn_label}")
        try:
            with pending_conversation_follow_up_hint_scope(
                self.runtime,
                active=listen_source == "follow_up",
            ):
                return self._complete_realtime_turn(
                    transcript_seed=transcript_seed.strip() or "[voice input]",
                    turn_label=turn_label,
                    initial_source=initial_source,
                    listen_source=listen_source,
                    proactive_trigger=proactive_trigger,
                    turn_started=turn_started,
                    capture_ms=capture_ms,
                    stt_ms=stt_ms,
                    turn_runner=lambda on_audio_chunk, on_output_text_delta: self.realtime_session.run_audio_turn(
                        audio_pcm,
                        conversation=self._conversation_context_for_turn_label(turn_label),
                        on_audio_chunk=on_audio_chunk,
                        on_output_text_delta=on_output_text_delta,
                    ),
                    realtime_started=realtime_started,
                )
        finally:
            self._current_turn_audio_pcm = None
            self._current_turn_audio_sample_rate = None

    def _run_single_text_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        turn_started = time.monotonic()
        self._begin_text_turn_listening(
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        self._notify_voice_orchestrator_state("thinking", detail=listen_source)
        realtime_started = time.monotonic()
        with pending_conversation_follow_up_hint_scope(
            self.runtime,
            active=listen_source == "follow_up",
        ):
            return self._complete_realtime_turn(
                transcript_seed=transcript,
                turn_label=None,
                initial_source=listen_source,
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                turn_started=turn_started,
                capture_ms=0,
                stt_ms=-1,
                turn_runner=lambda on_audio_chunk, on_output_text_delta: self.realtime_session.run_text_turn(
                    transcript,
                    conversation=self._conversation_context_for_turn_label(None),
                    on_audio_chunk=on_audio_chunk,
                    on_output_text_delta=on_output_text_delta,
                ),
                realtime_started=realtime_started,
            )

    def _begin_text_turn_listening(
        self,
        *,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> None:
        """Reuse an already-open listening window before seeded text turns."""

        if getattr(self.runtime.status, "value", None) == "listening":
            self._trace_event(
                "text_turn_reuse_listening_state",
                kind="branch",
                details={"listen_source": listen_source},
            )
            return
        self.runtime.begin_listening(
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )

    def _begin_audio_turn_listening(
        self,
        *,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> None:
        """Reuse an already-open listening window before audio turns."""

        if getattr(self.runtime.status, "value", None) == "listening":
            self._trace_event(
                "audio_turn_reuse_listening_state",
                kind="branch",
                details={"listen_source": listen_source},
            )
            return
        if listen_source == "button":
            self.runtime.press_green_button()
            return
        self.runtime.begin_listening(
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )

    def _run_interrupt_follow_up_turn(self) -> bool:
        listening_window = self.runtime.listening_window(
            initial_source="button",
            follow_up=True,
        )
        return self._run_single_audio_turn(
            initial_source="button",
            follow_up=True,
            listening_window=listening_window,
            listen_source="interrupt",
            proactive_trigger=None,
            speech_start_chunks=self.config.audio_follow_up_speech_start_chunks,
            ignore_initial_ms=0,
            timeout_emit_key="interrupt_follow_up_timeout",
            timeout_message="Interrupt follow-up listening window expired.",
            play_initial_beep=False,
        )

    def _start_answer_interrupt_watcher(
        self,
        *,
        interrupt_event: Event,
        stop_event: Event,
        transcript_holder: list[str],
        answer_started: Callable[[], bool],
        on_interrupt: Callable[[], None] | None = None,
    ) -> Thread | None:
        if self._voice_orchestrator_owns_live_listening():
            self.emit("interrupt_watcher_remote_owned=true")
            return None
        if not self.config.turn_controller_interrupt_enabled:
            return None
        stt_provider = self._interrupt_stt_provider()
        if stt_provider is None:
            return None
        try:
            sampler = self._ambient_sampler()
        except Exception as exc:
            self.emit(f"interrupt_sampler_failed={type(exc).__name__}")
            return None

        min_active_ratio = max(0.0, float(self.config.turn_controller_interrupt_min_active_ratio))
        min_transcript_chars = max(
            1, int(self.config.turn_controller_interrupt_min_transcript_chars)
        )
        required_windows = max(
            1, int(self.config.turn_controller_interrupt_consecutive_windows)
        )
        window_ms = max(120, int(self.config.turn_controller_interrupt_window_ms))
        poll_s = max(0.02, int(self.config.turn_controller_interrupt_poll_ms) / 1000.0)
        min_transcribe_interval_s = max(
            0.0,
            float(
                self._config_value(
                    "turn_controller_interrupt_min_transcribe_interval_ms",
                    0,
                )
            )
            / 1000.0,
        )

        def worker() -> None:
            consecutive_confirmed = 0
            saw_candidate = False
            last_transcribe_at = 0.0
            while not stop_event.is_set() and not interrupt_event.is_set():
                if not answer_started():
                    if stop_event.wait(0.02):
                        return
                    continue
                try:
                    raw_window = sampler.sample_window(duration_ms=window_ms)
                    window = self._prepare_interrupt_window(raw_window)
                except Exception as exc:
                    self.emit(f"interrupt_sampler_failed={type(exc).__name__}")
                    return
                sample = window.sample
                speech_like = (
                    sample.active_chunk_count > 0 and sample.active_ratio >= min_active_ratio
                )
                if not speech_like:
                    if saw_candidate:
                        self.emit("false_interrupt_recovered=true")
                    consecutive_confirmed = 0
                    saw_candidate = False
                    if stop_event.wait(poll_s):
                        return
                    continue
                saw_candidate = True
                now = time.monotonic()
                if min_transcribe_interval_s > 0 and (now - last_transcribe_at) < min_transcribe_interval_s:
                    if stop_event.wait(poll_s):
                        return
                    continue
                transcript = ""
                try:
                    last_transcribe_at = now
                    wav_bytes = pcm16_to_wav_bytes(
                        window.pcm_bytes,
                        sample_rate=window.sample_rate,
                        channels=window.channels,
                    )
                    transcript = stt_provider.transcribe(
                        wav_bytes,
                        filename="interrupt.wav",
                        content_type="audio/wav",
                        language=self.config.deepgram_stt_language,
                    ).strip()
                except Exception as exc:
                    self.emit(f"interrupt_transcribe_failed={type(exc).__name__}")
                normalized = normalize_turn_text(transcript).replace(" ", "")
                if len(normalized) < min_transcript_chars:
                    consecutive_confirmed = 0
                    if stop_event.wait(poll_s):
                        return
                    continue
                if not self._confirm_interrupt_candidate(
                    window=window,
                    transcript=transcript,
                    sample=sample,
                ):
                    consecutive_confirmed = 0
                    if stop_event.wait(poll_s):
                        return
                    continue
                consecutive_confirmed += 1
                self._emit_text_observation("interrupt_candidate_transcript", transcript)
                self.emit(f"interrupt_candidate_windows={consecutive_confirmed}")
                if consecutive_confirmed >= required_windows:
                    transcript_holder[0] = transcript
                    interrupt_event.set()
                    if callable(on_interrupt):
                        try:
                            on_interrupt()
                        except Exception as exc:
                            self._trace_event(
                                "interrupt_callback_failed",
                                kind="error",
                                level="ERROR",
                                details={"error_type": type(exc).__name__},
                            )
                    self.emit("user_interrupt_detected=true")
                    return
                if stop_event.wait(poll_s):
                    return

        thread = Thread(target=worker, daemon=True, name="twinr-answer-interrupt")
        thread.start()
        return thread

    def _complete_realtime_turn(
        self,
        *,
        transcript_seed: str,
        turn_label: str | None,
        initial_source: str,
        listen_source: str,
        proactive_trigger: str | None,
        turn_started: float,
        capture_ms: int,
        stt_ms: int,
        turn_runner,
        realtime_started: float,
    ) -> bool:
        self.runtime.submit_transcript(transcript_seed)
        self._emit_status(force=True)
        self._notify_voice_orchestrator_state("thinking", detail=listen_source)
        stop_processing_feedback = self._start_working_feedback_loop("processing")
        workflow_trace_id = current_workflow_trace_id()
        mark_voice_turn_supervisor_ready(trace_id=workflow_trace_id)

        def stop_answering_feedback() -> None:
            return None

        queue_max_chunks = max(2, int(self._config_value("realtime_playback_queue_max_chunks", 24)))
        queue_put_timeout_s = max(
            0.02,
            float(self._config_value("realtime_playback_queue_put_timeout_ms", 100)) / 1000.0,
        )
        backpressure_log_cooldown_s = max(
            0.1,
            float(self._config_value("realtime_playback_backpressure_log_cooldown_ms", 1000))
            / 1000.0,
        )

        audio_chunks: Queue[bytes | None] = Queue(maxsize=queue_max_chunks)
        playback_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        first_audio_ms_override: int | None = None
        answer_started = False
        interrupt_event = Event()
        interrupt_stop_event = Event()
        interrupt_transcript_holder = [""]
        interrupt_thread: Thread | None = None
        playback_started = False
        turn = None
        turn_error: Exception | None = None
        played_audio_bytes = [0]
        backpressure_last_emit_at = [0.0]
        interrupt_detected_at = [None]
        self._set_answer_interrupt_event(interrupt_event)

        def estimate_played_audio_ms() -> int | None:
            if first_audio_at[0] is None:
                return None
            sample_rate = max(1, int(self.config.openai_realtime_input_sample_rate))
            channels = max(1, int(self.config.audio_channels))
            bytes_per_ms = (sample_rate * channels * 2) / 1000.0
            yielded_ms = int(played_audio_bytes[0] / bytes_per_ms) if bytes_per_ms > 0 else 0
            elapsed_ms = max(0, int((time.monotonic() - first_audio_at[0]) * 1000))
            return min(yielded_ms, elapsed_ms)

        def signal_playback_stop() -> None:
            if not playback_started:
                return
            while True:
                try:
                    audio_chunks.put_nowait(None)
                    return
                except Full:
                    try:
                        audio_chunks.get_nowait()
                    except Empty:
                        return

        def begin_answering() -> None:
            nonlocal answer_started, stop_answering_feedback, interrupt_thread
            if answer_started:
                return
            stop_processing_feedback()
            self.runtime.begin_answering()
            self._emit_status(force=True)
            self._notify_voice_orchestrator_state(
                "speaking",
                detail=listen_source,
                follow_up_allowed=self._follow_up_allowed_for_source(
                    initial_source=initial_source
                ),
            )
            stop_answering_feedback = self._start_working_feedback_loop("answering")
            answer_started = True
            if interrupt_thread is None:
                def handle_interrupt() -> None:
                    interrupt_detected_at[0] = time.monotonic()
                    self._request_realtime_response_cancel(turn=turn)
                    signal_playback_stop()

                interrupt_thread = self._start_answer_interrupt_watcher(
                    interrupt_event=interrupt_event,
                    stop_event=interrupt_stop_event,
                    transcript_holder=interrupt_transcript_holder,
                    answer_started=lambda: answer_started,
                    on_interrupt=handle_interrupt,
                )

        def playback_generator():
            while True:
                chunk = audio_chunks.get()
                if chunk is None:
                    return
                played_audio_bytes[0] += len(chunk)
                yield chunk

        def playback_worker() -> None:
            try:
                self.player.play_pcm16_chunks(
                    playback_generator(),
                    sample_rate=self.config.openai_realtime_input_sample_rate,
                    channels=self.config.audio_channels,
                    should_stop=interrupt_event.is_set,
                )
            except Exception as exc:
                playback_error.append(exc)

        worker: Thread | None = None

        def ensure_playback_started() -> None:
            nonlocal worker, playback_started
            if playback_started:
                return
            worker = Thread(target=playback_worker, daemon=True)
            worker.start()
            playback_started = True

        def enqueue_audio_chunk(chunk: bytes) -> bool:
            while not interrupt_event.is_set() and not interrupt_stop_event.is_set():
                try:
                    audio_chunks.put(chunk, timeout=queue_put_timeout_s)
                    return True
                except Full:
                    now = time.monotonic()
                    if (now - backpressure_last_emit_at[0]) >= backpressure_log_cooldown_s:
                        backpressure_last_emit_at[0] = now
                        self.emit("realtime_playback_backpressure=true")
                        self.emit(f"realtime_playback_queue_max_chunks={queue_max_chunks}")
            return False

        def on_audio_chunk(chunk: bytes) -> None:
            begin_answering()
            if interrupt_event.is_set():
                return
            if first_audio_at[0] is None:
                stop_answering_feedback()
                first_audio_at[0] = time.monotonic()
                mark_voice_turn_tts_started(trace_id=workflow_trace_id)
                ensure_playback_started()
            enqueue_audio_chunk(chunk)

        def on_output_text_delta(_delta: str) -> None:
            if interrupt_event.is_set():
                return
            begin_answering()

        try:
            with self.realtime_session:
                turn = turn_runner(on_audio_chunk, on_output_text_delta)
        except Exception as exc:
            turn_error = exc
        finally:
            stop_processing_feedback()
            interrupt_stop_event.set()
            self._clear_answer_interrupt_event(interrupt_event)
            if playback_started:
                signal_playback_stop()
        if interrupt_thread is not None:
            interrupt_thread.join(timeout=1.0)
            if interrupt_thread.is_alive():
                self.emit("interrupt_watcher_join_timeout=true")
        if worker is not None:
            playback_join_timeout_s = max(
                0.1,
                float(getattr(self.config, "realtime_playback_join_timeout_s", 5.0)),
            )
            worker.join(timeout=playback_join_timeout_s)
            if worker.is_alive():
                playback_error.append(
                    TimeoutError(
                        f"Playback worker did not exit within {playback_join_timeout_s:.2f}s"
                    )
                )
                self.emit("playback_join_timeout=true")
        realtime_ms = int((time.monotonic() - realtime_started) * 1000)
        self.emit("realtime_turn_provider_complete=true")
        self.emit(f"realtime_turn_answer_started={str(answer_started).lower()}")
        self.emit(
            f"realtime_turn_provider_audio_received={str(first_audio_at[0] is not None).lower()}"
        )
        self.emit(f"realtime_turn_provider_playback_started={str(playback_started).lower()}")

        fatal_error: Exception | None = turn_error
        if fatal_error is None and playback_error:
            fatal_error = playback_error[0]
        if fatal_error is None and turn is None:
            fatal_error = RuntimeError("Realtime turn did not return a result")

        response_text = ""
        provider_truncated_text: str | None = None
        provider_synced_interrupt = False
        played_audio_ms = estimate_played_audio_ms()
        if played_audio_ms is not None:
            self.emit(f"realtime_played_audio_ms={played_audio_ms}")

        try:
            if fatal_error is not None:
                self._cleanup_failed_realtime_turn(
                    listen_source=listen_source,
                    error=fatal_error,
                    turn_label=turn_label,
                )
                raise fatal_error

            response_text = turn.response_text or ""
            if first_audio_at[0] is None and response_text.strip():
                self.emit("realtime_provider_audio_missing=true")
                raise RuntimeError("Realtime turn returned text without provider audio.")
        finally:
            stop_answering_feedback()
            self._stop_working_feedback()

        if interrupt_event.is_set():
            provider_synced_interrupt, provider_truncated_text = (
                self._sync_realtime_interrupt_with_provider(
                    turn=turn,
                    played_audio_ms=played_audio_ms,
                )
            )

        final_transcript = ((turn.transcript or "").strip() or transcript_seed)
        self.runtime.last_transcript = final_transcript
        self._emit_text_observation("transcript", final_transcript)
        if not answer_started:
            begin_answering()

        response_text_for_runtime = response_text
        if interrupt_event.is_set():
            response_text_for_runtime = self._assistant_text_after_interrupt(
                response_text=response_text,
                turn=turn,
                played_audio_ms=played_audio_ms,
                provider_truncated_text=provider_truncated_text,
            )
            if response_text_for_runtime != response_text:
                self.emit("assistant_response_context_trimmed=true")
                if not provider_synced_interrupt:
                    self.emit("assistant_response_context_trim_reason=local_interrupt_trim")

        force_close = True
        follow_up_mode = "disabled"
        rearm_remote_follow_up = False
        self.emit("realtime_turn_finalize_start=true")
        try:
            if interrupt_event.is_set() and not response_text_for_runtime.strip():
                self.runtime.finalize_interrupted_turn(
                    source="conversation",
                    modality="voice",
                )
                self.emit("assistant_response_omitted_after_interrupt=true")
                if turn.response_id:
                    self.emit(f"openai_response_id={turn.response_id}")
                self._record_usage(
                    request_kind="realtime_conversation",
                    source="realtime_loop",
                    model=turn.model or self.config.openai_realtime_model,
                    response_id=turn.response_id,
                    request_id=None,
                    used_web_search=None,
                    token_usage=turn.token_usage,
                    transcript=self._usage_transcript_for_recording(final_transcript),
                    request_source=listen_source,
                    proactive_trigger=proactive_trigger,
                )
                force_close = False
                follow_up_mode = self._voice_orchestrator_follow_up_mode(
                    initial_source=initial_source
                )
            else:
                answer = self.runtime.finalize_agent_turn(response_text_for_runtime)
                self._emit_text_observation("response", answer)
                if turn.response_id:
                    self.emit(f"openai_response_id={turn.response_id}")
                self._record_usage(
                    request_kind="realtime_conversation",
                    source="realtime_loop",
                    model=turn.model or self.config.openai_realtime_model,
                    response_id=turn.response_id,
                    request_id=None,
                    used_web_search=None,
                    token_usage=turn.token_usage,
                    transcript=self._usage_transcript_for_recording(final_transcript),
                    request_source=listen_source,
                    proactive_trigger=proactive_trigger,
                )
                force_close = turn.end_conversation or self._follow_up_vetoed_by_closure(
                    user_transcript=final_transcript,
                    assistant_response=answer,
                    request_source=initial_source,
                    proactive_trigger=proactive_trigger,
                )
                follow_up_mode = self._voice_orchestrator_follow_up_mode(
                    initial_source=initial_source
                )
                rearm_remote_follow_up = (
                    not force_close and follow_up_mode == "remote" and not interrupt_event.is_set()
                )
        finally:
            if rearm_remote_follow_up:
                self.emit("realtime_turn_finish_path=rearm_follow_up")
                self.runtime.rearm_follow_up(request_source="follow_up")
            else:
                self.emit("realtime_turn_finish_path=finish_speaking")
                self.runtime.finish_speaking()
            self._emit_status(force=True)
        self.emit("realtime_turn_finalize_done=true")
        if turn.end_conversation:
            self.emit("conversation_ended=true")
        elif force_close:
            self.emit("conversation_follow_up_vetoed=closure")
        if interrupt_event.is_set():
            self.emit("assistant_interrupted=true")
            if interrupt_detected_at[0] is not None:
                self.emit(
                    f"timing_interrupt_after_answer_start_ms="
                    f"{max(0, int((interrupt_detected_at[0] - (first_audio_at[0] or turn_started)) * 1000))}"
                )
            if interrupt_transcript_holder[0].strip():
                self._emit_text_observation(
                    "interrupt_transcript",
                    interrupt_transcript_holder[0].strip(),
                )
            interrupt_text_len, interrupt_text_digest = self._text_observation_digest(
                interrupt_transcript_holder[0].strip()
            )
            self._record_event(
                "user_interrupt_detected",
                "Twinr stopped the current spoken answer after detecting a user interruption.",
                transcript_len=interrupt_text_len,
                transcript_sha256=interrupt_text_digest,
                request_source=listen_source,
                turn_label=turn_label,
                provider_synced=provider_synced_interrupt,
                played_audio_ms=played_audio_ms,
            )
        self.emit(f"timing_capture_ms={capture_ms}")
        if stt_ms >= 0:
            self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_realtime_ms={realtime_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_ms_override is not None:
            self.emit(f"timing_first_audio_ms={first_audio_ms_override}")
        elif first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        emit_voice_turn_latency_breakdown(
            emit=self.emit,
            trace_event=self._trace_event,
            trace_id=workflow_trace_id,
        )
        if interrupt_event.is_set() and not force_close:
            return self._run_interrupt_follow_up_turn()
        if not force_close and follow_up_mode == "remote":
            self._acknowledge_follow_up_open(request_source=listen_source)
            self._notify_voice_orchestrator_state(
                "follow_up_open",
                detail=listen_source,
                follow_up_allowed=True,
            )
            return True
        self._notify_voice_orchestrator_state("waiting", detail=listen_source)
        return not force_close

    def _acknowledge_voice_activation(self) -> None:
        if self.voice_orchestrator is not None:
            self._play_listen_beep()
            self.emit("voice_activation_ack=earcon")
            self.emit("voice_activation_ack_cached=true")
            self.emit("timing_voice_activation_ack_tts_ms=0")
            self._record_event(
                "voice_activation_acknowledged",
                "Twinr confirmed a remote voice activation with an earcon before opening hands-free listening.",
                prompt="earcon",
            )
            return
        prompt = self.runtime.begin_voice_activation_prompt("Ja?")
        self._emit_status(force=True)
        cached_audio = self._cached_voice_activation_ack_wav_bytes()
        tts_started = time.monotonic()
        try:
            self._play_listen_beep()
            if cached_audio is None:
                cached_audio = self._build_voice_activation_ack_wav_bytes(prompt)
                with self._voice_activation_ack_cache_lock:
                    if self._voice_activation_ack_wav_bytes is None:
                        self._voice_activation_ack_wav_bytes = cached_audio
            self.playback_coordinator.play_wav_bytes(
                owner="voice_activation_ack",
                priority=PlaybackPriority.SPEECH,
                wav_bytes=cached_audio,
            )
        finally:
            self.runtime.finish_speaking()
            self._emit_status(force=True)
        self.emit(f"voice_activation_ack={prompt}")
        self.emit(f"voice_activation_ack_cached={str(cached_audio is not None).lower()}")
        self.emit(
            f"timing_voice_activation_ack_tts_ms={int((time.monotonic() - tts_started) * 1000)}"
        )
        self._record_event(
            "voice_activation_acknowledged",
            "Twinr confirmed a remote voice activation before opening hands-free listening.",
            prompt=prompt,
        )

    def _ensure_voice_activation_ack_prefetch_started(self) -> None:
        with self._voice_activation_ack_cache_lock:
            if self._voice_activation_ack_wav_bytes is not None:
                return
            if self._voice_activation_ack_prefetch_started:
                return
            self._voice_activation_ack_prefetch_started = True
            self._voice_activation_ack_prefetch_thread = Thread(
                target=self._prime_voice_activation_ack_cache,
                name="twinr-voice-activation-ack",
                daemon=True,
            )
            self._voice_activation_ack_prefetch_thread.start()

    def _cached_voice_activation_ack_wav_bytes(self) -> bytes | None:
        with self._voice_activation_ack_cache_lock:
            return self._voice_activation_ack_wav_bytes

    def _prime_voice_activation_ack_cache(self) -> None:
        try:
            audio_bytes = self._build_voice_activation_ack_wav_bytes("Ja?")
        except Exception as exc:
            with self._voice_activation_ack_cache_lock:
                self._voice_activation_ack_prefetch_started = False
                self._voice_activation_ack_prefetch_thread = None
            self.emit(f"voice_activation_ack_prefetch_failed={type(exc).__name__}")
            return
        with self._voice_activation_ack_cache_lock:
            self._voice_activation_ack_wav_bytes = audio_bytes
            self._voice_activation_ack_prefetch_started = False
            self._voice_activation_ack_prefetch_thread = None

    def _build_voice_activation_ack_wav_bytes(self, prompt: str) -> bytes:
        return normalize_wav_playback_level(self.tts_provider.synthesize(prompt))

    def _listening_window_speech_start_chunks(
        self, *, initial_source: str, follow_up: bool
    ) -> int | None:
        if initial_source == "button" and not follow_up:
            return None
        return self.config.audio_follow_up_speech_start_chunks

    def _listening_window_ignore_initial_ms(self, *, initial_source: str, follow_up: bool) -> int:
        if initial_source == "button" and not follow_up:
            return 0
        return self.config.audio_follow_up_ignore_ms

    def _listening_timeout_emit_key(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "follow_up_timeout"
        if initial_source == "proactive":
            return "proactive_listen_timeout"
        if initial_source == "voice_activation":
            return "voice_activation_listen_timeout"
        return "listen_timeout"

    def _listening_timeout_message(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "Follow-up listening window expired."
        if initial_source == "proactive":
            return "Hands-free listening window after a proactive prompt expired."
        if initial_source == "voice_activation":
            return "Voice-activation listening window expired."
        return "Listening timed out before speech started."

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        try:
            if self.print_lane.is_busy():
                self.emit("print_skipped=busy")
                self._record_event(
                    "print_skipped",
                    "Print request ignored because another print job is already in progress.",
                )
                return
            response_to_print = self.runtime.prepare_background_button_print_request()
            conversation = self.runtime.conversation_context()
            request = PrintLaneRequest(
                conversation=conversation,
                focus_hint=self.runtime.last_transcript,
                direct_text=response_to_print,
                request_source="button",
                usage_source="realtime_loop",
                printer_queue=self.config.printer_queue,
                multimodal_source="realtime_print",
            )
            if not self.print_lane.submit(request):
                self.emit("print_skipped=busy")
                self._record_event(
                    "print_skipped",
                    "Print request could not be queued because another print job won the race.",
                )
                return
            self.emit("print_lane=queued")
            self._emit_status(force=True)
        except Exception:
            self._emit_status(force=True)
            raise
