"""Streaming capture and transcript assembly helpers for the realtime loop."""

# CHANGELOG: 2026-03-27
# BUG-1: Serialized controller/shared-state access across callback and capture threads to
#        eliminate turn-decision races and inconsistent transcript hints.
# BUG-2: Added bounded finalize deadlines plus explicit recovery paths so stalled streaming
#        providers no longer hang the realtime loop indefinitely.
# BUG-3: Promote transport send failures into buffered-audio recovery instead of silently
#        accepting truncated or empty transcripts.
# SEC-1: Raw partial transcripts are no longer emitted by default; telemetry is sanitized
#        and fingerprinted unless an explicit opt-in flag enables full transcript logging.
# SEC-2: Per-turn audio buffering is now bounded to prevent memory-exhaustion / availability
#        failures on constrained deployments such as Raspberry Pi 4.
# IMP-1: Opportunistically uses provider control hooks for mid-stream configuration, stream
#        flush / CloseStream, and keepalive when the session implementation exposes them.
# IMP-2: Added optional hybrid fallback hooks so cloud-stream degradation can recover from
#        buffered PCM through provider or on-device STT backends.
# IMP-3: Added safer transcript telemetry and transcript-stability helpers tuned for modern
#        voice-agent streaming pipelines.


from __future__ import annotations

from hashlib import sha1
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, cast

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveListeningWindow
from twinr.agent.base_agent.conversation.decision_core import normalize_turn_text
from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController
from twinr.agent.base_agent.contracts import (
    StreamingSpeechEndpointEvent,
    StreamingSpeechToTextProvider,
)
from twinr.hardware.audio import SpeechCaptureResult


class TwinrRealtimeTurnCaptureMixin:
    """Own streaming-STT capture and transcript recovery for one turn."""

    _STREAMING_BYTES_PER_SAMPLE = 2
    _DEFAULT_MAX_BUFFERED_AUDIO_MS = 45_000
    _DEFAULT_FINALIZE_TIMEOUT_MS = 6_500
    _DEFAULT_KEEPALIVE_INTERVAL_S = 3.5
    _DEFAULT_PARTIAL_PREVIEW_CHARS = 0
    _EARLY_SNAPSHOT_POLL_S = 0.02

    def _cfg(self, name: str, default: Any) -> Any:
        config = getattr(self, "config", None)
        return getattr(config, name, default)

    def _clean_transcript_text(self, value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _sanitize_for_log(self, value: str | None) -> str:
        if not value:
            return ""
        return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()

    def _text_sha12(self, value: str | None) -> str | None:
        cleaned = (value or "").strip()
        if not cleaned:
            return None
        return sha1(cleaned.encode("utf-8")).hexdigest()[:12]

    def _trace_text_payload(self, value: str | None) -> dict[str, Any]:
        cleaned = self._sanitize_for_log(value)
        preview_chars = max(
            0,
            int(self._cfg("streaming_transcript_preview_chars", self._DEFAULT_PARTIAL_PREVIEW_CHARS)),
        )
        payload: dict[str, Any] = {
            "text_len": len(cleaned),
            "text_sha12": self._text_sha12(cleaned),
        }
        if preview_chars > 0 and cleaned:
            payload["preview"] = cleaned[:preview_chars]
        return payload

    def _emit_sanitized_partial(self, text: str) -> None:
        cleaned = self._sanitize_for_log(text)
        if not cleaned:
            return
        if bool(self._cfg("streaming_emit_partial_transcripts", False)):
            self.emit(f"stt_partial={cleaned}")
            return
        # BREAKING: stt_partial is redacted by default to avoid logging spoken PII into
        # low-trust device logs; opt in via config.streaming_emit_partial_transcripts=true.
        self.emit("stt_partial=[redacted]")
        digest = self._text_sha12(cleaned)
        if digest:
            self.emit(f"stt_partial_sha12={digest}")
        self.emit(f"stt_partial_len={len(cleaned)}")

    def _streaming_capture_budget_bytes(self) -> int:
        sample_rate = max(1, int(self._recorder_sample_rate()))
        channels = max(1, int(self.config.audio_channels))
        max_audio_ms = max(
            1_000,
            int(self._cfg("streaming_capture_max_audio_ms", self._DEFAULT_MAX_BUFFERED_AUDIO_MS)),
        )
        return int(
            (sample_rate * channels * self._STREAMING_BYTES_PER_SAMPLE * max_audio_ms) / 1000
        )

    def _streaming_finalize_timeout_ms(self) -> int:
        return max(
            250,
            int(self._cfg("streaming_finalize_timeout_ms", self._DEFAULT_FINALIZE_TIMEOUT_MS)),
        )

    def _streaming_keepalive_interval_s(self) -> float:
        return max(
            1.0,
            float(
                self._cfg(
                    "streaming_session_keepalive_interval_s",
                    self._DEFAULT_KEEPALIVE_INTERVAL_S,
                )
            ),
        )

    def _best_effort_streaming_transcript_hint(
        self,
        *,
        partial_text: str,
        controller: StreamingTurnController | None,
    ) -> str:
        return self.streaming_transcript_verifier_runtime.best_effort_streaming_transcript_hint(
            partial_text=partial_text,
            controller=controller,
        )

    def _maybe_recover_low_evidence_streaming_transcript(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        capture_result: SpeechCaptureResult | None,
        transcript: str,
        saw_interim: bool,
        capture_ms: int,
    ) -> str:
        return self.streaming_transcript_verifier_runtime.maybe_recover_low_evidence_streaming_transcript(
            stt_provider=stt_provider,
            capture_result=capture_result,
            transcript=transcript,
            saw_interim=saw_interim,
            capture_ms=capture_ms,
        )

    def _captured_audio_duration_ms(
        self,
        *,
        capture_result: SpeechCaptureResult | None,
    ) -> int:
        return self.streaming_transcript_verifier_runtime.captured_audio_duration_ms(
            capture_result=capture_result,
        )

    def _should_verify_streaming_transcript(
        self,
        *,
        transcript: str,
        capture_result: SpeechCaptureResult | None,
        capture_ms: int,
        saw_speech_final: bool,
        saw_utterance_end: bool,
        confidence: float | None,
    ) -> bool:
        return self.streaming_transcript_verifier_runtime.verification_gate(
            transcript=transcript,
            capture_result=capture_result,
            capture_ms=capture_ms,
            saw_speech_final=saw_speech_final,
            saw_utterance_end=saw_utterance_end,
            confidence=confidence,
        ).should_verify

    def _build_streaming_transcript_verifier_prompt(
        self,
        *,
        transcript_hint: str,
    ) -> str:
        return self.streaming_transcript_verifier_runtime.build_streaming_transcript_verifier_prompt(
            transcript_hint=transcript_hint,
        )

    def _maybe_verify_streaming_transcript(
        self,
        *,
        capture_result: SpeechCaptureResult | None,
        transcript: str,
        capture_ms: int,
        saw_speech_final: bool,
        saw_utterance_end: bool,
        confidence: float | None,
    ) -> str:
        return self.streaming_transcript_verifier_runtime.maybe_verify_streaming_transcript(
            capture_result=capture_result,
            transcript=transcript,
            capture_ms=capture_ms,
            saw_speech_final=saw_speech_final,
            saw_utterance_end=saw_utterance_end,
            confidence=confidence,
        )

    def _on_streaming_stt_interim(self, text: str) -> None:
        del text

    def _on_streaming_stt_endpoint(self, event: StreamingSpeechEndpointEvent) -> None:
        del event

    def _collect_streaming_keyterms(self) -> list[str]:
        keyterms: list[str] = []
        keyterm_sources = (
            self._cfg("streaming_keyterms", None),
            self._cfg("deepgram_streaming_keyterms", None),
        )
        for source in keyterm_sources:
            if not source:
                continue
            if isinstance(source, str):
                keyterms.append(source)
                continue
            try:
                for item in source:
                    if item:
                        keyterms.append(str(item))
            except TypeError:
                keyterms.append(str(source))
        supplier = getattr(self, "_streaming_keyterms_for_turn", None)
        if callable(supplier):
            try:
                supplier_callable = cast(Callable[[], Any], supplier)
                supplied = supplier_callable()  # pylint: disable=not-callable
            except Exception as exc:
                self.emit(f"stt_streaming_keyterms_failed={type(exc).__name__}")
                supplied = None
            if supplied:
                if isinstance(supplied, str):
                    keyterms.append(supplied)
                else:
                    try:
                        for item in supplied:
                            if item:
                                keyterms.append(str(item))
                    except TypeError:
                        keyterms.append(str(supplied))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in keyterms:
            cleaned = self._sanitize_for_log(item)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        limit = max(0, int(self._cfg("streaming_keyterms_limit", 32)))
        return deduped[:limit] if limit else deduped

    def _maybe_configure_streaming_session(
        self,
        *,
        session: Any,
        listening_window: AdaptiveListeningWindow,
    ) -> None:
        configure = getattr(session, "configure", None)
        if not callable(configure):
            return
        thresholds: dict[str, Any] = {}
        if bool(
            self._cfg(
                "streaming_auto_configure_eot_timeout",
                bool(self._cfg("deepgram_streaming_stop_on_utterance_end", False)),
            )
        ):
            eot_timeout_ms = max(
                int(getattr(listening_window, "speech_pause_ms", 0) or 0),
                int(getattr(listening_window, "pause_grace_ms", 0) or 0),
            )
            if eot_timeout_ms > 0:
                thresholds["eot_timeout_ms"] = min(max(eot_timeout_ms, 500), 10_000)
        threshold_attr_names = (
            ("flux_eot_threshold", "eot_threshold"),
            ("deepgram_flux_eot_threshold", "eot_threshold"),
            ("flux_eager_eot_threshold", "eager_eot_threshold"),
            ("deepgram_flux_eager_eot_threshold", "eager_eot_threshold"),
        )
        for attr_name, threshold_name in threshold_attr_names:
            raw_value = self._cfg(attr_name, None)
            if raw_value is None:
                continue
            thresholds[threshold_name] = float(raw_value)
        payload: dict[str, Any] = {}
        if thresholds:
            payload["thresholds"] = thresholds
        keyterms = self._collect_streaming_keyterms()
        if keyterms:
            payload["keyterms"] = keyterms
        if not payload:
            return
        try:
            try:
                configure(**payload)
            except TypeError:
                configure(payload)
            self._trace_event(
                "turn_controller_session_configured",
                kind="io",
                details={
                    "threshold_keys": sorted(thresholds.keys()),
                    "keyterms_count": len(keyterms),
                },
            )
        except Exception as exc:
            self.emit(f"stt_streaming_configure_failed={type(exc).__name__}")
            self._trace_event(
                "turn_controller_session_configure_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def _maybe_send_stream_keepalive(self, *, session: Any) -> None:
        for method_name in ("keepalive", "send_keepalive", "keep_alive"):
            method = getattr(session, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                return
            except Exception as exc:
                self.emit(f"stt_streaming_keepalive_failed={type(exc).__name__}")
                self._trace_event(
                    "turn_controller_keepalive_failed",
                    kind="exception",
                    level="WARN",
                    details={"error_type": type(exc).__name__, "method": method_name},
                )
                return

    def _request_stream_flush(self, *, session: Any) -> str | None:
        for method_name in (
            "close_stream",
            "request_close_stream",
            "send_close_stream",
            "flush",
            "finish_writing",
            "end_of_stream",
        ):
            method = getattr(session, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                self._trace_event(
                    "turn_controller_stream_flush_requested",
                    kind="io",
                    details={"method": method_name},
                )
                return method_name
            except TypeError:
                continue
            except Exception as exc:
                self.emit(f"stt_streaming_flush_failed={type(exc).__name__}")
                self._trace_event(
                    "turn_controller_stream_flush_failed",
                    kind="exception",
                    level="WARN",
                    details={"error_type": type(exc).__name__, "method": method_name},
                )
                return None
        return None

    def _finalize_streaming_session(self, *, session: Any):
        timeout_ms = self._streaming_finalize_timeout_ms()
        result_box: dict[str, Any] = {}
        done = threading.Event()

        def runner() -> None:
            try:
                result_box["result"] = session.finalize()
            except Exception as exc:
                result_box["exception"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=runner,
            name="twinr-streaming-finalize",
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        keepalive_at = time.monotonic() + self._streaming_keepalive_interval_s()
        while True:
            now = time.monotonic()
            remaining_s = deadline - now
            if remaining_s <= 0:
                break
            if done.wait(min(0.25, remaining_s)):
                if "exception" in result_box:
                    raise result_box["exception"]
                return result_box.get("result")
            now = time.monotonic()
            if now >= keepalive_at:
                self._maybe_send_stream_keepalive(session=session)
                keepalive_at = now + self._streaming_keepalive_interval_s()
        try:
            close_method = getattr(session, "close", None)
            if callable(close_method):
                close_method()
        except Exception as close_exc:
            self.emit(f"stt_streaming_force_close_failed={type(close_exc).__name__}")
            self._trace_event(
                "turn_controller_force_close_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(close_exc).__name__},
            )
        raise TimeoutError(f"streaming finalize exceeded {timeout_ms}ms")

    def _coerce_fallback_transcript(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        for attr_name in ("transcript", "text"):
            text = getattr(value, attr_name, None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        alternatives = getattr(value, "alternatives", None)
        if alternatives:
            for alternative in alternatives:
                text = self._coerce_fallback_transcript(alternative)
                if text:
                    return text
        if isinstance(value, dict):
            for key in ("transcript", "text"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            results = value.get("results")
            if isinstance(results, dict):
                channels = results.get("channels")
                if isinstance(channels, list):
                    for channel in channels:
                        if not isinstance(channel, dict):
                            continue
                        alternatives = channel.get("alternatives")
                        if isinstance(alternatives, list):
                            for alternative in alternatives:
                                text = self._coerce_fallback_transcript(alternative)
                                if text:
                                    return text
        if isinstance(value, list):
            for item in value:
                text = self._coerce_fallback_transcript(item)
                if text:
                    return text
        return ""

    def _call_fallback_method(self, method: Any, *, pcm_bytes: bytes) -> str:
        call_specs = (
            {
                "pcm_bytes": pcm_bytes,
                "sample_rate": self._recorder_sample_rate(),
                "channels": self.config.audio_channels,
                "language": self.config.deepgram_stt_language,
            },
            {
                "audio_bytes": pcm_bytes,
                "sample_rate": self._recorder_sample_rate(),
                "channels": self.config.audio_channels,
                "language": self.config.deepgram_stt_language,
            },
            {
                "pcm": pcm_bytes,
                "sample_rate": self._recorder_sample_rate(),
                "channels": self.config.audio_channels,
                "language": self.config.deepgram_stt_language,
            },
        )
        for kwargs in call_specs:
            try:
                transcript = self._coerce_fallback_transcript(method(**kwargs))
            except TypeError:
                continue
            if transcript:
                return transcript
        try:
            return self._coerce_fallback_transcript(method(pcm_bytes))
        except TypeError:
            return ""
        except Exception:
            raise
        return ""

    def _attempt_pcm_transcript_fallback(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        capture_result: SpeechCaptureResult | None,
    ) -> str:
        if capture_result is None or not getattr(capture_result, "pcm_bytes", b""):
            return ""
        fallback_targets: list[Any] = []
        for attr_name in (
            "local_streaming_stt_fallback_runtime",
            "streaming_stt_fallback_runtime",
            "on_device_stt_runtime",
        ):
            target = getattr(self, attr_name, None)
            if target is not None and target not in fallback_targets:
                fallback_targets.append(target)
        if stt_provider not in fallback_targets:
            fallback_targets.append(stt_provider)

        for target in fallback_targets:
            for method_name in ("transcribe_pcm", "transcribe_bytes", "transcribe_audio", "transcribe"):
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    transcript = self._call_fallback_method(
                        method,
                        pcm_bytes=capture_result.pcm_bytes,
                    )
                except Exception as exc:
                    self.emit(f"stt_pcm_fallback_failed={type(exc).__name__}")
                    self._trace_event(
                        "turn_controller_pcm_fallback_failed",
                        kind="exception",
                        level="WARN",
                        details={
                            "error_type": type(exc).__name__,
                            "method": method_name,
                            "target_type": type(target).__name__,
                        },
                    )
                    continue
                if transcript:
                    self._trace_event(
                        "turn_controller_pcm_fallback_succeeded",
                        kind="branch",
                        details={
                            "method": method_name,
                            "target_type": type(target).__name__,
                            "transcript_len": len(transcript),
                        },
                    )
                    return transcript
        return ""

    def _recover_after_streaming_failure(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        capture_result: SpeechCaptureResult | None,
        transcript_hint: str,
        saw_interim: bool,
        capture_ms: int,
    ) -> str:
        transcript = (transcript_hint or "").strip()
        transcript = self._maybe_recover_low_evidence_streaming_transcript(
            stt_provider=stt_provider,
            capture_result=capture_result,
            transcript=transcript,
            saw_interim=saw_interim,
            capture_ms=capture_ms,
        )
        if transcript:
            return transcript
        transcript = self._attempt_pcm_transcript_fallback(
            stt_provider=stt_provider,
            capture_result=capture_result,
        )
        return transcript.strip()

    def _capture_and_transcribe_with_turn_controller(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        listening_window: AdaptiveListeningWindow,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ) -> tuple[SpeechCaptureResult, str, int, int, str | None]:
        self._trace_event(
            "turn_controller_capture_started",
            kind="span_start",
            details={
                "speech_start_chunks": speech_start_chunks,
                "ignore_initial_ms": ignore_initial_ms,
                "pause_ms": listening_window.speech_pause_ms,
                "start_timeout_s": listening_window.start_timeout_s,
            },
        )
        partial_cache = [""]
        streamed_pcm = bytearray()
        controller = self._build_streaming_turn_controller()
        session = None
        capture_result: SpeechCaptureResult | None = None
        capture_ms = 0
        early_transcript_hint = ""
        shared_lock = threading.RLock()
        streaming_send_error: list[Exception] = []
        audio_budget_bytes = self._streaming_capture_budget_bytes()
        capture_truncated = [False]
        capture_truncation_emitted = [False]

        def current_turn_label() -> str | None:
            if controller is None:
                return None
            with shared_lock:
                decision = controller.last_decision()
            return decision.label if decision is not None else None

        def current_partial_text() -> str:
            with shared_lock:
                return partial_cache[0]

        def best_effort_transcript_hint() -> str:
            with shared_lock:
                return self._best_effort_streaming_transcript_hint(
                    partial_text=partial_cache[0],
                    controller=controller,
                )

        def send_chunk(chunk: bytes) -> None:
            if not chunk:
                return
            chunk_to_send = chunk
            overflowed = False
            with shared_lock:
                remaining = audio_budget_bytes - len(streamed_pcm)
                if remaining <= 0:
                    capture_truncated[0] = True
                    overflowed = True
                    chunk_to_send = b""
                elif len(chunk) > remaining:
                    capture_truncated[0] = True
                    overflowed = True
                    chunk_to_send = chunk[:remaining]
                    streamed_pcm.extend(chunk_to_send)
                else:
                    streamed_pcm.extend(chunk)
                send_blocked = session is None or bool(streaming_send_error)
            if overflowed and not capture_truncation_emitted[0]:
                capture_truncation_emitted[0] = True
                self.emit("turn_controller_capture_hard_limit_reached=true")
                self._trace_event(
                    "turn_controller_capture_hard_limit_reached",
                    kind="branch",
                    level="WARN",
                    details={
                        "buffered_pcm_bytes": len(streamed_pcm),
                        "budget_bytes": audio_budget_bytes,
                    },
                )
            if send_blocked or not chunk_to_send:
                return
            try:
                session.send_pcm(chunk_to_send)
            except Exception as exc:
                with shared_lock:
                    streaming_send_error.append(exc)
                self.emit(f"turn_controller_stream_send_failed={type(exc).__name__}")
                self._trace_event(
                    "turn_controller_stream_send_failed",
                    kind="exception",
                    level="WARN",
                    details={"error_type": type(exc).__name__, "chunk_size": len(chunk_to_send)},
                )

        def on_interim(text: str) -> None:
            transcript_text = self._clean_transcript_text(text)
            if not transcript_text:
                return
            with shared_lock:
                if transcript_text == partial_cache[0]:
                    return
                partial_cache[0] = transcript_text
                if controller is not None:
                    controller.on_interim(transcript_text)
            self._on_streaming_stt_interim(transcript_text)
            self._emit_sanitized_partial(transcript_text)
            self._trace_event(
                "turn_controller_interim_received",
                kind="io",
                details=self._trace_text_payload(transcript_text),
            )

        def on_endpoint(event: StreamingSpeechEndpointEvent) -> None:
            if controller is not None:
                with shared_lock:
                    controller.on_endpoint(event)
            self._on_streaming_stt_endpoint(event)
            endpoint_transcript = self._clean_transcript_text(
                str(getattr(event, "transcript", "") or "")
            )
            details = self._trace_text_payload(endpoint_transcript)
            details.update(
                {
                    "speech_final": bool(getattr(event, "speech_final", False)),
                    "utterance_end": bool(getattr(event, "utterance_end", False)),
                }
            )
            self._trace_event(
                "turn_controller_endpoint_received",
                kind="io",
                details=details,
            )

        try:
            session = stt_provider.start_streaming_session(
                sample_rate=self._recorder_sample_rate(),
                channels=self.config.audio_channels,
                language=self.config.deepgram_stt_language,
                on_interim=on_interim,
                on_endpoint=on_endpoint,
            )
            self._trace_event(
                "turn_controller_session_started",
                kind="io",
                details={
                    "sample_rate": self._recorder_sample_rate(),
                    "channels": self.config.audio_channels,
                },
            )
            self._maybe_configure_streaming_session(
                session=session,
                listening_window=listening_window,
            )
        except Exception:
            if controller is not None:
                try:
                    controller.close()
                except Exception as close_exc:
                    self.emit(f"turn_controller_close_failed={type(close_exc).__name__}")
            raise
        try:
            capture_started = time.monotonic()
            try:

                def should_stop_capture() -> bool:
                    with shared_lock:
                        controller_requested = bool(
                            controller is not None and controller.should_stop_capture()
                        )
                        budget_exhausted = capture_truncated[0]
                    return (
                        controller_requested
                        or budget_exhausted
                        or self._active_turn_stop_requested()
                    )

                capture_result = self.recorder.capture_pcm_until_pause_with_options(
                    pause_ms=listening_window.speech_pause_ms,
                    start_timeout_s=listening_window.start_timeout_s,
                    speech_start_chunks=speech_start_chunks,
                    ignore_initial_ms=ignore_initial_ms,
                    pause_grace_ms=listening_window.pause_grace_ms,
                    on_chunk=send_chunk,
                    should_stop=should_stop_capture,
                )
            except RuntimeError:
                transcript_hint = best_effort_transcript_hint()
                with shared_lock:
                    buffered_pcm = bytes(streamed_pcm)
                if not transcript_hint or not buffered_pcm:
                    raise
                capture_result = SpeechCaptureResult(
                    pcm_bytes=buffered_pcm,
                    speech_started_after_ms=0,
                    resumed_after_pause_count=0,
                )
                self.emit("turn_controller_capture_recovered=true")
                self._trace_event(
                    "turn_controller_capture_recovered",
                    kind="branch",
                    details={
                        "transcript_hint_len": len(transcript_hint),
                        "pcm_bytes": len(buffered_pcm),
                    },
                )
            capture_ms = int((time.monotonic() - capture_started) * 1000)
            if capture_result is None:
                raise RuntimeError("capture_pcm_until_pause_with_options returned no result")
            self._trace_event(
                "turn_controller_capture_completed",
                kind="span_end",
                details={"pcm_bytes": len(capture_result.pcm_bytes)},
                kpi={"duration_ms": capture_ms},
            )

            with shared_lock:
                send_error = streaming_send_error[0] if streaming_send_error else None
            if send_error is not None:
                transcript = self._recover_after_streaming_failure(
                    stt_provider=stt_provider,
                    capture_result=capture_result,
                    transcript_hint=best_effort_transcript_hint(),
                    saw_interim=bool(current_partial_text()),
                    capture_ms=capture_ms,
                )
                transcript = self._maybe_verify_streaming_transcript(
                    capture_result=capture_result,
                    transcript=transcript,
                    capture_ms=capture_ms,
                    saw_speech_final=False,
                    saw_utterance_end=False,
                    confidence=None,
                )
                self.emit("stt_streaming_recovered_from_transport=true")
                self._trace_event(
                    "turn_controller_transport_recovered",
                    kind="branch",
                    level="WARN",
                    details={
                        "error_type": type(send_error).__name__,
                        "transcript_len": len(transcript),
                    },
                )
                return capture_result, transcript, capture_ms, -1, current_turn_label()

            early_result = None
            try:
                early_result = self._early_streaming_transcription_result(
                    session=session,
                    controller=controller,
                    controller_lock=shared_lock,
                )
            except Exception as exc:
                self.emit(f"stt_streaming_early_fallback={type(exc).__name__}")
            if early_result is not None:
                if early_result.request_id:
                    self.emit(f"stt_request_id={early_result.request_id}")
                self.emit(f"stt_streaming_interim={str(early_result.saw_interim).lower()}")
                self.emit(
                    f"stt_streaming_speech_final={str(early_result.saw_speech_final).lower()}"
                )
                self.emit(
                    f"stt_streaming_utterance_end={str(early_result.saw_utterance_end).lower()}"
                )
                self.emit(
                    "stt_streaming_early=true"
                    if bool(getattr(early_result, "saw_speech_final", False))
                    else "stt_streaming_early_hint=true"
                )
                self._trace_event(
                    "turn_controller_early_snapshot_available",
                    kind="cache",
                    details={
                        "speech_final": bool(getattr(early_result, "saw_speech_final", False)),
                        "utterance_end": bool(
                            getattr(early_result, "saw_utterance_end", False)
                        ),
                        "interim": bool(getattr(early_result, "saw_interim", False)),
                        "transcript_len": len(
                            self._clean_transcript_text(
                                str(getattr(early_result, "transcript", "") or "")
                            )
                        ),
                    },
                )
                early_transcript_hint = self._maybe_recover_low_evidence_streaming_transcript(
                    stt_provider=stt_provider,
                    capture_result=capture_result,
                    transcript=self._clean_transcript_text(
                        str(getattr(early_result, "transcript", "") or "")
                    ),
                    saw_interim=bool(getattr(early_result, "saw_interim", False)),
                    capture_ms=capture_ms,
                )
                self.emit("stt_streaming_deferred_until_finalize=true")
                self._trace_event(
                    "turn_controller_early_snapshot_deferred",
                    kind="branch",
                    details={
                        "transcript_hint_len": len(early_transcript_hint),
                        "speech_final": bool(getattr(early_result, "saw_speech_final", False)),
                    },
                )

            self._request_stream_flush(session=session)
            stt_started = time.monotonic()
            try:
                result = self._finalize_streaming_session(session=session)
            except Exception as exc:
                transcript = self._recover_after_streaming_failure(
                    stt_provider=stt_provider,
                    capture_result=capture_result,
                    transcript_hint=early_transcript_hint or best_effort_transcript_hint(),
                    saw_interim=bool(current_partial_text()),
                    capture_ms=capture_ms,
                )
                transcript = self._maybe_verify_streaming_transcript(
                    capture_result=capture_result,
                    transcript=transcript,
                    capture_ms=capture_ms,
                    saw_speech_final=False,
                    saw_utterance_end=False,
                    confidence=None,
                )
                if capture_result is not None:
                    self.emit(f"turn_controller_finalize_recovered={type(exc).__name__}")
                    self._trace_event(
                        "turn_controller_finalize_recovered",
                        kind="branch",
                        level="WARN",
                        details={"error_type": type(exc).__name__, "hint_len": len(transcript)},
                    )
                    return capture_result, transcript, capture_ms, -1, current_turn_label()
                raise
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            self._trace_event(
                "turn_controller_finalize_completed",
                kind="llm_call",
                details={
                    "request_id": result.request_id,
                    "transcript_len": len(self._clean_transcript_text(result.transcript or "")),
                },
                kpi={"duration_ms": stt_ms},
            )
            if result.request_id:
                self.emit(f"stt_request_id={result.request_id}")
            self.emit(f"stt_streaming_interim={str(result.saw_interim).lower()}")
            self.emit(f"stt_streaming_speech_final={str(result.saw_speech_final).lower()}")
            self.emit(f"stt_streaming_utterance_end={str(result.saw_utterance_end).lower()}")
            transcript = self._clean_transcript_text(result.transcript or "")
            if not transcript:
                transcript = early_transcript_hint or best_effort_transcript_hint()
            transcript = self._maybe_recover_low_evidence_streaming_transcript(
                stt_provider=stt_provider,
                capture_result=capture_result,
                transcript=transcript,
                saw_interim=bool(getattr(result, "saw_interim", False)),
                capture_ms=capture_ms,
            )
            transcript = self._maybe_verify_streaming_transcript(
                capture_result=capture_result,
                transcript=transcript,
                capture_ms=capture_ms,
                saw_speech_final=bool(getattr(result, "saw_speech_final", False)),
                saw_utterance_end=bool(getattr(result, "saw_utterance_end", False)),
                confidence=getattr(result, "confidence", None),
            )
            self._trace_event(
                "turn_controller_transcript_ready",
                kind="observation",
                details={
                    "turn_label": current_turn_label(),
                    "transcript_chars": len(transcript),
                    "transcript_words": len(transcript.split()) if transcript else 0,
                    "transcript_sha12": self._text_sha12(transcript),
                },
                kpi={"capture_ms": capture_ms, "stt_ms": stt_ms},
            )
            return capture_result, transcript, capture_ms, stt_ms, current_turn_label()
        finally:
            if session is not None:
                try:
                    close_method = getattr(session, "close", None)
                    if callable(close_method):
                        close_method()
                except Exception as close_exc:
                    self.emit(f"stt_session_close_failed={type(close_exc).__name__}")
                    self._trace_event(
                        "turn_controller_session_close_failed",
                        kind="exception",
                        level="WARN",
                        details={"error_type": type(close_exc).__name__},
                    )
            if controller is not None:
                try:
                    with shared_lock:
                        controller.close()
                except Exception as close_exc:
                    self.emit(f"turn_controller_close_failed={type(close_exc).__name__}")
                    self._trace_event(
                        "turn_controller_close_failed",
                        kind="exception",
                        level="WARN",
                        details={"error_type": type(close_exc).__name__},
                    )
            self._trace_event("turn_controller_capture_finished", kind="span_end", details={})

    def _early_streaming_transcription_result(
        self,
        *,
        session,
        controller: StreamingTurnController | None,
        controller_lock: threading.RLock | None = None,
    ):
        if not self.config.streaming_early_transcript_enabled:
            return None
        snapshot_fn = getattr(session, "snapshot", None)
        if snapshot_fn is None:
            return None
        deadline = time.monotonic() + (
            max(0, int(self.config.streaming_early_transcript_wait_ms)) / 1000.0
        )
        keepalive_at = time.monotonic() + self._streaming_keepalive_interval_s()
        while True:
            snapshot = snapshot_fn()
            early_result = self._coerce_early_streaming_snapshot(
                snapshot=snapshot,
                controller=controller,
                controller_lock=controller_lock,
            )
            if early_result is not None:
                return early_result
            now = time.monotonic()
            if now >= deadline:
                return None
            if now >= keepalive_at:
                self._maybe_send_stream_keepalive(session=session)
                keepalive_at = now + self._streaming_keepalive_interval_s()
            self.sleep(self._EARLY_SNAPSHOT_POLL_S)

    def _coerce_early_streaming_snapshot(
        self,
        *,
        snapshot,
        controller: StreamingTurnController | None,
        controller_lock: threading.RLock | None = None,
    ):
        transcript = self._clean_transcript_text(str(getattr(snapshot, "transcript", "") or ""))
        min_chars = max(1, int(self.config.streaming_early_transcript_min_chars))
        if len(transcript) < min_chars:
            return None
        if getattr(snapshot, "saw_speech_final", False):
            if not bool(getattr(snapshot, "saw_interim", False)):
                return None
            coerced = SimpleNamespace(
                request_id=getattr(snapshot, "request_id", None),
                transcript=transcript,
                saw_interim=bool(getattr(snapshot, "saw_interim", False)),
                saw_speech_final=bool(getattr(snapshot, "saw_speech_final", False)),
                saw_utterance_end=bool(getattr(snapshot, "saw_utterance_end", False)),
                confidence=getattr(snapshot, "confidence", None),
            )
            return coerced
        if not (
            self.config.deepgram_streaming_stop_on_utterance_end
            and getattr(snapshot, "saw_utterance_end", False)
            and controller is not None
        ):
            return None
        if controller_lock is not None:
            with controller_lock:
                decision = controller.last_decision()
        else:
            decision = controller.last_decision()
        if decision is None or decision.decision != "end_turn":
            return None
        if decision.confidence < self.config.turn_controller_fast_endpoint_min_confidence:
            return None
        decision_transcript = str(getattr(decision, "transcript", "") or "")
        if normalize_turn_text(decision_transcript) != normalize_turn_text(transcript):
            return None
        return SimpleNamespace(
            request_id=getattr(snapshot, "request_id", None),
            transcript=transcript,
            saw_interim=bool(getattr(snapshot, "saw_interim", False)),
            saw_speech_final=bool(getattr(snapshot, "saw_speech_final", False)),
            saw_utterance_end=bool(getattr(snapshot, "saw_utterance_end", False)),
            confidence=getattr(snapshot, "confidence", None),
        )
