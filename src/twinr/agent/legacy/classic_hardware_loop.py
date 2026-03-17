"""Run the classic press-to-talk hardware workflow loop."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from threading import Lock, RLock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveListeningWindow
from twinr.agent.base_agent.contracts import (
    AgentTextProvider,
    CombinedSpeechAgentProvider,
    CompositeSpeechAgentProvider,
    SpeechToTextProvider,
    StreamingSpeechToTextProvider,
    TextToSpeechProvider,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.turn_controller import (
    StreamingTurnController,
    ToolCallingTurnDecisionEvaluator,
    _normalize_turn_text,
)
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.tools import (
    RealtimeToolExecutor,
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
    bind_realtime_tool_handlers,
    realtime_tool_names,
)
from twinr.hardware.audio import SilenceDetectedRecorder, SpeechCaptureResult, WaveAudioPlayer, pcm16_to_wav_bytes
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.ops.usage import TwinrUsageStore
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.working_feedback import WorkingFeedbackKind, start_working_feedback_loop
from twinr.proactive import (
    ProactiveGovernorCandidate,
    ProactiveGovernorReservation,
    SocialTriggerDecision,
    build_default_proactive_monitor,
    is_safety_trigger,
    proactive_observation_facts,
    proactive_prompt_mode,
)
from twinr.providers.openai import OpenAIBackend, OpenAIImageInput, OpenAIProviderBundle
from twinr.providers.factory import build_streaming_provider_bundle


_DEFAULT_TTS_WORKER_JOIN_TIMEOUT_S = 15.0
_DEFAULT_EMIT_LINE_MAX_CHARS = 4096
_DEFAULT_REFERENCE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_ALLOWED_REFERENCE_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _default_emit(line: str) -> None:
    """Print one workflow telemetry line to stdout."""
    print(line, flush=True)


class TwinrHardwareLoop:
    """Coordinate the non-realtime Twinr hardware loop.

    This loop owns button-driven turn capture, proactive prompting, print
    requests, and the classic request/response provider path.
    """

    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        backend: CombinedSpeechAgentProvider | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        agent_provider: AgentTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        tool_agent_provider: ToolCallingAgentProvider | None = None,
        turn_stt_provider: StreamingSpeechToTextProvider | None = None,
        turn_tool_agent_provider: ToolCallingAgentProvider | None = None,
        tool_turn_loop: ToolCallingStreamingLoop | None = None,
        button_monitor=None,
        recorder: SilenceDetectedRecorder | None = None,
        player: WaveAudioPlayer | None = None,
        printer: RawReceiptPrinter | None = None,
        camera: V4L2StillCamera | None = None,
        usage_store: TwinrUsageStore | None = None,
        voice_profile_monitor: VoiceProfileMonitor | None = None,
        proactive_monitor=None,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        error_reset_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime or TwinrRuntime(config=config)
        # AUDIT-FIX(#6): Wrap the injected emitter once so logging failures never crash the hardware loop.
        self._emit_lock = Lock()
        self._unsafe_emit = emit or _default_emit
        self.emit: Callable[[str], None] = self._safe_emit
        # AUDIT-FIX(#1): Serialize stateful hardware/runtime operations across button, proactive, reminder, and reload paths.
        self._operation_lock = RLock()
        self._owns_tool_turn_loop = tool_turn_loop is None

        openai_bundle: OpenAIProviderBundle | None = None
        if isinstance(backend, OpenAIBackend):
            openai_bundle = OpenAIProviderBundle.from_backend(backend)
        elif backend is None and (stt_provider is None or agent_provider is None or tts_provider is None):
            openai_bundle = OpenAIProviderBundle.from_config(config)

        self.stt_provider = stt_provider or backend or (openai_bundle.stt if openai_bundle is not None else None)
        self.agent_provider = agent_provider or backend or (openai_bundle.agent if openai_bundle is not None else None)
        self.tts_provider = tts_provider or backend or (openai_bundle.tts if openai_bundle is not None else None)
        if self.stt_provider is None or self.agent_provider is None or self.tts_provider is None:
            raise ValueError("TwinrHardwareLoop requires STT, agent, and TTS providers")
        self.backend = backend or (
            openai_bundle.combined
            if openai_bundle is not None
            else CompositeSpeechAgentProvider(
                stt=self.stt_provider,
                agent=self.agent_provider,
                tts=self.tts_provider,
            )
        )
        self.print_backend = self.backend
        self.tool_agent_provider = tool_agent_provider or (openai_bundle.tool_agent if openai_bundle is not None else None)
        provider_bundle = None
        if self.config.turn_controller_enabled and (
            turn_stt_provider is None or turn_tool_agent_provider is None
        ):
            try:
                provider_bundle = build_streaming_provider_bundle(config)
            except Exception as exc:
                # AUDIT-FIX(#8): Surface provider bundle construction failures instead of silently swallowing them.
                self.emit(f"turn_provider_bundle_error={self._format_exception(exc)}")
                provider_bundle = None
        self.turn_stt_provider = turn_stt_provider or (
            provider_bundle.stt
            if provider_bundle is not None and isinstance(provider_bundle.stt, StreamingSpeechToTextProvider)
            else None
        )
        self.turn_tool_agent_provider = turn_tool_agent_provider or self.tool_agent_provider or (
            provider_bundle.tool_agent if provider_bundle is not None else None
        )
        self.turn_decision_evaluator = (
            ToolCallingTurnDecisionEvaluator(
                config=config,
                provider=self.turn_tool_agent_provider,
            )
            if self.turn_tool_agent_provider is not None and self.config.turn_controller_enabled
            else None
        )
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder.from_config(config)
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self.voice_profile_monitor = voice_profile_monitor or VoiceProfileMonitor.from_config(config)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.playback_coordinator = PlaybackCoordinator(
            self.player,
            emit=self.emit,
            io_lock=self._audio_lock,
        )
        self._current_turn_audio_pcm: bytes | None = None
        self._current_turn_audio_sample_rate: int = self.config.audio_sample_rate
        self.sleep = sleep
        self.error_reset_seconds = self._coerce_non_negative_float(error_reset_seconds, 1.0, minimum=0.0)
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self._next_long_term_memory_proactive_check_at: float = 0.0
        self._working_feedback_stop: Callable[[], None] | None = None
        self._working_feedback_generation: int = 0
        self.tool_executor = RealtimeToolExecutor(self)
        if self.tool_agent_provider is None:
            self.tool_turn_loop = tool_turn_loop
        else:
            self.tool_turn_loop = tool_turn_loop or self._build_tool_turn_loop(config)
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
            emit=self.emit,
        )

    # AUDIT-FIX(#6): Keep log lines single-line and bounded to avoid log injection and runaway debug output.
    def _safe_emit(self, line: str) -> None:
        sanitized = self._sanitize_emit_line(line)
        try:
            self._unsafe_emit(sanitized)
        except Exception as exc:
            try:
                with self._emit_lock:
                    _default_emit(f"emit_error={self._sanitize_emit_line(self._format_exception(exc))}")
                    _default_emit(sanitized)
            except Exception:
                return

    def _sanitize_emit_line(self, line: object) -> str:
        text = str(line)
        text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\x00", "")
        max_chars = self._coerce_positive_int(
            getattr(self.config, "emit_line_max_chars", _DEFAULT_EMIT_LINE_MAX_CHARS),
            _DEFAULT_EMIT_LINE_MAX_CHARS,
            minimum=256,
        )
        if len(text) > max_chars:
            return f"{text[: max_chars - 3]}..."
        return text

    def _format_exception(self, exc: BaseException) -> str:
        return self._sanitize_emit_line(f"{type(exc).__name__}: {exc}")

    def _coerce_non_negative_float(
        self,
        raw_value: object,
        default: float,
        *,
        minimum: float = 0.0,
    ) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        if value < minimum:
            return minimum
        return value

    def _coerce_positive_int(
        self,
        raw_value: object,
        default: int,
        *,
        minimum: int = 1,
    ) -> int:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        if value < minimum:
            return minimum
        return value

    def _build_tool_turn_loop(self, config: TwinrConfig) -> ToolCallingStreamingLoop:
        if self.tool_agent_provider is None:
            raise ValueError("Tool turn loop requested without a tool agent provider")
        tool_schemas = (
            build_compact_agent_tool_schemas(realtime_tool_names())
            if (config.llm_provider or "").strip().lower() == "groq"
            else build_agent_tool_schemas(realtime_tool_names())
        )
        return ToolCallingStreamingLoop(
            provider=self.tool_agent_provider,
            tool_handlers=bind_realtime_tool_handlers(self.tool_executor),
            tool_schemas=tool_schemas,
            stream_final_only=((config.llm_provider or "").strip().lower() == "groq"),
        )

    def _with_try_operation_lock(self) -> bool:
        return self._operation_lock.acquire(blocking=False)

    def _release_try_operation_lock(self) -> None:
        self._operation_lock.release()

    def _recorder_sample_rate(self) -> int:
        return int(getattr(self.recorder, "sample_rate", self.config.audio_sample_rate))

    def _turn_controller_conversation(self) -> tuple[tuple[str, str], ...]:
        max_turns = max(0, int(self.config.turn_controller_context_turns))
        conversation = self.runtime.provider_conversation_context()
        if max_turns <= 0 or len(conversation) <= max_turns:
            return conversation
        return conversation[-max_turns:]

    def _build_streaming_turn_controller(self) -> StreamingTurnController | None:
        if self.turn_decision_evaluator is None or not self.config.turn_controller_enabled:
            return None
        return StreamingTurnController(
            config=self.config,
            evaluator=self.turn_decision_evaluator,
            conversation_factory=self._turn_controller_conversation,
            emit=self.emit,
        )

    def _should_recover_turn_controller_capture(
        self,
        *,
        exc: RuntimeError,
        controller: StreamingTurnController | None,
        transcript_hint: str,
        streamed_pcm: bytearray,
    ) -> bool:
        # AUDIT-FIX(#11): Only recover when streaming STT already heard real speech.
        # A local "no speech detected" timeout can still be recoverable if the turn controller
        # has already produced an explicit end-turn decision from streamed audio.
        if not transcript_hint or not streamed_pcm:
            return False
        if controller is not None:
            decision = controller.last_decision()
            if decision is not None and decision.decision == "end_turn":
                return True
        if self._is_no_speech_timeout(exc):
            return False
        lowered = str(exc).strip().lower()
        return any(
            marker in lowered
            for marker in (
                "stop capture",
                "capture stopped",
                "stop requested",
                "end turn",
                "endpoint",
            )
        )

    def _capture_and_transcribe_with_turn_controller(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        listening_window: AdaptiveListeningWindow,
    ) -> tuple[SpeechCaptureResult, str, int, int]:
        partial_cache = [""]
        streamed_pcm = bytearray()
        controller = self._build_streaming_turn_controller()

        def send_chunk(chunk: bytes) -> None:
            streamed_pcm.extend(chunk)
            session.send_pcm(chunk)

        def on_interim(text: str) -> None:
            cleaned = text.strip()
            if not cleaned or cleaned == partial_cache[0]:
                return
            partial_cache[0] = cleaned
            if controller is not None:
                controller.on_interim(cleaned)
            self.emit(f"stt_partial={cleaned}")

        try:
            session = stt_provider.start_streaming_session(
                sample_rate=self._recorder_sample_rate(),
                channels=self.config.audio_channels,
                language=self.config.deepgram_stt_language,
                on_interim=on_interim,
                on_endpoint=(controller.on_endpoint if controller is not None else None),
            )
        except Exception:
            if controller is not None:
                self._best_effort_close(controller, label="turn_controller")
            raise
        try:
            capture_started = time.monotonic()
            try:
                capture_result = self.recorder.capture_pcm_until_pause_with_options(
                    pause_ms=listening_window.speech_pause_ms,
                    start_timeout_s=listening_window.start_timeout_s,
                    pause_grace_ms=listening_window.pause_grace_ms,
                    on_chunk=send_chunk,
                    should_stop=(controller.should_stop_capture if controller is not None else None),
                )
            except RuntimeError as exc:
                transcript_hint = partial_cache[0].strip()
                if controller is not None:
                    controller_hint = controller.latest_transcript()
                    if controller_hint:
                        transcript_hint = controller_hint
                if not self._should_recover_turn_controller_capture(
                    exc=exc,
                    controller=controller,
                    transcript_hint=transcript_hint,
                    streamed_pcm=streamed_pcm,
                ):
                    raise
                capture_result = SpeechCaptureResult(
                    pcm_bytes=bytes(streamed_pcm),
                    speech_started_after_ms=0,
                    resumed_after_pause_count=0,
                )
                self.emit("turn_controller_capture_recovered=true")
            capture_ms = int((time.monotonic() - capture_started) * 1000)
            early_result = self._early_streaming_transcription_result(
                session=session,
                controller=controller,
            )
            if early_result is not None:
                if early_result.request_id:
                    self.emit(f"stt_request_id={early_result.request_id}")
                self.emit(f"stt_streaming_interim={str(early_result.saw_interim).lower()}")
                self.emit(f"stt_streaming_speech_final={str(early_result.saw_speech_final).lower()}")
                self.emit(f"stt_streaming_utterance_end={str(early_result.saw_utterance_end).lower()}")
                self.emit("stt_streaming_early=true")
                return capture_result, early_result.transcript.strip(), capture_ms, 0
            stt_started = time.monotonic()
            result = session.finalize()
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            if result.request_id:
                self.emit(f"stt_request_id={result.request_id}")
            self.emit(f"stt_streaming_interim={str(result.saw_interim).lower()}")
            self.emit(f"stt_streaming_speech_final={str(result.saw_speech_final).lower()}")
            self.emit(f"stt_streaming_utterance_end={str(result.saw_utterance_end).lower()}")
            transcript = result.transcript.strip()
            if not transcript and controller is not None:
                transcript = controller.latest_transcript()
            if not transcript:
                transcript = partial_cache[0].strip()
            return capture_result, transcript, capture_ms, stt_ms
        finally:
            # AUDIT-FIX(#8): Cleanup must not raise and mask the real recorder/STT error.
            self._best_effort_close(session, label="streaming_stt_session")
            if controller is not None:
                self._best_effort_close(controller, label="turn_controller")

    def _best_effort_close(self, closeable: object, *, label: str) -> None:
        close_fn = getattr(closeable, "close", None)
        if not callable(close_fn):
            return
        try:
            close_fn()
        except Exception as exc:
            self.emit(f"{label}_close_error={self._format_exception(exc)}")

    def _early_streaming_transcription_result(
        self,
        *,
        session,
        controller: StreamingTurnController | None,
    ):
        if not self.config.streaming_early_transcript_enabled:
            return None
        snapshot_fn = getattr(session, "snapshot", None)
        if snapshot_fn is None:
            return None
        deadline = time.monotonic() + (max(0, int(self.config.streaming_early_transcript_wait_ms)) / 1000.0)
        while True:
            try:
                snapshot = snapshot_fn()
            except Exception as exc:
                # AUDIT-FIX(#8): Snapshot failures should degrade to the normal finalize path instead of crashing the turn.
                self.emit(f"stt_snapshot_error={self._format_exception(exc)}")
                return None
            early_result = self._coerce_early_streaming_snapshot(
                snapshot=snapshot,
                controller=controller,
            )
            if early_result is not None:
                return early_result
            if time.monotonic() >= deadline:
                return None
            self.sleep(0.02)

    def _coerce_early_streaming_snapshot(
        self,
        *,
        snapshot,
        controller: StreamingTurnController | None,
    ):
        transcript = snapshot.transcript.strip()
        min_chars = max(1, int(self.config.streaming_early_transcript_min_chars))
        if len(transcript) < min_chars:
            return None
        if snapshot.saw_speech_final:
            if not snapshot.saw_interim:
                return None
            return snapshot
        if not (
            self.config.deepgram_streaming_stop_on_utterance_end
            and snapshot.saw_utterance_end
            and controller is not None
        ):
            return None
        decision = controller.last_decision()
        if decision is None or decision.decision != "end_turn":
            return None
        if decision.confidence < self.config.turn_controller_fast_endpoint_min_confidence:
            return None
        if _normalize_turn_text(decision.transcript) != _normalize_turn_text(transcript):
            return None
        return snapshot

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        poll_timeout = self._coerce_non_negative_float(poll_timeout, 0.25, minimum=0.01)
        started_at = time.monotonic()
        self._emit_status(force=True)
        with ExitStack() as stack:
            monitor = stack.enter_context(self.button_monitor)
            if self.proactive_monitor is not None:
                stack.enter_context(self.proactive_monitor)
            while True:
                if duration_s is not None and time.monotonic() - started_at >= duration_s:
                    return 0
                try:
                    # AUDIT-FIX(#5): Poll failures used to crash the whole loop while idle.
                    event = monitor.poll(timeout=poll_timeout)
                except Exception as exc:
                    self._handle_error(exc)
                    continue
                if event is None:
                    try:
                        # AUDIT-FIX(#5): Reminder/proactive paths must also fail closed instead of taking down the process.
                        if self._maybe_deliver_due_reminder():
                            continue
                        self._maybe_run_long_term_memory_proactive()
                    except Exception as exc:
                        self._handle_error(exc)
                    continue
                if event.action != ButtonAction.PRESSED:
                    continue
                self.emit(f"button={event.name}")
                self._record_event(
                    "button_pressed",
                    f"Physical button `{event.name}` was pressed.",
                    button=event.name,
                    line_offset=getattr(event, "line_offset", None),
                )
                self.handle_button_press(event.name)

    def handle_button_press(self, button_name: str) -> None:
        # AUDIT-FIX(#1): Button-triggered turns need exclusive access to runtime state and hardware.
        with self._operation_lock:
            try:
                if button_name == "green":
                    self._handle_green_turn()
                    return
                if button_name == "yellow":
                    self._handle_print_turn()
                    return
                raise ValueError(f"Unsupported button: {button_name}")
            except Exception as exc:
                self._handle_error(exc)

    def handle_social_trigger(self, trigger: SocialTriggerDecision) -> bool:
        # AUDIT-FIX(#1): Social prompts are opportunistic, so skip instead of waiting behind an active user operation.
        if not self._with_try_operation_lock():
            self.emit("social_trigger_skipped=busy_lock")
            self._record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because another operation was in progress.",
                trigger=trigger.trigger_id,
                reason="operation_in_progress",
                prompt=trigger.prompt,
                priority=int(trigger.priority),
            )
            return False
        try:
            if self.runtime.status.value != "waiting":
                self.emit("social_trigger_skipped=busy")
                self._record_event(
                    "social_trigger_skipped",
                    "Social trigger prompt was skipped because Twinr was not idle.",
                    trigger=trigger.trigger_id,
                    reason=trigger.reason,
                    prompt=trigger.prompt,
                    priority=int(trigger.priority),
                )
                return False

            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id=trigger.trigger_id,
                    summary=trigger.prompt,
                    priority=int(trigger.priority),
                    presence_session_id=self._current_presence_session_id(),
                    safety_exempt=is_safety_trigger(trigger.trigger_id),
                    counts_toward_presence_budget=not is_safety_trigger(trigger.trigger_id),
                )
            )
            if governor_reservation is None:
                return False

            phrase_response = None
            prompt_mode = proactive_prompt_mode(trigger)
            prompt_text = trigger.prompt
            try:
                if prompt_mode == "llm":
                    try:
                        phrase_response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                            trigger_id=trigger.trigger_id,
                            reason=trigger.reason,
                            default_prompt=trigger.prompt,
                            priority=int(trigger.priority),
                            conversation=self.runtime.provider_conversation_context(),
                            recent_prompts=self._recent_proactive_prompts(trigger_id=trigger.trigger_id),
                            observation_facts=proactive_observation_facts(trigger),
                        )
                        candidate_prompt = phrase_response.text.strip()
                        if candidate_prompt:
                            prompt_text = candidate_prompt
                        else:
                            prompt_mode = "default_fallback"
                            self.emit("social_prompt_fallback=empty_phrase")
                    except Exception as exc:
                        prompt_mode = "default_fallback"
                        self.emit("social_prompt_fallback=default")
                        self.emit(f"social_prompt_phrase_error={self._format_exception(exc)}")
                        self._record_event(
                            "social_trigger_phrase_fallback",
                            "Twinr fell back to the default proactive prompt after proactive phrasing failed.",
                            level="warning",
                            trigger=trigger.trigger_id,
                            error=self._format_exception(exc),
                        )

                prompt = self.runtime.begin_proactive_prompt(prompt_text)
                self._emit_status(force=True)
                turn_started = time.monotonic()
                tts_ms, first_audio_ms = self._speak_full_answer(prompt, turn_started=turn_started)
                self.runtime.finish_speaking()
                self._emit_status(force=True)
                self.runtime.proactive_governor.mark_delivered(governor_reservation)
                self.emit(f"social_trigger={trigger.trigger_id}")
                self.emit(f"social_trigger_priority={int(trigger.priority)}")
                self.emit(f"social_prompt_mode={prompt_mode}")
                self.emit(f"social_prompt={prompt}")
                if phrase_response is not None:
                    if phrase_response.response_id:
                        self.emit(f"social_response_id={phrase_response.response_id}")
                    if phrase_response.request_id:
                        self.emit(f"social_request_id={phrase_response.request_id}")
                    self._record_usage(
                        request_kind="proactive_prompt",
                        source="hardware_loop",
                        model=phrase_response.model,
                        response_id=phrase_response.response_id,
                        request_id=phrase_response.request_id,
                        used_web_search=False,
                        token_usage=phrase_response.token_usage,
                        proactive_trigger=trigger.trigger_id,
                    )
                self.emit(f"timing_social_tts_ms={tts_ms}")
                if first_audio_ms is not None:
                    self.emit(f"timing_social_first_audio_ms={first_audio_ms}")
                self._record_event(
                    "social_trigger_prompted",
                    "Twinr spoke a proactive social prompt.",
                    trigger=trigger.trigger_id,
                    reason=trigger.reason,
                    priority=int(trigger.priority),
                    prompt=prompt,
                    default_prompt=trigger.prompt,
                    prompt_mode=prompt_mode,
                )
                return True
            except Exception as exc:
                if self.runtime.status.value == "answering":
                    self.runtime.finish_speaking()
                    self._emit_status(force=True)
                self.runtime.proactive_governor.mark_skipped(
                    governor_reservation,
                    reason=f"delivery_failed: {exc}",
                )
                # AUDIT-FIX(#9): Never re-raise from the proactive callback path, or the monitor thread can die permanently.
                self.emit(f"social_trigger_error={self._format_exception(exc)}")
                self._record_event(
                    "social_trigger_failed",
                    "Social trigger delivery failed.",
                    level="error",
                    trigger=trigger.trigger_id,
                    error=self._format_exception(exc),
                )
                return False
        finally:
            self._release_try_operation_lock()

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        for entry in reversed(self.runtime.ops_events.tail(limit=50)):
            if entry.get("event") not in {"social_trigger_prompted", "longterm_proactive_prompted"}:
                continue
            data = entry.get("data", {})
            if data.get("trigger") != trigger_id:
                continue
            prompt = str(data.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= limit:
                break
        return tuple(prompts)

    def _current_presence_session_id(self) -> int | None:
        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        snapshot = None if coordinator is None else getattr(coordinator, "latest_presence_snapshot", None)
        if snapshot is None or not getattr(snapshot, "armed", False):
            return None
        return getattr(snapshot, "session_id", None)

    def _reserve_governed_prompt(
        self,
        candidate: ProactiveGovernorCandidate,
    ) -> ProactiveGovernorReservation | None:
        decision = self.runtime.proactive_governor.try_reserve(candidate)
        if decision.allowed:
            return decision.reservation
        emit_prefix = {
            "social": "social_trigger",
            "longterm": "longterm_proactive",
            "reminder": "reminder",
            "automation": "automation",
        }.get(candidate.source_kind, candidate.source_kind)
        self.emit(f"{emit_prefix}_skipped={decision.reason}")
        self._record_event(
            "proactive_governor_blocked",
            "Proactive delivery was blocked by the shared governor policy.",
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            summary=candidate.summary,
            reason=decision.reason,
            channel=candidate.channel,
            priority=int(candidate.priority),
            presence_session_id=candidate.presence_session_id,
        )
        return None

    def _handle_green_turn(self) -> None:
        turn_started = time.monotonic()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        listening_window = self.runtime.listening_window(initial_source="button", follow_up=False)
        capture_started = time.monotonic()
        try:
            with self._audio_lock:
                transcript = ""
                stt_ms = -1
                if isinstance(self.turn_stt_provider, StreamingSpeechToTextProvider):
                    try:
                        capture_result, transcript, capture_ms, stt_ms = (
                            self._capture_and_transcribe_with_turn_controller(
                                stt_provider=self.turn_stt_provider,
                                listening_window=listening_window,
                            )
                        )
                    except Exception as exc:
                        self.emit(f"turn_controller_fallback={type(exc).__name__}")
                        capture_result = self.recorder.capture_pcm_until_pause_with_options(
                            pause_ms=listening_window.speech_pause_ms,
                            start_timeout_s=listening_window.start_timeout_s,
                            pause_grace_ms=listening_window.pause_grace_ms,
                        )
                        capture_ms = int((time.monotonic() - capture_started) * 1000)
                else:
                    capture_result = self.recorder.capture_pcm_until_pause_with_options(
                        pause_ms=listening_window.speech_pause_ms,
                        start_timeout_s=listening_window.start_timeout_s,
                        pause_grace_ms=listening_window.pause_grace_ms,
                    )
                    capture_ms = int((time.monotonic() - capture_started) * 1000)
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.remember_listen_timeout(initial_source="button", follow_up=False)
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self.emit("listen_timeout=true")
            self._record_event(
                "listen_timeout",
                "Listening timed out before speech started.",
                request_source="button",
            )
            return
        self.runtime.remember_listen_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        recorder_sample_rate = self._recorder_sample_rate()
        audio_bytes = pcm16_to_wav_bytes(
            capture_result.pcm_bytes,
            sample_rate=recorder_sample_rate,
            channels=self.config.audio_channels,
        )
        self._update_voice_assessment_from_wav(audio_bytes)
        self._current_turn_audio_pcm = capture_result.pcm_bytes
        self._current_turn_audio_sample_rate = recorder_sample_rate

        try:
            if stt_ms < 0:
                stt_started = time.monotonic()
                try:
                    transcript = self.stt_provider.transcribe(
                        audio_bytes,
                        filename="twinr-listen.wav",
                        content_type="audio/wav",
                    ).strip()
                except Exception as exc:
                    self._record_event("stt_failed", "Speech-to-text failed.", level="error", error=self._format_exception(exc))
                    raise
                stt_ms = int((time.monotonic() - stt_started) * 1000)
            if not transcript:
                self._record_event("stt_failed", "Speech-to-text returned an empty transcript.", level="error")
                raise RuntimeError("Speech-to-text returned an empty transcript")

            self.emit(f"transcript={transcript}")
            self.runtime.submit_transcript(transcript)
            self._emit_status(force=True)
            stop_processing_feedback = self._start_working_feedback_loop("processing")

            llm_started = time.monotonic()
            spoken_segments: Queue[str | None] = Queue()
            tts_error: list[Exception] = []
            tts_error_lock = Lock()
            first_audio_at: list[float | None] = [None]
            first_audio_lock = Lock()
            answer_started = False
            pending_segment = ""
            segment_lock = Lock()
            worker_started = False

            def mark_tts_error(exc: Exception) -> None:
                with tts_error_lock:
                    tts_error.append(exc)

            def start_answering_locked() -> None:
                nonlocal answer_started
                if answer_started:
                    return
                stop_processing_feedback()
                self.runtime.begin_answering()
                self._emit_status(force=True)
                answer_started = True

            def tts_worker() -> None:
                while True:
                    segment = spoken_segments.get()
                    if segment is None:
                        return
                    try:
                        def mark_first_chunk():
                            for chunk in self.tts_provider.synthesize_stream(segment):
                                with first_audio_lock:
                                    if first_audio_at[0] is None:
                                        first_audio_at[0] = time.monotonic()
                                yield chunk

                        self.playback_coordinator.play_wav_chunks(
                            owner="classic_turn_tts_stream",
                            priority=PlaybackPriority.SPEECH,
                            chunks=mark_first_chunk(),
                        )
                    except Exception as exc:
                        mark_tts_error(exc)
                        return

            worker = Thread(target=tts_worker, daemon=True)
            worker.start()
            worker_started = True

            def queue_ready_segments(delta: str) -> None:
                nonlocal pending_segment
                # AUDIT-FIX(#2): Protect streaming callback state because providers may call on_text_delta from a worker thread.
                with segment_lock:
                    pending_segment += delta
                    while True:
                        boundary = self._segment_boundary(pending_segment)
                        if boundary is None:
                            return
                        segment = pending_segment[:boundary].strip()
                        pending_segment = pending_segment[boundary:].lstrip()
                        if not segment:
                            continue
                        start_answering_locked()
                        spoken_segments.put(segment)

            def flush_pending_segment() -> None:
                nonlocal pending_segment
                with segment_lock:
                    segment = pending_segment.strip()
                    pending_segment = ""
                    if not segment:
                        return
                    start_answering_locked()
                    spoken_segments.put(segment)

            try:
                if self.tool_turn_loop is not None:
                    tool_response = self.tool_turn_loop.run(
                        transcript,
                        conversation=self.runtime.provider_conversation_context(),
                        instructions=(
                            build_compact_tool_agent_instructions(
                                self.config,
                                extra_instructions=self.config.openai_realtime_instructions,
                            )
                            if (self.config.llm_provider or "").strip().lower() == "groq"
                            else build_tool_agent_instructions(
                                self.config,
                                extra_instructions=self.config.openai_realtime_instructions,
                            )
                        ),
                        allow_web_search=False,
                        on_text_delta=queue_ready_segments,
                    )
                    llm_ms = int((time.monotonic() - llm_started) * 1000)
                    if not tool_response.text.strip():
                        raise RuntimeError("Tool-calling turn completed without text output")
                    with segment_lock:
                        if self.runtime.status.value == "printing":
                            self.runtime.resume_answering_after_print()
                            self._emit_status(force=True)
                            answer_started = True
                    answer = self.runtime.finalize_agent_turn(tool_response.text)
                    flush_pending_segment()
                    response = tool_response
                else:
                    allow_web_search = self._resolve_web_search_mode()
                    direct_response = self.agent_provider.respond_streaming(
                        transcript,
                        conversation=self.runtime.provider_conversation_context(),
                        allow_web_search=allow_web_search,
                        on_text_delta=queue_ready_segments,
                    )
                    llm_ms = int((time.monotonic() - llm_started) * 1000)
                    if direct_response.used_web_search:
                        self.runtime.remember_search_result(
                            question=transcript,
                            answer=direct_response.text,
                        )
                    answer = self.runtime.finalize_agent_turn(direct_response.text)
                    flush_pending_segment()
                    response = direct_response
            finally:
                if worker_started:
                    spoken_segments.put(None)
                stop_processing_feedback()

            tts_started = time.monotonic()
            if worker_started:
                worker_timeout_s = self._coerce_non_negative_float(
                    getattr(self.config, "tts_worker_join_timeout_s", _DEFAULT_TTS_WORKER_JOIN_TIMEOUT_S),
                    _DEFAULT_TTS_WORKER_JOIN_TIMEOUT_S,
                    minimum=0.1,
                )
                # AUDIT-FIX(#3): Never block forever waiting for a stuck audio backend thread.
                worker.join(timeout=worker_timeout_s)
                if worker.is_alive():
                    self._best_effort_stop_player()
                    mark_tts_error(RuntimeError("Text-to-speech playback worker did not exit before timeout"))
                    self._record_event(
                        "tts_worker_timeout",
                        "The text-to-speech playback worker did not exit before timeout.",
                        level="error",
                        timeout_s=worker_timeout_s,
                    )
            tts_ms = int((time.monotonic() - tts_started) * 1000)
            with tts_error_lock:
                worker_error = tts_error[0] if tts_error else None
            if worker_error is not None:
                raise worker_error
            with segment_lock:
                if not answer_started:
                    stop_processing_feedback()
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
            self.emit(f"response={answer}")
            if response.response_id:
                self.emit(f"agent_response_id={response.response_id}")
            if response.request_id:
                self.emit(f"agent_request_id={response.request_id}")
            if self.tool_turn_loop is not None:
                self.emit(f"agent_tool_rounds={response.rounds}")
                self.emit(f"agent_tool_calls={len(response.tool_calls)}")
                self.emit(f"agent_used_web_search={str(response.used_web_search).lower()}")
            else:
                self.emit(f"openai_allow_web_search={str(allow_web_search).lower()}")
                self.emit(f"openai_used_web_search={str(response.used_web_search).lower()}")
            self._record_usage(
                request_kind="conversation",
                source="hardware_loop",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=response.used_web_search,
                token_usage=response.token_usage,
                transcript=transcript,
                tool_rounds=getattr(response, "rounds", None),
                tool_calls=len(getattr(response, "tool_calls", ())),
            )
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            self.emit(f"timing_capture_ms={capture_ms}")
            self.emit(f"timing_stt_ms={stt_ms}")
            self.emit(f"timing_llm_ms={llm_ms}")
            self.emit(f"timing_tts_ms={tts_ms}")
            self.emit("timing_playback_ms=streamed")
            with first_audio_lock:
                first_audio_value = first_audio_at[0]
            if first_audio_value is not None:
                self.emit(f"timing_first_audio_ms={int((first_audio_value - turn_started) * 1000)}")
            self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        finally:
            self._current_turn_audio_pcm = None

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        try:
            composed = self.agent_provider.compose_print_job_with_metadata(
                conversation=self.runtime.provider_conversation_context(),
                focus_hint=self.runtime.last_transcript,
                direct_text=response_to_print,
                request_source="button",
            )
            print_job = self.printer.print_text(composed.text)
        except Exception as exc:
            self._record_event(
                "print_failed",
                "Print composition or delivery failed.",
                level="error",
                error=self._format_exception(exc),
            )
            raise
        self.emit(f"print_text={composed.text}")
        if composed.response_id:
            self.emit(f"print_response_id={composed.response_id}")
        self._record_usage(
            request_kind="print",
            source="hardware_loop",
            model=composed.model,
            response_id=composed.response_id,
            request_id=composed.request_id,
            used_web_search=False,
            token_usage=composed.token_usage,
            request_source="button",
        )
        if print_job:
            self.emit(f"print_job={print_job}")
        self._record_event(
            "print_job_sent",
            "Print job was sent to the configured printer.",
            queue=self.config.printer_queue,
            job=print_job,
        )
        # AUDIT-FIX(#7): Long-term memory evidence is observability, not part of the primary print success path.
        self._enqueue_multimodal_evidence(
            event_name="print_completed",
            modality="printer",
            source="hardware_print",
            message="Printed Twinr output was delivered from the hardware loop.",
            data={
                "request_source": "button",
                "queue": self.config.printer_queue,
                "job": print_job or "",
            },
        )

        self.runtime.finish_printing()
        self._emit_status(force=True)
        self._last_print_request_at = time.monotonic()

    def _best_effort_stop_player(self) -> None:
        stop_fn = getattr(self.player, "stop", None)
        if not callable(stop_fn):
            return
        try:
            stop_fn()
        except Exception as exc:
            self.emit(f"player_stop_error={self._format_exception(exc)}")

    def _handle_error(self, exc: Exception) -> None:
        safe_error = self._format_exception(exc)
        # AUDIT-FIX(#6): The recovery path itself must be defensive and must stop any lingering auxiliary audio work.
        with self._operation_lock:
            self._stop_working_feedback()
            self._best_effort_stop_player()
            self._record_event(
                "hardware_loop_error",
                "Hardware loop operation failed.",
                level="error",
                error=safe_error,
            )
            try:
                self.runtime.fail(safe_error)
            except Exception as fail_exc:
                self.emit(f"runtime_fail_error={self._format_exception(fail_exc)}")
            self._emit_status(force=True)
            self.emit(f"error={safe_error}")
            if self.error_reset_seconds > 0:
                try:
                    self.sleep(self.error_reset_seconds)
                except Exception as sleep_exc:
                    self.emit(f"error_sleep_failed={self._format_exception(sleep_exc)}")
            try:
                self.runtime.reset_error()
            except Exception as reset_exc:
                self.emit(f"runtime_reset_error={self._format_exception(reset_exc)}")
            self._emit_status(force=True)

    def _emit_status(self, *, force: bool = False) -> None:
        try:
            status = self.runtime.status.value
        except Exception as exc:
            self.emit(f"status_read_error={self._format_exception(exc)}")
            return
        if force or status != self._last_status:
            self.emit(f"status={status}")
            self._last_status = status

    def _apply_config_if_supported(self, component: object, updated_config: TwinrConfig, *, seen: set[int]) -> None:
        if component is None:
            return
        component_id = id(component)
        if component_id in seen:
            return
        seen.add(component_id)
        if not hasattr(component, "config"):
            return
        try:
            setattr(component, "config", updated_config)
        except Exception as exc:
            self.emit(f"live_config_apply_error={self._format_exception(exc)}")

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        # AUDIT-FIX(#10): Live config updates must be serialized and must not assume every dependency exposes a writable .config attribute.
        with self._operation_lock:
            self.runtime.apply_live_config(updated_config)
            self.config = updated_config
            self._current_turn_audio_sample_rate = updated_config.audio_sample_rate
            seen: set[int] = set()
            for component in (
                self.stt_provider,
                self.agent_provider,
                self.tts_provider,
                self.print_backend,
                self.tool_agent_provider,
                self.turn_stt_provider,
                self.turn_tool_agent_provider,
                self.recorder,
                self.player,
                self.printer,
                self.camera,
                self.button_monitor,
                self.usage_store,
                self.voice_profile_monitor,
                self.proactive_monitor,
            ):
                self._apply_config_if_supported(component, updated_config, seen=seen)
            if self._owns_tool_turn_loop and self.tool_agent_provider is not None:
                self.tool_turn_loop = self._build_tool_turn_loop(updated_config)
            self.turn_decision_evaluator = (
                ToolCallingTurnDecisionEvaluator(
                    config=updated_config,
                    provider=self.turn_tool_agent_provider,
                )
                if self.turn_tool_agent_provider is not None and updated_config.turn_controller_enabled
                else None
            )

    def _start_working_feedback_loop(self, kind: WorkingFeedbackKind) -> Callable[[], None]:
        previous_stop = self._working_feedback_stop
        if callable(previous_stop):
            try:
                previous_stop()
            except Exception as exc:
                # AUDIT-FIX(#6): Working-feedback helper failures must not derail the primary voice turn.
                self.emit(f"working_feedback_stop_error={self._format_exception(exc)}")
        generation = self._working_feedback_generation + 1
        try:
            stop = start_working_feedback_loop(
                self.player,
                kind=kind,
                sample_rate=self.config.audio_sample_rate,
                emit=self.emit,
                delay_override_ms=(
                    self.config.processing_feedback_delay_ms
                    if kind == "processing"
                    else None
                ),
                playback_coordinator=self.playback_coordinator,
            )
        except Exception as exc:
            self.emit(f"working_feedback_start_error={self._format_exception(exc)}")
            self._working_feedback_generation = generation
            self._working_feedback_stop = None
            return lambda: None
        self._working_feedback_generation = generation
        self._working_feedback_stop = stop

        def stop_current() -> None:
            if self._working_feedback_generation != generation:
                return
            active_stop = self._working_feedback_stop
            self._working_feedback_stop = None
            if callable(active_stop):
                try:
                    active_stop()
                except Exception as exc:
                    self.emit(f"working_feedback_stop_error={self._format_exception(exc)}")

        return stop_current

    def _stop_working_feedback(self) -> None:
        active_stop = self._working_feedback_stop
        self._working_feedback_stop = None
        if callable(active_stop):
            try:
                active_stop()
            except Exception as exc:
                self.emit(f"working_feedback_stop_error={self._format_exception(exc)}")

    def _update_voice_assessment_from_wav(self, audio_bytes: bytes) -> None:
        try:
            assessment = self.voice_profile_monitor.assess_wav_bytes(audio_bytes)
        except Exception as exc:
            self.emit(f"voice_profile_error={self._format_exception(exc)}")
            return
        if not assessment.should_persist:
            return
        try:
            # AUDIT-FIX(#7): Voice-profile persistence is secondary and must not abort the user turn on storage failure.
            self.runtime.update_user_voice_assessment(
                status=assessment.status,
                confidence=assessment.confidence,
                checked_at=assessment.checked_at,
            )
        except Exception as exc:
            self.emit(f"voice_profile_persist_error={self._format_exception(exc)}")
            return
        self.emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self.emit(f"voice_profile_confidence={assessment.confidence:.2f}")

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        try:
            # AUDIT-FIX(#7): Observability writes must never take down the customer-facing flow when the file-backed store is flaky.
            self.runtime.ops_events.append(event=event, message=message, level=level, data=data)
        except Exception as exc:
            self.emit(f"ops_event_error={self._format_exception(exc)}")

    def _record_usage(
        self,
        *,
        request_kind: str,
        source: str,
        model: str | None,
        response_id: str | None,
        request_id: str | None,
        used_web_search: bool | None,
        token_usage,
        **metadata: object,
    ) -> None:
        try:
            self.usage_store.append(
                source=source,
                request_kind=request_kind,
                model=model,
                response_id=response_id,
                request_id=request_id,
                used_web_search=used_web_search,
                token_usage=token_usage,
                metadata=metadata,
            )
        except Exception as exc:
            self.emit(f"usage_store_error={self._format_exception(exc)}")

    def _enqueue_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str,
        message: str,
        data: dict[str, object],
    ) -> None:
        try:
            self.runtime.long_term_memory.enqueue_multimodal_evidence(
                event_name=event_name,
                modality=modality,
                source=source,
                message=message,
                data=data,
            )
        except Exception as exc:
            self.emit(f"multimodal_evidence_error={self._format_exception(exc)}")

    def _maybe_deliver_due_reminder(self) -> bool:
        if not self._with_try_operation_lock():
            return False
        try:
            now_monotonic = time.monotonic()
            reminder_interval_s = self._coerce_non_negative_float(
                getattr(self.config, "reminder_poll_interval_s", 0.0),
                0.0,
                minimum=0.1,
            )
            if now_monotonic < self._next_reminder_check_at:
                return False
            # AUDIT-FIX(#12): Clamp poll intervals so bad env values cannot create a busy loop or crash numeric comparisons.
            self._next_reminder_check_at = now_monotonic + reminder_interval_s
            if self.runtime.status.value != "waiting":
                return False
            preview_entries = self.runtime.peek_due_reminders(limit=1)
            if not preview_entries:
                return False
            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="reminder",
                    source_id=preview_entries[0].reminder_id,
                    summary=preview_entries[0].summary,
                    priority=80,
                    presence_session_id=self._current_presence_session_id(),
                    safety_exempt=False,
                    counts_toward_presence_budget=False,
                )
            )
            if governor_reservation is None:
                return False
            due_entries = self.runtime.reserve_due_reminders(limit=1)
            if not due_entries:
                self.runtime.proactive_governor.cancel(governor_reservation)
                return False
            if due_entries[0].reminder_id != preview_entries[0].reminder_id:
                self.runtime.proactive_governor.cancel(governor_reservation)
                return False
            return self._deliver_due_reminder(due_entries[0], governor_reservation=governor_reservation)
        finally:
            self._release_try_operation_lock()

    def _maybe_run_long_term_memory_proactive(self) -> bool:
        if not self._with_try_operation_lock():
            return False
        try:
            now_monotonic = time.monotonic()
            proactive_interval_s = self._coerce_non_negative_float(
                getattr(self.config, "long_term_memory_proactive_poll_interval_s", 0.0),
                0.0,
                minimum=0.1,
            )
            if now_monotonic < self._next_long_term_memory_proactive_check_at:
                return False
            self._next_long_term_memory_proactive_check_at = now_monotonic + proactive_interval_s
            if self.runtime.status.value != "waiting":
                return False
            preview = self.runtime.preview_long_term_proactive_candidate()
            if preview is None:
                return False
            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="longterm",
                    source_id=preview.candidate_id,
                    summary=preview.summary,
                    priority=max(1, min(99, int(preview.confidence * 100))),
                    presence_session_id=self._current_presence_session_id(),
                    safety_exempt=False,
                    counts_toward_presence_budget=True,
                )
            )
            if governor_reservation is None:
                return False
            reservation = self.runtime.reserve_specific_long_term_proactive_candidate(preview)
            if reservation is None or reservation.candidate.candidate_id != preview.candidate_id:
                self.runtime.proactive_governor.cancel(governor_reservation)
                return False
            candidate = reservation.candidate
            response = None
            prompt_mode = "default"
            prompt_text = candidate.summary
            trigger_id = f"longterm:{candidate.candidate_id}"
            try:
                try:
                    response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                        trigger_id=trigger_id,
                        reason=candidate.rationale,
                        default_prompt=candidate.summary,
                        priority=max(1, min(99, int(candidate.confidence * 100))),
                        conversation=self.runtime.provider_conversation_context(),
                        recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                        observation_facts=(
                            f"candidate_kind={candidate.kind}",
                            f"sensitivity={candidate.sensitivity}",
                        ),
                    )
                    candidate_prompt = response.text.strip()
                    if candidate_prompt:
                        prompt_text = candidate_prompt
                        prompt_mode = "llm"
                    else:
                        prompt_mode = "default_fallback"
                except Exception as exc:
                    prompt_mode = "default_fallback"
                    self.emit("longterm_proactive_phrase_fallback=default")
                    self.emit(f"longterm_proactive_phrase_error={self._format_exception(exc)}")
                    self._record_event(
                        "longterm_proactive_phrase_fallback",
                        "Twinr fell back to the default long-term proactive prompt after phrasing failed.",
                        level="warning",
                        candidate_id=candidate.candidate_id,
                        error=self._format_exception(exc),
                    )

                prompt = self.runtime.begin_proactive_prompt(prompt_text)
                self._emit_status(force=True)
                tts_ms, first_audio_ms = self._speak_full_answer(prompt, turn_started=time.monotonic())
                self.runtime.finish_speaking()
                self._emit_status(force=True)
                self.runtime.mark_long_term_proactive_candidate_delivered(
                    reservation,
                    prompt_text=prompt,
                )
                self.runtime.proactive_governor.mark_delivered(governor_reservation)
                self.emit(f"longterm_proactive_candidate={candidate.candidate_id}")
                self.emit(f"longterm_proactive_kind={candidate.kind}")
                self.emit(f"longterm_proactive_prompt_mode={prompt_mode}")
                self.emit(f"longterm_proactive_prompt={prompt}")
                self.emit(f"timing_longterm_proactive_tts_ms={tts_ms}")
                if first_audio_ms is not None:
                    self.emit(f"timing_longterm_proactive_first_audio_ms={first_audio_ms}")
                if response is not None:
                    self._record_usage(
                        request_kind="longterm_proactive_prompt",
                        source="hardware_loop",
                        model=response.model,
                        response_id=response.response_id,
                        request_id=response.request_id,
                        used_web_search=False,
                        token_usage=response.token_usage,
                        proactive_trigger=trigger_id,
                    )
                self._record_event(
                    "longterm_proactive_prompted",
                    "Twinr spoke a proactive prompt derived from long-term memory.",
                    trigger=trigger_id,
                    candidate_id=candidate.candidate_id,
                    candidate_kind=candidate.kind,
                    prompt=prompt,
                    prompt_mode=prompt_mode,
                    rationale=candidate.rationale,
                )
                return True
            except Exception as exc:
                if self.runtime.status.value == "answering":
                    self.runtime.finish_speaking()
                    self._emit_status(force=True)
                self.runtime.mark_long_term_proactive_candidate_skipped(
                    reservation,
                    reason=f"delivery_failed: {exc}",
                )
                self.runtime.proactive_governor.mark_skipped(
                    governor_reservation,
                    reason=f"delivery_failed: {exc}",
                )
                self.emit(f"longterm_proactive_error={self._format_exception(exc)}")
                return False
        finally:
            self._release_try_operation_lock()

    def _deliver_due_reminder(
        self,
        reminder,
        *,
        governor_reservation: ProactiveGovernorReservation,
    ) -> bool:
        response = None
        spoken_prompt = ""
        try:
            response = self.agent_provider.phrase_due_reminder_with_metadata(reminder)
            spoken_prompt = self.runtime.begin_reminder_prompt(response.text)
            self._emit_status(force=True)
            tts_ms, first_audio_ms = self._speak_full_answer(spoken_prompt, turn_started=time.monotonic())
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            delivered = self.runtime.mark_reminder_delivered(reminder.reminder_id)
            self.runtime.proactive_governor.mark_delivered(governor_reservation)
            self.emit("reminder_delivered=true")
            self.emit(f"reminder_due_at={delivered.due_at.isoformat()}")
            self.emit(f"reminder_text={spoken_prompt}")
            if response.response_id:
                self.emit(f"reminder_response_id={response.response_id}")
            if response.request_id:
                self.emit(f"reminder_request_id={response.request_id}")
            self.emit(f"timing_reminder_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self.emit(f"timing_reminder_first_audio_ms={first_audio_ms}")
            self._record_usage(
                request_kind="reminder_delivery",
                source="hardware_loop",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=False,
                token_usage=response.token_usage,
                reminder_id=delivered.reminder_id,
                reminder_kind=delivered.kind,
            )
            return True
        except Exception as exc:
            if self.runtime.status.value == "answering":
                self.runtime.finish_speaking()
                self._emit_status(force=True)
            self.runtime.mark_reminder_failed(reminder.reminder_id, error=str(exc))
            self.runtime.proactive_governor.mark_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            self.emit(f"reminder_error={self._format_exception(exc)}")
            return False

    def _resolve_web_search_mode(self) -> bool | None:
        # AUDIT-FIX(#12): Config may be unset or None during live reload; coerce safely before string ops.
        mode = str(getattr(self.config, "conversation_web_search", "") or "").strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        return None

    def _build_vision_images(self) -> list[OpenAIImageInput]:
        with self._camera_lock:
            capture = self.camera.capture_photo(filename="camera-capture.png")
        self.emit(f"camera_device={capture.source_device}")
        self.emit(f"camera_input_format={capture.input_format or 'default'}")
        self.emit(f"camera_capture_bytes={len(capture.data)}")
        # AUDIT-FIX(#7): Camera evidence enqueue must not fail the primary vision request.
        self._enqueue_multimodal_evidence(
            event_name="camera_capture",
            modality="camera",
            source="camera_tool",
            message="Live camera frame captured for device interaction.",
            data={
                "purpose": "vision_inspection",
                "source_device": capture.source_device,
                "input_format": capture.input_format or "default",
            },
        )

        images = [
            OpenAIImageInput(
                data=capture.data,
                content_type=capture.content_type,
                filename=capture.filename,
                label="Image 1: live camera frame from the device.",
            )
        ]
        reference_image = self._load_reference_image()
        if reference_image is not None:
            images.append(reference_image)
        return images

    def _contains_symlink_component(self, path: Path) -> bool:
        if path.is_absolute():
            current = Path(path.anchor)
            parts = path.parts[1:]
        else:
            current = Path.cwd()
            parts = path.parts
        for part in parts:
            current = current / part
            try:
                if current.is_symlink():
                    return True
            except OSError:
                return True
        return False

    def _resolve_reference_image_path(self, raw_path: str) -> Path | None:
        try:
            candidate = Path(raw_path).expanduser()
            # AUDIT-FIX(#4): Reject symlinks and non-regular files so a config-path swap cannot exfiltrate arbitrary local files.
            if self._contains_symlink_component(candidate):
                self.emit("vision_reference_rejected=symlink")
                return None
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file():
                self.emit("vision_reference_rejected=not_file")
                return None
            if resolved.suffix.lower() not in _ALLOWED_REFERENCE_IMAGE_SUFFIXES:
                self.emit(f"vision_reference_rejected=bad_suffix:{resolved.suffix.lower()}")
                return None
            max_bytes = self._coerce_positive_int(
                getattr(self.config, "vision_reference_image_max_bytes", _DEFAULT_REFERENCE_IMAGE_MAX_BYTES),
                _DEFAULT_REFERENCE_IMAGE_MAX_BYTES,
                minimum=1024,
            )
            if resolved.stat().st_size > max_bytes:
                self.emit("vision_reference_rejected=too_large")
                return None
            return resolved
        except FileNotFoundError:
            self.emit("vision_reference_missing=true")
            return None
        except OSError as exc:
            self.emit(f"vision_reference_error={self._format_exception(exc)}")
            return None

    def _load_reference_image(self) -> OpenAIImageInput | None:
        raw_path = (self.config.vision_reference_image_path or "").strip()
        if not raw_path:
            return None
        path = self._resolve_reference_image_path(raw_path)
        if path is None:
            return None
        self.emit(f"vision_reference_image={path.name}")
        return OpenAIImageInput.from_path(
            path,
            label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
        )

    def _build_vision_prompt(self, transcript: str, *, include_reference: bool) -> str:
        if include_reference:
            return (
                "This request includes camera input. "
                "Image 1 is the current live camera frame from the device. "
                "Image 2 is a stored reference image of the main user. "
                "Use the reference image only when the user's question depends on whether the live image shows that user. "
                "If identity is uncertain, say that clearly. "
                "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
                f"User request: {transcript.strip()}"
            )
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Answer from what is actually visible. "
            "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
            f"User request: {transcript.strip()}"
        )

    def _speak_full_answer(self, text: str, *, turn_started: float) -> tuple[int, int | None]:
        first_audio_at: list[float | None] = [None]
        playback_error: list[Exception] = []

        def mark_first_chunk():
            for chunk in self.tts_provider.synthesize_stream(text):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        def playback_worker() -> None:
            try:
                self.playback_coordinator.play_wav_chunks(
                    owner="classic_answer_tts",
                    priority=PlaybackPriority.SPEECH,
                    chunks=mark_first_chunk(),
                )
            except Exception as exc:
                playback_error.append(exc)

        tts_started = time.monotonic()
        worker = Thread(target=playback_worker, daemon=True)
        worker.start()
        worker_timeout_s = self._coerce_non_negative_float(
            getattr(self.config, "tts_worker_join_timeout_s", _DEFAULT_TTS_WORKER_JOIN_TIMEOUT_S),
            _DEFAULT_TTS_WORKER_JOIN_TIMEOUT_S,
            minimum=0.1,
        )
        # AUDIT-FIX(#3): Apply the same playback timeout guard to proactive/reminder speech as to streamed conversation speech.
        worker.join(timeout=worker_timeout_s)
        if worker.is_alive():
            self._best_effort_stop_player()
            raise RuntimeError("Synchronous text-to-speech playback did not exit before timeout")
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if playback_error:
            raise playback_error[0]
        if first_audio_at[0] is None:
            return tts_ms, None
        return tts_ms, int((first_audio_at[0] - turn_started) * 1000)

    def _segment_boundary(self, text: str) -> int | None:
        for index, character in enumerate(text):
            if character not in ".?!":
                continue
            previous_character = text[index - 1] if index > 0 else ""
            next_character = text[index + 1] if index + 1 < len(text) else ""
            # AUDIT-FIX(#13): Avoid splitting mid-number/mid-token, which made TTS choppy for decimals and abbreviations.
            if character == "." and previous_character.isdigit() and next_character.isdigit():
                continue
            if next_character and next_character not in {' ', '\n', '"', "'", ')', ']', '}'}:
                continue
            return index + 1
        if len(text) >= 140:
            return len(text)
        return None

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        # AUDIT-FIX(#11): Match timeout cases more defensively so provider wording changes do not break listen-timeout handling.
        lowered = str(exc).strip().lower()
        return (
            isinstance(exc, TimeoutError)
            or ("no speech" in lowered and "timeout" in lowered)
            or "timed out waiting for speech" in lowered
            or "before speech started" in lowered
        )

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        cooldown_s = self._coerce_non_negative_float(
            getattr(self.config, "print_button_cooldown_s", 0.0),
            0.0,
            minimum=0.0,
        )
        return (time.monotonic() - self._last_print_request_at) < cooldown_s
