"""Run the realtime hardware workflow for live Twinr conversations."""

from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha1
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveListeningWindow
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentTextProvider,
    CombinedSpeechAgentProvider,
    CompositeSpeechAgentProvider,
    SpeechToTextProvider,
    StreamingSpeechEndpointEvent,
    StreamingSpeechToTextProvider,
    TextToSpeechProvider,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.conversation.turn_controller import (
    StreamingTurnController,
    ToolCallingTurnDecisionEvaluator,
    _normalize_turn_text,
)
from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureEvaluation,
    ConversationClosureDecision,
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.tools import RealtimeToolExecutor, bind_realtime_tool_handlers
from twinr.hardware.audio import (
    AmbientAudioSampler,
    SilenceDetectedRecorder,
    SpeechCaptureResult,
    WaveAudioPlayer,
    pcm16_to_wav_bytes,
)
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.ops.usage import TwinrUsageStore
from twinr.proactive import SocialTriggerDecision, WakewordMatch, build_default_proactive_monitor
from twinr.providers.openai import OpenAIProviderBundle
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai.realtime import OpenAIRealtimeSession
from twinr.agent.workflows.realtime_runtime.background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.required_remote_watch import RequiredRemoteDependencyWatch
from twinr.agent.workflows.button_dispatch import ButtonPressDispatcher
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.listen_timeout_diagnostics import (
    diagnostics_from_exception,
    emit_listen_timeout_diagnostics,
)
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.print_lane import PrintLaneRequest, TwinrPrintLane
from twinr.agent.workflows.realtime_runtime.support import TwinrRealtimeSupportMixin, _default_emit
from twinr.agent.workflows.realtime_runner_tools import TwinrRealtimeToolDelegatesMixin
from twinr.agent.workflows.streaming_transcript_verifier import StreamingTranscriptVerifierRuntime
from twinr.agent.workflows.turn_guidance import TurnGuidanceRuntime


class TwinrRealtimeHardwareLoop(
    TwinrRealtimeBackgroundMixin,
    TwinrRealtimeToolDelegatesMixin,
    TwinrRealtimeSupportMixin,
):
    """Coordinate realtime sessions, wakeword entry, and background delivery.

    This loop owns the live conversation session lifecycle, print-lane handoff,
    and the wakeword/button entry points for the realtime runtime path.
    """

    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        realtime_session: OpenAIRealtimeSession | None = None,
        print_backend: CombinedSpeechAgentProvider | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        agent_provider: AgentTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        turn_stt_provider: StreamingSpeechToTextProvider | None = None,
        turn_tool_agent_provider: ToolCallingAgentProvider | None = None,
        verification_stt_provider: SpeechToTextProvider | None = None,
        conversation_closure_evaluator: ToolCallingConversationClosureEvaluator | None = None,
        button_monitor=None,
        recorder: SilenceDetectedRecorder | None = None,
        player: WaveAudioPlayer | None = None,
        printer: RawReceiptPrinter | None = None,
        camera: V4L2StillCamera | None = None,
        usage_store: TwinrUsageStore | None = None,
        voice_profile_monitor: VoiceProfileMonitor | None = None,
        ambient_audio_sampler: AmbientAudioSampler | None = None,
        proactive_monitor=None,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        error_reset_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime or TwinrRuntime(config=config)
        openai_bundle: OpenAIProviderBundle | None = None
        if print_backend is None and (stt_provider is None or agent_provider is None or tts_provider is None):
            openai_bundle = OpenAIProviderBundle.from_config(config)
        self.stt_provider = stt_provider or print_backend or (openai_bundle.stt if openai_bundle is not None else None)
        self.agent_provider = agent_provider or print_backend or (openai_bundle.agent if openai_bundle is not None else None)
        self.tts_provider = tts_provider or print_backend or (openai_bundle.tts if openai_bundle is not None else None)
        if self.stt_provider is None or self.agent_provider is None or self.tts_provider is None:
            raise ValueError("TwinrRealtimeHardwareLoop requires STT, agent, and TTS providers")
        self.transcript_verifier_provider = verification_stt_provider
        self.print_backend = print_backend or (
            openai_bundle.combined
            if openai_bundle is not None
            else CompositeSpeechAgentProvider(
                stt=self.stt_provider,
                agent=self.agent_provider,
                tts=self.tts_provider,
            )
        )
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder(
            device=config.audio_input_device,
            sample_rate=config.openai_realtime_input_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=config.audio_speech_start_chunks,
            start_timeout_s=config.audio_start_timeout_s,
            max_record_seconds=config.audio_max_record_seconds,
        )
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self.voice_profile_monitor = voice_profile_monitor or VoiceProfileMonitor.from_config(config)
        self._ambient_audio_sampler = ambient_audio_sampler
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.playback_coordinator = PlaybackCoordinator(
            self.player,
            emit=emit or _default_emit,
            io_lock=self._audio_lock,
        )
        self._active_turn_stop_lock = Lock()
        self._active_turn_stop_event: Event | None = None
        self._conversation_session_lock = Lock()  # AUDIT-FIX(#1): Serialize session entry across button, wakeword, and proactive threads.
        self._current_turn_audio_pcm: bytes | None = None
        self._current_turn_audio_sample_rate: int = self.config.openai_realtime_input_sample_rate
        self.tool_executor = RealtimeToolExecutor(self)
        provider_bundle = None
        if (self.config.turn_controller_enabled or self.config.conversation_closure_guard_enabled) and (
            turn_stt_provider is None or turn_tool_agent_provider is None
        ):
            try:
                provider_bundle = build_streaming_provider_bundle(config)
            except Exception:
                provider_bundle = None
        self.turn_stt_provider = turn_stt_provider or (
            provider_bundle.stt
            if provider_bundle is not None and isinstance(provider_bundle.stt, StreamingSpeechToTextProvider)
            else None
        )
        self.turn_tool_agent_provider = turn_tool_agent_provider or (
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
        self.conversation_closure_evaluator = (
            conversation_closure_evaluator
            or (
                ToolCallingConversationClosureEvaluator(
                    config=config,
                    provider=self.turn_tool_agent_provider,
                )
                if self.turn_tool_agent_provider is not None and self.config.conversation_closure_guard_enabled
                else None
            )
        )
        self.realtime_session = realtime_session or OpenAIRealtimeSession(
            config=config,
            tool_handlers=bind_realtime_tool_handlers(self.tool_executor),
        )
        self.emit = emit or _default_emit
        self.turn_guidance_runtime = TurnGuidanceRuntime(self)
        self.streaming_transcript_verifier_runtime = StreamingTranscriptVerifierRuntime(self)
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self._next_automation_check_at: float = 0.0
        self._next_long_term_memory_proactive_check_at: float = 0.0
        self._working_feedback_stop: Callable[[], None] | None = None
        self._working_feedback_generation: int = 0
        self._conversation_session_active = False
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue()
        self._latest_sensor_observation_facts: dict[str, object] | None = None
        self._wakeword_ack_cache_lock = Lock()
        self._wakeword_ack_wav_bytes: bytes | None = None
        self._wakeword_ack_prefetch_started = False
        self._wakeword_ack_prefetch_thread: Thread | None = None
        self.print_lane = TwinrPrintLane(
            backend=self.print_backend,
            printer=self.printer,
            emit=self.emit,
            record_event=self._record_event,
            record_usage=self._record_usage,
            start_feedback_loop=lambda kind: self._start_working_feedback_loop(kind),
            format_exception=self._safe_error_text,
            on_print_submitted=self._mark_print_submitted,
            enqueue_multimodal_evidence=self.runtime.long_term_memory.enqueue_multimodal_evidence,
        )
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.print_backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
            wakeword_handler=self.handle_wakeword_match,
            idle_predicate=self._background_work_allowed,
            observation_handler=self.handle_sensor_observation,
            emit=self.emit,
        )
        self.workflow_forensics = WorkflowForensics.from_env(
            project_root=Path(self.config.project_root),
            service=self.__class__.__name__,
        )
        self._workflow_active_trace_id: str | None = None
        self._trace_event(
            "workflow_loop_initialized",
            kind="run_start",
            details={
                "class": self.__class__.__name__,
                "project_root": self.config.project_root,
                "stt_provider": type(self.stt_provider).__name__,
                "agent_provider": type(self.agent_provider).__name__,
                "tts_provider": type(self.tts_provider).__name__,
            },
        )

    def _record_event(self, *args, **kwargs):
        try:
            return super()._record_event(*args, **kwargs)
        except Exception as exc:
            emit = getattr(self, "emit", None)
            if callable(emit):
                try:
                    emit(f"record_event_failed={type(exc).__name__}")  # AUDIT-FIX(#5): Telemetry persistence is secondary and must never abort user-facing flows.
                except Exception:
                    self._trace_event(
                        "record_event_emit_failed",
                        kind="error",
                        level="ERROR",
                        details={"error_type": type(exc).__name__},
                    )
            return None

    def _record_usage(self, *args, **kwargs):
        try:
            return super()._record_usage(*args, **kwargs)
        except Exception as exc:
            emit = getattr(self, "emit", None)
            if callable(emit):
                try:
                    emit(f"record_usage_failed={type(exc).__name__}")  # AUDIT-FIX(#5): Usage accounting failures must degrade quietly instead of surfacing as interaction errors.
                except Exception:
                    self._trace_event(
                        "record_usage_emit_failed",
                        kind="error",
                        level="ERROR",
                        details={"error_type": type(exc).__name__},
                    )
            return None

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        started_at = time.monotonic()
        self._trace_event(
            "workflow_run_loop_entered",
            kind="run_start",
            details={"duration_s": duration_s, "poll_timeout": poll_timeout},
        )
        self._required_remote_dependency_error_active = bool(
            self._remote_dependency_is_required()
            and getattr(getattr(self.runtime, "status", None), "value", None) == "error"
        )
        self._required_remote_dependency_cached_ready = self._required_remote_dependency_current_ready()
        if self._remote_dependency_is_required() and self._required_remote_dependency_uses_watchdog_artifact():
            self._required_remote_dependency_cached_ready = False
        self._required_remote_dependency_watch = RequiredRemoteDependencyWatch(
            interval_s=self._required_remote_dependency_interval_seconds(),
            refresh=lambda force: self._refresh_required_remote_dependency(
                force=force,
                force_sync=False,
            ),
            emit=self.emit,
            trace_event=lambda msg, details=None: self._trace_event(msg, details=details),
        )
        self._required_remote_dependency_watch.start()
        self._trace_event(
            "required_remote_watch_started",
            kind="queue",
            details={"interval_s": self._required_remote_dependency_interval_seconds()},
        )
        self._emit_status(force=True)
        self._ensure_wakeword_ack_prefetch_started()
        try:
            safe_poll_timeout = max(0.0, float(poll_timeout))
        except (TypeError, ValueError):
            safe_poll_timeout = 0.25  # AUDIT-FIX(#4): Sanitize invalid poll timeouts instead of crashing before the loop starts.
        with ExitStack() as stack:
            monitor = stack.enter_context(self.button_monitor)
            if self.proactive_monitor is not None:
                stack.enter_context(self.proactive_monitor)
            button_dispatcher = ButtonPressDispatcher(
                handle_press=self.handle_button_press,
                interrupt_current=self._request_active_turn_interrupt,
                emit=self.emit,
                trace_event=lambda msg, details=None: self._trace_event(msg, details=details),
            )
            housekeeping_stop, housekeeping_thread = self._start_idle_housekeeping_worker(
                poll_timeout=safe_poll_timeout
            )
            try:
                while True:
                    try:  # AUDIT-FIX(#4): Keep the hardware loop alive when button polling or button handling raises.
                        if duration_s is not None and time.monotonic() - started_at >= duration_s:
                            self._trace_event(
                                "workflow_run_duration_elapsed",
                                kind="run_end",
                                details={"duration_s": duration_s},
                            )
                            return 0
                        if not self._required_remote_dependency_current_ready():
                            self._trace_event(
                                "workflow_poll_skipped_remote_not_ready",
                                kind="branch",
                                details={"poll_timeout": safe_poll_timeout},
                            )
                            self._request_required_remote_dependency_refresh()
                            self.sleep(min(0.25, safe_poll_timeout or 0.25))
                            continue
                        event = monitor.poll(timeout=safe_poll_timeout)
                        if event is None:
                            continue
                        if event.action != ButtonAction.PRESSED:
                            self._trace_event(
                                "button_event_ignored_non_press",
                                kind="branch",
                                details={"name": event.name, "action": str(event.action)},
                            )
                            continue
                        self.emit(f"button={event.name}")
                        self._trace_event(
                            "button_press_received",
                            kind="io",
                            details={"name": event.name, "line_offset": event.line_offset},
                        )
                        self._record_event(
                            "button_pressed",
                            f"Physical button `{event.name}` was pressed.",
                            button=event.name,
                            line_offset=event.line_offset,
                        )
                        button_dispatcher.submit(event.name)
                    except Exception as exc:
                        self._handle_error(exc)
            finally:
                housekeeping_stop.set()
                housekeeping_thread.join(timeout=max(0.5, safe_poll_timeout + 0.25))
                button_dispatcher.close(timeout_s=max(0.5, safe_poll_timeout + 0.25))
                self._required_remote_dependency_watch.stop(
                    timeout_s=max(0.5, safe_poll_timeout + 0.25)
                )
                self._trace_event("workflow_run_loop_exiting", kind="run_end", details={})
                self.workflow_forensics.close()

    def _run_idle_housekeeping_cycle(self) -> bool:
        if not self._required_remote_dependency_current_ready():
            return False
        if self.print_lane.is_busy():
            return False
        if self._maybe_deliver_due_reminder():
            return True
        if self._maybe_run_due_automation():
            return True
        if self._maybe_run_sensor_automation():
            return True
        if self._maybe_run_long_term_memory_proactive():
            return True
        return False

    def _start_idle_housekeeping_worker(self, *, poll_timeout: float) -> tuple[Event, Thread]:
        stop_event = Event()
        idle_sleep_s = max(0.02, min(0.25, poll_timeout or 0.25))

        def worker() -> None:
            while not stop_event.is_set():
                try:
                    did_work = self._run_idle_housekeeping_cycle()
                except Exception as exc:  # AUDIT-FIX(#10): Contain housekeeping-thread faults so hardware button monitoring stays alive.
                    self._handle_error(exc)
                    did_work = False
                if did_work:
                    continue
                stop_event.wait(idle_sleep_s)

        thread = Thread(target=worker, daemon=True, name="twinr-realtime-housekeeping")
        thread.start()
        return stop_event, thread

    def wait_for_print_lane_idle(self, timeout_s: float = 1.0) -> bool:
        return self.print_lane.wait_for_idle(timeout_s=timeout_s)

    def _mark_print_submitted(self) -> None:
        self._last_print_request_at = time.monotonic()

    def handle_button_press(self, button_name: str) -> None:
        try:
            self._trace_event(
                "handle_button_press_entered",
                kind="io",
                details={"button_name": button_name},
            )
            if not self._required_remote_dependency_current_ready():
                self._trace_event(
                    "button_press_blocked_remote_not_ready",
                    kind="invariant",
                    level="WARN",
                    details={"button_name": button_name},
                )
                self._request_required_remote_dependency_refresh()
                return
            if button_name == "green":
                self._trace_decision(
                    "button_press_routed",
                    question="Which button workflow path should run?",
                    selected={"id": "green", "summary": "Start or interrupt conversation"},
                    options=[
                        {"id": "green", "summary": "Conversation turn"},
                        {"id": "yellow", "summary": "Print lane"},
                    ],
                    context={"button_name": button_name},
                    guardrails=["remote_ready_required"],
                )
                self._handle_green_turn()
                return
            if button_name == "yellow":
                self._trace_decision(
                    "button_press_routed",
                    question="Which button workflow path should run?",
                    selected={"id": "yellow", "summary": "Print lane"},
                    options=[
                        {"id": "green", "summary": "Conversation turn"},
                        {"id": "yellow", "summary": "Print lane"},
                    ],
                    context={"button_name": button_name},
                    guardrails=["remote_ready_required"],
                )
                self._handle_print_turn()
                return
            raise ValueError(f"Unsupported button: {button_name}")
        except Exception as exc:
            self._handle_error(exc)

    def handle_wakeword_match(self, match: WakewordMatch) -> bool:
        try:  # AUDIT-FIX(#4): Prevent background wakeword handlers from dying on transient runtime or provider failures.
            if not self._required_remote_dependency_current_ready():
                self._request_required_remote_dependency_refresh()
                return False
            if not self._background_work_allowed():
                skip_reason = "busy" if self.runtime.status.value != "waiting" else "conversation_active"
                self.emit(f"wakeword_skipped={skip_reason}")
                self._record_event(
                    "wakeword_skipped",
                    "Wakeword was detected but Twinr was not idle enough to open a turn.",
                    skip_reason=skip_reason,
                    matched_phrase=match.matched_phrase,
                )
                return False
            self.emit(f"wakeword_phrase={match.matched_phrase or 'unknown'}")
            seed_transcript = (match.remaining_text or "").strip()  # AUDIT-FIX(#7): Ignore whitespace-only wakeword spillover instead of opening an empty text turn.
            if seed_transcript:
                self.emit("wakeword_mode=direct_text")
            else:
                self.emit("wakeword_mode=listen")
            self._record_event(
                "wakeword_detected",
                "Wakeword matched while Twinr was attentive.",
                matched_phrase=match.matched_phrase,
                transcript_preview=match.transcript[:160],
                remaining_text=seed_transcript or None,
            )
            play_initial_beep = True
            if not seed_transcript:
                self._acknowledge_wakeword()
                play_initial_beep = False
            return self._run_conversation_session(
                initial_source="wakeword",
                seed_transcript=seed_transcript or None,
                play_initial_beep=play_initial_beep,
            )
        except Exception as exc:
            self._handle_error(exc)
            return False

    def _handle_green_turn(self) -> None:
        self._run_conversation_session(initial_source="button")

    def _run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> bool:
        try:  # AUDIT-FIX(#4): Keep proactive background monitoring alive when hands-free follow-up fails.
            self.emit("proactive_listen=true")
            self._record_event(
                "proactive_listen_started",
                "Twinr opened a hands-free listening window after a proactive prompt.",
                trigger=trigger.trigger_id,
                timeout_s=self.config.conversation_follow_up_timeout_s,
            )
            return self._run_conversation_session(
                initial_source="proactive",
                proactive_trigger=trigger.trigger_id,
            )
        except Exception as exc:
            self._handle_error(exc)
            return False

    def _follow_up_allowed_for_source(self, *, initial_source: str) -> bool:
        if not self.config.conversation_follow_up_enabled:
            return False
        if initial_source == "proactive":
            return bool(self.config.conversation_follow_up_after_proactive_enabled)
        return True

    def _follow_up_vetoed_by_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        evaluation = self._evaluate_follow_up_closure(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )
        return self._apply_follow_up_closure_evaluation(
            evaluation=evaluation,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )

    def _evaluate_follow_up_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> ConversationClosureEvaluation:
        evaluator = self.conversation_closure_evaluator
        if evaluator is None or not self.config.conversation_closure_guard_enabled:
            return ConversationClosureEvaluation()
        if not self._follow_up_allowed_for_source(initial_source=request_source):
            return ConversationClosureEvaluation()
        try:
            decision = evaluator.evaluate(
                user_transcript=user_transcript,
                assistant_response=assistant_response,
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                conversation=self.runtime.conversation_context(),
            )
        except Exception as exc:
            return ConversationClosureEvaluation(error_type=type(exc).__name__)
        return ConversationClosureEvaluation(decision=decision)

    def _apply_follow_up_closure_evaluation(
        self,
        *,
        evaluation: ConversationClosureEvaluation,
        request_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        if evaluation.error_type:
            self.emit(f"conversation_closure_fallback={evaluation.error_type}")
            return False
        decision = evaluation.decision
        if decision is None:
            return False
        self._emit_closure_decision(decision)
        if not decision.close_now:
            return False
        min_confidence = max(0.0, min(1.0, float(self.config.conversation_closure_min_confidence)))
        if decision.confidence < min_confidence:
            self.emit("conversation_closure_below_threshold=true")
            return False
        self._record_event(
            "conversation_closure_detected",
            "Twinr suppressed automatic follow-up listening because the exchange clearly ended for now.",
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            confidence=decision.confidence,
            reason=decision.reason,
        )
        return True

    def _emit_closure_decision(self, decision: ConversationClosureDecision) -> None:
        self.emit(f"conversation_closure_close_now={str(decision.close_now).lower()}")
        self.emit(f"conversation_closure_confidence={decision.confidence:.3f}")
        self.emit(f"conversation_closure_reason={decision.reason}")

    def _run_conversation_session(
        self,
        *,
        initial_source: str,
        proactive_trigger: str | None = None,
        seed_transcript: str | None = None,
        play_initial_beep: bool = True,
    ) -> bool:
        if not self._required_remote_dependency_current_ready():
            self._trace_event(
                "conversation_session_blocked_remote_not_ready",
                kind="invariant",
                level="WARN",
                details={"initial_source": initial_source, "proactive_trigger": proactive_trigger},
            )
            self._request_required_remote_dependency_refresh()
            return False
        if not self._conversation_session_lock.acquire(blocking=False):
            self._trace_event(
                "conversation_session_lock_busy",
                kind="queue",
                details={"initial_source": initial_source, "proactive_trigger": proactive_trigger},
            )
            self.emit("conversation_session_skipped=busy")
            self._record_event(
                "conversation_session_skipped",
                "A new conversation trigger arrived while another session was still active.",
                initial_source=initial_source,
                proactive_trigger=proactive_trigger,
            )
            return False
        trace_id = self._new_workflow_trace_id()
        self._workflow_trace_set_active(trace_id)
        stop_event = Event()
        self._set_active_turn_stop_event(stop_event)
        self._conversation_session_active = True  # AUDIT-FIX(#1): Mark active only while this thread owns the session lock.
        try:
            self._trace_event(
                "conversation_session_started",
                kind="span_start",
                details={
                    "initial_source": initial_source,
                    "proactive_trigger": proactive_trigger,
                    "seed_present": bool(seed_transcript and seed_transcript.strip()),
                },
                trace_id=trace_id,
            )
            follow_up = False
            normalized_seed = seed_transcript.strip() if seed_transcript is not None else None  # AUDIT-FIX(#7): Normalize transcript seeds once before branching.
            if normalized_seed:
                self._trace_event(
                    "conversation_seed_transcript_branch",
                    kind="branch",
                    details={"length": len(normalized_seed), "initial_source": initial_source},
                    trace_id=trace_id,
                )
                if self._run_single_text_turn(
                    transcript=normalized_seed,
                    listen_source=initial_source,
                    proactive_trigger=proactive_trigger,
                ):
                    if self._follow_up_allowed_for_source(initial_source=initial_source):
                        follow_up = True
                    else:
                        return True
                else:
                    return True
            while True:
                listening_window = self.runtime.listening_window(
                    initial_source=initial_source,
                    follow_up=follow_up,
                )
                self._trace_event(
                    "conversation_listening_window_selected",
                    kind="decision",
                    details={
                        "initial_source": initial_source,
                        "follow_up": follow_up,
                        "pause_ms": listening_window.speech_pause_ms,
                        "start_timeout_s": listening_window.start_timeout_s,
                        "pause_grace_ms": listening_window.pause_grace_ms,
                    },
                    trace_id=trace_id,
                )
                if self._run_single_audio_turn(
                    initial_source=initial_source,
                    follow_up=follow_up,
                    listening_window=listening_window,
                    listen_source="follow_up" if follow_up else initial_source,
                    proactive_trigger=None if follow_up else proactive_trigger,
                    speech_start_chunks=self._listening_window_speech_start_chunks(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    ignore_initial_ms=self._listening_window_ignore_initial_ms(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    timeout_emit_key=self._listening_timeout_emit_key(initial_source=initial_source, follow_up=follow_up),
                    timeout_message=self._listening_timeout_message(initial_source=initial_source, follow_up=follow_up),
                    play_initial_beep=True if follow_up else play_initial_beep,
                ):
                    if self._follow_up_allowed_for_source(initial_source=initial_source):
                        if self._active_turn_stop_requested():
                            self._cancel_interrupted_turn()
                            return False
                        self._trace_event(
                            "conversation_follow_up_continues",
                            kind="branch",
                            details={"initial_source": initial_source},
                            trace_id=trace_id,
                        )
                        follow_up = True
                        continue
                return True
        finally:
            self._conversation_session_active = False
            self._clear_active_turn_stop_event(stop_event)
            self._conversation_session_lock.release()
            self._trace_event(
                "conversation_session_finished",
                kind="span_end",
                details={"initial_source": initial_source},
                trace_id=trace_id,
            )
            self._workflow_trace_set_active(None)

    def _set_active_turn_stop_event(self, stop_event: Event) -> None:
        with self._active_turn_stop_lock:
            self._active_turn_stop_event = stop_event
        self._trace_event(
            "active_turn_stop_event_set",
            kind="mutation",
            details={"stop_event_id": id(stop_event)},
        )

    def _clear_active_turn_stop_event(self, stop_event: Event) -> None:
        with self._active_turn_stop_lock:
            if self._active_turn_stop_event is stop_event:
                self._active_turn_stop_event = None
        self._trace_event(
            "active_turn_stop_event_cleared",
            kind="mutation",
            details={"stop_event_id": id(stop_event)},
        )

    def _active_turn_stop_requested(self) -> bool:
        with self._active_turn_stop_lock:
            stop_event = self._active_turn_stop_event
        return bool(stop_event is not None and stop_event.is_set())

    def _request_active_turn_interrupt(self, source: str = "button") -> bool:
        with self._active_turn_stop_lock:
            stop_event = self._active_turn_stop_event
        if stop_event is None or stop_event.is_set():
            self._trace_event(
                "turn_interrupt_request_ignored",
                kind="branch",
                details={"source": source, "has_stop_event": stop_event is not None},
            )
            return False
        stop_event.set()
        self._best_effort_stop_player()
        self._stop_working_feedback()
        self.emit(f"turn_interrupt_requested={source}")
        self._trace_event(
            "turn_interrupt_requested",
            kind="mutation",
            details={"source": source, "stop_event_id": id(stop_event)},
        )
        return True

    def _cancel_interrupted_turn(self) -> None:
        self.runtime.cancel_listening()
        self._emit_status(force=True)
        self.emit("turn_interrupted=true")
        self._trace_event("turn_interrupt_canceled_current_turn", kind="branch", details={})

    def _recorder_sample_rate(self) -> int:
        return int(
            getattr(
                self.recorder,
                "sample_rate",
                self.config.openai_realtime_input_sample_rate,
            )
        )

    def _turn_controller_conversation(self) -> tuple[tuple[str, str], ...]:
        """Return the bounded turn-controller conversation from the guidance runtime."""

        return self.turn_guidance_runtime.controller_conversation()

    def _build_streaming_turn_controller(self) -> StreamingTurnController | None:
        """Build the turn controller through the extracted guidance runtime."""

        return self.turn_guidance_runtime.build_streaming_turn_controller()

    def _turn_guidance_messages(self, turn_label: str | None) -> tuple[tuple[str, str], ...]:
        """Return label-specific guidance through the extracted guidance runtime."""

        return self.turn_guidance_runtime.guidance_messages(turn_label)

    def _conversation_context_for_turn_label(self, turn_label: str | None) -> tuple[tuple[str, str], ...]:
        """Return guided conversation context through the extracted guidance runtime."""

        return self.turn_guidance_runtime.conversation_context_for_turn_label(turn_label)

    def _interrupt_stt_provider(self) -> SpeechToTextProvider | None:
        provider = self.turn_stt_provider or self.stt_provider
        if provider is None or not callable(getattr(provider, "transcribe", None)):
            return None
        return provider

    def _ambient_sampler(self) -> AmbientAudioSampler:
        sampler = self._ambient_audio_sampler
        if sampler is None:
            sampler = AmbientAudioSampler.from_config(self.config)
            self._ambient_audio_sampler = sampler
        return sampler

    def _best_effort_streaming_transcript_hint(
        self,
        *,
        partial_text: str,
        controller: StreamingTurnController | None,
    ) -> str:
        """Return the current best transcript hint from the verifier runtime."""

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
        """Recover weak streaming transcripts through the verifier runtime."""

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
        """Return the PCM-derived audio duration through the verifier runtime."""

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
        """Return the verifier-runtime gate result as a compatibility boolean."""

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
        """Build the verifier prompt through the extracted verifier runtime."""

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
        """Run the verifier through the extracted verifier runtime."""

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
        streaming_send_error: list[Exception] = []
        early_transcript_hint = ""

        def current_turn_label() -> str | None:
            if controller is None:
                return None
            decision = controller.last_decision()
            return decision.label if decision is not None else None

        def send_chunk(chunk: bytes) -> None:
            streamed_pcm.extend(chunk)
            if session is None or streaming_send_error:
                return
            try:
                session.send_pcm(chunk)
            except Exception as exc:
                streaming_send_error.append(exc)
                self.emit(f"turn_controller_stream_send_failed={type(exc).__name__}")  # AUDIT-FIX(#2): Keep the original user audio even if streaming STT transport drops mid-capture.
                self._trace_event(
                    "turn_controller_stream_send_failed",
                    kind="exception",
                    level="WARN",
                    details={"error_type": type(exc).__name__, "chunk_size": len(chunk)},
                )

        def on_interim(text: str) -> None:
            cleaned = text.strip()
            if not cleaned or cleaned == partial_cache[0]:
                return
            partial_cache[0] = cleaned
            self._on_streaming_stt_interim(cleaned)
            if controller is not None:
                controller.on_interim(cleaned)
            self.emit(f"stt_partial={cleaned}")
            self._trace_event(
                "turn_controller_interim_received",
                kind="io",
                details={"text_len": len(cleaned), "preview": cleaned[:80]},
            )

        def on_endpoint(event: StreamingSpeechEndpointEvent) -> None:
            if controller is not None:
                controller.on_endpoint(event)
            self._on_streaming_stt_endpoint(event)
            self._trace_event(
                "turn_controller_endpoint_received",
                kind="io",
                details={
                    "text_len": len(str(getattr(event, "transcript", "") or "").strip()),
                    "speech_final": bool(getattr(event, "speech_final", False)),
                    "utterance_end": bool(getattr(event, "utterance_end", False)),
                },
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
                details={"sample_rate": self._recorder_sample_rate(), "channels": self.config.audio_channels},
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
                    controller_requested = bool(
                        controller is not None and controller.should_stop_capture()
                    )
                    return controller_requested or self._active_turn_stop_requested()

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
                transcript_hint = self._best_effort_streaming_transcript_hint(
                    partial_text=partial_cache[0],
                    controller=controller,
                )
                if not transcript_hint or not streamed_pcm:
                    raise
                capture_result = SpeechCaptureResult(
                    pcm_bytes=bytes(streamed_pcm),
                    speech_started_after_ms=0,
                    resumed_after_pause_count=0,
                )
                self.emit("turn_controller_capture_recovered=true")
                self._trace_event(
                    "turn_controller_capture_recovered",
                    kind="branch",
                    details={"transcript_hint_len": len(transcript_hint), "pcm_bytes": len(streamed_pcm)},
                )
            capture_ms = int((time.monotonic() - capture_started) * 1000)
            self._trace_event(
                "turn_controller_capture_completed",
                kind="span_end",
                details={"pcm_bytes": len(capture_result.pcm_bytes)},
                kpi={"duration_ms": capture_ms},
            )
            early_result = None
            try:
                early_result = self._early_streaming_transcription_result(
                    session=session,
                    controller=controller,
                )
            except Exception as exc:
                self.emit(f"stt_streaming_early_fallback={type(exc).__name__}")  # AUDIT-FIX(#2): Do not throw away captured audio when snapshot inspection fails.
            if early_result is not None:
                if early_result.request_id:
                    self.emit(f"stt_request_id={early_result.request_id}")
                self.emit(f"stt_streaming_interim={str(early_result.saw_interim).lower()}")
                self.emit(f"stt_streaming_speech_final={str(early_result.saw_speech_final).lower()}")
                self.emit(f"stt_streaming_utterance_end={str(early_result.saw_utterance_end).lower()}")
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
                        "utterance_end": bool(getattr(early_result, "saw_utterance_end", False)),
                        "interim": bool(getattr(early_result, "saw_interim", False)),
                        "transcript_len": len(str(getattr(early_result, "transcript", "") or "").strip()),
                    },
                )
                early_transcript_hint = self._maybe_recover_low_evidence_streaming_transcript(
                    stt_provider=stt_provider,
                    capture_result=capture_result,
                    transcript=early_result.transcript.strip(),
                    saw_interim=bool(getattr(early_result, "saw_interim", False)),
                    capture_ms=capture_ms,
                )
                if bool(getattr(early_result, "saw_speech_final", False)):
                    transcript = self._maybe_verify_streaming_transcript(
                        capture_result=capture_result,
                        transcript=early_transcript_hint,
                        capture_ms=capture_ms,
                        saw_speech_final=True,
                        saw_utterance_end=bool(getattr(early_result, "saw_utterance_end", False)),
                        confidence=getattr(early_result, "confidence", None),
                    )
                    self._trace_event(
                        "turn_controller_returned_early_snapshot",
                        kind="branch",
                        details={"transcript_len": len(transcript)},
                    )
                    return capture_result, transcript, capture_ms, 0, current_turn_label()
                self.emit("stt_streaming_deferred_until_finalize=true")
                self._trace_event(
                    "turn_controller_early_snapshot_deferred",
                    kind="branch",
                    details={"transcript_hint_len": len(early_transcript_hint)},
                )

            stt_started = time.monotonic()
            try:
                result = session.finalize()
            except Exception as exc:
                transcript_hint = self._best_effort_streaming_transcript_hint(
                    partial_text=partial_cache[0],
                    controller=controller,
                )
                if capture_result is not None:
                    self.emit(f"turn_controller_finalize_recovered={type(exc).__name__}")  # AUDIT-FIX(#2): Preserve the original capture and continue with realtime audio even when seed STT finalization fails.
                    self._trace_event(
                        "turn_controller_finalize_recovered",
                        kind="branch",
                        level="WARN",
                        details={"error_type": type(exc).__name__, "hint_len": len(transcript_hint)},
                    )
                    return capture_result, transcript_hint, capture_ms, -1, current_turn_label()
                raise
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            self._trace_event(
                "turn_controller_finalize_completed",
                kind="llm_call",
                details={"request_id": result.request_id, "transcript_len": len((result.transcript or '').strip())},
                kpi={"duration_ms": stt_ms},
            )
            if result.request_id:
                self.emit(f"stt_request_id={result.request_id}")
            self.emit(f"stt_streaming_interim={str(result.saw_interim).lower()}")
            self.emit(f"stt_streaming_speech_final={str(result.saw_speech_final).lower()}")
            self.emit(f"stt_streaming_utterance_end={str(result.saw_utterance_end).lower()}")
            transcript = (result.transcript or "").strip()
            if not transcript:
                transcript = early_transcript_hint or self._best_effort_streaming_transcript_hint(
                    partial_text=partial_cache[0],
                    controller=controller,
                )
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
                    "transcript_sha12": (
                        sha1(transcript.encode("utf-8")).hexdigest()[:12]
                        if transcript
                        else None
                    ),
                },
                kpi={"capture_ms": capture_ms, "stt_ms": stt_ms},
            )
            return capture_result, transcript, capture_ms, stt_ms, current_turn_label()
        finally:
            try:
                session.close()
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
    ):
        if not self.config.streaming_early_transcript_enabled:
            return None
        snapshot_fn = getattr(session, "snapshot", None)
        if snapshot_fn is None:
            return None
        deadline = time.monotonic() + (max(0, int(self.config.streaming_early_transcript_wait_ms)) / 1000.0)
        while True:
            snapshot = snapshot_fn()
            early_result = self._coerce_early_streaming_snapshot(
                snapshot=snapshot,
                controller=controller,
            )
            if early_result is not None:
                return early_result
            if time.monotonic() >= deadline:
                return None
            self.sleep(0.02)  # AUDIT-FIX(#8): Respect the injected sleep hook for tests and controlled shutdown behavior.

    def _coerce_early_streaming_snapshot(
        self,
        *,
        snapshot,
        controller: StreamingTurnController | None,
    ):
        transcript = str(getattr(snapshot, "transcript", "") or "").strip()
        min_chars = max(1, int(self.config.streaming_early_transcript_min_chars))
        if len(transcript) < min_chars:
            return None
        if getattr(snapshot, "saw_speech_final", False):
            if not bool(getattr(snapshot, "saw_interim", False)):
                return None
            return snapshot
        if not (
            self.config.deepgram_streaming_stop_on_utterance_end
            and getattr(snapshot, "saw_utterance_end", False)
            and controller is not None
        ):
            return None
        decision = controller.last_decision()
        if decision is None or decision.decision != "end_turn":
            return None
        if decision.confidence < self.config.turn_controller_fast_endpoint_min_confidence:
            return None
        decision_transcript = str(getattr(decision, "transcript", "") or "")
        if _normalize_turn_text(decision_transcript) != _normalize_turn_text(transcript):
            return None
        return snapshot

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
        if listen_source == "button":
            self.runtime.press_green_button()
        else:
            self.runtime.begin_listening(
                request_source=listen_source,
                proactive_trigger=proactive_trigger,
            )
        self._emit_status(force=True)
        if play_initial_beep:
            self._play_listen_beep()

        capture_started = time.monotonic()
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
                        raise  # AUDIT-FIX(#2): Preserve the normal no-speech timeout path instead of starting a second capture window.
                    except Exception as exc:
                        self.emit(f"turn_controller_fallback={type(exc).__name__}")
                        capture_result = self.recorder.capture_pcm_until_pause_with_options(
                            pause_ms=listening_window.speech_pause_ms,
                            start_timeout_s=listening_window.start_timeout_s,
                            speech_start_chunks=speech_start_chunks,
                            ignore_initial_ms=ignore_initial_ms,
                            pause_grace_ms=listening_window.pause_grace_ms,
                            should_stop=self._active_turn_stop_requested,
                        )
                        capture_ms = int((time.monotonic() - capture_started) * 1000)
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
            self.emit(f"voice_assessment_failed={type(exc).__name__}")  # AUDIT-FIX(#5): Optional voice assessment must not abort the primary conversation path.
        realtime_started = time.monotonic()
        self._current_turn_audio_pcm = audio_pcm
        self._current_turn_audio_sample_rate = self._recorder_sample_rate()
        if turn_label:
            self.emit(f"turn_controller_selected_label={turn_label}")
        try:
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

    def _run_single_text_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        turn_started = time.monotonic()
        self.runtime.begin_listening(
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        realtime_started = time.monotonic()
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
        min_transcript_chars = max(1, int(self.config.turn_controller_interrupt_min_transcript_chars))
        required_windows = max(1, int(self.config.turn_controller_interrupt_consecutive_windows))
        window_ms = max(120, int(self.config.turn_controller_interrupt_window_ms))
        poll_s = max(0.02, int(self.config.turn_controller_interrupt_poll_ms) / 1000.0)

        def worker() -> None:
            consecutive_confirmed = 0
            saw_candidate = False
            while not stop_event.is_set() and not interrupt_event.is_set():
                if not answer_started():
                    if stop_event.wait(0.02):
                        return
                    continue
                try:
                    window = sampler.sample_window(duration_ms=window_ms)
                except Exception as exc:
                    self.emit(f"interrupt_sampler_failed={type(exc).__name__}")
                    return
                sample = window.sample
                speech_like = sample.active_chunk_count > 0 and sample.active_ratio >= min_active_ratio
                if not speech_like:
                    if saw_candidate:
                        self.emit("false_interrupt_recovered=true")
                    consecutive_confirmed = 0
                    saw_candidate = False
                    if stop_event.wait(poll_s):
                        return
                    continue
                saw_candidate = True
                transcript = ""
                try:
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
                if len(_normalize_turn_text(transcript).replace(" ", "")) < min_transcript_chars:
                    if stop_event.wait(poll_s):
                        return
                    continue
                consecutive_confirmed += 1
                self.emit(f"interrupt_candidate_transcript={transcript}")
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
        stop_processing_feedback = self._start_working_feedback_loop("processing")
        stop_answering_feedback: Callable[[], None] = lambda: None

        audio_chunks: Queue[bytes | None] = Queue()
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

        def begin_answering() -> None:
            nonlocal answer_started, stop_answering_feedback, interrupt_thread
            if answer_started:
                return
            stop_processing_feedback()
            self.runtime.begin_answering()
            self._emit_status(force=True)
            stop_answering_feedback = self._start_working_feedback_loop("answering")
            answer_started = True
            if interrupt_thread is None:
                interrupt_thread = self._start_answer_interrupt_watcher(
                    interrupt_event=interrupt_event,
                    stop_event=interrupt_stop_event,
                    transcript_holder=interrupt_transcript_holder,
                    answer_started=lambda: answer_started,
                    on_interrupt=lambda: audio_chunks.put(None),
                )

        def playback_generator():
            while True:
                chunk = audio_chunks.get()
                if chunk is None:
                    return
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

        def on_audio_chunk(chunk: bytes) -> None:
            begin_answering()
            if interrupt_event.is_set():
                return
            if first_audio_at[0] is None:
                stop_answering_feedback()
                first_audio_at[0] = time.monotonic()
                ensure_playback_started()
            audio_chunks.put(chunk)

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
            if playback_started:
                audio_chunks.put(None)
        if interrupt_thread is not None:
            interrupt_thread.join(timeout=1.0)
            if interrupt_thread.is_alive():
                self.emit("interrupt_watcher_join_timeout=true")
        if worker is not None:
            playback_join_timeout_s = max(
                0.1,
                float(getattr(self.config, "realtime_playback_join_timeout_s", 5.0)),
            )
            worker.join(timeout=playback_join_timeout_s)  # AUDIT-FIX(#3): Bound playback-thread joins so wedged audio backends cannot freeze the whole loop forever.
            if worker.is_alive():
                playback_error.append(
                    TimeoutError(
                        f"Playback worker did not exit within {playback_join_timeout_s:.2f}s"
                    )
                )
                self.emit("playback_join_timeout=true")
        realtime_ms = int((time.monotonic() - realtime_started) * 1000)
        try:
            if turn_error is not None:
                raise turn_error
            if playback_error:
                raise playback_error[0]
            if turn is None:
                raise RuntimeError("Realtime turn did not return a result")

            response_text = turn.response_text or ""
            if first_audio_at[0] is None and response_text.strip():
                if not answer_started:
                    begin_answering()
                self.emit("realtime_audio_fallback=true")
                fallback_tts_ms, fallback_first_audio_ms = self._play_streaming_tts_with_feedback(
                    response_text,
                    turn_started=turn_started,
                    should_stop=interrupt_event.is_set,
                )
                self.emit(f"timing_tts_fallback_ms={fallback_tts_ms}")
                if fallback_first_audio_ms is not None:
                    first_audio_ms_override = fallback_first_audio_ms
        finally:
            stop_answering_feedback()
            self._stop_working_feedback()

        final_transcript = ((turn.transcript or "").strip() or transcript_seed)
        self.runtime.last_transcript = final_transcript
        self.emit(f"transcript={final_transcript}")
        if not answer_started:
            begin_answering()
        try:
            answer = self.runtime.finalize_agent_turn(response_text)
            self.emit(f"response={answer}")
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
                transcript=final_transcript,
                request_source=listen_source,
                proactive_trigger=proactive_trigger,
            )
        finally:
            self.runtime.finish_speaking()  # AUDIT-FIX(#6): Always leave the runtime in a non-speaking state even if post-processing or usage accounting fails.
            self._emit_status(force=True)
        force_close = turn.end_conversation or self._follow_up_vetoed_by_closure(
            user_transcript=final_transcript,
            assistant_response=answer,
            request_source=initial_source,
            proactive_trigger=proactive_trigger,
        )
        if turn.end_conversation:
            self.emit("conversation_ended=true")
        elif force_close:
            self.emit("conversation_follow_up_vetoed=closure")
        if interrupt_event.is_set():
            self.emit("assistant_interrupted=true")
            if interrupt_transcript_holder[0].strip():
                self.emit(f"interrupt_transcript={interrupt_transcript_holder[0].strip()}")
            self._record_event(
                "user_interrupt_detected",
                "Twinr stopped the current spoken answer after detecting a user interruption.",
                transcript_preview=interrupt_transcript_holder[0].strip() or None,
                request_source=listen_source,
                turn_label=turn_label,
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
        if interrupt_event.is_set() and not force_close:
            return self._run_interrupt_follow_up_turn()
        return not force_close

    def _acknowledge_wakeword(self) -> None:
        prompt = self.runtime.begin_wakeword_prompt("Ja?")
        self._emit_status(force=True)
        cached_audio = self._cached_wakeword_ack_wav_bytes()
        tts_started = time.monotonic()
        try:
            self._play_listen_beep()
            if cached_audio is not None:
                self.playback_coordinator.play_wav_bytes(
                    owner="wakeword_ack",
                    priority=PlaybackPriority.SPEECH,
                    wav_bytes=cached_audio,
                )
            else:
                self._ensure_wakeword_ack_prefetch_started()
                self.playback_coordinator.play_wav_chunks(
                    owner="wakeword_ack",
                    priority=PlaybackPriority.SPEECH,
                    chunks=self.tts_provider.synthesize_stream(prompt),
                )
        finally:
            self.runtime.finish_speaking()  # AUDIT-FIX(#6): Ensure wakeword acknowledgment failures do not strand runtime state in "speaking".
            self._emit_status(force=True)
        self.emit(f"wakeword_ack={prompt}")
        self.emit(f"wakeword_ack_cached={str(cached_audio is not None).lower()}")
        self.emit(f"timing_wakeword_ack_tts_ms={int((time.monotonic() - tts_started) * 1000)}")
        self._record_event(
            "wakeword_acknowledged",
            "Twinr confirmed a wakeword before opening hands-free listening.",
            prompt=prompt,
        )

    def _ensure_wakeword_ack_prefetch_started(self) -> None:
        if not self.config.wakeword_enabled:
            return
        with self._wakeword_ack_cache_lock:
            if self._wakeword_ack_wav_bytes is not None:
                return
            if self._wakeword_ack_prefetch_started:
                return
            self._wakeword_ack_prefetch_started = True
            self._wakeword_ack_prefetch_thread = Thread(
                target=self._prime_wakeword_ack_cache,
                name="twinr-wakeword-ack",
                daemon=True,
            )
            self._wakeword_ack_prefetch_thread.start()

    def _cached_wakeword_ack_wav_bytes(self) -> bytes | None:
        with self._wakeword_ack_cache_lock:
            return self._wakeword_ack_wav_bytes

    def _prime_wakeword_ack_cache(self) -> None:
        try:
            audio_bytes = self.tts_provider.synthesize("Ja?")
        except Exception as exc:
            with self._wakeword_ack_cache_lock:
                self._wakeword_ack_prefetch_started = False  # AUDIT-FIX(#9): Allow retry after transient TTS/network failures instead of disabling the cache for the process lifetime.
                self._wakeword_ack_prefetch_thread = None
            self.emit(f"wakeword_ack_prefetch_failed={type(exc).__name__}")
            return
        with self._wakeword_ack_cache_lock:
            self._wakeword_ack_wav_bytes = audio_bytes
            self._wakeword_ack_prefetch_thread = None

    def _listening_window_speech_start_chunks(self, *, initial_source: str, follow_up: bool) -> int | None:
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
        if initial_source == "wakeword":
            return "wakeword_listen_timeout"
        return "listen_timeout"

    def _listening_timeout_message(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "Follow-up listening window expired."
        if initial_source == "proactive":
            return "Hands-free listening window after a proactive prompt expired."
        if initial_source == "wakeword":
            return "Wakeword listening window expired."
        return "Listening timed out before speech started."

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        try:
            if self.print_lane.is_busy():
                self.emit("print_skipped=busy")
                self._record_event("print_skipped", "Print request ignored because another print job is already in progress.")
                return
            response_to_print = self.runtime.prepare_background_button_print_request()
            # Yellow-button prints should stay local and immediate. Pulling the
            # full provider context here can block on remote long-term memory
            # retrieval even though the print lane already has the direct text.
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
                self._record_event("print_skipped", "Print request could not be queued because another print job won the race.")
                return
            self.emit("print_lane=queued")
            self._emit_status(force=True)
        except Exception:
            self._emit_status(force=True)
            raise
