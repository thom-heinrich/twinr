"""Run the speculative streaming workflow with dual-lane tool orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    FirstWordProvider,
    FirstWordReply,
    SupervisorDecision,
    ToolCallingAgentProvider,
    StreamingSpeechToTextProvider,
)
from twinr.agent.base_agent.prompting.personality import load_supervisor_loop_instructions
from twinr.agent.base_agent.conversation.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.agent.tools import (
    DualLaneToolLoop,
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_first_word_instructions,
    build_compact_agent_tool_schemas,
    build_supervisor_decision_instructions,
    build_specialist_tool_agent_instructions,
    bind_realtime_tool_handlers,
    realtime_tool_names,
)
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.streaming_capture import (
    StreamingAudioTurnRequest,
    StreamingCaptureController,
)
from twinr.agent.workflows.streaming_lane_planner import StreamingLanePlanner
from twinr.agent.workflows.streaming_semantic_router import StreamingSemanticRouterRuntime
from twinr.agent.workflows.streaming_speculation import StreamingSpeculationController
from twinr.agent.workflows.streaming_turn_coordinator import (
    StreamingTurnCoordinator,
    StreamingTurnCoordinatorHooks,
    StreamingTurnLanePlan,
    StreamingTurnRequest,
    StreamingTurnSpeechServices,
)
from twinr.agent.workflows.streaming_turn_orchestrator import StreamingTurnTimeoutPolicy
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai import (
    OpenAIBackend,
    OpenAIFirstWordProvider,
    OpenAIProviderBundle,
    OpenAISupervisorDecisionProvider,
    OpenAIToolCallingAgentProvider,
)


class _StreamingSessionPlaceholder:
    """Carry config for the realtime parent without opening a realtime session."""

    def __init__(self, config: TwinrConfig) -> None:
        self.config = config


class TwinrStreamingHardwareLoop(TwinrRealtimeHardwareLoop):
    """Extend the realtime loop with streaming STT and speculative speech.

    The streaming loop adds dual-lane tool execution, interruptible speech
    output, and speculative first-word and supervisor decision warmups.
    """

    def __init__(
        self,
        config: TwinrConfig,
        *,
        tool_agent_provider: ToolCallingAgentProvider | None = None,
        streaming_turn_loop: ToolCallingStreamingLoop | None = None,
        verification_stt_provider=None,
        **kwargs,
    ) -> None:
        verifier_provider = verification_stt_provider
        if (
            tool_agent_provider is None
            and kwargs.get("print_backend") is None
            and (
                kwargs.get("stt_provider") is None
                or kwargs.get("agent_provider") is None
                or kwargs.get("tts_provider") is None
            )
        ):
            provider_bundle = build_streaming_provider_bundle(config)
            kwargs.setdefault("print_backend", provider_bundle.print_backend)
            kwargs.setdefault("stt_provider", provider_bundle.stt)
            kwargs.setdefault("agent_provider", provider_bundle.agent)
            kwargs.setdefault("tts_provider", provider_bundle.tts)
            tool_agent_provider = provider_bundle.tool_agent
            if (
                verifier_provider is None
                and self._should_enable_streaming_transcript_verifier(config, kwargs.get("stt_provider"))
            ):
                verifier_backend = OpenAIBackend(
                    config=replace(
                        config,
                        openai_stt_model=config.streaming_transcript_verifier_model,
                    )
                )
                verifier_provider = OpenAIProviderBundle.from_backend(verifier_backend).stt
        if kwargs.get("recorder") is None and (config.stt_provider or "").strip().lower() == "deepgram":
            kwargs["recorder"] = SilenceDetectedRecorder(
                device=config.audio_input_device,
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                chunk_ms=config.audio_chunk_ms,
                preroll_ms=config.audio_preroll_ms,
                speech_threshold=config.audio_speech_threshold,
                speech_start_chunks=config.audio_speech_start_chunks,
                start_timeout_s=config.audio_start_timeout_s,
                max_record_seconds=config.audio_max_record_seconds,
                dynamic_pause_enabled=config.audio_dynamic_pause_enabled,
                dynamic_pause_short_utterance_max_ms=config.audio_dynamic_pause_short_utterance_max_ms,
                dynamic_pause_long_utterance_min_ms=config.audio_dynamic_pause_long_utterance_min_ms,
                dynamic_pause_short_pause_bonus_ms=config.audio_dynamic_pause_short_pause_bonus_ms,
                dynamic_pause_short_pause_grace_bonus_ms=config.audio_dynamic_pause_short_pause_grace_bonus_ms,
                dynamic_pause_long_pause_penalty_ms=config.audio_dynamic_pause_long_pause_penalty_ms,
                dynamic_pause_long_pause_grace_penalty_ms=config.audio_dynamic_pause_long_pause_grace_penalty_ms,
            )
        resolved_tool_agent = tool_agent_provider
        if resolved_tool_agent is None:
            raise ValueError("TwinrStreamingHardwareLoop requires a tool-capable agent provider")

        super().__init__(
            config,
            realtime_session=_StreamingSessionPlaceholder(config),
            turn_stt_provider=(
                kwargs.get("stt_provider")
                if isinstance(kwargs.get("stt_provider"), StreamingSpeechToTextProvider)
                else None
            ),
            turn_tool_agent_provider=resolved_tool_agent,
            verification_stt_provider=verifier_provider,
            **kwargs,
        )
        self.tool_agent_provider = resolved_tool_agent
        self._tool_handlers = bind_realtime_tool_handlers(self.tool_executor)
        tool_schemas = (
            build_compact_agent_tool_schemas(realtime_tool_names())
            if (self.config.llm_provider or "").strip().lower() == "groq"
            else build_agent_tool_schemas(realtime_tool_names())
        )
        self.streaming_turn_loop = streaming_turn_loop or self._build_streaming_turn_loop(
            tool_schemas=tool_schemas,
        )
        self.first_word_provider: FirstWordProvider | None = getattr(self, "first_word_provider", None)
        self._streaming_capture = StreamingCaptureController(self)
        self._streaming_speculation = StreamingSpeculationController(self)
        self._streaming_lane_planner = StreamingLanePlanner(self)
        self._streaming_semantic_router = StreamingSemanticRouterRuntime(self)
        self._prime_supervisor_decision_cache()
        self._prime_first_word_cache()
        self._trace_event(
            "streaming_workflow_initialized",
            kind="run_start",
            details={
                "dual_lane": isinstance(self.streaming_turn_loop, DualLaneToolLoop),
                "first_word_enabled": bool(self.config.streaming_first_word_enabled),
                "stt_provider": type(self.stt_provider).__name__,
                "tool_agent_provider": type(self.tool_agent_provider).__name__,
            },
        )

    @staticmethod
    def _should_enable_streaming_transcript_verifier(config: TwinrConfig, stt_provider) -> bool:
        if not bool(getattr(config, "streaming_transcript_verifier_enabled", True)):
            return False
        if not (config.openai_api_key or "").strip():
            return False
        if (config.stt_provider or "").strip().lower() != "deepgram":
            return False
        return isinstance(stt_provider, StreamingSpeechToTextProvider)

    def _build_streaming_turn_loop(
        self,
        *,
        tool_schemas,
    ):
        self.first_word_provider = None
        llm_name = (self.config.llm_provider or "").strip().lower()
        if (
            llm_name == "openai"
            and self.config.streaming_dual_lane_enabled
            and isinstance(self.tool_agent_provider, OpenAIToolCallingAgentProvider)
        ):
            supervisor_tool_names = ("end_conversation",)
            supervisor_tool_handlers = {
                name: handler
                for name, handler in self._tool_handlers.items()
                if name in supervisor_tool_names
            }
            supervisor_tool_schemas = build_agent_tool_schemas(supervisor_tool_names)
            supervisor_backend = OpenAIBackend(config=self.config)
            supervisor_decision_backend = OpenAIBackend(config=self.config)
            specialist_backend = OpenAIBackend(config=self.config)
            first_word_backend = OpenAIBackend(config=self.config)
            supervisor_provider = OpenAIToolCallingAgentProvider(
                supervisor_backend,
                model_override=self.config.streaming_supervisor_model,
                reasoning_effort_override=self.config.streaming_supervisor_reasoning_effort,
                base_instructions_override=load_supervisor_loop_instructions(self.config),
                replace_base_instructions=True,
            )
            supervisor_decision_provider = OpenAISupervisorDecisionProvider(
                supervisor_decision_backend,
                model_override=self.config.streaming_supervisor_model,
                reasoning_effort_override=self.config.streaming_supervisor_reasoning_effort,
                base_instructions_override=load_supervisor_loop_instructions(self.config),
                replace_base_instructions=True,
            )
            specialist_provider = OpenAIToolCallingAgentProvider(
                specialist_backend,
                model_override=self.config.streaming_specialist_model,
                reasoning_effort_override=self.config.streaming_specialist_reasoning_effort,
            )
            if self.config.streaming_first_word_enabled:
                self.first_word_provider = OpenAIFirstWordProvider(
                    first_word_backend,
                    model_override=self.config.streaming_first_word_model,
                    reasoning_effort_override=self.config.streaming_first_word_reasoning_effort,
                    base_instructions_override=build_first_word_instructions(
                        self.config,
                        extra_instructions=self.config.openai_realtime_instructions,
                    ),
                    replace_base_instructions=True,
                )
            return DualLaneToolLoop(
                supervisor_provider=supervisor_provider,
                specialist_provider=specialist_provider,
                tool_handlers=self._tool_handlers,
                tool_schemas=tool_schemas,
                supervisor_decision_provider=supervisor_decision_provider,
                first_word_provider=self.first_word_provider,
                supervisor_tool_handlers=supervisor_tool_handlers,
                supervisor_tool_schemas=supervisor_tool_schemas,
                supervisor_instructions=build_supervisor_decision_instructions(
                    self.config,
                    extra_instructions=self.config.openai_realtime_instructions,
                ),
                specialist_instructions=build_specialist_tool_agent_instructions(
                    self.config,
                    extra_instructions=self.config.openai_realtime_instructions,
                ),
                max_rounds=6,
                trace_event=self._trace_event,
                trace_decision=self._trace_decision,
            )
        return ToolCallingStreamingLoop(
            provider=self.tool_agent_provider,
            tool_handlers=self._tool_handlers,
            tool_schemas=tool_schemas,
            stream_final_only=(llm_name == "groq"),
        )

    def _recorder_sample_rate(self) -> int:
        return int(getattr(self.recorder, "sample_rate", self.config.audio_sample_rate))

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        self.config = updated_config
        self.runtime.apply_live_config(updated_config)
        seen: set[int] = set()
        for provider in (
            self.stt_provider,
            self.agent_provider,
            self.tts_provider,
            self.print_backend,
            self.first_word_provider,
            self.tool_agent_provider,
            self.turn_stt_provider,
            self.turn_tool_agent_provider,
        ):
            if provider is None:
                continue
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            provider.config = updated_config
        self.turn_decision_evaluator = (
            ToolCallingTurnDecisionEvaluator(
                config=updated_config,
                provider=self.turn_tool_agent_provider,
            )
            if self.turn_tool_agent_provider is not None and updated_config.turn_controller_enabled
            else None
        )
        self.realtime_session.config = updated_config
        tool_schemas = (
            build_compact_agent_tool_schemas(realtime_tool_names())
            if (self.config.llm_provider or "").strip().lower() == "groq"
            else build_agent_tool_schemas(realtime_tool_names())
        )
        self.streaming_turn_loop = self._build_streaming_turn_loop(
            tool_schemas=tool_schemas,
        )
        self._streaming_semantic_router.reload()
        self._reset_speculative_supervisor_decision()
        self._supervisor_cache_prewarmed = False
        self._first_word_cache_prewarmed = False
        self._prime_supervisor_decision_cache()
        self._prime_first_word_cache()

    def _reset_speculative_supervisor_decision(self) -> None:
        self._streaming_speculation.reset()

    def _capture_and_transcribe_streaming(
        self,
        *,
        listening_window,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ):
        return self._streaming_capture.capture_and_transcribe_streaming(
            listening_window=listening_window,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
        )

    def _on_streaming_stt_interim(self, text: str) -> None:
        self._streaming_capture.handle_stt_interim(text)

    def _on_streaming_stt_endpoint(self, event) -> None:
        self._streaming_capture.handle_stt_endpoint(event)

    def _maybe_start_speculative_first_word(self, text: str) -> None:
        self._streaming_speculation.maybe_start_first_word(text)

    def _consume_speculative_first_word(self, transcript: str) -> FirstWordReply | None:
        return self._streaming_speculation.consume_first_word(transcript)

    def _maybe_start_speculative_supervisor_decision(self, text: str) -> None:
        self._streaming_speculation.maybe_start_supervisor_decision(text)

    def _consume_speculative_supervisor_decision(self, transcript: str) -> SupervisorDecision | None:
        return self._streaming_speculation.consume_supervisor_decision(transcript)

    def _wait_for_speculative_supervisor_decision(
        self,
        transcript: str,
        *,
        wait_ms: int | None = None,
    ) -> SupervisorDecision | None:
        return self._streaming_speculation.wait_for_supervisor_decision(
            transcript,
            wait_ms=wait_ms,
        )

    def _has_shared_speculative_supervisor_decision(self, transcript: str) -> bool:
        return self._streaming_speculation.has_shared_supervisor_decision(transcript)

    def _prime_supervisor_decision_cache(self) -> None:
        self._streaming_speculation.prime_supervisor_decision_cache()

    def _prime_first_word_cache(self) -> None:
        self._streaming_speculation.prime_first_word_cache()

    def _generate_first_word_reply(
        self,
        transcript: str,
        *,
        instructions: str | None = None,
    ) -> FirstWordReply | None:
        return self._streaming_speculation.generate_first_word_reply(
            transcript,
            instructions=instructions,
        )

    def _dual_lane_prefers_supervisor_bridge(self) -> bool:
        return self._streaming_speculation.dual_lane_prefers_supervisor_bridge()

    def _store_supervisor_decision(
        self,
        *,
        transcript: str,
        decision: SupervisorDecision | None,
    ) -> None:
        self._streaming_speculation.store_supervisor_decision(
            transcript=transcript,
            decision=decision,
        )

    def _generate_supervisor_bridge_reply(
        self,
        transcript: str,
        *,
        instructions: str | None,
    ) -> FirstWordReply | None:
        return self._streaming_speculation.generate_supervisor_bridge_reply(
            transcript,
            instructions=instructions,
        )

    def _streaming_turn_timeout_policy(
        self,
        *,
        decision_hint=None,
    ) -> StreamingTurnTimeoutPolicy:
        return self._streaming_lane_planner.streaming_turn_timeout_policy(
            decision_hint=decision_hint,
        )

    def _dual_lane_bridge_reply_from_decision(
        self,
        prefetched_decision: SupervisorDecision | None,
    ) -> FirstWordReply | None:
        return self._streaming_speculation.dual_lane_bridge_reply_from_decision(prefetched_decision)

    def _resolve_local_semantic_route(self, transcript: str):
        resolution = self._streaming_semantic_router.resolve_transcript(transcript)
        if resolution is not None and resolution.supervisor_decision is not None:
            self._store_supervisor_decision(
                transcript=transcript,
                decision=resolution.supervisor_decision,
            )
        return resolution

    def _build_streaming_turn_lane_plan(self, transcript: str) -> StreamingTurnLanePlan:
        return self._streaming_lane_planner.build_turn_lane_plan(transcript)

    def _run_dual_lane_final_response(
        self,
        transcript: str,
        *,
        turn_instructions: str | None,
        prefetched_decision: SupervisorDecision | None = None,
    ):
        return self._streaming_lane_planner.run_dual_lane_final_response(
            transcript,
            turn_instructions=turn_instructions,
            prefetched_decision=prefetched_decision,
        )

    def _run_single_audio_turn(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        listening_window,
        listen_source: str,
        proactive_trigger: str | None,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
        timeout_emit_key: str,
        timeout_message: str,
        play_initial_beep: bool,
    ) -> bool:
        return self._streaming_capture.run_audio_turn(
            StreamingAudioTurnRequest(
                initial_source=initial_source,
                follow_up=follow_up,
                listening_window=listening_window,
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                speech_start_chunks=speech_start_chunks,
                ignore_initial_ms=ignore_initial_ms,
                timeout_emit_key=timeout_emit_key,
                timeout_message=timeout_message,
                play_initial_beep=play_initial_beep,
            )
        )

    def _run_single_text_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        self._reset_speculative_supervisor_decision()
        turn_started = time.monotonic()
        self._trace_event(
            "streaming_text_turn_started",
            kind="span_start",
            details={"listen_source": listen_source, "proactive_trigger": proactive_trigger, "transcript_len": len(transcript)},
        )
        self.runtime.begin_listening(
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        result = self._complete_streaming_turn(
            transcript=transcript,
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
            turn_started=turn_started,
            capture_ms=0,
            stt_ms=0,
            allow_follow_up_rearm=False,
        )
        self._trace_event(
            "streaming_text_turn_finished",
            kind="span_end",
            details={"result": result},
            kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
        )
        return result

    def _complete_streaming_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
        turn_started: float,
        capture_ms: int,
        stt_ms: int,
        allow_follow_up_rearm: bool,
    ) -> bool:
        self._trace_event(
            "streaming_turn_completion_started",
            kind="span_start",
            details={"listen_source": listen_source, "proactive_trigger": proactive_trigger, "transcript_len": len(transcript)},
        )
        coordinator = StreamingTurnCoordinator(
            config=self.config,
            runtime=self.runtime,
            request=StreamingTurnRequest(
                transcript=transcript,
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
                allow_follow_up_rearm=allow_follow_up_rearm,
            ),
            lane_plan_factory=lambda: self._build_streaming_turn_lane_plan(transcript),
            speech_services=StreamingTurnSpeechServices(
                tts_provider=self.tts_provider,
                player=self.player,
                playback_coordinator=self.playback_coordinator,
                segment_boundary=self._segment_boundary,
            ),
            hooks=StreamingTurnCoordinatorHooks(
                emit=self.emit,
                emit_status=lambda: self._emit_status(force=True),
                trace_event=self._trace_event,
                trace_decision=self._trace_decision,
                start_processing_feedback_loop=self._start_working_feedback_loop,
                is_search_feedback_active=lambda: callable(getattr(self, "_search_feedback_stop", None)),
                stop_search_feedback=self._stop_search_feedback,
                should_stop=self._active_turn_stop_requested,
                request_turn_stop=self._signal_active_turn_stop,
                cancel_interrupted_turn=self._cancel_interrupted_turn,
                record_usage=self._record_usage,
                evaluate_follow_up_closure=self._evaluate_follow_up_closure,
                apply_follow_up_closure_evaluation=self._apply_follow_up_closure_evaluation,
            ),
        )
        try:
            outcome = coordinator.execute()
        except InterruptedError:
            return False
        return outcome.keep_listening

    def _segment_boundary(self, text: str) -> int | None:
        clause_min_chars = max(16, int(self.config.streaming_tts_clause_min_chars))
        soft_segment_chars = max(clause_min_chars + 12, int(self.config.streaming_tts_soft_segment_chars))
        hard_segment_chars = max(soft_segment_chars, int(self.config.streaming_tts_hard_segment_chars))

        for index, character in enumerate(text):
            if character in ".?!":
                return index + 1
        if len(text) >= clause_min_chars:
            for index, character in enumerate(text):
                if index + 1 < clause_min_chars:
                    continue
                if character in ",;:":
                    return index + 1
        if len(text) >= soft_segment_chars:
            boundary = self._last_whitespace_before(text, hard_segment_chars)
            if boundary is not None and boundary >= clause_min_chars:
                return boundary
        if len(text) >= hard_segment_chars:
            return len(text)
        return None

    def _last_whitespace_before(self, text: str, limit: int) -> int | None:
        upper_bound = min(len(text), limit)
        for index in range(upper_bound - 1, -1, -1):
            if text[index].isspace():
                return index + 1
        return None
