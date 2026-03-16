"""Run the speculative streaming workflow with dual-lane tool orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread
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
from twinr.agent.base_agent.conversation.turn_controller import _normalize_turn_text
from twinr.agent.base_agent.conversation.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.agent.tools import (
    DualLaneToolLoop,
    SpeechLaneDelta,
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_first_word_instructions,
    build_compact_agent_tool_schemas,
    build_compact_tool_agent_instructions,
    build_supervisor_decision_instructions,
    build_specialist_tool_agent_instructions,
    build_tool_agent_instructions,
    bind_realtime_tool_handlers,
    realtime_tool_names,
)
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.speech_output import InterruptibleSpeechOutput
from twinr.agent.workflows.streaming_turn_orchestrator import (
    StreamingTurnOrchestrator,
    StreamingTurnTimeoutPolicy,
)
from twinr.hardware.audio import SilenceDetectedRecorder, pcm16_to_wav_bytes
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
        self._speculative_supervisor_lock = Lock()
        self._speculative_supervisor_done = Event()
        self._speculative_supervisor_started = False
        self._speculative_supervisor_transcript = ""
        self._speculative_supervisor_decision: SupervisorDecision | None = None
        self._speculative_first_word_lock = Lock()
        self._speculative_first_word_done = Event()
        self._speculative_first_word_started = False
        self._speculative_first_word_transcript = ""
        self._speculative_first_word_reply: FirstWordReply | None = None
        self._speculative_handoff_lock = Lock()
        self._speculative_handoff_done = Event()
        self._speculative_handoff_started = False
        self._speculative_handoff_transcript = ""
        self._speculative_handoff_decision: SupervisorDecision | None = None
        self._speculative_handoff_result = None
        self._supervisor_cache_prewarmed = False
        self._first_word_cache_prewarmed = False
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
        self._reset_speculative_supervisor_decision()
        self._supervisor_cache_prewarmed = False
        self._first_word_cache_prewarmed = False
        self._prime_supervisor_decision_cache()
        self._prime_first_word_cache()

    def _reset_speculative_supervisor_decision(self) -> None:
        with self._speculative_supervisor_lock:
            self._speculative_supervisor_done = Event()
            self._speculative_supervisor_started = False
            self._speculative_supervisor_transcript = ""
            self._speculative_supervisor_decision = None
        with self._speculative_first_word_lock:
            self._speculative_first_word_done = Event()
            self._speculative_first_word_started = False
            self._speculative_first_word_transcript = ""
            self._speculative_first_word_reply = None
        with self._speculative_handoff_lock:
            self._speculative_handoff_done = Event()
            self._speculative_handoff_started = False
            self._speculative_handoff_transcript = ""
            self._speculative_handoff_decision = None
            self._speculative_handoff_result = None

    def _capture_and_transcribe_streaming(
        self,
        *,
        listening_window,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ):
        self._reset_speculative_supervisor_decision()
        self._trace_event(
            "streaming_capture_cycle_reset_speculation",
            kind="branch",
            details={
                "speech_start_chunks": speech_start_chunks,
                "ignore_initial_ms": ignore_initial_ms,
                "pause_ms": listening_window.speech_pause_ms,
            },
        )
        return super()._capture_and_transcribe_streaming(
            listening_window=listening_window,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
        )

    def _on_streaming_stt_interim(self, text: str) -> None:
        self._trace_event(
            "streaming_stt_interim_received",
            kind="io",
            details={"text_len": len(text.strip()), "preview": text.strip()[:80]},
        )
        self._maybe_start_speculative_first_word(text)
        self._maybe_start_speculative_supervisor_decision(text)

    def _on_streaming_stt_endpoint(self, event) -> None:
        transcript = str(getattr(event, "transcript", "") or "").strip()
        self._trace_event(
            "streaming_stt_endpoint_received",
            kind="io",
            details={
                "text_len": len(transcript),
                "speech_final": bool(getattr(event, "speech_final", False)),
                "utterance_end": bool(getattr(event, "utterance_end", False)),
            },
        )
        if transcript:
            self._maybe_start_speculative_first_word(transcript)
            self._maybe_start_speculative_supervisor_decision(transcript)

    def _maybe_start_speculative_first_word(self, text: str) -> None:
        if not self.config.streaming_first_word_enabled:
            return
        if isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        if not self.config.streaming_first_word_prefetch_enabled:
            return
        provider = self.first_word_provider
        if provider is None:
            return
        cleaned = text.strip()
        if len(cleaned) < max(1, int(self.config.streaming_first_word_prefetch_min_chars)):
            self._trace_event(
                "speculative_first_word_skipped_short_text",
                kind="branch",
                details={"text_len": len(cleaned)},
            )
            return
        with self._speculative_first_word_lock:
            if self._speculative_first_word_started:
                self._trace_event(
                    "speculative_first_word_skipped_already_started",
                    kind="branch",
                    details={"text_len": len(cleaned)},
                )
                return
            self._speculative_first_word_started = True
            self._speculative_first_word_transcript = cleaned
            done_event = self._speculative_first_word_done
            conversation = self.runtime.first_word_provider_conversation_context()
        self._trace_event(
            "speculative_first_word_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )
        worker = Thread(
            target=self._speculative_first_word_worker,
            args=(provider, cleaned, conversation, done_event),
            daemon=True,
        )
        worker.start()

    def _speculative_first_word_worker(
        self,
        provider: FirstWordProvider,
        transcript: str,
        conversation,
        done_event: Event,
    ) -> None:
        reply: FirstWordReply | None = None
        try:
            reply = provider.reply(
                transcript,
                conversation=conversation,
                instructions=None,
            )
        except Exception as exc:
            self.emit(f"speculative_first_word_failed={type(exc).__name__}")
            self._trace_event(
                "speculative_first_word_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with self._speculative_first_word_lock:
                if self._speculative_first_word_transcript == transcript:
                    self._speculative_first_word_reply = reply
            done_event.set()
            self._trace_event(
                "speculative_first_word_worker_completed",
                kind="queue",
                details={"hit": reply is not None, "text_len": len(transcript)},
            )

    def _consume_speculative_first_word(self, transcript: str) -> FirstWordReply | None:
        if not self.config.streaming_first_word_enabled:
            return None
        with self._speculative_first_word_lock:
            if not self._speculative_first_word_started:
                return None
            done_event = self._speculative_first_word_done
            seeded_transcript = self._speculative_first_word_transcript
        wait_ms = max(0, int(self.config.streaming_first_word_prefetch_wait_ms))
        if wait_ms > 0 and not done_event.is_set():
            done_event.wait(wait_ms / 1000.0)
        with self._speculative_first_word_lock:
            reply = self._speculative_first_word_reply
        if reply is None:
            return None
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return None
        if not (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        ):
            return None
        self.emit("speculative_first_word_hit=true")
        self._trace_event(
            "speculative_first_word_consumed",
            kind="cache",
            details={"seed_len": len(seeded_transcript), "final_len": len(transcript)},
        )
        return reply

    def _maybe_start_speculative_supervisor_decision(self, text: str) -> None:
        if not self.config.streaming_supervisor_prefetch_enabled:
            return
        if not isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        provider = getattr(self.streaming_turn_loop, "supervisor_decision_provider", None)
        if provider is None:
            return
        cleaned = text.strip()
        if len(cleaned) < max(1, int(self.config.streaming_supervisor_prefetch_min_chars)):
            self._trace_event(
                "speculative_supervisor_skipped_short_text",
                kind="branch",
                details={"text_len": len(cleaned)},
            )
            return
        with self._speculative_supervisor_lock:
            if self._speculative_supervisor_started:
                self._trace_event(
                    "speculative_supervisor_skipped_already_started",
                    kind="branch",
                    details={"text_len": len(cleaned)},
                )
                return
            self._speculative_supervisor_started = True
            self._speculative_supervisor_transcript = cleaned
            done_event = self._speculative_supervisor_done
            supervisor_conversation = self.runtime.supervisor_provider_conversation_context()
            supervisor_instructions = self.streaming_turn_loop.supervisor_instructions
        self._trace_event(
            "speculative_supervisor_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )
        worker = Thread(
            target=self._speculative_supervisor_worker,
            args=(provider, cleaned, supervisor_conversation, supervisor_instructions, done_event),
            daemon=True,
        )
        worker.start()

    def _speculative_supervisor_worker(
        self,
        provider,
        transcript: str,
        conversation,
        instructions: str,
        done_event: Event,
    ) -> None:
        decision: SupervisorDecision | None = None
        try:
            decision = provider.decide(
                transcript,
                conversation=conversation,
                instructions=instructions,
            )
        except Exception as exc:
            self.emit(f"speculative_supervisor_failed={type(exc).__name__}")
            self._trace_event(
                "speculative_supervisor_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with self._speculative_supervisor_lock:
                if self._speculative_supervisor_transcript == transcript:
                    self._speculative_supervisor_decision = decision
            self._maybe_start_speculative_handoff(transcript, decision)
            done_event.set()
            self._trace_event(
                "speculative_supervisor_worker_completed",
                kind="queue",
                details={
                    "decision_action": getattr(decision, "action", None) if decision is not None else None,
                    "text_len": len(transcript),
                },
            )

    def _consume_speculative_supervisor_decision(self, transcript: str) -> SupervisorDecision | None:
        if not self.config.streaming_supervisor_prefetch_enabled:
            return None
        with self._speculative_supervisor_lock:
            if not self._speculative_supervisor_started:
                return None
            done_event = self._speculative_supervisor_done
            seeded_transcript = self._speculative_supervisor_transcript
        wait_ms = max(0, int(self.config.streaming_supervisor_prefetch_wait_ms))
        if wait_ms > 0 and not done_event.is_set():
            done_event.wait(wait_ms / 1000.0)
        with self._speculative_supervisor_lock:
            decision = self._speculative_supervisor_decision
        if decision is None:
            return None
        action = str(getattr(decision, "action", "") or "").strip().lower()
        if action not in {"direct", "handoff", "end_conversation"}:
            return None
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return None
        if not (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        ):
            return None
        self.emit("speculative_supervisor_hit=true")
        self._trace_event(
            "speculative_supervisor_consumed",
            kind="cache",
            details={
                "action": action,
                "seed_len": len(seeded_transcript),
                "final_len": len(transcript),
            },
        )
        return decision

    def _maybe_start_speculative_handoff(
        self,
        transcript: str,
        decision: SupervisorDecision | None,
    ) -> None:
        if not isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        if decision is None or decision.action != "handoff":
            return
        if str(getattr(decision, "kind", "") or "").strip().lower() != "search":
            self._trace_event(
                "speculative_handoff_skipped_non_search",
                kind="branch",
                details={"action": getattr(decision, "action", None), "kind": getattr(decision, "kind", None)},
            )
            return
        cleaned = transcript.strip()
        if not cleaned:
            return
        with self._speculative_handoff_lock:
            if self._speculative_handoff_started:
                self._trace_event(
                    "speculative_handoff_skipped_already_started",
                    kind="branch",
                    details={"text_len": len(cleaned)},
                )
                return
            self._speculative_handoff_started = True
            self._speculative_handoff_transcript = cleaned
            self._speculative_handoff_decision = decision
            done_event = self._speculative_handoff_done
            specialist_conversation = self.runtime.tool_provider_conversation_context()
        self._trace_event(
            "speculative_handoff_worker_started",
            kind="queue",
            details={"text_len": len(cleaned), "kind": getattr(decision, "kind", None)},
        )
        worker = Thread(
            target=self._speculative_handoff_worker,
            args=(cleaned, decision, specialist_conversation, done_event),
            daemon=True,
        )
        worker.start()

    def _speculative_handoff_worker(
        self,
        transcript: str,
        decision: SupervisorDecision,
        specialist_conversation,
        done_event: Event,
    ) -> None:
        result = None
        try:
            result = self.streaming_turn_loop.run_handoff_only(
                transcript,
                conversation=specialist_conversation,
                specialist_conversation=specialist_conversation,
                handoff=decision,
                instructions=None,
                allow_web_search=False,
                on_text_delta=None,
                on_lane_text_delta=None,
                emit_filler=False,
            )
        except Exception as exc:
            self.emit(f"speculative_handoff_failed={type(exc).__name__}")
            self._trace_event(
                "speculative_handoff_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with self._speculative_handoff_lock:
                if self._speculative_handoff_transcript == transcript:
                    self._speculative_handoff_result = result
            done_event.set()
            self._trace_event(
                "speculative_handoff_worker_completed",
                kind="queue",
                details={"has_result": result is not None, "text_len": len(transcript)},
            )

    def _consume_speculative_handoff_result(
        self,
        transcript: str,
        decision: SupervisorDecision | None,
        *,
        wait_for_completion: bool,
    ):
        if decision is None or decision.action != "handoff":
            return None
        with self._speculative_handoff_lock:
            if not self._speculative_handoff_started:
                return None
            done_event = self._speculative_handoff_done
            seeded_transcript = self._speculative_handoff_transcript
            seeded_decision = self._speculative_handoff_decision
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return None
        if not (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        ):
            return None
        if seeded_decision is None:
            return None
        if str(getattr(seeded_decision, "kind", "") or "").strip().lower() != str(getattr(decision, "kind", "") or "").strip().lower():
            return None
        if wait_for_completion and not done_event.is_set():
            done_event.wait()
        with self._speculative_handoff_lock:
            result = self._speculative_handoff_result
        self._trace_event(
            "speculative_handoff_consumed",
            kind="cache",
            details={
                "wait_for_completion": wait_for_completion,
                "has_result": result is not None,
                "seed_len": len(seeded_transcript),
                "final_len": len(transcript),
            },
        )
        return result

    def _prime_supervisor_decision_cache(self) -> None:
        if self._supervisor_cache_prewarmed:
            return
        if not isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        provider = getattr(self.streaming_turn_loop, "supervisor_decision_provider", None)
        if provider is None:
            return
        try:
            provider.decide(
                "Sag bitte nur kurz Hallo.",
                conversation=(),
                instructions=self.streaming_turn_loop.supervisor_instructions,
            )
            self._supervisor_cache_prewarmed = True
            self._trace_event("supervisor_cache_prewarmed", kind="cache", details={})
        except Exception as exc:
            self.emit(f"supervisor_cache_prewarm_failed={type(exc).__name__}")
            self._trace_event(
                "supervisor_cache_prewarm_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def _prime_first_word_cache(self) -> None:
        if self._first_word_cache_prewarmed:
            return
        if isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        provider = self.first_word_provider
        if provider is None:
            return
        try:
            provider.reply(
                "Sag bitte nur kurz Hallo.",
                conversation=(),
                instructions=None,
            )
            self._first_word_cache_prewarmed = True
            self._trace_event("first_word_cache_prewarmed", kind="cache", details={})
        except Exception as exc:
            self.emit(f"first_word_cache_prewarm_failed={type(exc).__name__}")
            self._trace_event(
                "first_word_cache_prewarm_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def _generate_first_word_reply(self, transcript: str) -> FirstWordReply | None:
        if not self.config.streaming_first_word_enabled:
            return None
        provider = self.first_word_provider
        if provider is None:
            return None
        try:
            reply = provider.reply(
                transcript,
                conversation=self.runtime.first_word_provider_conversation_context(),
                instructions=None,
            )
            self._trace_event(
                "first_word_reply_generated",
                kind="llm_call",
                details={"text_len": len(transcript), "mode": getattr(reply, "mode", None)},
            )
            return reply
        except Exception as exc:
            self.emit(f"first_word_sync_failed={type(exc).__name__}")
            self._trace_event(
                "first_word_reply_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
            return None

    def _streaming_turn_timeout_policy(self) -> StreamingTurnTimeoutPolicy:
        """Build the bounded timeout policy for one parallel dual-lane turn."""

        watchdog_ms = max(25, int(self.config.streaming_final_lane_watchdog_timeout_ms))
        hard_timeout_ms = max(watchdog_ms, int(self.config.streaming_final_lane_hard_timeout_ms))
        return StreamingTurnTimeoutPolicy(
            bridge_reply_timeout_ms=max(0, int(self.config.streaming_bridge_reply_timeout_ms)),
            final_lane_watchdog_timeout_ms=watchdog_ms,
            final_lane_hard_timeout_ms=hard_timeout_ms,
            first_audio_gate_ms=max(0, int(self.config.streaming_first_word_final_lane_wait_ms)),
        )

    def _dual_lane_bridge_reply_from_decision(
        self,
        prefetched_decision: SupervisorDecision | None,
    ) -> FirstWordReply | None:
        """Build a bridge reply directly from a resolved supervisor decision."""

        fallback_text = ""
        fallback_mode = "filler"
        if prefetched_decision is not None:
            action = str(getattr(prefetched_decision, "action", "") or "").strip().lower()
            if action in {"direct", "end_conversation"}:
                fallback_text = str(
                    getattr(prefetched_decision, "spoken_reply", None)
                    or getattr(prefetched_decision, "spoken_ack", None)
                    or ""
                ).strip()
                fallback_mode = "direct" if fallback_text else fallback_mode
            elif action == "handoff":
                fallback_text = str(getattr(prefetched_decision, "spoken_ack", None) or "").strip()
        if not fallback_text:
            return None
        return FirstWordReply(
            mode=fallback_mode,
            spoken_text=fallback_text,
            response_id=getattr(prefetched_decision, "response_id", None),
            request_id=getattr(prefetched_decision, "request_id", None),
            model=getattr(prefetched_decision, "model", None),
            token_usage=getattr(prefetched_decision, "token_usage", None),
        )

    def _dual_lane_bridge_fallback_reply(
        self,
        prefetched_decision: SupervisorDecision | None,
    ) -> FirstWordReply | None:
        """Build a bounded bridge reply for stalled first-word or search turns."""

        decision_reply = self._dual_lane_bridge_reply_from_decision(prefetched_decision)
        if decision_reply is not None:
            return decision_reply
        if not isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return None
        fallback_text = str(getattr(self.streaming_turn_loop, "default_spoken_ack", "") or "").strip()
        if not fallback_text:
            return None
        return FirstWordReply(
            mode="filler",
            spoken_text=fallback_text,
            response_id=None,
            request_id=None,
            model=None,
            token_usage=None,
        )

    def _dual_lane_error_reply(self) -> str:
        """Return the localized fallback reply for bounded lane failures."""

        if isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return str(getattr(self.streaming_turn_loop, "default_error_reply", "") or "").strip()
        return "Das hat gerade nicht geklappt. Bitte versuche es noch einmal."

    def _run_dual_lane_final_response(
        self,
        transcript: str,
        *,
        turn_instructions: str | None,
        prefetched_decision: SupervisorDecision | None = None,
    ):
        if prefetched_decision is None:
            prefetched_decision = self._consume_speculative_supervisor_decision(transcript)
        search_context = self.runtime.search_provider_conversation_context()
        supervisor_context = self.runtime.supervisor_provider_conversation_context()
        if prefetched_decision is not None and getattr(prefetched_decision, "action", None) == "handoff":
            self._trace_decision(
                "dual_lane_final_path_selected",
                question="Which final-lane execution path should run?",
                selected={"id": "prefetched_handoff", "summary": "Reuse prefetched search handoff"},
                options=[
                    {"id": "prefetched_handoff", "summary": "Reuse prefetched handoff result or rerun search worker"},
                    {"id": "resolve_supervisor", "summary": "Resolve supervisor decision synchronously"},
                    {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                ],
                context={"transcript_len": len(transcript), "prefetched": True},
                guardrails=["search_handoff_short_path"],
            )
            fast_result = self._consume_speculative_handoff_result(
                transcript,
                prefetched_decision,
                wait_for_completion=False,
            )
            if fast_result is not None:
                return fast_result
            response = self._consume_speculative_handoff_result(
                transcript,
                prefetched_decision,
                wait_for_completion=True,
            )
            if response is not None:
                return response
            return self.streaming_turn_loop.run_handoff_only(
                transcript,
                conversation=search_context,
                specialist_conversation=search_context,
                handoff=prefetched_decision,
                instructions=turn_instructions,
                allow_web_search=False,
                on_text_delta=None,
                on_lane_text_delta=None,
                emit_filler=False,
            )
        if (
            isinstance(self.streaming_turn_loop, DualLaneToolLoop)
            and getattr(self.streaming_turn_loop, "supervisor_decision_provider", None) is not None
        ):
            resolved_decision = prefetched_decision or self.streaming_turn_loop.resolve_supervisor_decision(
                transcript,
                conversation=supervisor_context,
                instructions=turn_instructions,
            )
            if resolved_decision is not None:
                action = str(getattr(resolved_decision, "action", "") or "").strip().lower()
                if action == "handoff" and str(getattr(resolved_decision, "kind", "") or "").strip().lower() == "search":
                    self._trace_decision(
                        "dual_lane_final_path_selected",
                        question="Which final-lane execution path should run?",
                        selected={"id": "resolved_search_handoff", "summary": "Run direct search handoff"},
                        options=[
                            {"id": "resolved_search_handoff", "summary": "Run search handoff only"},
                            {"id": "resolved_direct", "summary": "Run resolved direct reply"},
                            {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                        ],
                        context={"transcript_len": len(transcript), "action": action},
                        guardrails=["search_handoff_short_path"],
                    )
                    return self.streaming_turn_loop.run_handoff_only(
                        transcript,
                        conversation=search_context,
                        specialist_conversation=search_context,
                        handoff=resolved_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
                        emit_filler=False,
                    )
                if action in {"direct", "end_conversation"}:
                    self._trace_decision(
                        "dual_lane_final_path_selected",
                        question="Which final-lane execution path should run?",
                        selected={"id": action, "summary": "Run resolved direct supervisor path"},
                        options=[
                            {"id": "direct", "summary": "Speak direct answer"},
                            {"id": "end_conversation", "summary": "Speak answer and close session"},
                            {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                        ],
                        context={"transcript_len": len(transcript), "action": action},
                    )
                    return self.streaming_turn_loop.run(
                        transcript,
                        conversation=search_context,
                        supervisor_conversation=supervisor_context,
                        prefetched_decision=resolved_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
                    )
        self._trace_decision(
            "dual_lane_final_path_selected",
            question="Which final-lane execution path should run?",
            selected={"id": "generic_tool_loop", "summary": "Run generic tool loop"},
            options=[
                {"id": "prefetched_handoff", "summary": "Reuse search handoff"},
                {"id": "resolved_supervisor", "summary": "Resolve supervisor route"},
                {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
            ],
            context={"transcript_len": len(transcript), "prefetched": prefetched_decision is not None},
        )
        return self.streaming_turn_loop.run(
            transcript,
            conversation=self.runtime.tool_provider_conversation_context(),
            supervisor_conversation=supervisor_context,
            prefetched_decision=prefetched_decision,
            instructions=turn_instructions,
            allow_web_search=False,
            on_text_delta=None,
            on_lane_text_delta=None,
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
        turn_started = time.monotonic()
        self._trace_event(
            "streaming_audio_turn_started",
            kind="span_start",
            details={
                "initial_source": initial_source,
                "follow_up": follow_up,
                "listen_source": listen_source,
                "play_initial_beep": play_initial_beep,
            },
        )
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
                self._trace_event(
                    "streaming_audio_capture_started",
                    kind="span_start",
                    details={
                        "streaming_stt": isinstance(self.stt_provider, StreamingSpeechToTextProvider),
                        "speech_start_chunks": speech_start_chunks,
                        "ignore_initial_ms": ignore_initial_ms,
                    },
                )
                if isinstance(self.stt_provider, StreamingSpeechToTextProvider):
                    try:
                        self._trace_event("streaming_audio_capture_path", kind="decision", details={"path": "streaming_stt"})
                        capture_result, transcript, capture_ms, stt_ms, _turn_label = self._capture_and_transcribe_streaming(
                            listening_window=listening_window,
                            speech_start_chunks=speech_start_chunks,
                            ignore_initial_ms=ignore_initial_ms,
                        )
                    except RuntimeError as exc:
                        if self._is_no_speech_timeout(exc):
                            raise
                        self.emit(f"turn_controller_fallback={type(exc).__name__}")
                        self._trace_event(
                            "streaming_audio_capture_fallback",
                            kind="branch",
                            level="WARN",
                            details={"error_type": type(exc).__name__, "path": "streaming_stt"},
                        )
                        capture_result = self.recorder.capture_pcm_until_pause_with_options(
                            pause_ms=listening_window.speech_pause_ms,
                            start_timeout_s=listening_window.start_timeout_s,
                            speech_start_chunks=speech_start_chunks,
                            ignore_initial_ms=ignore_initial_ms,
                            pause_grace_ms=listening_window.pause_grace_ms,
                            should_stop=self._active_turn_stop_requested,
                        )
                        capture_ms = int((time.monotonic() - capture_started) * 1000)
                        stt_ms = -1
                        transcript = ""
                    except Exception as exc:
                        self.emit(f"turn_controller_fallback={type(exc).__name__}")
                        self._trace_event(
                            "streaming_audio_capture_fallback",
                            kind="branch",
                            level="WARN",
                            details={"error_type": type(exc).__name__, "path": "streaming_stt"},
                        )
                        capture_result = self.recorder.capture_pcm_until_pause_with_options(
                            pause_ms=listening_window.speech_pause_ms,
                            start_timeout_s=listening_window.start_timeout_s,
                            speech_start_chunks=speech_start_chunks,
                            ignore_initial_ms=ignore_initial_ms,
                            pause_grace_ms=listening_window.pause_grace_ms,
                            should_stop=self._active_turn_stop_requested,
                        )
                        capture_ms = int((time.monotonic() - capture_started) * 1000)
                        stt_ms = -1
                        transcript = ""
                else:
                    self._trace_event("streaming_audio_capture_path", kind="decision", details={"path": "recorder_only"})
                    capture_result = self.recorder.capture_pcm_until_pause_with_options(
                        pause_ms=listening_window.speech_pause_ms,
                        start_timeout_s=listening_window.start_timeout_s,
                        speech_start_chunks=speech_start_chunks,
                        ignore_initial_ms=ignore_initial_ms,
                        pause_grace_ms=listening_window.pause_grace_ms,
                        should_stop=self._active_turn_stop_requested,
                    )
                    capture_ms = int((time.monotonic() - capture_started) * 1000)
                    stt_ms = -1
                    transcript = ""
                self._trace_event(
                    "streaming_audio_capture_completed",
                    kind="span_end",
                    details={
                        "pcm_bytes": len(capture_result.pcm_bytes),
                        "speech_started_after_ms": capture_result.speech_started_after_ms,
                        "resumed_after_pause_count": capture_result.resumed_after_pause_count,
                    },
                    kpi={"duration_ms": capture_ms},
                )
        except RuntimeError as exc:
            if self._active_turn_stop_requested():
                self._cancel_interrupted_turn()
                self._trace_event("streaming_audio_turn_interrupted_during_capture", kind="branch", details={})
                return False
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.remember_listen_timeout(
                initial_source=initial_source,
                follow_up=follow_up,
            )
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self.emit(f"{timeout_emit_key}=true")
            self._record_event("listen_timeout", timeout_message, request_source=listen_source)
            self._trace_event(
                "streaming_audio_turn_timeout_no_speech",
                kind="warning",
                level="WARN",
                details={"timeout_emit_key": timeout_emit_key, "listen_source": listen_source},
            )
            return False
        audio_pcm = capture_result.pcm_bytes
        self.runtime.remember_listen_capture(
            initial_source=initial_source,
            follow_up=follow_up,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        self._update_voice_assessment_from_pcm(audio_pcm)

        recorder_sample_rate = self._recorder_sample_rate()
        self._current_turn_audio_pcm = audio_pcm
        self._current_turn_audio_sample_rate = recorder_sample_rate
        try:
            if stt_ms < 0:
                audio_bytes = pcm16_to_wav_bytes(
                    audio_pcm,
                    sample_rate=recorder_sample_rate,
                    channels=self.config.audio_channels,
                )
                stt_started = time.monotonic()
                self._trace_event(
                    "streaming_batch_stt_started",
                    kind="llm_call",
                    details={"audio_bytes": len(audio_bytes)},
                )
                transcript = self.stt_provider.transcribe(
                    audio_bytes,
                    filename="twinr-streaming-listen.wav",
                    content_type="audio/wav",
                ).strip()
                stt_ms = int((time.monotonic() - stt_started) * 1000)
                self._trace_event(
                    "streaming_batch_stt_completed",
                    kind="llm_call",
                    details={"transcript_len": len(transcript)},
                    kpi={"duration_ms": stt_ms},
                )
            if not transcript:
                self._trace_event(
                    "streaming_transcript_empty_after_stt",
                    kind="warning",
                    level="WARN",
                    details={"capture_ms": capture_ms, "stt_ms": stt_ms},
                )
                raise RuntimeError("Speech-to-text returned an empty transcript")
            self.emit(f"transcript={transcript}")
            self._trace_event(
                "streaming_transcript_ready",
                kind="observation",
                details={"transcript_len": len(transcript), "capture_ms": capture_ms, "stt_ms": stt_ms},
            )
            return self._complete_streaming_turn(
                transcript=transcript,
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
            )
        finally:
            self._current_turn_audio_pcm = None
            self._trace_event(
                "streaming_audio_turn_finished",
                kind="span_end",
                kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
            )

    def _capture_and_transcribe_streaming(
        self,
        *,
        listening_window,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
    ):
        return self._capture_and_transcribe_with_turn_controller(
            stt_provider=self.stt_provider,
            listening_window=listening_window,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
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
    ) -> bool:
        self._trace_event(
            "streaming_turn_completion_started",
            kind="span_start",
            details={"listen_source": listen_source, "proactive_trigger": proactive_trigger, "transcript_len": len(transcript)},
        )
        self.runtime.submit_transcript(transcript)
        self._emit_status(force=True)
        stop_processing_feedback: Callable[[], None] = lambda: None
        processing_feedback_started = False

        def ensure_processing_feedback() -> None:
            nonlocal stop_processing_feedback, processing_feedback_started
            if processing_feedback_started:
                return
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            processing_feedback_started = True
            self._trace_event("streaming_processing_feedback_started", kind="io", details={})

        first_audio_at: list[float | None] = [None]
        answer_started = False

        def queue_ready_segments(delta: str) -> None:
            self._trace_event("streaming_text_delta_queued", kind="queue", details={"delta_len": len(delta)})
            speech_output.submit_text_delta(delta)

        def queue_lane_segments(delta: SpeechLaneDelta) -> None:
            self._trace_event(
                "streaming_lane_delta_queued",
                kind="queue",
                details={"delta_len": len(delta.text), "replace_current": delta.replace_current, "atomic": delta.atomic},
            )
            speech_output.submit_lane_delta(delta)

        def mark_answering_started() -> None:
            nonlocal answer_started
            if answer_started:
                return
            if self.runtime.status.value == "answering":
                answer_started = True
                stop_processing_feedback()
                return
            stop_processing_feedback()
            self.runtime.begin_answering()
            self._emit_status(force=True)
            answer_started = True
            self._trace_event("streaming_answering_started", kind="mutation", details={})

        def mark_first_audio() -> None:
            if first_audio_at[0] is None:
                first_audio_at[0] = time.monotonic()
                self._trace_event(
                    "streaming_first_audio_observed",
                    kind="metric",
                    kpi={"duration_ms": round((first_audio_at[0] - turn_started) * 1000.0, 3)},
                )

        speech_output = InterruptibleSpeechOutput(
            tts_provider=self.tts_provider,
            player=self.player,
            chunk_size=max(512, int(self.config.openai_tts_stream_chunk_size)),
            segment_boundary=self._segment_boundary,
            on_speaking_started=mark_answering_started,
            on_first_audio=mark_first_audio,
            on_preempt=lambda: self.emit("tts_lane_preempted=true"),
            playback_lock=self._audio_lock,
            should_stop=self._active_turn_stop_requested,
            trace_event=lambda msg, details=None: self._trace_event(msg, details=details),
        )
        self._trace_event(
            "streaming_speech_output_created",
            kind="queue",
            details={"chunk_size": max(512, int(self.config.openai_tts_stream_chunk_size))},
        )

        llm_started = time.monotonic()
        first_word_reply: FirstWordReply | None = None
        first_word_source = "none"
        try:
            if isinstance(self.streaming_turn_loop, DualLaneToolLoop):
                turn_instructions = None
            else:
                turn_instructions = (
                    build_compact_tool_agent_instructions(
                        self.config,
                        extra_instructions=self.config.openai_realtime_instructions,
                    )
                    if (self.config.llm_provider or "").strip().lower() == "groq"
                    else build_tool_agent_instructions(
                        self.config,
                        extra_instructions=self.config.openai_realtime_instructions,
                    )
                )
            if isinstance(self.streaming_turn_loop, DualLaneToolLoop):
                self._maybe_start_speculative_supervisor_decision(transcript)
                prefetched_decision = self._consume_speculative_supervisor_decision(transcript)
                first_word_reply = self._dual_lane_bridge_reply_from_decision(prefetched_decision)
                if first_word_reply is not None:
                    first_word_source = "supervisor_prefetched"
                self._trace_event(
                    "streaming_dual_lane_starting",
                    kind="decision",
                    details={
                        "prefetched_decision": getattr(prefetched_decision, "action", None) if prefetched_decision is not None else None,
                        "prefetched_first_word": first_word_reply is not None,
                    },
                )
                orchestrator = StreamingTurnOrchestrator(
                    timeout_policy=self._streaming_turn_timeout_policy(),
                    queue_lane_delta=queue_lane_segments,
                    wait_for_first_audio=speech_output.wait_for_first_audio,
                    ensure_processing_feedback=ensure_processing_feedback,
                    emit=self.emit,
                    should_stop=self._active_turn_stop_requested,
                )
                try:
                    lane_outcome = orchestrator.execute(
                        prefetched_first_word=first_word_reply,
                        prefetched_first_word_source=first_word_source,
                        generate_first_word=None,
                        bridge_fallback_reply=self._dual_lane_bridge_fallback_reply(prefetched_decision),
                        run_final_lane=lambda: self._run_dual_lane_final_response(
                            transcript,
                            turn_instructions=turn_instructions,
                            prefetched_decision=prefetched_decision,
                        ),
                        final_timeout_reply=self._dual_lane_error_reply(),
                        final_error_reply=self._dual_lane_error_reply(),
                    )
                except InterruptedError:
                    self._cancel_interrupted_turn()
                    return False
                first_word_reply = lane_outcome.first_word_reply
                first_word_source = lane_outcome.first_word_source
                if first_word_reply is not None:
                    self.emit(f"first_word_mode={first_word_reply.mode}")
                    self.emit(f"first_word_source={first_word_source}")
                    self._trace_event(
                        "streaming_first_word_selected",
                        kind="decision",
                        details={"mode": first_word_reply.mode, "source": first_word_source},
                    )
                response = lane_outcome.response
            else:
                ensure_processing_feedback()
                self._trace_event("streaming_single_lane_tool_loop_starting", kind="decision", details={})
                response = self.streaming_turn_loop.run(
                    transcript,
                    conversation=self.runtime.tool_provider_conversation_context(),
                    instructions=turn_instructions,
                    allow_web_search=False,
                    on_text_delta=queue_ready_segments,
                )
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            self._trace_event(
                "streaming_llm_lane_completed",
                kind="llm_call",
                details={"response_text_len": len(response.text), "tool_calls": len(response.tool_calls)},
                kpi={"duration_ms": llm_ms},
            )
            if self._active_turn_stop_requested():
                self._cancel_interrupted_turn()
                self._trace_event("streaming_turn_interrupted_after_llm", kind="branch", details={})
                return False
            if not response.text.strip():
                self._trace_event("streaming_turn_empty_response", kind="warning", level="WARN", details={})
                raise RuntimeError("Streaming tool loop completed without text output")
            if self.runtime.status.value == "printing":
                self.runtime.resume_answering_after_print()
                self._emit_status(force=True)
                answer_started = True
                self._trace_event("streaming_print_resume_answering", kind="mutation", details={})
            answer = self.runtime.finalize_agent_turn(response.text)
            self._trace_event(
                "streaming_agent_turn_finalized",
                kind="mutation",
                details={"answer_len": len(answer), "response_text_len": len(response.text)},
            )
            speech_output.flush()
        finally:
            try:
                close_timeout_s = max(
                    0.1,
                    float(getattr(self.config, "tts_worker_join_timeout_s", 30.0)),
                )
            except (TypeError, ValueError):
                close_timeout_s = 30.0
            interrupted = self._active_turn_stop_requested()
            if interrupted:
                stopped = speech_output.abort(timeout_s=min(0.25, close_timeout_s))
                if not stopped:
                    self.emit("speech_output_abort_timeout=true")
                self._trace_event(
                    "streaming_speech_output_abort_path",
                    kind="branch",
                    details={"stopped": stopped, "close_timeout_s": close_timeout_s},
                )
            else:
                speech_output.close(
                    timeout_s=close_timeout_s,
                )
                self._trace_event(
                    "streaming_speech_output_close_path",
                    kind="branch",
                    details={"close_timeout_s": close_timeout_s},
                )
            stop_processing_feedback()

        if self._active_turn_stop_requested():
            self._cancel_interrupted_turn()
            self._trace_event("streaming_turn_interrupted_after_playback", kind="branch", details={})
            return False
        speech_output.raise_if_error()
        if not answer_started:
            stop_processing_feedback()
            self.runtime.begin_answering()
            self._emit_status(force=True)
            self._trace_event("streaming_answering_started_late", kind="mutation", details={})

        self.emit(f"response={answer}")
        if response.response_id:
            self.emit(f"agent_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"agent_request_id={response.request_id}")
        self.emit(f"agent_tool_rounds={response.rounds}")
        self.emit(f"agent_tool_calls={len(response.tool_calls)}")
        self.emit(f"agent_used_web_search={str(response.used_web_search).lower()}")
        self._record_usage(
            request_kind="conversation",
            source="streaming_loop",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=response.used_web_search,
            token_usage=response.token_usage,
            transcript=transcript,
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
            tool_rounds=response.rounds,
            tool_calls=len(response.tool_calls),
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_llm_ms={llm_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        end_conversation = any(call.name == "end_conversation" for call in response.tool_calls)
        force_close = end_conversation or self._follow_up_vetoed_by_closure(
            user_transcript=transcript,
            assistant_response=answer,
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        if end_conversation:
            self.emit("conversation_ended=true")
        elif force_close:
            self.emit("conversation_follow_up_vetoed=closure")
        self._trace_decision(
            "streaming_follow_up_decision",
            question="Should the session remain open for a follow-up turn?",
            selected={"id": "close" if force_close else "continue", "summary": "close session" if force_close else "keep session open"},
            options=[
                {"id": "continue", "summary": "Allow follow-up listening"},
                {"id": "close", "summary": "Close session now"},
            ],
            context={
                "end_conversation_tool": end_conversation,
                "listen_source": listen_source,
                "proactive_trigger": proactive_trigger,
            },
            guardrails=["closure_evaluator", "end_conversation_tool"],
        )
        self._trace_event(
            "streaming_turn_completion_finished",
            kind="span_end",
            details={"force_close": force_close, "end_conversation": end_conversation},
            kpi={
                "capture_ms": capture_ms,
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "total_ms": round((time.monotonic() - turn_started) * 1000.0, 3),
            },
        )
        return not force_close

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
