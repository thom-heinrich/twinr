# CHANGELOG: 2026-03-28
# BUG-1: Fixed partial provider auto-resolution so custom tool providers no longer leave STT/agent/TTS/print dependencies unset.
# BUG-2: Fixed live-config reload to refresh derived turn/verifier providers, recorder state, and voice-orchestrator recovery on coordinator failures.
# BUG-3: Fixed unsafe sentence segmentation that split on decimals/abbreviations, causing incorrect or confusing streamed TTS output.
# BUG-4: Seeded text turns now reuse an already-open `listening` state so remote
#        follow-up transcript commits cannot reopen listening and surface a false
#        runtime `error` after follow-up rearm.
# SEC-1: Hardened live .env reload against symlink/permission attacks and reduced TOCTOU risk by loading from a validated file snapshot.
# IMP-1: Added transactional live reload with reconfiguration locking and managed-component rebuilds for stable long-running Pi deployments.
# IMP-2: Added async speculative cache warmup with bounded join and speculation locking to cut cold-start latency without racing cache state.
# IMP-3: Exposed dual-lane round budget through config and improved tracing around warmup/reload/failure paths.
# BUG-5: Re-prewarm the processing feedback media clip after live config reloads so the dragon THINKING cue does
#        not regress back to a cold-cache tone fallback on the next turn.

"""Run the speculative streaming workflow with dual-lane tool orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
import os
import stat
import tempfile
import threading
import time
from typing import Any, Callable, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.follow_up_context import (
    pending_conversation_follow_up_hint_scope,
)
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
)
from twinr.agent.workflows.forensics import current_workflow_trace_id
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
from twinr.agent.workflows.voice_turn_latency import emit_voice_turn_latency_breakdown
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.ops.process_memory import record_streaming_memory_phase_best_effort
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai import (
    OpenAIBackend,
    OpenAIFirstWordProvider,
    OpenAIProviderBundle,
    OpenAISupervisorDecisionProvider,
    OpenAIToolCallingAgentProvider,
)


@dataclass
class _ResolvedStreamingDependencies:
    kwargs: dict[str, Any]
    tool_agent_provider: ToolCallingAgentProvider | None
    verification_stt_provider: Any | None
    managed_components: dict[str, bool]


class _StreamingSessionPlaceholder:
    """Carry config for the realtime parent without opening a realtime session."""

    def __init__(self, config: TwinrConfig) -> None:
        self.config = config


class TwinrStreamingHardwareLoop(TwinrRealtimeHardwareLoop):
    """Extend the realtime loop with streaming STT and speculative speech.

    The streaming loop adds dual-lane tool execution, interruptible speech
    output, and speculative first-word and supervisor decision warmups.
    """

    _COMMON_NON_TERMINAL_ABBREVIATIONS = frozenset(
        {
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "prof.",
            "sr.",
            "jr.",
            "st.",
            "vs.",
            "etc.",
            "e.g.",
            "i.e.",
            "ca.",
            "z.b.",
            "bzw.",
            "d.h.",
            "u.a.",
            "u.s.w.",
            "nr.",
            "bsp.",
        }
    )

    @staticmethod
    def _record_constructor_memory_phase(
        config: TwinrConfig,
        *,
        label: str,
        owner_label: str,
        owner_detail: str,
    ) -> None:
        """Attach one best-effort constructor checkpoint to the streaming PID snapshot."""

        record_streaming_memory_phase_best_effort(
            config,
            label=label,
            owner_label=owner_label,
            owner_detail=owner_detail,
        )

    def __init__(
        self,
        config: TwinrConfig,
        *,
        tool_agent_provider: ToolCallingAgentProvider | None = None,
        streaming_turn_loop: ToolCallingStreamingLoop | None = None,
        verification_stt_provider=None,
        **kwargs,
    ) -> None:
        self._reconfigure_lock = threading.RLock()
        self._speculation_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._warmup_generation = 0
        self._warmup_futures: dict[str, Future[Any]] = {}
        self._warmup_executor: ThreadPoolExecutor | None = None
        self._managed_components: dict[str, bool] = {}
        self._managed_streaming_turn_loop = streaming_turn_loop is None
        self._supervisor_cache_prewarmed = False
        self._first_word_cache_prewarmed = False

        self._record_constructor_memory_phase(
            config,
            label="streaming_loop.hardware_loop.constructor_entered",
            owner_label="streaming_loop.hardware_loop.constructor",
            owner_detail="TwinrStreamingHardwareLoop constructor entered before runtime dependency resolution.",
        )
        resolved_dependencies = self._resolve_runtime_dependencies(
            config=config,
            tool_agent_provider=tool_agent_provider,
            verification_stt_provider=verification_stt_provider,
            kwargs=kwargs,
        )
        kwargs = resolved_dependencies.kwargs
        verifier_provider = resolved_dependencies.verification_stt_provider
        resolved_tool_agent = resolved_dependencies.tool_agent_provider
        self._managed_components = resolved_dependencies.managed_components
        if resolved_tool_agent is None:
            raise ValueError("TwinrStreamingHardwareLoop requires a tool-capable agent provider")
        self._record_constructor_memory_phase(
            config,
            label="streaming_loop.hardware_loop.dependencies_resolved",
            owner_label="streaming_loop.hardware_loop.dependencies",
            owner_detail="TwinrStreamingHardwareLoop resolved its runtime dependency bundle and managed-component map.",
        )

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
        self._record_constructor_memory_phase(
            config,
            label="streaming_loop.hardware_loop.super_init_ready",
            owner_label="streaming_loop.hardware_loop.super_init",
            owner_detail="TwinrStreamingHardwareLoop finished the shared realtime bootstrap super().__init__ path.",
        )
        self.tool_agent_provider = resolved_tool_agent
        self.verification_stt_provider = verifier_provider
        tool_schemas = self._build_runtime_tool_schemas()
        self.streaming_turn_loop = streaming_turn_loop or self._build_streaming_turn_loop(
            tool_schemas=tool_schemas,
        )
        self._record_constructor_memory_phase(
            config,
            label="streaming_loop.hardware_loop.turn_loop_ready",
            owner_label="streaming_loop.hardware_loop.turn_loop",
            owner_detail="TwinrStreamingHardwareLoop built the streaming turn loop and runtime tool schemas.",
        )
        self.first_word_provider: FirstWordProvider | None = getattr(self, "first_word_provider", None)
        self._streaming_capture = StreamingCaptureController(self)
        self._streaming_speculation = StreamingSpeculationController(self)
        self._streaming_lane_planner = StreamingLanePlanner(self)
        self._streaming_semantic_router = StreamingSemanticRouterRuntime(self)
        self._schedule_speculative_warmups()
        self._record_constructor_memory_phase(
            config,
            label="streaming_loop.hardware_loop.streaming_features_ready",
            owner_label="streaming_loop.hardware_loop.streaming_features",
            owner_detail="TwinrStreamingHardwareLoop finished capture/speculation/lane/router startup and scheduled speculative warmups.",
        )
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

    def close(self, *, timeout_s: float = 1.0) -> None:
        """Release streaming-loop resources, including speculative warmups."""

        warmup_lock = getattr(self, "_warmup_lock", None)
        warmup_futures = getattr(self, "_warmup_futures", None)
        executor = None
        if warmup_lock is not None and isinstance(warmup_futures, dict):
            with warmup_lock:
                executor = getattr(self, "_warmup_executor", None)
                self._warmup_generation = getattr(self, "_warmup_generation", 0) + 1
                for future in warmup_futures.values():
                    future.cancel()
                warmup_futures.clear()
                self._warmup_executor = None
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        super().close(timeout_s=timeout_s)

    def __del__(self) -> None:
        try:
            self.close(timeout_s=0.2)
        except Exception:
            pass

    def _resolve_runtime_dependencies(
        self,
        *,
        config: TwinrConfig,
        tool_agent_provider: ToolCallingAgentProvider | None,
        verification_stt_provider,
        kwargs: dict[str, Any],
    ) -> _ResolvedStreamingDependencies:
        resolved_kwargs = dict(kwargs)
        managed_components = {
            "print_backend": False,
            "stt_provider": False,
            "agent_provider": False,
            "tts_provider": False,
            "tool_agent_provider": False,
            "recorder": False,
            "verification_stt_provider": False,
        }

        need_bundle = tool_agent_provider is None or any(
            resolved_kwargs.get(name) is None
            for name in ("print_backend", "stt_provider", "agent_provider", "tts_provider")
        )
        provider_bundle = build_streaming_provider_bundle(config) if need_bundle else None
        if provider_bundle is not None:
            for kwarg_name, bundle_name in (
                ("print_backend", "print_backend"),
                ("stt_provider", "stt"),
                ("agent_provider", "agent"),
                ("tts_provider", "tts"),
            ):
                if resolved_kwargs.get(kwarg_name) is None:
                    resolved_kwargs[kwarg_name] = getattr(provider_bundle, bundle_name)
                    managed_components[kwarg_name] = True
            if tool_agent_provider is None:
                tool_agent_provider = provider_bundle.tool_agent
                managed_components["tool_agent_provider"] = True

        stt_provider = resolved_kwargs.get("stt_provider")
        if (
            verification_stt_provider is None
            and self._should_enable_streaming_transcript_verifier(config, stt_provider)
        ):
            verification_stt_provider = self._build_streaming_transcript_verifier_provider(config)
            managed_components["verification_stt_provider"] = True

        if resolved_kwargs.get("recorder") is None and self._should_use_deepgram_recorder(config):
            resolved_kwargs["recorder"] = self._build_default_deepgram_recorder(config)
            managed_components["recorder"] = True

        return _ResolvedStreamingDependencies(
            kwargs=resolved_kwargs,
            tool_agent_provider=tool_agent_provider,
            verification_stt_provider=verification_stt_provider,
            managed_components=managed_components,
        )

    @staticmethod
    def _should_use_deepgram_recorder(config: TwinrConfig) -> bool:
        return (config.stt_provider or "").strip().lower() == "deepgram"

    @staticmethod
    def _build_default_deepgram_recorder(config: TwinrConfig) -> SilenceDetectedRecorder:
        return SilenceDetectedRecorder(
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

    @staticmethod
    def _build_streaming_transcript_verifier_provider(config: TwinrConfig):
        verifier_backend = OpenAIBackend(
            config=replace(
                config,
                openai_stt_model=config.streaming_transcript_verifier_model,
            )
        )
        return OpenAIProviderBundle.from_backend(verifier_backend).stt

    @staticmethod
    def _resolve_turn_stt_provider(stt_provider):
        if isinstance(stt_provider, StreamingSpeechToTextProvider):
            return stt_provider
        return None

    @staticmethod
    def _should_enable_streaming_transcript_verifier(config: TwinrConfig, stt_provider) -> bool:
        if not bool(getattr(config, "streaming_transcript_verifier_enabled", True)):
            return False
        if not (config.openai_api_key or "").strip():
            return False
        if (config.stt_provider or "").strip().lower() != "deepgram":
            return False
        return isinstance(stt_provider, StreamingSpeechToTextProvider)

    def _streaming_tool_max_rounds(self) -> int:
        return max(1, int(getattr(self.config, "streaming_tool_max_rounds", 6)))

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
                max_rounds=self._streaming_tool_max_rounds(),
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

    def _allow_insecure_live_env_reload(self) -> bool:
        return os.getenv("TWINR_ALLOW_INSECURE_LIVE_ENV_RELOAD", "").strip() == "1"

    def _read_live_env_payload(self, env_path: Path) -> str:
        candidate = env_path.expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Live config path does not exist: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"Live config path must be a regular file: {candidate}")

        insecure_reload_allowed = self._allow_insecure_live_env_reload()
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if not insecure_reload_allowed and hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(os.fspath(candidate), flags)
        except OSError as exc:
            if not insecure_reload_allowed and candidate.is_symlink():
                raise PermissionError(
                    f"Live config file must not be a symlink: {candidate}"
                ) from exc
            raise
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"Live config path must be a regular file: {candidate}")
            if not insecure_reload_allowed:
                if hasattr(os, "getuid") and file_stat.st_uid not in {0, os.getuid()}:
                    raise PermissionError(
                        f"Live config file must be owned by root or the current user: {candidate}"
                    )
                if stat.S_IMODE(file_stat.st_mode) & 0o022:
                    raise PermissionError(
                        f"Live config file must not be group/world writable: {candidate}"
                    )
                if not hasattr(os, "O_NOFOLLOW") and candidate.is_symlink():
                    raise PermissionError(
                        f"Live config file must not be a symlink: {candidate}"
                    )
            with os.fdopen(fd, "r", encoding="utf-8", errors="surrogateescape") as handle:
                fd = -1
                return handle.read()
        finally:
            if fd >= 0:
                os.close(fd)

    # BREAKING: live config reload now rejects symlinked, foreign-owned, or group/world-writable
    # .env files by default. Set TWINR_ALLOW_INSECURE_LIVE_ENV_RELOAD=1 only for controlled legacy setups.
    def _load_live_config_from_env_file(self, env_path: Path) -> TwinrConfig:
        payload = self._read_live_env_payload(env_path)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".env",
                prefix="twinr-live-config-",
                encoding="utf-8",
                errors="surrogateescape",
                delete=False,
            ) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            return TwinrConfig.from_env(temp_path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def _snapshot_reload_state(self) -> dict[str, Any]:
        fields = (
            "config",
            "print_backend",
            "stt_provider",
            "agent_provider",
            "tts_provider",
            "recorder",
            "tool_agent_provider",
            "turn_stt_provider",
            "turn_tool_agent_provider",
            "verification_stt_provider",
            "turn_decision_evaluator",
            "streaming_turn_loop",
            "first_word_provider",
            "_supervisor_cache_prewarmed",
            "_first_word_cache_prewarmed",
        )
        return {name: getattr(self, name, None) for name in fields}

    def _restore_reload_state(self, snapshot: dict[str, Any]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, value)
        self.runtime.apply_live_config(snapshot["config"])
        self._update_provider_configs(snapshot["config"])
        self._apply_recorder_live_config(snapshot["config"])
        self.realtime_session.config = snapshot["config"]
        self._refresh_runtime_tool_surface()
        self._streaming_semantic_router.reload()

    def _prepare_reload_overrides(self, updated_config: TwinrConfig) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        needs_bundle = any(
            self._managed_components.get(name, False)
            for name in (
                "print_backend",
                "stt_provider",
                "agent_provider",
                "tts_provider",
                "tool_agent_provider",
            )
        )
        provider_bundle = build_streaming_provider_bundle(updated_config) if needs_bundle else None
        if provider_bundle is not None:
            for attr_name, bundle_attr_name in (
                ("print_backend", "print_backend"),
                ("stt_provider", "stt"),
                ("agent_provider", "agent"),
                ("tts_provider", "tts"),
            ):
                if self._managed_components.get(attr_name, False):
                    overrides[attr_name] = getattr(provider_bundle, bundle_attr_name)
            if self._managed_components.get("tool_agent_provider", False):
                overrides["tool_agent_provider"] = provider_bundle.tool_agent

        if self._managed_components.get("recorder", False) and self._should_use_deepgram_recorder(updated_config):
            overrides["recorder"] = self._build_default_deepgram_recorder(updated_config)

        effective_stt_provider = overrides.get("stt_provider", self.stt_provider)
        if self._managed_components.get("verification_stt_provider", False):
            if self._should_enable_streaming_transcript_verifier(updated_config, effective_stt_provider):
                overrides["verification_stt_provider"] = self._build_streaming_transcript_verifier_provider(
                    updated_config
                )
            else:
                overrides["verification_stt_provider"] = None

        return overrides

    def _update_provider_configs(self, updated_config: TwinrConfig) -> None:
        seen: set[int] = set()
        for provider in (
            getattr(self, "stt_provider", None),
            getattr(self, "agent_provider", None),
            getattr(self, "tts_provider", None),
            getattr(self, "print_backend", None),
            getattr(self, "first_word_provider", None),
            getattr(self, "tool_agent_provider", None),
            getattr(self, "turn_stt_provider", None),
            getattr(self, "turn_tool_agent_provider", None),
            getattr(self, "verification_stt_provider", None),
        ):
            if provider is None or not hasattr(provider, "config"):
                continue
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            provider.config = updated_config

    def _apply_recorder_live_config(self, updated_config: TwinrConfig) -> None:
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        if hasattr(recorder, "apply_live_config"):
            recorder.apply_live_config(updated_config)
            return
        if hasattr(recorder, "config"):
            recorder.config = updated_config

    def _invalidate_speculative_warmups(self) -> None:
        with self._warmup_lock:
            self._warmup_generation += 1
            for future in self._warmup_futures.values():
                future.cancel()
            self._warmup_futures.clear()

    def _speculative_warmup_join_ms(self) -> int:
        return max(0, int(getattr(self.config, "streaming_speculative_warmup_join_ms", 40)))

    def _schedule_speculative_warmups(self) -> None:
        if not self._managed_streaming_turn_loop:
            return
        if not isinstance(self.streaming_turn_loop, DualLaneToolLoop):
            return
        self._schedule_speculative_warmup(
            "supervisor_decision",
            self._prime_supervisor_decision_cache,
        )
        if self.first_word_provider is not None and bool(self.config.streaming_first_word_enabled):
            self._schedule_speculative_warmup(
                "first_word",
                self._prime_first_word_cache,
            )

    def _schedule_speculative_warmup(
        self,
        name: str,
        runner: Callable[[], None],
    ) -> None:
        with self._warmup_lock:
            existing = self._warmup_futures.get(name)
            if existing is not None and not existing.done():
                return
            generation = self._warmup_generation
            executor = self._warmup_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="twinr-streaming-warmup",
                )
                self._warmup_executor = executor
            self._warmup_futures[name] = executor.submit(
                self._run_speculative_warmup,
                generation,
                name,
                runner,
            )

    def _run_speculative_warmup(
        self,
        generation: int,
        name: str,
        runner: Callable[[], None],
    ) -> None:
        started = time.monotonic()
        self._trace_event(
            "streaming_speculative_warmup_started",
            kind="span_start",
            details={"name": name, "generation": generation},
        )
        try:
            if generation != self._warmup_generation:
                return
            runner()
            if generation != self._warmup_generation:
                return
            self._trace_event(
                "streaming_speculative_warmup_finished",
                kind="span_end",
                details={"name": name, "generation": generation},
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )
        except Exception as exc:
            self._trace_event(
                "streaming_speculative_warmup_failed",
                kind="error",
                details={"name": name, "generation": generation, "error": repr(exc)},
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )

    def _wait_for_speculative_warmup(
        self,
        name: str,
        *,
        wait_ms: int | None = None,
    ) -> None:
        timeout_ms = self._speculative_warmup_join_ms() if wait_ms is None else max(0, int(wait_ms))
        if timeout_ms <= 0:
            return
        with self._warmup_lock:
            future = self._warmup_futures.get(name)
        if future is None or future.done():
            return
        try:
            future.result(timeout=timeout_ms / 1000.0)
        except FutureTimeoutError:
            return
        except Exception:
            return

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = self._load_live_config_from_env_file(env_path)
        overrides = self._prepare_reload_overrides(updated_config)
        with self._reconfigure_lock:
            snapshot = self._snapshot_reload_state()
            self._invalidate_speculative_warmups()
            try:
                self.config = updated_config
                for attr_name, value in overrides.items():
                    setattr(self, attr_name, value)
                self.turn_tool_agent_provider = self.tool_agent_provider
                self.turn_stt_provider = self._resolve_turn_stt_provider(self.stt_provider)
                if "verification_stt_provider" in overrides:
                    self.verification_stt_provider = overrides["verification_stt_provider"]
                self.runtime.apply_live_config(updated_config)
                self._update_provider_configs(updated_config)
                self._apply_recorder_live_config(updated_config)
                self.turn_decision_evaluator = (
                    ToolCallingTurnDecisionEvaluator(
                        config=updated_config,
                        provider=self.turn_tool_agent_provider,
                    )
                    if self.turn_tool_agent_provider is not None and updated_config.turn_controller_enabled
                    else None
                )
                self._refresh_runtime_tool_surface()
                self.realtime_session.config = updated_config
                tool_schemas = self._build_runtime_tool_schemas()
                self.streaming_turn_loop = self._build_streaming_turn_loop(
                    tool_schemas=tool_schemas,
                )
                self._streaming_semantic_router.reload()
                self._reset_speculative_supervisor_decision()
                self._supervisor_cache_prewarmed = False
                self._first_word_cache_prewarmed = False
            except Exception:
                self._restore_reload_state(snapshot)
                self._schedule_speculative_warmups()
                raise
            self._schedule_speculative_warmups()
            self._prewarm_working_feedback_media("processing")

    def _build_runtime_tool_schemas(self):
        tool_names = self._runtime_tool_names
        if (self.config.llm_provider or "").strip().lower() == "groq":
            return build_compact_agent_tool_schemas(tool_names)
        return build_agent_tool_schemas(tool_names)

    def _sync_streaming_turn_loop_tool_surface(self) -> None:
        """Refresh the active streaming loop after the runtime tool surface changes."""

        streaming_turn_loop = getattr(self, "streaming_turn_loop", None)
        if streaming_turn_loop is None:
            return
        tool_schemas = tuple(self._build_runtime_tool_schemas())
        setattr(streaming_turn_loop, "tool_handlers", dict(self._tool_handlers))
        setattr(streaming_turn_loop, "tool_schemas", tool_schemas)
        reset_supervisor = getattr(self, "_reset_speculative_supervisor_decision", None)
        if callable(reset_supervisor):
            reset_supervisor()

    def _reset_speculative_supervisor_decision(self) -> None:
        with self._speculation_lock:
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
        with self._speculation_lock:
            self._streaming_speculation.maybe_start_first_word(text)

    def _consume_speculative_first_word(self, transcript: str) -> FirstWordReply | None:
        self._wait_for_speculative_warmup("first_word")
        with self._speculation_lock:
            return self._streaming_speculation.consume_first_word(transcript)

    def _maybe_start_speculative_supervisor_decision(self, text: str) -> None:
        with self._speculation_lock:
            self._streaming_speculation.maybe_start_supervisor_decision(text)

    def _maybe_start_local_semantic_router_warmup(self, text: str) -> None:
        self._streaming_semantic_router.maybe_start_warmup(text)

    def _maybe_start_speculative_long_term_context(
        self,
        text: str,
        *,
        final_transcript: bool,
    ) -> None:
        long_term_memory = getattr(getattr(self, "runtime", None), "long_term_memory", None)
        if long_term_memory is None:
            return
        prewarm_provider_context = getattr(long_term_memory, "prewarm_provider_context", None)
        if not callable(prewarm_provider_context):
            return
        prewarm_provider_context_fn = cast(Callable[..., object], prewarm_provider_context)
        # Pylint does not narrow through the callable() guard on this getattr()-resolved hook.
        # The runtime contract here is intentional and already checked above.
        # pylint: disable=not-callable
        try:
            scheduled = prewarm_provider_context_fn(
                text,
                rewrite_query=final_transcript,
                sticky=False,
            )
        except Exception as exc:
            self._trace_event(
                "streaming_speculative_longterm_context_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "final_transcript": final_transcript,
                    "text_len": len(str(text or "")),
                },
            )
            return
        # pylint: enable=not-callable
        self._trace_event(
            "streaming_speculative_longterm_context_requested",
            kind="cache",
            details={
                "final_transcript": final_transcript,
                "scheduled": bool(scheduled),
                "text_len": len(str(text or "")),
            },
        )

    def _consume_speculative_supervisor_decision(self, transcript: str) -> SupervisorDecision | None:
        self._wait_for_speculative_warmup("supervisor_decision")
        with self._speculation_lock:
            return self._streaming_speculation.consume_supervisor_decision(transcript)

    def _wait_for_speculative_supervisor_decision(
        self,
        transcript: str,
        *,
        wait_ms: int | None = None,
    ) -> SupervisorDecision | None:
        self._wait_for_speculative_warmup("supervisor_decision", wait_ms=wait_ms)
        with self._speculation_lock:
            return self._streaming_speculation.wait_for_supervisor_decision(
                transcript,
                wait_ms=wait_ms,
            )

    def _has_shared_speculative_supervisor_decision(self, transcript: str) -> bool:
        with self._speculation_lock:
            return self._streaming_speculation.has_shared_supervisor_decision(transcript)

    def _prime_supervisor_decision_cache(self) -> None:
        with self._speculation_lock:
            self._streaming_speculation.prime_supervisor_decision_cache()
            self._supervisor_cache_prewarmed = True

    def _prime_first_word_cache(self) -> None:
        with self._speculation_lock:
            self._streaming_speculation.prime_first_word_cache()
            self._first_word_cache_prewarmed = True

    def _generate_first_word_reply(
        self,
        transcript: str,
        *,
        instructions: str | None = None,
    ) -> FirstWordReply | None:
        self._wait_for_speculative_warmup("first_word")
        with self._speculation_lock:
            return self._streaming_speculation.generate_first_word_reply(
                transcript,
                instructions=instructions,
            )

    def _dual_lane_prefers_supervisor_bridge(self) -> bool:
        with self._speculation_lock:
            return self._streaming_speculation.dual_lane_prefers_supervisor_bridge()

    def _store_supervisor_decision(
        self,
        *,
        transcript: str,
        decision: SupervisorDecision | None,
    ) -> None:
        with self._speculation_lock:
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
        self._wait_for_speculative_warmup("supervisor_decision")
        with self._speculation_lock:
            return self._streaming_speculation.generate_supervisor_bridge_reply(
                transcript,
                instructions=instructions,
            )

    def _streaming_turn_timeout_policy(
        self,
        *,
        decision_hint=None,
        assume_unresolved_supervisor_handoff: bool = False,
    ) -> StreamingTurnTimeoutPolicy:
        return self._streaming_lane_planner.streaming_turn_timeout_policy(
            decision_hint=decision_hint,
            assume_unresolved_supervisor_handoff=assume_unresolved_supervisor_handoff,
        )

    def _dual_lane_bridge_reply_from_decision(
        self,
        prefetched_decision: SupervisorDecision | None,
    ) -> FirstWordReply | None:
        with self._speculation_lock:
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
        with pending_conversation_follow_up_hint_scope(
            self.runtime,
            active=listen_source == "follow_up",
        ):
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
        allow_remote_follow_up_rearm = self._voice_orchestrator_handles_follow_up(
            initial_source=listen_source,
        )
        self._trace_event(
            "streaming_text_turn_started",
            kind="span_start",
            details={
                "listen_source": listen_source,
                "proactive_trigger": proactive_trigger,
                "transcript_len": len(transcript),
            },
        )
        self._begin_text_turn_listening(
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        try:
            with pending_conversation_follow_up_hint_scope(
                self.runtime,
                active=listen_source == "follow_up",
            ):
                result = self._complete_streaming_turn(
                    transcript=transcript,
                    listen_source=listen_source,
                    proactive_trigger=proactive_trigger,
                    turn_started=turn_started,
                    capture_ms=0,
                    stt_ms=0,
                    allow_follow_up_rearm=allow_remote_follow_up_rearm,
                )
        except Exception as exc:
            self._trace_event(
                "streaming_text_turn_failed",
                kind="error",
                details={
                    "listen_source": listen_source,
                    "proactive_trigger": proactive_trigger,
                    "error": repr(exc),
                },
                kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
            )
            raise
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
            details={
                "listen_source": listen_source,
                "proactive_trigger": proactive_trigger,
                "transcript_len": len(transcript),
            },
        )
        if self.voice_orchestrator is not None:
            self._notify_voice_orchestrator_state("thinking", detail=listen_source)
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
                workflow_trace_id=current_workflow_trace_id(),
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
                follow_up_rearm_allowed_now=lambda request_source: self._follow_up_allowed_for_source(
                    initial_source=request_source
                ),
                emit_turn_latency_breakdown=lambda trace_id: emit_voice_turn_latency_breakdown(
                    emit=self.emit,
                    trace_event=self._trace_event,
                    trace_id=trace_id,
                ),
            ),
        )
        try:
            outcome = coordinator.execute()
        except InterruptedError:
            if self.voice_orchestrator is not None:
                self._notify_voice_orchestrator_state("waiting", detail=listen_source)
            return False
        except Exception as exc:
            self._trace_event(
                "streaming_turn_completion_failed",
                kind="error",
                details={
                    "listen_source": listen_source,
                    "proactive_trigger": proactive_trigger,
                    "error": repr(exc),
                },
                kpi={"duration_ms": round((time.monotonic() - turn_started) * 1000.0, 3)},
            )
            if self.voice_orchestrator is not None:
                self._notify_voice_orchestrator_state("waiting", detail=listen_source)
            raise
        if self.voice_orchestrator is not None:
            remote_follow_up = (
                outcome.keep_listening
                and allow_follow_up_rearm
                and self._voice_orchestrator_handles_follow_up(initial_source=listen_source)
            )
            if outcome.keep_listening and remote_follow_up:
                self._notify_voice_orchestrator_state(
                    "follow_up_open",
                    detail=listen_source,
                    follow_up_allowed=True,
                )
            else:
                self._notify_voice_orchestrator_state("waiting", detail=listen_source)
        return outcome.keep_listening

    def _segment_boundary(self, text: str) -> int | None:
        clause_min_chars = max(16, int(self.config.streaming_tts_clause_min_chars))
        soft_segment_chars = max(clause_min_chars + 12, int(self.config.streaming_tts_soft_segment_chars))
        hard_segment_chars = max(soft_segment_chars, int(self.config.streaming_tts_hard_segment_chars))

        strong_boundary = self._find_strong_segment_boundary(text, min_chars=clause_min_chars)
        if strong_boundary is not None:
            return strong_boundary

        clause_boundary = self._find_clause_segment_boundary(text, min_chars=clause_min_chars)
        if clause_boundary is not None:
            return clause_boundary

        if len(text) >= soft_segment_chars:
            boundary = self._last_whitespace_before(text, hard_segment_chars)
            if boundary is not None and boundary >= clause_min_chars:
                return boundary
        if len(text) >= hard_segment_chars:
            return len(text)
        return None

    def _find_strong_segment_boundary(self, text: str, *, min_chars: int) -> int | None:
        for index, character in enumerate(text):
            if index + 1 < min_chars:
                continue
            if character in "!?":
                return index + 1
            if character == "." and self._is_sentence_period(text, index):
                return index + 1
        return None

    def _find_clause_segment_boundary(self, text: str, *, min_chars: int) -> int | None:
        for index, character in enumerate(text):
            if index + 1 < min_chars:
                continue
            if character in ",;:":
                next_char = self._next_non_space_char(text, index + 1)
                if next_char is not None and next_char.isdigit() and text[index - 1].isdigit():
                    continue
                return index + 1
        return None

    def _is_sentence_period(self, text: str, index: int) -> bool:
        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if previous_char.isdigit() and next_char.isdigit():
            return False

        context_start = max(0, index - 6)
        context = text[context_start : index + 1].strip().lower()
        if any(context.endswith(abbreviation) for abbreviation in self._COMMON_NON_TERMINAL_ABBREVIATIONS):
            return False

        token_start = index - 1
        while token_start >= 0 and text[token_start].isalpha():
            token_start -= 1
        token = text[token_start + 1 : index]
        if len(token) == 1 and token.isalpha():
            return False

        next_non_space = self._next_non_space_char(text, index + 1)
        if next_non_space is not None and next_non_space.islower():
            return False

        return True

    @staticmethod
    def _next_non_space_char(text: str, start: int) -> str | None:
        for index in range(start, len(text)):
            if not text[index].isspace():
                return text[index]
        return None

    def _last_whitespace_before(self, text: str, limit: int) -> int | None:
        upper_bound = min(len(text), limit)
        for index in range(upper_bound - 1, -1, -1):
            if text[index].isspace():
                return index + 1
        return None
