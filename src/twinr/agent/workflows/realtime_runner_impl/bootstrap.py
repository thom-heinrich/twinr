# CHANGELOG: 2026-03-27
# BUG-1: project_root is now normalized once at bootstrap time so smart-home adapters and forensics paths are no longer cwd-dependent.
# BUG-2: Turn-control bootstrap now reuses already-injected streaming/tool-capable providers and surfaces optional-bundle failures instead of silently degrading.
# BUG-3: Realtime tool-surface refresh now syncs into an already-created realtime session so refreshed handlers are not left stale.
# SEC-1: Sensitive realtime tools are now least-privilege by default and are withheld until voice identity or explicit authorization is available.
# IMP-1: Added 2026 Realtime session feature negotiation (semantic/server VAD, idle timeout, noise reduction, truncation retention ratio, transcription hints) when supported by the session wrapper.
# IMP-2: Added optional ambient-audio-sampler auto-bootstrap plus stronger config validation and boot diagnostics for Raspberry Pi deployments.
# BUG-4: Prewarm the processing feedback media clip during loop bootstrap so the first THINKING cue can hit the
#        dragon MP3 path without waiting for a cold ffmpeg render on demand.

"""Bootstrap helpers for the realtime workflow loop."""

# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from queue import Queue
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable, cast
import copy
import inspect
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentTextProvider,
    CombinedSpeechAgentProvider,
    CompositeSpeechAgentProvider,
    SpeechToTextProvider,
    StreamingSpeechToTextProvider,
    TextToSpeechProvider,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.conversation.closure import ConversationClosureEvaluator
from twinr.agent.base_agent.conversation.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.tools import RealtimeToolExecutor
from twinr.agent.tools.runtime.availability import (
    available_realtime_tool_names,
    bind_available_realtime_tool_handlers,
)
from twinr.agent.workflows import voice_identity_runtime
from twinr.agent.workflows.follow_up_steering import FollowUpSteeringRuntime
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator
from twinr.agent.workflows.print_lane import TwinrPrintLane
from twinr.agent.workflows.realtime_runtime.support import _default_emit
from twinr.agent.workflows.remote_transcript_commit import RemoteTranscriptCommitCoordinator
from twinr.agent.workflows.startup_boot_sound import start_startup_boot_sound
from twinr.agent.workflows.streaming_transcript_verifier import StreamingTranscriptVerifierRuntime
from twinr.agent.workflows.turn_guidance import TurnGuidanceRuntime
from twinr.agent.workflows.voice_orchestrator import EdgeVoiceOrchestrator
from twinr.hardware.audio import AmbientAudioSampler, SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import configured_button_monitor
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.integrations import build_smart_home_hub_adapter
from twinr.integrations.smarthome import SmartHomeObservation, SmartHomeSensorWorker
from twinr.ops.usage import TwinrUsageStore
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext
from twinr.proactive import build_default_proactive_monitor
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai import OpenAIProviderBundle
from twinr.providers.openai.realtime import OpenAIRealtimeSession


_DEFAULT_SENSITIVE_REALTIME_TOOL_FRAGMENTS: tuple[str, ...] = (
    "camera",
    "photo",
    "snapshot",
    "image",
    "print",
    "printer",
    "smart_home",
    "smarthome",
    "door",
    "lock",
    "unlock",
    "alarm",
    "garage",
    "thermostat",
    "light",
    "lights",
    "call",
    "sms",
    "email",
    "message",
    "notify",
    "share",
    "transcript",
    "memory",
    "contact",
)
_OPENAI_REALTIME_PCM_SAMPLE_RATE = 24_000


class TwinrRealtimeBootstrapMixin:
    """Initialize the realtime loop and its static collaborators."""

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
        conversation_closure_evaluator: ConversationClosureEvaluator | None = None,
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
        self.emit = emit or _default_emit
        self._project_root = self._normalize_project_root(self.config.project_root)
        self._bootstrap_notice_once_codes: set[str] = set()
        self._tool_surface_refresh_lock = RLock()
        self._tool_surface_authorized = bool(
            getattr(self.config, "realtime_sensitive_tools_start_authorized", False)
        )
        self._tool_surface_authorized_reason: str | None = (
            "config"
            if self._tool_surface_authorized
            else None
        )
        self._close_lock = RLock()
        self._closed = False
        self._runtime_tool_policy_state: tuple[bool, tuple[str, ...], tuple[str, ...]] | None = None
        self._deferred_sensitive_tool_names: tuple[str, ...] = ()
        self._validate_bootstrap_config(error_reset_seconds=error_reset_seconds)

        self.runtime = runtime or TwinrRuntime(config=config)
        openai_bundle: OpenAIProviderBundle | None = None
        if print_backend is None and (
            stt_provider is None or agent_provider is None or tts_provider is None
        ):
            openai_bundle = OpenAIProviderBundle.from_config(config)
        self.stt_provider = stt_provider or print_backend or (
            openai_bundle.stt if openai_bundle is not None else None
        )
        self.agent_provider = agent_provider or print_backend or (
            openai_bundle.agent if openai_bundle is not None else None
        )
        self.tts_provider = tts_provider or print_backend or (
            openai_bundle.tts if openai_bundle is not None else None
        )
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
        self._ambient_audio_sampler = self._resolve_ambient_audio_sampler(ambient_audio_sampler)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.playback_coordinator = PlaybackCoordinator(
            self.player,
            emit=self.emit,
            io_lock=self._audio_lock,
        )
        self._prewarm_working_feedback_media("processing")
        self._active_turn_stop_lock = Lock()
        self._active_turn_stop_event: Event | None = None
        self._active_turn_stop_reason: str | None = None
        self._answer_interrupt_lock = Lock()
        self._answer_interrupt_event: Event | None = None
        self._conversation_session_lock = Lock()
        self._remote_transcript_commits = RemoteTranscriptCommitCoordinator(
            # The current edge/server transcript-commit contract still delivers
            # only `source` + `transcript`. Until wait_id/item_id travel end to
            # end, legacy source correlation must stay enabled here or remote
            # same-stream listening can never resolve on the Pi.
            allow_legacy_source_match=True,
        )
        self._current_turn_audio_pcm: bytes | None = None
        self._current_turn_audio_sample_rate: int = self.config.openai_realtime_input_sample_rate
        self.tool_executor = RealtimeToolExecutor(self)
        self._runtime_tool_names: tuple[str, ...] = ()
        self._tool_handlers: dict[str, Callable[..., Any]] = {}
        self._refresh_runtime_tool_surface()

        provider_bundle = None
        provider_bundle_error: Exception | None = None
        should_build_turn_bundle = bool(
            self.config.turn_controller_enabled or self.config.conversation_closure_guard_enabled
        )
        if should_build_turn_bundle and (turn_stt_provider is None or turn_tool_agent_provider is None):
            try:
                provider_bundle = build_streaming_provider_bundle(config)
            except Exception as exc:
                provider_bundle_error = exc
                provider_bundle = None
                self._emit_bootstrap_notice(
                    "streaming_provider_bundle_unavailable",
                    f"{type(exc).__name__}",
                    details={
                        "turn_controller_enabled": bool(self.config.turn_controller_enabled),
                        "conversation_closure_guard_enabled": bool(
                            self.config.conversation_closure_guard_enabled
                        ),
                    },
                )
                if bool(
                    getattr(
                        self.config,
                        "realtime_bootstrap_strict_optional_dependencies",
                        False,
                    )
                ):
                    raise

        self.turn_stt_provider = turn_stt_provider or self._coerce_provider_from_contract(
            provider_bundle.stt if provider_bundle is not None else None,
            StreamingSpeechToTextProvider,
        ) or self._coerce_provider_from_contract(
            self.stt_provider,
            StreamingSpeechToTextProvider,
        )
        self.turn_tool_agent_provider = turn_tool_agent_provider or self._coerce_provider_from_contract(
            provider_bundle.tool_agent if provider_bundle is not None else None,
            ToolCallingAgentProvider,
        ) or self._coerce_provider_from_contract(
            self.agent_provider,
            ToolCallingAgentProvider,
        )

        if should_build_turn_bundle and provider_bundle is None and provider_bundle_error is None:
            self._emit_bootstrap_notice(
                "streaming_provider_bundle_missing",
                "turn_controller_features_requested_without_optional_bundle",
                details={
                    "turn_controller_enabled": bool(self.config.turn_controller_enabled),
                    "conversation_closure_guard_enabled": bool(
                        self.config.conversation_closure_guard_enabled
                    ),
                },
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
            or self._build_conversation_closure_evaluator(config)
        )

        if realtime_session is not None:
            self.realtime_session = realtime_session
        else:
            self.realtime_session = self._build_realtime_session()
        self._sync_realtime_session_tool_handlers()

        self.turn_guidance_runtime = TurnGuidanceRuntime(self)
        self.follow_up_steering_runtime = FollowUpSteeringRuntime(self)
        self.streaming_transcript_verifier_runtime = StreamingTranscriptVerifierRuntime(self)
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._voice_orchestrator_runtime_state_lock = RLock()
        self._last_voice_orchestrator_runtime_state: tuple[str, str | None, bool] | None = None
        self._last_voice_orchestrator_intent_context: VoiceRuntimeIntentContext | None = None
        self._last_voice_orchestrator_quiet_until_utc: str | None = None
        self._last_voice_identity_profile_revision: str | None = None
        self._last_voice_identity_profile_count: int = 0
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self._next_automation_check_at: float = 0.0
        self._next_long_term_memory_proactive_check_at: float = 0.0
        self._working_feedback_stop: Callable[[], None] | None = None
        self._working_feedback_generation: int = 0
        self._conversation_session_active = False
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue(maxsize=1)
        self._latest_sensor_observation_facts: dict[str, object] | None = None
        self._voice_activation_ack_cache_lock = Lock()
        self._voice_activation_ack_wav_bytes: bytes | None = None
        self._voice_activation_ack_prefetch_started = False
        self._voice_activation_ack_prefetch_thread: Thread | None = None
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
        self.workflow_forensics = WorkflowForensics.from_env(
            project_root=self._project_root,
            service=self.__class__.__name__,
        )
        self.voice_orchestrator = (
            EdgeVoiceOrchestrator(
                config=config,
                emit=self.emit,
                playback_coordinator=self.playback_coordinator,
                on_voice_activation=self.handle_voice_activation,
                on_transcript_committed=self.handle_remote_transcript_committed,
                on_follow_up_closed=self.handle_remote_follow_up_closed,
                on_barge_in_interrupt=lambda: self._request_answer_interrupt("voice_orchestrator"),
                on_recent_remote_audio=(
                    lambda pcm_bytes, source: voice_identity_runtime.update_household_voice_assessment_from_pcm(
                        self,
                        pcm_bytes,
                        source=source,
                    )
                ),
                forensics=self.workflow_forensics,
            )
            if bool(getattr(config, "voice_orchestrator_enabled", False))
            else None
        )
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.print_backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
            gesture_wakeup_handler=self.handle_gesture_wakeup,
            idle_predicate=self._background_work_allowed,
            observation_handler=self.handle_sensor_observation,
            live_context_handler=self.handle_live_sensor_context,
            emit=self.emit,
        )
        self._workflow_active_trace_id: str | None = None
        self._trace_event(
            "workflow_loop_initialized",
            kind="run_start",
            details={
                "class": self.__class__.__name__,
                "project_root": str(self._project_root),
                "stt_provider": type(self.stt_provider).__name__,
                "agent_provider": type(self.agent_provider).__name__,
                "tts_provider": type(self.tts_provider).__name__,
                "turn_stt_provider": (
                    type(self.turn_stt_provider).__name__
                    if self.turn_stt_provider is not None
                    else None
                ),
                "turn_tool_agent_provider": (
                    type(self.turn_tool_agent_provider).__name__
                    if self.turn_tool_agent_provider is not None
                    else None
                ),
                "realtime_tool_count": len(self._runtime_tool_names),
                "deferred_sensitive_tool_count": len(self._deferred_sensitive_tool_names),
                "ambient_audio_sampler": (
                    type(self._ambient_audio_sampler).__name__
                    if self._ambient_audio_sampler is not None
                    else None
                ),
            },
        )

    def _normalize_project_root(self, raw_project_root: str | Path) -> Path:
        project_root = Path(raw_project_root).expanduser()
        try:
            return project_root.resolve(strict=False)
        except TypeError:
            return project_root.resolve()
        except OSError:
            return project_root.absolute()

    def _validate_bootstrap_config(self, *, error_reset_seconds: float) -> None:
        if float(error_reset_seconds) < 0:
            raise ValueError("error_reset_seconds must be >= 0")

        numeric_checks: tuple[tuple[str, float, bool], ...] = (
            ("audio_chunk_ms", float(getattr(self.config, "audio_chunk_ms", 0)), True),
            ("audio_preroll_ms", float(getattr(self.config, "audio_preroll_ms", 0)), False),
            ("audio_speech_start_chunks", float(getattr(self.config, "audio_speech_start_chunks", 0)), True),
            ("audio_start_timeout_s", float(getattr(self.config, "audio_start_timeout_s", 0)), False),
            ("audio_max_record_seconds", float(getattr(self.config, "audio_max_record_seconds", 0)), True),
            ("openai_realtime_input_sample_rate", float(getattr(self.config, "openai_realtime_input_sample_rate", 0)), True),
            ("audio_channels", float(getattr(self.config, "audio_channels", 0)), True),
        )
        for field_name, value, strictly_positive in numeric_checks:
            if strictly_positive and value <= 0:
                raise ValueError(f"{field_name} must be > 0")
            if not strictly_positive and value < 0:
                raise ValueError(f"{field_name} must be >= 0")

        realtime_sample_rate = int(getattr(self.config, "openai_realtime_input_sample_rate", 0))
        audio_channels = int(getattr(self.config, "audio_channels", 0))
        if realtime_sample_rate != _OPENAI_REALTIME_PCM_SAMPLE_RATE:
            self._emit_bootstrap_notice(
                "nonstandard_realtime_sample_rate",
                str(realtime_sample_rate),
                details={
                    "recommended_pcm_sample_rate": _OPENAI_REALTIME_PCM_SAMPLE_RATE,
                },
            )
        if audio_channels != 1:
            self._emit_bootstrap_notice(
                "non_mono_realtime_input",
                str(audio_channels),
                details={"recommended_channels": 1},
            )

    def _emit_bootstrap_notice(
        self,
        code: str,
        message: str,
        *,
        level: str = "WARNING",
        details: Mapping[str, object] | None = None,
        once: bool = True,
    ) -> None:
        if once and code in self._bootstrap_notice_once_codes:
            return
        if once:
            self._bootstrap_notice_once_codes.add(code)
        try:
            self.emit(f"{code}={message}")
        except Exception:
            pass
        trace_event = getattr(self, "_trace_event", None)
        if callable(trace_event):
            try:
                trace_event_callable = cast(Callable[..., Any], trace_event)
                trace_payload = {"message": message}
                if details:
                    trace_payload.update(dict(details))
                trace_event_callable(  # pylint: disable=not-callable
                    code,
                    kind="bootstrap",
                    level=level,
                    details=trace_payload,
                )
            except Exception:
                pass

    def _resolve_ambient_audio_sampler(
        self,
        ambient_audio_sampler: AmbientAudioSampler | None,
    ) -> AmbientAudioSampler | None:
        if ambient_audio_sampler is not None:
            return ambient_audio_sampler
        if not bool(getattr(self.config, "ambient_audio_sampler_enabled", False)):
            return None
        factory = getattr(AmbientAudioSampler, "from_config", None)
        if callable(factory):
            try:
                return factory(self.config)
            except Exception as exc:
                self._emit_bootstrap_notice(
                    "ambient_audio_sampler_init_failed",
                    f"{type(exc).__name__}",
                )
        return None

    @staticmethod
    def _protocol_callable_names(protocol_type: type[object]) -> tuple[str, ...]:
        names: list[str] = []
        for name, member in protocol_type.__dict__.items():
            if name.startswith("_"):
                continue
            if callable(member):
                names.append(name)
        return tuple(names)

    def _coerce_provider_from_contract(self, provider: object | None, protocol_type: type[object]):
        if provider is None:
            return None
        try:
            if isinstance(provider, protocol_type):
                return provider
        except TypeError:
            pass
        required_callables = self._protocol_callable_names(protocol_type)
        if required_callables and all(callable(getattr(provider, name, None)) for name in required_callables):
            return provider
        return None

    @staticmethod
    def _coerce_name_tuple(value: object | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items: Iterable[object] = value.replace(";", ",").split(",")
        elif isinstance(value, Mapping):
            items = value.keys()
        elif isinstance(value, Iterable):
            items = value
        else:
            items = (value,)
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
        return tuple(result)

    def _realtime_sensitive_tools_require_identity(self) -> bool:
        # BREAKING: secure-by-default posture for ambient assistants. Set
        # config.realtime_sensitive_tools_require_identity = False to restore the
        # legacy always-on tool surface.
        return bool(getattr(self.config, "realtime_sensitive_tools_require_identity", True))

    def _is_sensitive_tool_access_authorized(self) -> bool:
        if not self._realtime_sensitive_tools_require_identity():
            return True
        if self._tool_surface_authorized:
            return True
        return bool(getattr(self, "_last_voice_identity_profile_revision", None))

    def authorize_realtime_sensitive_tools(self, reason: str = "explicit") -> tuple[str, ...]:
        with self._tool_surface_refresh_lock:
            self._tool_surface_authorized = True
            self._tool_surface_authorized_reason = reason
            self._refresh_runtime_tool_surface()
            return self._runtime_tool_names

    def deauthorize_realtime_sensitive_tools(self, reason: str = "explicit") -> tuple[str, ...]:
        with self._tool_surface_refresh_lock:
            self._tool_surface_authorized = False
            self._tool_surface_authorized_reason = reason
            self._refresh_runtime_tool_surface()
            return self._runtime_tool_names

    def _sensitive_tool_fragments(self) -> tuple[str, ...]:
        configured = self._coerce_name_tuple(
            getattr(self.config, "realtime_sensitive_tool_fragments", None)
        )
        if configured:
            return tuple(fragment.lower() for fragment in configured)
        return _DEFAULT_SENSITIVE_REALTIME_TOOL_FRAGMENTS

    def _explicit_sensitive_tool_names(self) -> set[str]:
        return {
            name.lower()
            for name in self._coerce_name_tuple(
                getattr(self.config, "realtime_sensitive_tool_names", None)
            )
        }

    def _is_sensitive_tool_name(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        if lowered in self._explicit_sensitive_tool_names():
            return True
        return any(fragment in lowered for fragment in self._sensitive_tool_fragments())

    def _apply_runtime_tool_policy(self, tool_names: Iterable[str]) -> tuple[str, ...]:
        deduped = tuple(dict.fromkeys(str(name) for name in tool_names))
        allowlist = set(self._coerce_name_tuple(getattr(self.config, "realtime_tool_allowlist", None)))
        denylist = set(self._coerce_name_tuple(getattr(self.config, "realtime_tool_denylist", None)))

        filtered = deduped
        if allowlist:
            filtered = tuple(name for name in filtered if name in allowlist)
        if denylist:
            filtered = tuple(name for name in filtered if name not in denylist)

        if self._is_sensitive_tool_access_authorized():
            self._deferred_sensitive_tool_names = ()
            return filtered

        allowed: list[str] = []
        deferred: list[str] = []
        for name in filtered:
            if self._is_sensitive_tool_name(name):
                deferred.append(name)
            else:
                allowed.append(name)
        self._deferred_sensitive_tool_names = tuple(deferred)
        return tuple(allowed)

    def _build_realtime_turn_detection_payload(self) -> dict[str, object]:
        # BREAKING: on wrappers that support session defaults, we prefer semantic
        # VAD by default in 2026 because it reduces premature cutoffs and is less
        # pause-driven than legacy silence endpointing.
        mode = str(
            getattr(self.config, "realtime_turn_detection_mode", "semantic_vad")
        ).strip().lower()
        create_response = bool(
            getattr(self.config, "realtime_turn_detection_create_response", True)
        )
        interrupt_response = bool(
            getattr(self.config, "realtime_turn_detection_interrupt_response", True)
        )

        if mode == "semantic_vad":
            payload: dict[str, object] = {
                "type": "semantic_vad",
                "eagerness": str(
                    getattr(self.config, "realtime_turn_detection_eagerness", "auto")
                ).strip().lower(),
                "create_response": create_response,
                "interrupt_response": interrupt_response,
            }
            return payload

        payload = {
            "type": "server_vad",
            "threshold": float(getattr(self.config, "realtime_server_vad_threshold", 0.5)),
            "prefix_padding_ms": int(
                getattr(self.config, "realtime_server_vad_prefix_padding_ms", 300)
            ),
            "silence_duration_ms": int(
                getattr(self.config, "realtime_server_vad_silence_duration_ms", 500)
            ),
            "create_response": create_response,
            "interrupt_response": interrupt_response,
        }
        idle_timeout_ms = getattr(self.config, "realtime_idle_timeout_ms", None)
        if idle_timeout_ms is not None:
            payload["idle_timeout_ms"] = int(idle_timeout_ms)
        return payload

    def _infer_realtime_noise_reduction_type(self) -> str | None:
        explicit = getattr(self.config, "realtime_noise_reduction_type", None) or getattr(
            self.config, "audio_noise_reduction_type", None
        )
        if explicit:
            normalized = str(explicit).strip().lower()
            if normalized in {"near_field", "far_field"}:
                return normalized
        if bool(getattr(self.config, "audio_far_field_microphone", False)):
            return "far_field"
        if bool(getattr(self.config, "audio_near_field_microphone", False)):
            return "near_field"
        return None

    def _build_realtime_input_transcription_payload(self) -> dict[str, object] | None:
        transcription_model = getattr(
            self.config,
            "realtime_input_transcription_model",
            None,
        )
        transcription_prompt = getattr(
            self.config,
            "realtime_input_transcription_prompt",
            None,
        )
        transcription_language = getattr(
            self.config,
            "realtime_input_transcription_language",
            None,
        )
        if not any((transcription_model, transcription_prompt, transcription_language)):
            return None
        payload: dict[str, object] = {}
        if transcription_model:
            payload["model"] = str(transcription_model)
        if transcription_prompt:
            payload["prompt"] = str(transcription_prompt)
        if transcription_language:
            payload["language"] = str(transcription_language)
        return payload

    def _build_realtime_session_payload(self) -> dict[str, object]:
        session_payload: dict[str, object] = {"type": "realtime"}

        audio_input: dict[str, object] = {
            "turn_detection": self._build_realtime_turn_detection_payload(),
        }
        realtime_sample_rate = int(getattr(self.config, "openai_realtime_input_sample_rate", 0))
        if realtime_sample_rate == _OPENAI_REALTIME_PCM_SAMPLE_RATE:
            audio_input["format"] = {"type": "audio/pcm", "rate": _OPENAI_REALTIME_PCM_SAMPLE_RATE}

        noise_reduction_type = self._infer_realtime_noise_reduction_type()
        if noise_reduction_type is not None:
            audio_input["noise_reduction"] = {"type": noise_reduction_type}

        transcription_payload = self._build_realtime_input_transcription_payload()
        if transcription_payload:
            audio_input["transcription"] = transcription_payload

        session_payload["audio"] = {"input": audio_input}

        retention_ratio = getattr(self.config, "realtime_truncation_retention_ratio", 0.8)
        if retention_ratio is not None:
            session_payload["truncation"] = {
                "type": "retention_ratio",
                "retention_ratio": float(retention_ratio),
            }

        tool_choice = getattr(self.config, "realtime_tool_choice", None)
        if tool_choice is not None:
            session_payload["tool_choice"] = tool_choice

        overrides = getattr(self.config, "realtime_session_overrides", None)
        if isinstance(overrides, Mapping):
            session_payload = self._deep_merge_dicts(session_payload, dict(overrides))
        return session_payload

    @staticmethod
    def _deep_merge_dicts(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
        merged = copy.deepcopy(dict(base))
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = TwinrRealtimeBootstrapMixin._deep_merge_dicts(
                    merged[key],
                    value,
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _callable_accepts_parameter(
        target: Callable[..., object] | type[object],
        parameter_name: str,
    ) -> bool:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        if parameter_name in signature.parameters:
            return True
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _build_realtime_session(self) -> OpenAIRealtimeSession:
        session_kwargs: dict[str, object] = {
            "config": self.config,
            "tool_handlers": dict(self._tool_handlers),
        }
        if self._callable_accepts_parameter(OpenAIRealtimeSession, "emit"):
            session_kwargs["emit"] = self.emit

        session_payload = self._build_realtime_session_payload()
        for parameter_name in (
            "session",
            "session_config",
            "initial_session",
            "initial_session_config",
            "session_options",
            "default_session",
            "session_defaults",
        ):
            if self._callable_accepts_parameter(OpenAIRealtimeSession, parameter_name):
                session_kwargs[parameter_name] = session_payload
                break

        session = OpenAIRealtimeSession(**session_kwargs)
        return session

    def _sync_realtime_session_tool_handlers(self) -> None:
        session = getattr(self, "realtime_session", None)
        if session is None:
            return
        handlers = dict(self._tool_handlers)
        last_error: Exception | None = None

        for method_name in ("set_tool_handlers", "update_tool_handlers", "replace_tool_handlers"):
            method = getattr(session, method_name, None)
            if not callable(method):
                continue
            try:
                method(handlers)
                return
            except TypeError:
                try:
                    method(tool_handlers=handlers)
                    return
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc

        for attribute_name in ("tool_handlers", "_tool_handlers"):
            if not hasattr(session, attribute_name):
                continue
            try:
                setattr(session, attribute_name, handlers)
                return
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            self._emit_bootstrap_notice(
                "realtime_tool_handler_sync_failed",
                f"{type(last_error).__name__}",
            )

    def _refresh_runtime_tool_surface(self) -> None:
        """Recompute the locally runnable realtime-tool surface."""

        with self._tool_surface_refresh_lock:
            available_tool_names = tuple(available_realtime_tool_names(self.config))
            effective_tool_names = self._apply_runtime_tool_policy(available_tool_names)
            self._runtime_tool_names = effective_tool_names
            self._tool_handlers = bind_available_realtime_tool_handlers(
                self.tool_executor,
                config=self.config,
                tool_names=self._runtime_tool_names,
            )
            self._sync_realtime_session_tool_handlers()
            streaming_surface_sync = cast(
                Callable[[], None] | None,
                getattr(self, "_sync_streaming_turn_loop_tool_surface", None),
            )
            if streaming_surface_sync is not None:
                streaming_surface_sync()

            policy_state = (
                self._is_sensitive_tool_access_authorized(),
                self._runtime_tool_names,
                self._deferred_sensitive_tool_names,
            )
            if policy_state != self._runtime_tool_policy_state:
                self._runtime_tool_policy_state = policy_state
                if self._deferred_sensitive_tool_names:
                    self._emit_bootstrap_notice(
                        "realtime_sensitive_tools_deferred",
                        ",".join(self._deferred_sensitive_tool_names),
                        details={
                            "authorized": False,
                            "reason": self._tool_surface_authorized_reason
                            or "voice_identity_required",
                        },
                        once=False,
                    )
                elif available_tool_names != effective_tool_names:
                    self._emit_bootstrap_notice(
                        "realtime_tool_surface_filtered",
                        ",".join(effective_tool_names),
                        details={"authorized": self._is_sensitive_tool_access_authorized()},
                        once=False,
                    )

    def refresh_runtime_tool_surface(self) -> tuple[str, ...]:
        self._refresh_runtime_tool_surface()
        return self._runtime_tool_names

    def _record_event(self, *args, **kwargs):
        try:
            return super()._record_event(*args, **kwargs)
        except Exception as exc:
            emit = getattr(self, "emit", None)
            if callable(emit):
                try:
                    emit(f"record_event_failed={type(exc).__name__}")
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
                    emit(f"record_usage_failed={type(exc).__name__}")
                except Exception:
                    self._trace_event(
                        "record_usage_emit_failed",
                        kind="error",
                        level="ERROR",
                        details={"error_type": type(exc).__name__},
                    )
            return None

    def _start_startup_boot_sound(self) -> None:
        start_startup_boot_sound(
            config=self.config,
            playback_coordinator=self.playback_coordinator,
            emit=self.emit,
        )

    def _build_managed_smart_home_adapter(self):
        return build_smart_home_hub_adapter(self._project_root)

    def _handle_smart_home_observation(self, observation: SmartHomeObservation) -> None:
        self.handle_sensor_observation(observation.facts, observation.event_names)

    def _start_smart_home_sensor_worker(self) -> SmartHomeSensorWorker | None:
        if not bool(getattr(self.config, "smart_home_background_worker_enabled", True)):
            return None
        worker = SmartHomeSensorWorker(
            adapter_loader=self._build_managed_smart_home_adapter,
            observation_callback=self._handle_smart_home_observation,
            idle_sleep_s=float(getattr(self.config, "smart_home_background_idle_sleep_s", 1.0)),
            retry_delay_s=float(getattr(self.config, "smart_home_background_retry_delay_s", 2.0)),
            batch_limit=int(getattr(self.config, "smart_home_background_batch_limit", 8)),
            emit=self.emit,
            record_event=lambda event_name, detail: self._record_event(
                event_name,
                detail,
                source="smart_home_sensor_worker",
            ),
        )
        worker.start()
        return worker

    def wait_for_print_lane_idle(self, timeout_s: float = 1.0) -> bool:
        return self.print_lane.wait_for_idle(timeout_s=timeout_s)

    def _mark_print_submitted(self) -> None:
        self._last_print_request_at = time.monotonic()

    def close(self, *, timeout_s: float = 1.0) -> None:
        """Release loop-owned resources for tests and bounded runtime shutdown."""

        try:
            timeout = max(0.05, float(timeout_s))
        except (TypeError, ValueError):
            timeout = 1.0

        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            return

        with close_lock:
            if getattr(self, "_closed", False):
                return
            self._closed = True

        playback_coordinator = getattr(self, "playback_coordinator", None)
        if playback_coordinator is not None:
            try:
                playback_coordinator.close(immediate=True, timeout_s=min(timeout, 2.0))
            except Exception:
                pass

        realtime_session = getattr(self, "realtime_session", None)
        realtime_session_close = getattr(realtime_session, "close", None)
        if callable(realtime_session_close):
            try:
                realtime_session_close()
            except Exception:
                pass

        runtime = getattr(self, "runtime", None)
        runtime_shutdown = getattr(runtime, "shutdown", None)
        if callable(runtime_shutdown):
            try:
                runtime_shutdown(timeout_s=min(timeout, 2.0))
            except Exception:
                pass

        workflow_forensics = getattr(self, "workflow_forensics", None)
        workflow_forensics_close = getattr(workflow_forensics, "close", None)
        if callable(workflow_forensics_close):
            try:
                workflow_forensics_close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close(timeout_s=0.2)
        except Exception:
            pass
