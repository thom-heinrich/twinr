"""Default proactive monitor builder for the refactored service package.

Purpose: assemble the production ``ProactiveCoordinator`` and
``ProactiveMonitorService`` from Twinr runtime dependencies while preserving
the legacy public build contract and monkeypatch surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler, AudioCaptureReadinessError
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.pir import configured_pir_monitor
from twinr.hardware.respeaker import (
    ScheduledReSpeakerSignalProvider,
    config_targets_respeaker,
)
from twinr.hardware.respeaker.signal_provider import ReSpeakerSignalProvider
from twinr.hardware.portrait_match import PortraitMatchProvider
from twinr.providers.openai import OpenAIBackend

from ...social.aideck_camera_provider import AIDeckOpenAIVisionObservationProvider
from ...social.engine import SocialTriggerDecision, SocialTriggerEngine
from ...social.local_camera_provider import LocalAICameraObservationProvider
from ...social.remote_camera_provider import (
    RemoteAICameraObservationProvider,
    RemoteFrameAICameraObservationProvider,
)
from ...social.observers import (
    AmbientAudioObservationProvider,
    NullAudioObservationProvider,
    OpenAIVisionObservationProvider,
    ReSpeakerAudioObservationProvider,
)
from ...social.vision_review import OpenAIProactiveVisionReviewer, ProactiveVisionFrameBuffer
from .coordinator import ProactiveCoordinator
from .coordinator_core import _NullSocialTriggerEngine
from .monitor import ProactiveMonitorService
from .compat import (
    _append_ops_event,
    _assistant_output_active,
    _exception_text,
    _preserve_local_attention_on_audio_block,
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
    _record_component_warning,
    _record_respeaker_dead_capture_blocker,
    _respeaker_capture_probe_duration_ms,
    _safe_emit,
)
from ..display_attention import resolve_display_attention_refresh_interval
from ..display_gesture_emoji import resolve_display_gesture_refresh_interval
from ..presence import PresenceSessionController
from ..respeaker_capture_gate import require_stable_respeaker_capture
from ..runtime_contract import (
    ReSpeakerRuntimeContractError,
    assess_respeaker_monitor_startup_contract,
)


@dataclass(frozen=True)
class BuildDefaultProactiveMonitorDependencies:
    """Factory surface passed in by the legacy wrapper for patch compatibility."""

    social_trigger_engine_cls: Any = SocialTriggerEngine
    configured_pir_monitor: Any = configured_pir_monitor
    presence_session_cls: Any = PresenceSessionController
    null_audio_observer_cls: Any = NullAudioObservationProvider
    ambient_audio_sampler_cls: Any = AmbientAudioSampler
    ambient_audio_observer_cls: Any = AmbientAudioObservationProvider
    base_signal_provider_cls: Any = ReSpeakerSignalProvider
    scheduled_signal_provider_cls: Any = ScheduledReSpeakerSignalProvider
    respeaker_audio_observer_cls: Any = ReSpeakerAudioObservationProvider
    openai_vision_provider_cls: Any = OpenAIVisionObservationProvider
    aideck_vision_provider_cls: Any = AIDeckOpenAIVisionObservationProvider
    local_vision_provider_cls: Any = LocalAICameraObservationProvider
    remote_proxy_vision_provider_cls: Any = RemoteAICameraObservationProvider
    remote_frame_vision_provider_cls: Any = RemoteFrameAICameraObservationProvider
    vision_reviewer_cls: Any = OpenAIProactiveVisionReviewer
    vision_frame_buffer_cls: Any = ProactiveVisionFrameBuffer
    portrait_match_provider_cls: Any = PortraitMatchProvider
    coordinator_cls: Any = ProactiveCoordinator
    monitor_service_cls: Any = ProactiveMonitorService


def build_default_proactive_monitor(
    *,
    config: TwinrConfig,
    runtime,
    backend: OpenAIBackend,
    camera: V4L2StillCamera,
    camera_lock: Lock | None,
    audio_lock: Lock | None,
    trigger_handler: Callable[[SocialTriggerDecision], bool],
    gesture_wakeup_handler: Callable[[Any], bool] | None = None,
    idle_predicate: Callable[[], bool] | None = None,
    observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
    live_context_handler: Callable[[dict[str, Any]], None] | None = None,
    emit: Callable[[str], None] | None = None,
    dependencies: BuildDefaultProactiveMonitorDependencies | None = None,
) -> ProactiveMonitorService | None:
    """Build the default proactive monitor stack from Twinr runtime services."""

    deps = dependencies or BuildDefaultProactiveMonitorDependencies()
    display_attention_enabled = (
        resolve_display_attention_refresh_interval(config) is not None
        and str(getattr(config, "display_driver", "") or "").strip().lower().startswith("hdmi")
    )
    display_gesture_enabled = (
        resolve_display_gesture_refresh_interval(config) is not None
        and str(getattr(config, "display_driver", "") or "").strip().lower().startswith("hdmi")
    )
    if (
        not config.proactive_enabled
        and not display_attention_enabled
        and not display_gesture_enabled
    ):
        return None

    try:
        engine = (
            deps.social_trigger_engine_cls.from_config(config)
            if config.proactive_enabled
            else _NullSocialTriggerEngine()
        )
    except Exception as exc:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="social_trigger_engine_init_failed",
            detail=(
                "The social trigger engine could not be initialized. "
                f"Proactive triggers were disabled: {_exception_text(exc)}"
            ),
        )
        engine = _NullSocialTriggerEngine()

    pir_monitor = None
    if config.pir_enabled:
        try:
            pir_monitor = deps.configured_pir_monitor(config)
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="pir_init_failed",
                detail=f"PIR monitoring could not be initialized: {_exception_text(exc)}",
            )
    else:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="pir_unconfigured",
            detail=(
                "PIR is not configured. Motion-gated camera inspection is disabled, "
                "but audio-only proactive logic can still run."
            ),
        )

    presence_session = None
    if config.voice_orchestrator_enabled:
        presence_session = deps.presence_session_cls(
            presence_grace_s=config.voice_orchestrator_follow_up_timeout_s,
            motion_grace_s=config.voice_orchestrator_follow_up_timeout_s,
            speech_grace_s=config.voice_orchestrator_follow_up_timeout_s,
        )

    audio_observer = deps.null_audio_observer_cls()
    audio_observer_fallback_factory: Callable[[], Any] | None = None
    distress_enabled = bool(config.proactive_enabled and config.proactive_audio_distress_enabled)
    shared_voice_capture_conflict = _proactive_pcm_capture_conflicts_with_voice_orchestrator(config)
    if shared_voice_capture_conflict and config.proactive_audio_enabled:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="proactive_audio_shared_capture_disabled",
            detail=(
                "Proactive PCM fallback is disabled because the voice orchestrator owns "
                "the same capture device. ReSpeaker host-control monitoring remains active "
                "when that hardware is targeted."
            ),
        )
    if config.proactive_audio_enabled:
        if shared_voice_capture_conflict:
            audio_observer = deps.null_audio_observer_cls()
        else:
            try:
                sampler = deps.ambient_audio_sampler_cls.from_config(config)
                audio_observer = deps.ambient_audio_observer_cls(
                    sampler=sampler,
                    audio_lock=audio_lock,
                    sample_ms=config.proactive_audio_sample_ms,
                    distress_enabled=distress_enabled,
                )
            except Exception as exc:
                _record_component_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="audio_sampler_init_failed",
                    detail=f"Ambient audio sampling could not be initialized: {_exception_text(exc)}",
                )
                audio_observer = deps.null_audio_observer_cls()

    respeaker_targeted = config_targets_respeaker(
        getattr(config, "audio_input_device", None),
        getattr(config, "proactive_audio_input_device", None),
    )
    if respeaker_targeted and config.proactive_audio_enabled:
        try:
            base_signal_provider = deps.base_signal_provider_cls(
                sensor_window_ms=config.proactive_audio_sample_ms,
                assistant_output_active_predicate=lambda: _assistant_output_active(runtime),
            )
            signal_provider = deps.scheduled_signal_provider_cls.from_config(
                config,
                provider=base_signal_provider,
            )
            initial_signal = signal_provider.observe()
            startup_contract = assess_respeaker_monitor_startup_contract(initial_signal)
            if startup_contract.blocking:
                blocker_code = startup_contract.blocker_code or "blocked"
                detail = startup_contract.detail or "ReSpeaker startup contract blocked monitor initialization."
                _safe_emit(emit, f"respeaker_runtime_blocker={blocker_code}")
                _append_ops_event(
                    runtime,
                    event="proactive_component_blocked",
                    level="error",
                    message="Proactive monitor startup was blocked by the ReSpeaker runtime contract.",
                    data={
                        "reason": startup_contract.ops_reason or "respeaker_startup_blocked",
                        "detail": detail,
                        "blocker_code": blocker_code,
                        "device_runtime_mode": initial_signal.device_runtime_mode,
                        "host_control_ready": initial_signal.host_control_ready,
                        "transport_reason": initial_signal.transport_reason,
                    },
                    emit=emit,
                )
                raise ReSpeakerRuntimeContractError(detail)
            if not initial_signal.host_control_ready:
                reason = initial_signal.transport_reason or "unknown_transport_state"
                detail = (
                    "ReSpeaker XVF3800 is configured for proactive/runtime audio, "
                    f"but host-control signals are degraded at startup: {reason}."
                )
                if initial_signal.requires_elevated_permissions:
                    detail += " USB permissions are likely missing for the runtime user."
                _record_component_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="respeaker_signal_provider_degraded",
                    detail=detail,
                )
            try:
                if not shared_voice_capture_conflict:
                    require_stable_respeaker_capture(
                        sampler=deps.ambient_audio_sampler_cls.from_config(config),
                        duration_ms=_respeaker_capture_probe_duration_ms(config),
                    )
            except AudioCaptureReadinessError as exc:
                detail = _record_respeaker_dead_capture_blocker(
                    runtime=runtime,
                    emit=emit,
                    probe=exc.probe,
                    stage="startup",
                    signal=initial_signal,
                )
                raise ReSpeakerRuntimeContractError(detail) from exc
            base_audio_observer = audio_observer
            audio_observer = deps.respeaker_audio_observer_cls(
                signal_provider=signal_provider,
                fallback_observer=base_audio_observer,
            )
            previous_factory = audio_observer_fallback_factory
            if previous_factory is not None:

                def _fallback_respeaker_audio_observer_factory() -> Any:
                    fallback_observer = previous_factory()
                    return deps.respeaker_audio_observer_cls(
                        signal_provider=signal_provider,
                        fallback_observer=fallback_observer,
                    )

                audio_observer_fallback_factory = _fallback_respeaker_audio_observer_factory
        except Exception as exc:
            if isinstance(exc, ReSpeakerRuntimeContractError):
                if not display_attention_enabled:
                    raise
                detail = _exception_text(exc)
                _preserve_local_attention_on_audio_block(
                    runtime=runtime,
                    emit=emit,
                    detail=detail,
                )
                audio_observer = deps.null_audio_observer_cls()
                audio_observer_fallback_factory = None
            else:
                _record_component_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="respeaker_signal_provider_init_failed",
                    detail=f"ReSpeaker signal-provider initialization failed: {_exception_text(exc)}",
                )

    vision_observer: Any | None = None
    if config.proactive_enabled or display_attention_enabled or display_gesture_enabled:
        try:
            provider_name = (getattr(config, "proactive_vision_provider", "local_first") or "local_first").strip().lower()
            if provider_name == "openai":
                vision_observer = deps.openai_vision_provider_cls(
                    backend=backend,
                    camera=camera,
                    camera_lock=camera_lock,
                )
            elif provider_name == "aideck_openai":
                vision_observer = deps.aideck_vision_provider_cls.from_config(
                    config,
                    backend=backend,
                    camera=camera,
                    camera_lock=camera_lock,
                )
            elif provider_name == "remote_proxy":
                vision_observer = deps.remote_proxy_vision_provider_cls.from_config(config)
            elif provider_name == "remote_frame":
                vision_observer = deps.remote_frame_vision_provider_cls.from_config(config)
            else:
                vision_observer = deps.local_vision_provider_cls.from_config(config)
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="vision_observer_init_failed",
                detail=f"Vision observation provider initialization failed: {_exception_text(exc)}",
            )

    vision_reviewer = None
    if config.proactive_enabled and config.proactive_vision_review_enabled and vision_observer is not None:
        try:
            vision_reviewer = deps.vision_reviewer_cls(
                backend=backend,
                frame_buffer=deps.vision_frame_buffer_cls(
                    max_items=config.proactive_vision_review_buffer_frames,
                ),
                max_frames=config.proactive_vision_review_max_frames,
                max_age_s=config.proactive_vision_review_max_age_s,
                min_spacing_s=config.proactive_vision_review_min_spacing_s,
            )
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="vision_reviewer_init_failed",
                detail=f"Buffered vision reviewer initialization failed: {_exception_text(exc)}",
            )

    portrait_match_provider = None
    if config.proactive_enabled and getattr(config, "portrait_match_enabled", True):
        try:
            portrait_match_provider = deps.portrait_match_provider_cls.from_config(
                config,
                camera=camera,
                camera_lock=camera_lock,
            )
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="portrait_match_provider_init_failed",
                detail=f"Local portrait-match provider initialization failed: {_exception_text(exc)}",
            )

    if (
        pir_monitor is None
        and isinstance(audio_observer, deps.null_audio_observer_cls)
        and vision_observer is None
    ):
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="no_operational_sensor_path",
            detail=(
                "No operational PIR, ambient audio, or HDMI attention camera path could be initialized. "
                "The proactive monitor was not started."
            ),
        )
        return None

    coordinator = deps.coordinator_cls(
        config=config,
        runtime=runtime,
        engine=engine,
        trigger_handler=trigger_handler,
        vision_observer=vision_observer,
        audio_observer=audio_observer,
        audio_observer_fallback_factory=audio_observer_fallback_factory,
        presence_session=presence_session,
        vision_reviewer=vision_reviewer,
        portrait_match_provider=portrait_match_provider,
        gesture_wakeup_handler=gesture_wakeup_handler,
        pir_monitor=pir_monitor,
        idle_predicate=idle_predicate,
        observation_handler=observation_handler,
        live_context_handler=live_context_handler,
        emit=emit,
    )
    return deps.monitor_service_cls(
        coordinator,
        poll_interval_s=config.proactive_poll_interval_s,
        emit=emit,
    )


__all__ = [
    "BuildDefaultProactiveMonitorDependencies",
    "build_default_proactive_monitor",
]
