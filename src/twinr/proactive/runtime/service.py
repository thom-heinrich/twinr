"""Compatibility shim for proactive runtime service exports.

##REFACTOR: 2026-03-27##

The implementation now lives under ``service_impl/``. Import from this module
exactly as before; no caller or systemd changes are required.
"""

# ruff: noqa: F401

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.pir import configured_pir_monitor
from twinr.hardware.portrait_match import PortraitMatchProvider
from twinr.hardware.respeaker import ScheduledReSpeakerSignalProvider
from twinr.hardware.respeaker.signal_provider import ReSpeakerSignalProvider
from twinr.providers.openai import OpenAIBackend

from ..social.aideck_camera_provider import AIDeckOpenAIVisionObservationProvider
from ..social.engine import SocialTriggerDecision, SocialTriggerEngine
from ..social.local_camera_provider import LocalAICameraObservationProvider
from ..social.observers import (
    AmbientAudioObservationProvider,
    NullAudioObservationProvider,
    OpenAIVisionObservationProvider,
    ReSpeakerAudioObservationProvider,
)
from ..social.vision_review import OpenAIProactiveVisionReviewer, ProactiveVisionFrameBuffer
from .presence import PresenceSessionController
from .service_impl import (
    BuildDefaultProactiveMonitorDependencies,
    ProactiveCoordinator,
    ProactiveMonitorService,
    ProactiveTickResult,
)
from .service_impl.builder import build_default_proactive_monitor as _build_default_proactive_monitor
from .service_impl.compat import (
    _ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
    _DEFAULT_CLOSE_JOIN_TIMEOUT_S,
    _DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES,
    _DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES,
    _LOGGER,
    _VISION_REVIEW_FAIL_OPEN_TRIGGERS,
    _append_ops_event,
    _assistant_output_active,
    _display_attention_refresh_allowed_runtime_status,
    _emit_key_value_line,
    _emit_token,
    _exception_text,
    _format_firmware_version,
    _normalize_optional_text,
    _normalize_text_tuple,
    _preserve_local_attention_on_audio_block,
    _proactive_audio_capture_device,
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
    _record_component_warning,
    _record_respeaker_dead_capture_blocker,
    _respeaker_capture_probe_duration_ms,
    _respeaker_dead_capture_payload,
    _round_optional_ratio,
    _round_optional_seconds,
    _safe_emit,
    _voice_orchestrator_capture_device,
)


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
) -> ProactiveMonitorService | None:
    """Build the default proactive monitor stack from Twinr runtime services."""

    dependencies = BuildDefaultProactiveMonitorDependencies(
        social_trigger_engine_cls=SocialTriggerEngine,
        configured_pir_monitor=configured_pir_monitor,
        presence_session_cls=PresenceSessionController,
        null_audio_observer_cls=NullAudioObservationProvider,
        ambient_audio_sampler_cls=AmbientAudioSampler,
        ambient_audio_observer_cls=AmbientAudioObservationProvider,
        base_signal_provider_cls=ReSpeakerSignalProvider,
        scheduled_signal_provider_cls=ScheduledReSpeakerSignalProvider,
        respeaker_audio_observer_cls=ReSpeakerAudioObservationProvider,
        openai_vision_provider_cls=OpenAIVisionObservationProvider,
        aideck_vision_provider_cls=AIDeckOpenAIVisionObservationProvider,
        local_vision_provider_cls=LocalAICameraObservationProvider,
        vision_reviewer_cls=OpenAIProactiveVisionReviewer,
        vision_frame_buffer_cls=ProactiveVisionFrameBuffer,
        portrait_match_provider_cls=PortraitMatchProvider,
        coordinator_cls=ProactiveCoordinator,
        monitor_service_cls=ProactiveMonitorService,
    )
    return _build_default_proactive_monitor(
        config=config,
        runtime=runtime,
        backend=backend,
        camera=camera,
        camera_lock=camera_lock,
        audio_lock=audio_lock,
        trigger_handler=trigger_handler,
        gesture_wakeup_handler=gesture_wakeup_handler,
        idle_predicate=idle_predicate,
        observation_handler=observation_handler,
        live_context_handler=live_context_handler,
        emit=emit,
        dependencies=dependencies,
    )


__all__ = [
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
