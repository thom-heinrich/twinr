# CHANGELOG: 2026-03-29
# BUG-1: Fixed the dead audio_observer_fallback_factory path so ambient/ReSpeaker
#        observers can be rebuilt after runtime faults instead of silently losing
#        audio resilience.
# BUG-2: Fixed false ReSpeaker startup failures caused by probing the PCM capture
#        device while an ambient observer could already be holding the same device.
# BUG-3: Fixed over-eager hard failure on ReSpeaker audio blockers; the monitor
#        now degrades to ambient audio or other non-audio sensor paths whenever a
#        viable runtime path still exists.
# BUG-4: Disabled the hdmi_wayland attention-refresh bootstrap path because Pi
#        evidence showed the hidden camera/servo lane still drives memory pressure
#        even when visible face publishing is already fail-closed there.
# BUG-6: Restored the hdmi_wayland attention-refresh fail-close after a later
#        regression drifted the hidden camera/servo lane back into the
#        transcript-first voice runtime and reintroduced transport backpressure.
# SEC-1: Hardened config-derived timing and buffer parameters to reduce practical
#        Pi 4 denial-of-service risk from invalid or hostile config values
#        (busy-loop polling, excessive frame buffers, or unbounded reviewer input).
# SEC-2: Sanitized and redacted exception details before emitting them to ops/log
#        sinks to reduce secret leakage and log/terminal injection risk.
# BUG-5: Removed proactive vision-provider fallback chaining so the configured
#        camera lane is now single-path and startup fails closed on init errors.
# IMP-1: Added 2026 Pi camera aliases (picamera2/libcamera/imx500/ai_camera)
#        while normalizing them onto the same strict local provider lane.
# IMP-2: Added reviewer-parameter normalization aligned with 2026 local-first
#        edge-AI deployment patterns on Raspberry Pi.

"""Default proactive monitor builder for the refactored service package.

Purpose: assemble the production ``ProactiveCoordinator`` and
``ProactiveMonitorService`` from Twinr runtime dependencies while preserving
the legacy public build contract and monkeypatch surface.
"""

from __future__ import annotations

import math
import re

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
    _proactive_respeaker_host_control_conflicts_with_voice_orchestrator,
    _record_component_warning,
    _record_respeaker_dead_capture_blocker,
    _respeaker_capture_probe_duration_ms,
    _safe_emit,
)
from ..display_attention import display_attention_refresh_backend_supported
from ..display_gesture_emoji import resolve_display_gesture_refresh_interval
from ..presence import PresenceSessionController
from ..respeaker_capture_gate import require_stable_respeaker_capture
from ..runtime_contract import (
    ReSpeakerRuntimeContractError,
    assess_respeaker_monitor_startup_contract,
)

_CONTROL_CHARS_RE = re.compile(r"[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "sk-***REDACTED***"),
    (re.compile(r"(?i)\b(bearer\s+)[^\s,;]+"), r"\1***REDACTED***"),
    (re.compile(r"(?i)\b(api[_-]?key\s*[=:]\s*)[^\s,;]+"), r"\1***REDACTED***"),
)

_MIN_POLL_INTERVAL_S = 0.05
_MAX_POLL_INTERVAL_S = 60.0
_MIN_AUDIO_SAMPLE_MS = 50
_MAX_AUDIO_SAMPLE_MS = 10_000
_MIN_CAPTURE_PROBE_MS = 50
_MAX_CAPTURE_PROBE_MS = 15_000
_MIN_VISION_REVIEW_BUFFER_FRAMES = 1
_MAX_VISION_REVIEW_BUFFER_FRAMES = 64
_MIN_VISION_REVIEW_MAX_FRAMES = 1
_MAX_VISION_REVIEW_MAX_FRAMES = 32
_MIN_VISION_REVIEW_MAX_AGE_S = 0.1
_MAX_VISION_REVIEW_MAX_AGE_S = 600.0
_MIN_VISION_REVIEW_MIN_SPACING_S = 0.0
_MAX_VISION_REVIEW_MIN_SPACING_S = 600.0
_MIN_PRESENCE_GRACE_S = 0.0
_MAX_PRESENCE_GRACE_S = 3_600.0


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
    vision_reviewer_cls: Any = OpenAIProactiveVisionReviewer
    vision_frame_buffer_cls: Any = ProactiveVisionFrameBuffer
    portrait_match_provider_cls: Any = PortraitMatchProvider
    coordinator_cls: Any = ProactiveCoordinator
    monitor_service_cls: Any = ProactiveMonitorService


def _sanitize_detail_text(detail: Any, *, max_len: int = 320) -> str:
    text = str(detail or "").strip()
    if not text:
        return "unknown error"
    text = _CONTROL_CHARS_RE.sub(" ", text)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _safe_exception_detail(exc: Exception) -> str:
    return _sanitize_detail_text(_exception_text(exc))


def _warn_config_normalized(
    *,
    runtime: Any,
    emit: Callable[[str], None] | None,
    field_name: str,
    original_value: Any,
    normalized_value: Any,
    minimum: Any,
    maximum: Any,
) -> None:
    _record_component_warning(
        runtime=runtime,
        emit=emit,
        reason="proactive_config_normalized",
        detail=(
            f"{field_name}={_sanitize_detail_text(original_value)!r} is outside the "
            f"safe runtime range [{minimum}, {maximum}] for this proactive monitor. "
            f"Using {normalized_value!r}."
        ),
    )


def _normalize_int_config(
    *,
    runtime: Any,
    emit: Callable[[str], None] | None,
    field_name: str,
    value: Any,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except Exception:
        normalized = minimum
        _warn_config_normalized(
            runtime=runtime,
            emit=emit,
            field_name=field_name,
            original_value=value,
            normalized_value=normalized,
            minimum=minimum,
            maximum=maximum,
        )
        return normalized
    if parsed < minimum or parsed > maximum:
        normalized = max(minimum, min(maximum, parsed))
        _warn_config_normalized(
            runtime=runtime,
            emit=emit,
            field_name=field_name,
            original_value=parsed,
            normalized_value=normalized,
            minimum=minimum,
            maximum=maximum,
        )
        return normalized
    return parsed


def _normalize_float_config(
    *,
    runtime: Any,
    emit: Callable[[str], None] | None,
    field_name: str,
    value: Any,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except Exception:
        normalized = minimum
        _warn_config_normalized(
            runtime=runtime,
            emit=emit,
            field_name=field_name,
            original_value=value,
            normalized_value=normalized,
            minimum=minimum,
            maximum=maximum,
        )
        return normalized
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        bounded = minimum if not math.isfinite(parsed) else max(minimum, min(maximum, parsed))
        _warn_config_normalized(
            runtime=runtime,
            emit=emit,
            field_name=field_name,
            original_value=parsed,
            normalized_value=bounded,
            minimum=minimum,
            maximum=maximum,
        )
        return bounded
    return parsed


def _normalize_vision_provider_name(raw_value: Any) -> str:
    provider_name = str(raw_value or "local").strip().lower()
    alias_map = {
        "local_first": "local",
        "local_camera": "local",
        "local_only": "local",
        "local_strict": "local",
        "picamera2": "local",
        "libcamera": "local",
        "rpicam": "local",
        "imx500": "local",
        "pi_ai_camera": "local",
        "ai_camera": "local",
        "aideck": "aideck_openai",
        "cloud": "openai",
        "openai_remote": "openai",
    }
    return alias_map.get(provider_name, provider_name)


def _make_ambient_audio_observer_factory(
    *,
    deps: BuildDefaultProactiveMonitorDependencies,
    config: TwinrConfig,
    audio_lock: Lock | None,
    sample_ms: int,
    distress_enabled: bool,
) -> Callable[[], Any]:
    def _factory() -> Any:
        sampler = deps.ambient_audio_sampler_cls.from_config(config)
        return deps.ambient_audio_observer_cls(
            sampler=sampler,
            audio_lock=audio_lock,
            sample_ms=sample_ms,
            distress_enabled=distress_enabled,
        )

    return _factory


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
    voice_path_enabled = bool(getattr(config, "voice_orchestrator_enabled", False))
    display_driver_is_hdmi = str(getattr(config, "display_driver", "") or "").strip().lower().startswith("hdmi")
    # Keep the attention-refresh lane only on HDMI backends that are presently
    # safe end-to-end. hdmi_wayland stays fail-closed here because the hidden
    # camera/servo lane has repeatedly reintroduced Pi memory pressure and
    # voice transport starvation in the transcript-first streaming PID.
    display_attention_enabled = display_attention_refresh_backend_supported(config=config)
    display_gesture_enabled = (
        not voice_path_enabled
        and
        resolve_display_gesture_refresh_interval(config) is not None
        and display_driver_is_hdmi
    )
    # Keep the transcript-first voice process single-purpose for proactive
    # prompts and the heavier HDMI gesture lane. Only the explicitly safe
    # attention-refresh backends stay available alongside live voice.
    vision_requested = bool(
        display_attention_enabled
        or (
            not voice_path_enabled
            and (config.proactive_enabled or display_gesture_enabled)
        )
    )
    if (
        not config.proactive_enabled
        and not display_attention_enabled
        and not display_gesture_enabled
    ):
        return None

    proactive_poll_interval_s = _normalize_float_config(
        runtime=runtime,
        emit=emit,
        field_name="proactive_poll_interval_s",
        value=getattr(config, "proactive_poll_interval_s", None),
        minimum=_MIN_POLL_INTERVAL_S,
        maximum=_MAX_POLL_INTERVAL_S,
    )
    proactive_audio_sample_ms = _normalize_int_config(
        runtime=runtime,
        emit=emit,
        field_name="proactive_audio_sample_ms",
        value=getattr(config, "proactive_audio_sample_ms", None),
        minimum=_MIN_AUDIO_SAMPLE_MS,
        maximum=_MAX_AUDIO_SAMPLE_MS,
    )
    presence_grace_s = _normalize_float_config(
        runtime=runtime,
        emit=emit,
        field_name="voice_orchestrator_follow_up_timeout_s",
        value=getattr(config, "voice_orchestrator_follow_up_timeout_s", None),
        minimum=_MIN_PRESENCE_GRACE_S,
        maximum=_MAX_PRESENCE_GRACE_S,
    )
    capture_probe_duration_ms = _normalize_int_config(
        runtime=runtime,
        emit=emit,
        field_name="respeaker_capture_probe_duration_ms",
        value=_respeaker_capture_probe_duration_ms(config),
        minimum=_MIN_CAPTURE_PROBE_MS,
        maximum=_MAX_CAPTURE_PROBE_MS,
    )

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
                f"Proactive triggers were disabled: {_safe_exception_detail(exc)}"
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
                detail=f"PIR monitoring could not be initialized: {_safe_exception_detail(exc)}",
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
            presence_grace_s=presence_grace_s,
            motion_grace_s=presence_grace_s,
            speech_grace_s=presence_grace_s,
        )

    audio_observer = deps.null_audio_observer_cls()
    audio_observer_fallback_factory: Callable[[], Any] | None = None
    ambient_audio_observer_factory: Callable[[], Any] | None = None
    distress_enabled = bool(config.proactive_enabled and config.proactive_audio_distress_enabled)
    shared_voice_capture_conflict = _proactive_pcm_capture_conflicts_with_voice_orchestrator(config)
    shared_respeaker_host_control_conflict = (
        _proactive_respeaker_host_control_conflicts_with_voice_orchestrator(config)
    )
    respeaker_targeted = config_targets_respeaker(
        getattr(config, "audio_input_device", None),
        getattr(config, "proactive_audio_input_device", None),
    )

    def _try_build_ambient_audio_observer(
        *,
        warning_reason: str,
        warning_prefix: str,
    ) -> tuple[Any | None, Callable[[], Any] | None]:
        if ambient_audio_observer_factory is None:
            return None, None
        try:
            observer = ambient_audio_observer_factory()
            return observer, ambient_audio_observer_factory
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason=warning_reason,
                detail=f"{warning_prefix}: {_safe_exception_detail(exc)}",
            )
            return None, None

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

    if config.proactive_audio_enabled and not shared_voice_capture_conflict:
        ambient_audio_observer_factory = _make_ambient_audio_observer_factory(
            deps=deps,
            config=config,
            audio_lock=audio_lock,
            sample_ms=proactive_audio_sample_ms,
            distress_enabled=distress_enabled,
        )
        if not respeaker_targeted:
            observer, observer_factory = _try_build_ambient_audio_observer(
                warning_reason="audio_sampler_init_failed",
                warning_prefix="Ambient audio sampling could not be initialized",
            )
            if observer is not None and observer_factory is not None:
                audio_observer = observer
                audio_observer_fallback_factory = observer_factory
            else:
                audio_observer = deps.null_audio_observer_cls()

    if respeaker_targeted and config.proactive_audio_enabled:
        if shared_respeaker_host_control_conflict:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="respeaker_signal_provider_disabled_for_voice_runtime",
                detail=(
                    "Proactive XVF3800 host-control monitoring was disabled because the "
                    "voice orchestrator owns the same ReSpeaker device inside the streaming "
                    "runtime. The proactive monitor stays camera/PIR-only so USB control "
                    "snapshots cannot starve live voice transport."
                ),
            )
        else:
            try:
                base_signal_provider = deps.base_signal_provider_cls(
                    sensor_window_ms=proactive_audio_sample_ms,
                    assistant_output_active_predicate=lambda: _assistant_output_active(runtime),
                )
                signal_provider = deps.scheduled_signal_provider_cls.from_config(
                    config,
                    provider=base_signal_provider,
                )
                initial_signal = signal_provider.observe()
                startup_contract = assess_respeaker_monitor_startup_contract(initial_signal)
                if startup_contract.blocking:
                    blocker_code = _sanitize_detail_text(startup_contract.blocker_code or "blocked", max_len=64)
                    detail = _sanitize_detail_text(
                        startup_contract.detail
                        or "ReSpeaker startup contract blocked monitor initialization."
                    )
                    _safe_emit(emit, f"respeaker_runtime_blocker={blocker_code}")
                    _append_ops_event(
                        runtime,
                        event="proactive_component_blocked",
                        level="error",
                        message="Proactive monitor startup was blocked by the ReSpeaker runtime contract.",
                        data={
                            "reason": _sanitize_detail_text(
                                startup_contract.ops_reason or "respeaker_startup_blocked",
                                max_len=96,
                            ),
                            "detail": detail,
                            "blocker_code": blocker_code,
                            "device_runtime_mode": _sanitize_detail_text(
                                initial_signal.device_runtime_mode,
                                max_len=96,
                            ),
                            "host_control_ready": bool(initial_signal.host_control_ready),
                            "transport_reason": _sanitize_detail_text(
                                initial_signal.transport_reason,
                                max_len=96,
                            ),
                        },
                        emit=emit,
                    )
                    raise ReSpeakerRuntimeContractError(detail)
                if not initial_signal.host_control_ready:
                    reason = _sanitize_detail_text(
                        initial_signal.transport_reason or "unknown_transport_state",
                        max_len=96,
                    )
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
                if ambient_audio_observer_factory is not None:
                    try:
                        require_stable_respeaker_capture(
                            sampler=deps.ambient_audio_sampler_cls.from_config(config),
                            duration_ms=capture_probe_duration_ms,
                        )
                    except AudioCaptureReadinessError as exc:
                        detail = _sanitize_detail_text(
                            _record_respeaker_dead_capture_blocker(
                                runtime=runtime,
                                emit=emit,
                                probe=exc.probe,
                                stage="startup",
                                signal=initial_signal,
                            )
                        )
                        raise ReSpeakerRuntimeContractError(detail) from exc

                base_audio_observer = deps.null_audio_observer_cls()
                if ambient_audio_observer_factory is not None:
                    observer, observer_factory = _try_build_ambient_audio_observer(
                        warning_reason="respeaker_pcm_fallback_init_failed",
                        warning_prefix="Ambient PCM fallback could not be initialized",
                    )
                    if observer is not None:
                        base_audio_observer = observer
                        audio_observer_fallback_factory = observer_factory

                audio_observer = deps.respeaker_audio_observer_cls(
                    signal_provider=signal_provider,
                    fallback_observer=base_audio_observer,
                )

                if audio_observer_fallback_factory is not None:
                    previous_factory = audio_observer_fallback_factory

                    def _fallback_respeaker_audio_observer_factory() -> Any:
                        fallback_observer = previous_factory()
                        return deps.respeaker_audio_observer_cls(
                            signal_provider=signal_provider,
                            fallback_observer=fallback_observer,
                        )

                    audio_observer_fallback_factory = _fallback_respeaker_audio_observer_factory

            except Exception as exc:
                if isinstance(exc, ReSpeakerRuntimeContractError):
                    detail = _safe_exception_detail(exc)
                    observer, observer_factory = _try_build_ambient_audio_observer(
                        warning_reason="respeaker_pcm_fallback_init_failed",
                        warning_prefix="Ambient PCM fallback could not be initialized after ReSpeaker startup failure",
                    )
                    if observer is not None and observer_factory is not None:
                        _record_component_warning(
                            runtime=runtime,
                            emit=emit,
                            reason="respeaker_runtime_degraded_to_pcm",
                            detail=(
                                "ReSpeaker host-control startup failed; degrading to direct PCM ambient "
                                f"audio monitoring: {detail}"
                            ),
                        )
                        audio_observer = observer
                        audio_observer_fallback_factory = observer_factory
                    elif display_attention_enabled:
                        _preserve_local_attention_on_audio_block(
                            runtime=runtime,
                            emit=emit,
                            detail=detail,
                        )
                        audio_observer = deps.null_audio_observer_cls()
                        audio_observer_fallback_factory = None
                    elif pir_monitor is not None or vision_requested or display_gesture_enabled:
                        _record_component_warning(
                            runtime=runtime,
                            emit=emit,
                            reason="proactive_audio_runtime_blocked",
                            detail=(
                                "Audio startup was blocked, but another non-audio proactive path remains "
                                f"available. Audio was disabled: {detail}"
                            ),
                        )
                        audio_observer = deps.null_audio_observer_cls()
                        audio_observer_fallback_factory = None
                    else:
                        raise
                else:
                    _record_component_warning(
                        runtime=runtime,
                        emit=emit,
                        reason="respeaker_signal_provider_init_failed",
                        detail=f"ReSpeaker signal-provider initialization failed: {_safe_exception_detail(exc)}",
                    )
                    observer, observer_factory = _try_build_ambient_audio_observer(
                        warning_reason="audio_sampler_init_failed",
                        warning_prefix="Ambient audio fallback could not be initialized after ReSpeaker provider failure",
                    )
                    if observer is not None and observer_factory is not None:
                        audio_observer = observer
                        audio_observer_fallback_factory = observer_factory

    vision_observer: Any | None = None
    if vision_requested:
        provider_name = _normalize_vision_provider_name(
            getattr(config, "proactive_vision_provider", "local")
        )
        try:
            if provider_name == "local":
                vision_observer = deps.local_vision_provider_cls.from_config(config)
            elif provider_name == "openai":
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
            elif provider_name in {"remote_proxy", "remote_frame"}:
                raise ValueError(
                    "Legacy helper-Pi proactive vision providers are no longer supported. "
                    "Twinr now requires local proactive vision on the main Pi."
                )
            else:
                raise ValueError(f"Unknown proactive vision provider kind: {provider_name}")
        except Exception as exc:
            detail = _safe_exception_detail(exc)
            _append_ops_event(
                runtime,
                event="proactive_component_blocked",
                level="error",
                message="Proactive monitor startup was blocked because the configured vision provider failed.",
                data={
                    "reason": "vision_observer_init_failed",
                    "provider": provider_name,
                    "detail": detail,
                },
                emit=emit,
            )
            raise RuntimeError(
                f"Vision observation provider {provider_name!r} failed to initialize: {detail}"
            ) from exc

    vision_reviewer = None
    if config.proactive_enabled and config.proactive_vision_review_enabled and vision_observer is not None:
        review_buffer_frames = _normalize_int_config(
            runtime=runtime,
            emit=emit,
            field_name="proactive_vision_review_buffer_frames",
            value=getattr(config, "proactive_vision_review_buffer_frames", None),
            minimum=_MIN_VISION_REVIEW_BUFFER_FRAMES,
            maximum=_MAX_VISION_REVIEW_BUFFER_FRAMES,
        )
        review_max_frames = _normalize_int_config(
            runtime=runtime,
            emit=emit,
            field_name="proactive_vision_review_max_frames",
            value=getattr(config, "proactive_vision_review_max_frames", None),
            minimum=_MIN_VISION_REVIEW_MAX_FRAMES,
            maximum=_MAX_VISION_REVIEW_MAX_FRAMES,
        )
        if review_max_frames > review_buffer_frames:
            _warn_config_normalized(
                runtime=runtime,
                emit=emit,
                field_name="proactive_vision_review_max_frames",
                original_value=review_max_frames,
                normalized_value=review_buffer_frames,
                minimum=_MIN_VISION_REVIEW_MAX_FRAMES,
                maximum=review_buffer_frames,
            )
            review_max_frames = review_buffer_frames

        review_max_age_s = _normalize_float_config(
            runtime=runtime,
            emit=emit,
            field_name="proactive_vision_review_max_age_s",
            value=getattr(config, "proactive_vision_review_max_age_s", None),
            minimum=_MIN_VISION_REVIEW_MAX_AGE_S,
            maximum=_MAX_VISION_REVIEW_MAX_AGE_S,
        )
        review_min_spacing_s = _normalize_float_config(
            runtime=runtime,
            emit=emit,
            field_name="proactive_vision_review_min_spacing_s",
            value=getattr(config, "proactive_vision_review_min_spacing_s", None),
            minimum=_MIN_VISION_REVIEW_MIN_SPACING_S,
            maximum=_MAX_VISION_REVIEW_MIN_SPACING_S,
        )
        if review_min_spacing_s > review_max_age_s:
            _warn_config_normalized(
                runtime=runtime,
                emit=emit,
                field_name="proactive_vision_review_min_spacing_s",
                original_value=review_min_spacing_s,
                normalized_value=review_max_age_s,
                minimum=_MIN_VISION_REVIEW_MIN_SPACING_S,
                maximum=review_max_age_s,
            )
            review_min_spacing_s = review_max_age_s

        try:
            vision_reviewer = deps.vision_reviewer_cls(
                backend=backend,
                frame_buffer=deps.vision_frame_buffer_cls(
                    max_items=review_buffer_frames,
                ),
                max_frames=review_max_frames,
                max_age_s=review_max_age_s,
                min_spacing_s=review_min_spacing_s,
            )
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="vision_reviewer_init_failed",
                detail=f"Buffered vision reviewer initialization failed: {_safe_exception_detail(exc)}",
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
                detail=f"Local portrait-match provider initialization failed: {_safe_exception_detail(exc)}",
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
        poll_interval_s=proactive_poll_interval_s,
        emit=emit,
    )


__all__ = [
    "BuildDefaultProactiveMonitorDependencies",
    "build_default_proactive_monitor",
]
