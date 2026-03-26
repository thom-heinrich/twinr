"""Coordinate proactive sensing, voice-presence arming, and monitor lifecycle.

This module assembles the runtime-facing proactive monitor used by Twinr
workflows. It wires sensor observers, presence-session
tracking, buffered vision review, automation observation export, and the
background worker lifecycle without owning the lower-level scoring or hardware
adapters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Callable, cast
import logging
import math
import time
import uuid

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span
from twinr.hardware.audio import (
    AudioCaptureReadinessError,
    AudioCaptureReadinessProbe,
    AmbientAudioSampler,
    capture_device_identity,
    resolve_capture_device,
)
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.camera_ai.gesture_forensics import GestureForensics
from twinr.hardware.pir import GpioPirMonitor, configured_pir_monitor
from twinr.hardware.servo_follow import AttentionServoController, AttentionServoDecision
from twinr.hardware.respeaker import (
    build_respeaker_claim_payloads,
    config_targets_respeaker,
    resolve_respeaker_indicator_state,
    ScheduledReSpeakerSignalProvider,
)
from twinr.hardware.portrait_match import PortraitMatchProvider
from twinr.hardware.respeaker.signal_provider import ReSpeakerSignalProvider
from twinr.providers.openai import OpenAIBackend

from ..social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurface, ProactiveCameraSurfaceUpdate
from ..social.aideck_camera_provider import AIDeckOpenAIVisionObservationProvider
from ..social.engine import SocialAudioObservation, SocialObservation, SocialTriggerDecision, SocialTriggerEngine, SocialVisionObservation
from ..social.local_camera_provider import LocalAICameraObservationProvider
from ..social.remote_camera_provider import (
    RemoteAICameraObservationProvider,
    RemoteFrameAICameraObservationProvider,
)
from ..social.observers import AmbientAudioObservationProvider, NullAudioObservationProvider, OpenAIVisionObservationProvider, ReSpeakerAudioObservationProvider
from ..social.observers import ProactiveAudioSnapshot
from ..social.vision_review import (
    OpenAIProactiveVisionReviewer,
    ProactiveVisionFrameBuffer,
    ProactiveVisionReview,
    is_reviewable_image_trigger,
)
from .affect_proxy import AffectProxySnapshot, derive_affect_proxy
from .ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from .attention_targeting import (
    MultimodalAttentionTargetSnapshot,
    MultimodalAttentionTargetTracker,
)
from .attention_debug_stream import AttentionDebugStream
from .display_attention import DisplayAttentionCuePublishResult, DisplayAttentionCuePublisher
from .display_attention_camera_fusion import DisplayAttentionCameraFusion
from .display_ambient_impulses import (
    DisplayAmbientImpulsePublishResult,
    DisplayAmbientImpulsePublisher,
)
from .display_attention import (
    display_attention_refresh_supported,
    resolve_display_attention_refresh_interval,
)
from .display_debug_signals import DisplayDebugSignalPublisher
from .display_gesture_emoji import (
    DisplayGestureEmojiDecision,
    DisplayGestureEmojiPublishResult,
    DisplayGestureEmojiPublisher,
    display_gesture_refresh_supported,
    resolve_display_gesture_refresh_interval,
)
from .gesture_ack_lane import GestureAckLane
from .gesture_debug_stream import GestureDebugStream
from .gesture_wakeup_priority import decide_gesture_wakeup_priority
from .gesture_wakeup_dispatcher import GestureWakeupDispatcher
from .gesture_wakeup_lane import GestureWakeupDecision, GestureWakeupLane
from .identity_fusion import (
    MultimodalIdentityFusionSnapshot,
    TemporalIdentityFusionTracker,
)
from .audio_policy import ReSpeakerAudioPolicySnapshot, ReSpeakerAudioPolicyTracker
from .known_user_hint import KnownUserHintSnapshot, derive_known_user_hint
from .multimodal_initiative import (
    ReSpeakerMultimodalInitiativeSnapshot,
    derive_respeaker_multimodal_initiative,
)
from .person_state import PersonStateSnapshot, derive_person_state
from .presence import PresenceSessionController, PresenceSessionSnapshot
from .pir_open_gate import open_pir_monitor_with_busy_retry
from .respeaker_capture_gate import require_stable_respeaker_capture
from .portrait_match import PortraitMatchSnapshot, derive_portrait_match
from .runtime_contract import (
    ReSpeakerRuntimeContractError,
    assess_respeaker_monitor_startup_contract,
    is_respeaker_runtime_hard_block,
)
from .safety_trigger_fusion import SafetyTriggerFusionBridge
from .speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)
from . import service_attention_helpers
from . import service_gesture_helpers

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime.runtime import TwinrRuntime

_VISION_REVIEW_FAIL_OPEN_TRIGGERS = frozenset(
    {"possible_fall", "floor_stillness", "distress_possible"}
)
_DEFAULT_CLOSE_JOIN_TIMEOUT_S = 5.0
_DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES = frozenset({"waiting", "listening", "processing", "answering"})
_DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES = frozenset({"error"})
_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S = 2.0

_LOGGER = logging.getLogger(__name__)


# AUDIT-FIX(#1): Isolate telemetry and log formatting so ops-event persistence or emit sinks cannot kill safety-critical monitoring.
def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    """Emit one telemetry line while suppressing sink failures."""

    if emit is None:
        return
    try:
        emit(line)
    except Exception:
        _LOGGER.warning("Proactive emit sink failed.", exc_info=True)
        return


# AUDIT-FIX(#1): Sanitize exception text before logging to avoid multiline/log-injection issues and overlong payloads.
def _exception_text(error: BaseException | object, *, limit: int = 240) -> str:
    """Normalize one exception payload into bounded log-safe text."""

    raw = str(error) if not isinstance(error, BaseException) else (str(error) or error.__class__.__name__)
    text = " ".join(raw.split())
    if not text:
        text = "unknown_error"
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _emit_token(value: object, *, limit: int = 96) -> str:
    """Render one bounded telemetry token for journal-friendly key=value lines."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value).lower()
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    text = " ".join(str(value).split())
    if not text:
        return "none"
    safe_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"_", "-", ".", ":", "/"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars) or "none"
    if len(safe) <= limit:
        return safe
    if limit <= 3:
        return safe[:limit]
    return safe[: limit - 3] + "..."


def _emit_key_value_line(prefix: str, /, **fields: object) -> str:
    """Build one stable key=value telemetry line for changed-only journal tracing."""

    parts = [prefix]
    for key, value in fields.items():
        parts.append(f"{key}={_emit_token(value)}")
    return " ".join(parts)


# AUDIT-FIX(#1): Degrade gracefully if file-backed ops-event persistence is unavailable.
def _append_ops_event(
    runtime: TwinrRuntime,
    *,
    event: str,
    message: str,
    data: dict[str, Any],
    emit: Callable[[str], None] | None = None,
    level: str | None = None,
) -> None:
    """Append one ops event without letting persistence failures escape."""

    kwargs: dict[str, Any] = {
        "event": event,
        "message": message,
        "data": data,
    }
    if level is not None:
        kwargs["level"] = level
    try:
        runtime.ops_events.append(**kwargs)
    except Exception:
        _safe_emit(emit, f"ops_event_append_failed={event}")


# AUDIT-FIX(#8): Normalize optional config strings so None/whitespace values do not crash subsystem initialization.
def _normalize_optional_text(*values: Any) -> str:
    """Return the first non-blank text value from a config-like list."""

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _proactive_audio_capture_device(config: TwinrConfig) -> str:
    """Return the device used by proactive ambient PCM sampling."""

    return resolve_capture_device(
        getattr(config, "proactive_audio_input_device", None),
        getattr(config, "audio_input_device", None),
    )


def _voice_orchestrator_capture_device(config: TwinrConfig) -> str:
    """Return the device used by the long-lived voice-orchestrator capture."""

    return resolve_capture_device(
        getattr(config, "voice_orchestrator_audio_device", None),
        getattr(config, "proactive_audio_input_device", None),
        getattr(config, "audio_input_device", None),
    )


def _proactive_pcm_capture_conflicts_with_voice_orchestrator(
    config: TwinrConfig,
) -> bool:
    """Return whether proactive PCM fallback would fight a shared voice capture."""

    if not bool(getattr(config, "voice_orchestrator_enabled", False)):
        return False
    return capture_device_identity(_proactive_audio_capture_device(config)) == capture_device_identity(
        _voice_orchestrator_capture_device(config)
    )


# AUDIT-FIX(#8): Normalize sequence-like config inputs from older env schemas before using them.
def _normalize_text_tuple(values: Any) -> tuple[str, ...]:
    """Normalize sequence-like config input to one tuple of non-blank strings."""

    if values is None:
        return ()
    if isinstance(values, (str, Path)):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _round_optional_seconds(value: float | None) -> float | None:
    """Round one optional duration to milliseconds for ops-safe payloads."""

    if value is None:
        return None
    return round(max(0.0, float(value)), 3)


def _round_optional_ratio(value: float | None) -> float | None:
    """Round one optional bounded ratio or score for ops-safe payloads."""

    if value is None:
        return None
    return round(float(value), 4)


def _format_firmware_version(version: tuple[int, int, int] | None) -> str | None:
    """Format one optional firmware tuple for ops/event payloads."""

    if version is None:
        return None
    return ".".join(str(int(part)) for part in version)


def _respeaker_capture_probe_duration_ms(config: TwinrConfig) -> int:
    """Return a short bounded capture probe window for ReSpeaker startup checks."""

    chunk_ms = max(20, int(getattr(config, "audio_chunk_ms", 100) or 100))
    requested_ms = max(chunk_ms, int(getattr(config, "proactive_audio_sample_ms", chunk_ms) or chunk_ms))
    return min(requested_ms, max(250, chunk_ms * 3))


def _display_attention_refresh_allowed_runtime_status(runtime_status_value: object) -> bool:
    """Return whether bounded local HDMI eye-follow may refresh in this runtime state."""

    normalized = str(runtime_status_value or "").strip().lower()
    if normalized in _DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES:
        return True
    # Keep required-remote outages fail-closed for the actual agent/runtime while
    # still allowing purely local HDMI face-follow cues to run from on-device
    # camera/audio signals.
    return normalized in _DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES


def _preserve_local_attention_on_audio_block(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    detail: str,
) -> None:
    """Keep local HDMI camera follow alive while marking the runtime as blocked.

    A broken ReSpeaker startup path must still surface as a hard runtime error,
    but it should not prevent Twinr from running purely local camera-driven
    HDMI cues such as eye-follow and gesture acknowledgements.
    """

    _record_component_warning(
        runtime=runtime,
        emit=emit,
        reason="respeaker_camera_follow_only",
        detail=(
            "ReSpeaker startup is blocked, so Twinr preserved only the local "
            f"HDMI camera-follow path: {detail}"
        ),
    )
    try:
        runtime.fail(detail)
    except Exception:
        _safe_emit(emit, f"runtime_fail_failed={_exception_text(detail)}")


def _respeaker_dead_capture_payload(
    *,
    probe: AudioCaptureReadinessProbe,
    stage: str,
    signal: object | None = None,
) -> dict[str, Any]:
    """Build one ops-safe payload for unreadable ReSpeaker capture failures."""

    payload: dict[str, Any] = {
        "stage": stage,
        "capture_device": probe.device,
        "capture_sample_rate": probe.sample_rate,
        "capture_channels": probe.channels,
        "capture_chunk_ms": probe.chunk_ms,
        "capture_probe_duration_ms": probe.duration_ms,
        "capture_probe_target_chunk_count": probe.target_chunk_count,
        "capture_probe_chunk_count": probe.captured_chunk_count,
        "capture_probe_bytes": probe.captured_bytes,
        "capture_probe_ready": probe.ready,
        "capture_probe_failure_reason": probe.failure_reason,
        "capture_probe_detail": probe.detail,
        "transport_reason": "capture_unreadable",
    }
    if signal is not None:
        payload.update(
            {
                "device_runtime_mode": getattr(signal, "device_runtime_mode", None),
                "host_control_ready": getattr(signal, "host_control_ready", None),
                "transport_reason": getattr(signal, "transport_reason", None) or "capture_unreadable",
                "firmware_version": _format_firmware_version(getattr(signal, "firmware_version", None)),
            }
        )
    return payload


def _record_respeaker_dead_capture_blocker(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    probe: AudioCaptureReadinessProbe,
    stage: str,
    signal: object | None = None,
) -> str:
    """Emit explicit alert/blocker events when ReSpeaker capture yields no frames."""

    payload = _respeaker_dead_capture_payload(
        probe=probe,
        stage=stage,
        signal=signal,
    )
    detail = (
        "ReSpeaker XVF3800 is enumerated, but the configured capture path yielded no readable "
        f"audio frames. {probe.detail or 'Twinr refuses to treat this path as ready.'}"
    )
    _safe_emit(emit, "respeaker_runtime_alert=capture_unknown")
    _append_ops_event(
        runtime,
        event="respeaker_runtime_alert",
        level="error",
        message=detail,
        data={
            **payload,
            "alert_code": "capture_unknown",
        },
        emit=emit,
    )
    _safe_emit(emit, "respeaker_runtime_blocker=dead_capture")
    _append_ops_event(
        runtime,
        event="respeaker_runtime_blocker",
        level="error",
        message="ReSpeaker capture is unreadable even though the device still enumerates.",
        data={
            **payload,
            "alert_code": "dead_capture",
            "blocker_code": "dead_capture",
        },
        emit=emit,
    )
    _append_ops_event(
        runtime,
        event="proactive_component_blocked",
        level="error",
        message="ReSpeaker unreadable capture blocked the proactive audio path.",
        data={
            **payload,
            "reason": (
                "respeaker_dead_capture_blocked"
                if stage == "startup"
                else "respeaker_dead_capture_runtime_blocked"
            ),
            "detail": detail,
            "blocker_code": "dead_capture",
        },
        emit=emit,
    )
    return detail


def _assistant_output_active(runtime: TwinrRuntime) -> bool:
    """Return whether Twinr is actively speaking right now."""

    try:
        return getattr(runtime.status, "value", None) == "answering"
    except Exception:
        return False

# AUDIT-FIX(#4): Log degraded-mode component startup instead of failing the whole monitor for optional hardware or provider issues.
def _record_component_warning(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    reason: str,
    detail: str,
) -> None:
    """Record one degraded-mode warning during proactive monitor setup."""

    _safe_emit(emit, f"proactive_component_warning={reason}")
    _append_ops_event(
        runtime,
        event="proactive_component_warning",
        level="warning",
        message="Proactive monitor is running in degraded mode.",
        data={
            "reason": reason,
            "detail": detail,
        },
        emit=emit,
    )


@dataclass(frozen=True, slots=True)
class ProactiveTickResult:
    """Describe the externally relevant outcome of one monitor tick."""

    decision: SocialTriggerDecision | None = None
    inspected: bool = False
    person_visible: bool = False


# AUDIT-FIX(#4): Provide a no-op trigger engine so audio/camera monitoring can still run when proactive triggers are disabled or engine setup fails.
class _NullSocialTriggerEngine:
    """Provide a no-op trigger engine when proactive triggers are disabled."""

    best_evaluation = None

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Return no proactive decision for the given observation."""

        return None


class ProactiveCoordinator:
    """Coordinate one proactive monitor tick across sensors and policies.

    The coordinator owns the runtime-facing orchestration for PIR, vision,
    ambient audio, presence sessions, trigger review, and
    automation observation export. Lower-level trigger scoring and voice
    matching remain in sibling packages.
    """

    def __init__(
        self,
        *,
        config: TwinrConfig,
        runtime: TwinrRuntime,
        engine: SocialTriggerEngine,
        trigger_handler: Callable[[SocialTriggerDecision], bool],
        vision_observer,
        pir_monitor: GpioPirMonitor | None = None,
        audio_observer=None,
        presence_session: PresenceSessionController | None = None,
        vision_reviewer: OpenAIProactiveVisionReviewer | None = None,
        portrait_match_provider: PortraitMatchProvider | None = None,
        gesture_wakeup_handler: Callable[[GestureWakeupDecision], bool] | None = None,
        idle_predicate: Callable[[], bool] | None = None,
        observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
        live_context_handler: Callable[[dict[str, Any]], None] | None = None,
        display_attention_publisher: DisplayAttentionCuePublisher | None = None,
        display_debug_signal_publisher: DisplayDebugSignalPublisher | None = None,
        display_gesture_emoji_publisher: DisplayGestureEmojiPublisher | None = None,
        display_ambient_impulse_publisher: DisplayAmbientImpulsePublisher | None = None,
        attention_servo_controller: AttentionServoController | None = None,
        display_attention_debug_stream: AttentionDebugStream | None = None,
        display_gesture_debug_stream: GestureDebugStream | None = None,
        emit: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        audio_observer_fallback_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize one coordinator from already-built dependencies."""

        self.config = config
        self.runtime = runtime
        self.engine = engine
        self.trigger_handler = trigger_handler
        self.vision_observer = vision_observer
        self.pir_monitor = pir_monitor
        self._null_audio_observer = NullAudioObservationProvider()  # AUDIT-FIX(#6): keep a guaranteed local fallback snapshot source for audio-path failures.
        self.audio_observer = audio_observer or self._null_audio_observer
        self._audio_observer_fallback_factory = audio_observer_fallback_factory
        self.presence_session = presence_session
        self.vision_reviewer = vision_reviewer
        self.portrait_match_provider = portrait_match_provider
        self.gesture_wakeup_handler = gesture_wakeup_handler
        self.idle_predicate = idle_predicate
        self.observation_handler = observation_handler
        self.live_context_handler = live_context_handler
        self.display_attention_publisher = display_attention_publisher or DisplayAttentionCuePublisher.from_config(
            config
        )
        self.display_debug_signal_publisher = display_debug_signal_publisher or DisplayDebugSignalPublisher.from_config(
            config
        )
        self.display_gesture_emoji_publisher = display_gesture_emoji_publisher or DisplayGestureEmojiPublisher.from_config(
            config
        )
        self.display_ambient_impulse_publisher = (
            display_ambient_impulse_publisher or DisplayAmbientImpulsePublisher.from_config(config)
        )
        self.attention_servo_controller = attention_servo_controller or AttentionServoController.from_config(config)
        self.display_attention_debug_stream = display_attention_debug_stream or AttentionDebugStream.from_config(
            config
        )
        self.display_gesture_debug_stream = display_gesture_debug_stream or GestureDebugStream.from_config(
            config
        )
        self.emit = emit or (lambda _line: None)
        self.clock = clock
        project_root = Path(getattr(config, "project_root", Path(__file__).resolve().parents[3])).expanduser()
        self._gesture_forensics = GestureForensics.from_env(
            project_root=project_root,
            service="ProactiveGestureRefresh",
        )
        self._attention_servo_forensic_trace_enabled = bool(
            getattr(config, "attention_servo_forensic_trace_enabled", False)
        )
        self._attention_servo_forensic_run_id = uuid.uuid4().hex
        self._attention_servo_forensic_tick_index = 0
        self._last_motion_at: float | None = None
        self._last_capture_at: float | None = None
        self._last_display_attention_refresh_at: float | None = None
        self._last_display_gesture_refresh_at: float | None = None
        self._last_audio_snapshot: ProactiveAudioSnapshot | None = None
        self._last_audio_snapshot_at: float | None = None
        self._last_display_attention_follow_key: tuple[object, ...] | None = None
        self._last_attention_follow_pipeline_key: tuple[object, ...] | None = None
        self._last_attention_servo_follow_key: tuple[object, ...] | None = None
        self._last_observation_key: tuple[object, ...] | None = None
        self._last_attention_vision_refresh_mode: str | None = None
        self._last_attention_audio_refresh_mode: str | None = None
        self._last_gesture_vision_refresh_mode: str | None = None
        self._last_display_attention_fusion_debug: dict[str, Any] | None = None
        self._camera_surface = ProactiveCameraSurface.from_config(config)
        self._display_attention_camera_surface = ProactiveCameraSurface.from_config(config)
        self._display_attention_camera_fusion = DisplayAttentionCameraFusion.from_config(config)
        self._safety_trigger_fusion = SafetyTriggerFusionBridge.from_config(
            config,
            engine=engine,
        )
        self._gesture_ack_lane = GestureAckLane.from_config(config)
        self._gesture_wakeup_lane = GestureWakeupLane.from_config(config)
        self._gesture_wakeup_dispatcher = GestureWakeupDispatcher(
            handle_decision=self._run_gesture_wakeup_handler,
        )
        self._last_sensor_flags: dict[str, bool] = {}
        self._speech_detected_since: float | None = None
        self._quiet_since: float | None = None
        self._last_presence_key: tuple[object, ...] | None = None
        self._last_respeaker_runtime_alert_code: str | None = None
        self._last_respeaker_runtime_blocker_code: str | None = None
        self._respeaker_targeted = config_targets_respeaker(
            getattr(config, "audio_input_device", None),
            getattr(config, "proactive_audio_input_device", None),
        )
        self._last_possible_fall_prompted_session_id: int | None = None
        self.latest_presence_snapshot: PresenceSessionSnapshot | None = None
        self.latest_audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None = None
        self.latest_speaker_association_snapshot: ReSpeakerSpeakerAssociationSnapshot | None = None
        self.latest_multimodal_initiative_snapshot: ReSpeakerMultimodalInitiativeSnapshot | None = None
        self.latest_ambiguous_room_guard_snapshot: AmbiguousRoomGuardSnapshot | None = None
        self.latest_identity_fusion_snapshot: MultimodalIdentityFusionSnapshot | None = None
        self.latest_portrait_match_snapshot: PortraitMatchSnapshot | None = None
        self.latest_known_user_hint_snapshot: KnownUserHintSnapshot | None = None
        self.latest_affect_proxy_snapshot: AffectProxySnapshot | None = None
        self.latest_attention_target_snapshot: MultimodalAttentionTargetSnapshot | None = None
        self.latest_person_state_snapshot: PersonStateSnapshot | None = None
        self.audio_policy_tracker = ReSpeakerAudioPolicyTracker.from_config(config)
        self.identity_fusion_tracker = TemporalIdentityFusionTracker.from_config(config)
        self.attention_target_tracker = MultimodalAttentionTargetTracker.from_config(config)

    # AUDIT-FIX(#1): Route all module-local emits through a non-throwing wrapper.
    def _emit(self, line: str) -> None:
        """Emit one coordinator-local telemetry line safely."""

        _safe_emit(self.emit, line)

    # AUDIT-FIX(#1): Route all module-local ops events through a non-throwing wrapper.
    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, Any],
        level: str | None = None,
    ) -> None:
        """Append one coordinator-local ops event safely."""

        _append_ops_event(
            self.runtime,
            event=event,
            message=message,
            data=data,
            level=level,
            emit=self.emit,
        )

    # AUDIT-FIX(#1): Standardize internal fault reporting so recovery code never re-throws while handling errors.
    def _record_fault(
        self,
        *,
        event: str,
        message: str,
        error: BaseException | object,
        data: dict[str, Any] | None = None,
        level: str = "error",
    ) -> None:
        """Record one recoverable coordinator fault in telemetry and ops events."""

        error_text = _exception_text(error)
        payload = dict(data or {})
        payload["error"] = error_text
        self._emit(f"{event}={error_text}")
        self._append_ops_event(
            event=event,
            level=level,
            message=message,
            data=payload,
        )

    # AUDIT-FIX(#6): Audio observation faults should degrade to a null snapshot instead of aborting the full monitoring cycle.
    def _observe_audio_safe(self):
        """Collect one ambient-audio snapshot with null fallback on failure."""

        try:
            snapshot = self.audio_observer.observe()
        except Exception as exc:
            self._record_fault(
                event="proactive_audio_observe_failed",
                message="Ambient audio observation failed; using a null audio snapshot for this tick.",
                error=exc,
            )
            if self._respeaker_targeted and isinstance(exc, AudioCaptureReadinessError):
                self._block_respeaker_dead_capture(exc)
            try:
                snapshot = self._null_audio_observer.observe()
            except Exception as fallback_exc:
                self._record_fault(
                    event="proactive_null_audio_observe_failed",
                    message="Null audio observation fallback failed.",
                    error=fallback_exc,
                )
                raise
        return self._store_audio_snapshot(snapshot=snapshot)

    def _store_audio_snapshot(
        self,
        *,
        snapshot: ProactiveAudioSnapshot,
        observed_at: float | None = None,
    ) -> ProactiveAudioSnapshot:
        """Remember the latest audio snapshot for fast local HCI refresh paths."""

        self._last_audio_snapshot = snapshot
        self._last_audio_snapshot_at = self.clock() if observed_at is None else float(observed_at)
        return snapshot

    def _observe_audio_for_attention_refresh(
        self,
        *,
        now: float,
    ) -> ProactiveAudioSnapshot:
        """Return a low-latency audio snapshot for HDMI attention refresh.

        The fast HDMI gaze/gesture loop must not block on full ambient PCM
        sampling windows. Prefer direct signal-only snapshots when the provider
        supports them, otherwise reuse a recent cached observation or fall back
        conservatively.
        """

        fast_observe = getattr(self.audio_observer, "observe_signal_only", None)
        if callable(fast_observe):
            fast_observe_fn = cast(Callable[[], ProactiveAudioSnapshot], fast_observe)
            try:
                snapshot = fast_observe_fn()  # pylint: disable=not-callable
            except Exception as exc:
                self._last_attention_audio_refresh_mode = "signal_only_failed_fallback"
                self._record_fault(
                    event="proactive_attention_audio_fast_observe_failed",
                    message="Fast audio observation failed during HDMI attention refresh; using cached/null audio instead.",
                    error=exc,
                )
                return self._attention_refresh_audio_fallback(now=now)
            self._last_attention_audio_refresh_mode = "signal_only"
            return self._store_audio_snapshot(snapshot=snapshot, observed_at=now)

        if isinstance(self.audio_observer, AmbientAudioObservationProvider):
            self._last_attention_audio_refresh_mode = "cache_or_null"
            return self._attention_refresh_audio_fallback(now=now)

        self._last_attention_audio_refresh_mode = "default_observe"
        return self._observe_audio_safe()

    def _attention_refresh_audio_fallback(
        self,
        *,
        now: float,
    ) -> ProactiveAudioSnapshot:
        """Return cached or null audio when the fast loop must stay non-blocking."""

        if (
            self._last_audio_snapshot is not None
            and self._last_audio_snapshot_at is not None
            and (now - self._last_audio_snapshot_at) <= _ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S
        ):
            self._last_attention_audio_refresh_mode = "cache"
            return self._last_audio_snapshot
        self._last_attention_audio_refresh_mode = "null"
        return self._store_audio_snapshot(
            snapshot=self._null_audio_observer.observe(),
            observed_at=now,
        )

    def _observe_vision_for_attention_refresh(self):
        """Collect one low-cost vision snapshot for the HDMI eye-follow path."""

        if self.vision_observer is None:
            self._last_attention_vision_refresh_mode = "missing"
            return None
        fast_observe = getattr(self.vision_observer, "observe_attention", None)
        if callable(fast_observe):
            try:
                self._last_attention_vision_refresh_mode = "attention_fast"
                return fast_observe()
            except Exception as exc:
                self._last_attention_vision_refresh_mode = "attention_fast_failed_default"
                self._record_fault(
                    event="proactive_attention_vision_fast_observe_failed",
                    message="Fast attention vision observation failed; falling back to the default vision observer.",
                    error=exc,
                )
        self._last_attention_vision_refresh_mode = "default_observe"
        return self._observe_vision_safe()

    def _observe_vision_with_method(self, observe_fn: Callable[[], Any]):
        """Call one vision-observer method while preserving the non-throwing contract."""

        try:
            return observe_fn()
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_specialized_observe_failed",
                message="Specialized vision observation failed; continuing without a refresh snapshot.",
                error=exc,
            )
            return None

    # AUDIT-FIX(#6): Vision observation faults should fall back to an uninspected tick so presence logic can continue.
    def _observe_vision_safe(self):
        """Collect one vision snapshot or return None on failure/unavailability."""

        if self.vision_observer is None:
            return None
        try:
            return self.vision_observer.observe()
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_observe_failed",
                message="Vision observation failed; continuing without a camera inspection this tick.",
                error=exc,
            )
            return None

    # AUDIT-FIX(#6): Snapshot buffering is optional; a buffer write failure must not suppress trigger evaluation.
    def _record_vision_snapshot_safe(self, snapshot) -> None:
        """Buffer one vision snapshot for later review when available."""

        if self.vision_reviewer is None:
            return
        if getattr(snapshot, "image", None) is None:
            return
        try:
            self.vision_reviewer.record_snapshot(snapshot)
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_snapshot_buffer_failed",
                message="Failed to buffer a proactive vision snapshot for later review.",
                error=exc,
            )

    # AUDIT-FIX(#5): Honor proactive_enabled as a hard privacy gate for camera-triggered proactive behavior.
    def _proactive_triggers_enabled(self) -> bool:
        """Return whether camera-driven proactive triggers may run."""

        return bool(self.config.proactive_enabled)

    # AUDIT-FIX(#2): Safety-critical triggers must fail open when review is unavailable, otherwise the device can miss a fall/distress prompt.
    def _should_fail_open_without_vision_review(self, decision: SocialTriggerDecision) -> bool:
        """Return whether missing review must not block this safety trigger."""

        return decision.trigger_id in _VISION_REVIEW_FAIL_OPEN_TRIGGERS

    # AUDIT-FIX(#6): Keep decision suppression logic centralized after presence checks.
    def _process_decision(
        self,
        *,
        now: float,
        decision: SocialTriggerDecision | None,
        observation: SocialObservation,
        inspected: bool,
        presence_snapshot: PresenceSessionSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None = None,
    ) -> ProactiveTickResult:
        """Apply final suppression, review, and dispatch to one trigger decision."""

        if decision is None:
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        blocked_reason = self._presence_session_block_reason(
            decision,
            presence_snapshot=presence_snapshot,
        )
        if blocked_reason is None:
            blocked_reason = self._audio_policy_block_reason(
                decision,
                presence_snapshot=presence_snapshot,
                audio_policy_snapshot=audio_policy_snapshot,
            )
        if blocked_reason is not None:
            if blocked_reason == "already_prompted_this_presence_session":
                self._record_trigger_skipped_presence_session(
                    decision,
                    presence_snapshot=presence_snapshot,
                    reason=blocked_reason,
                )
            else:
                self._record_trigger_skipped_audio_policy(
                    decision,
                    presence_snapshot=presence_snapshot,
                    audio_policy_snapshot=audio_policy_snapshot,
                    reason=blocked_reason,
                )
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        reviewed_decision, review = self._review_trigger(
            decision,
            observation=observation,
        )
        if reviewed_decision is None:
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        return self._handle_decision(
            reviewed_decision,
            observation=observation,
            inspected=inspected,
            presence_snapshot=presence_snapshot,
            review=review,
        )

    # AUDIT-FIX(#6): Reuse one absence-path implementation for normal no-inspection ticks and degraded vision failures.
    def _run_without_inspection(
        self,
        *,
        now: float,
        motion_active: bool,
        audio_snapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot,
        proactive_enabled: bool,
    ) -> ProactiveTickResult:
        """Run one proactive cycle when no fresh camera inspection is available."""

        observation, decision = self._feed_absence(
            now=now,
            motion_active=motion_active,
            audio_observation=audio_snapshot.observation,
            room_busy_or_overlapping=bool(
                audio_policy_snapshot is not None and audio_policy_snapshot.room_busy_or_overlapping
            ),
            evaluate_trigger=proactive_enabled,
        )
        presence_snapshot = self._observe_presence(
            now=now,
            person_visible=None,
            motion_active=motion_active,
            audio_observation=audio_snapshot.observation,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        self._dispatch_automation_observation(
            observation,
            inspected=False,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
            detected_trigger_id=None if decision is None else decision.trigger_id,
        )
        self._record_observation_if_changed(
            observation,
            inspected=False,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
            runtime_status_value="waiting",
        )
        result = self._process_decision(
            now=now,
            decision=decision,
            observation=observation,
            inspected=False,
            presence_snapshot=presence_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        self._maybe_publish_display_ambient_impulse(
            observed_at=now,
            runtime_status_value="waiting",
            tick_result=result,
            presence_active=bool(
                getattr(presence_snapshot, "armed", False)
                or getattr(observation.vision, "person_visible", False)
                or motion_active
            ),
        )
        return result

    def tick(self) -> ProactiveTickResult:
        """Run one proactive monitor tick across idle, sensor, and trigger paths."""

        now = self.clock()
        if self.idle_predicate is not None:
            try:
                if not self.idle_predicate():
                    return ProactiveTickResult()
            except Exception as exc:
                self._record_fault(
                    event="proactive_idle_predicate_failed",
                    message="Idle predicate failed; skipping this proactive tick.",
                    error=exc,
                )
                return ProactiveTickResult()
        motion_active = self._update_motion(now)
        try:
            runtime_status_value = self.runtime.status.value
            if runtime_status_value != "waiting":
                return self._run_busy_audio_only(
                    now=now,
                    motion_active=motion_active,
                    runtime_status_value=runtime_status_value,
                )
        except Exception as exc:
            self._record_fault(
                event="proactive_runtime_status_failed",
                message="Failed to read runtime status; skipping this proactive tick.",
                error=exc,
            )
            return ProactiveTickResult()

        proactive_enabled = self._proactive_triggers_enabled()
        inspect_requested = self._should_inspect(now, motion_active=motion_active)
        self._note_audio_observer_runtime_context(
            now=now,
            motion_active=motion_active,
            inspect_requested=inspect_requested,
            runtime_status_value="waiting",
        )
        audio_snapshot = self._observe_audio_safe()
        audio_policy_snapshot = self._observe_audio_policy(
            now=now,
            audio_observation=audio_snapshot.observation,
        )
        if not inspect_requested and self._should_bootstrap_inspect_from_speech(
            now=now,
            audio_observation=audio_snapshot.observation,
        ):
            inspect_requested = True
            self._emit("proactive_inspection_bootstrap=speech_detected")
        if not inspect_requested:
            return self._run_without_inspection(
                now=now,
                motion_active=motion_active,
                audio_snapshot=audio_snapshot,
                audio_policy_snapshot=audio_policy_snapshot,
                proactive_enabled=proactive_enabled,
            )

        snapshot = self._observe_vision_safe()
        if snapshot is None:
            return self._run_without_inspection(
                now=now,
                motion_active=motion_active,
                audio_snapshot=audio_snapshot,
                audio_policy_snapshot=audio_policy_snapshot,
                proactive_enabled=proactive_enabled,
            )

        self._last_capture_at = now
        self._record_vision_snapshot_safe(snapshot)
        low_motion = self._is_low_motion(now, motion_active=motion_active)
        observation = SocialObservation(
            observed_at=now,
            inspected=True,
            pir_motion_detected=motion_active,
            low_motion=low_motion,
            vision=snapshot.observation,
            audio=audio_snapshot.observation,
        )
        decision = (
            self._safety_trigger_fusion.observe(
                observation,
                room_busy_or_overlapping=bool(
                    audio_policy_snapshot is not None and audio_policy_snapshot.room_busy_or_overlapping
                ),
            )
            if proactive_enabled
            else None
        )  # AUDIT-FIX(#5): never evaluate proactive triggers when proactive mode is disabled.
        presence_snapshot = self._observe_presence(
            now=now,
            person_visible=snapshot.observation.person_visible,
            motion_active=motion_active,
            audio_observation=audio_snapshot.observation,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        self._dispatch_automation_observation(
            observation,
            inspected=True,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
            detected_trigger_id=None if decision is None else decision.trigger_id,
        )
        self._record_observation_if_changed(
            observation,
            inspected=True,
            vision_snapshot=snapshot,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
            runtime_status_value="waiting",
        )
        self._emit(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")  # AUDIT-FIX(#1): safe emit wrapper prevents telemetry failures from aborting monitor execution.
        self._emit(
            "proactive_speech_detected="
            f"{str(audio_snapshot.observation.speech_detected).lower()}"
        )
        if self.presence_session is not None:
            self._emit(f"voice_activation_armed={str(presence_snapshot.armed).lower()}")
        if audio_snapshot.observation.distress_detected is not None:
            self._emit(
                "proactive_distress_detected="
                f"{str(audio_snapshot.observation.distress_detected).lower()}"
            )
        if audio_snapshot.sample is not None:
            self._emit(f"proactive_audio_peak_rms={audio_snapshot.sample.peak_rms}")
        result = self._process_decision(
            now=now,
            decision=decision,
            observation=observation,
            inspected=True,
            presence_snapshot=presence_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        self._maybe_publish_display_ambient_impulse(
            observed_at=now,
            runtime_status_value="waiting",
            tick_result=result,
            presence_active=bool(
                getattr(presence_snapshot, "armed", False)
                or getattr(snapshot.observation, "person_visible", False)
                or motion_active
            ),
        )
        return result

    def _run_busy_audio_only(
        self,
        *,
        now: float,
        motion_active: bool,
        runtime_status_value: str,
    ) -> ProactiveTickResult:
        """Record bounded audio-only facts while Twinr is busy speaking."""

        self._note_audio_observer_runtime_context(
            now=now,
            motion_active=motion_active,
            inspect_requested=False,
            runtime_status_value=runtime_status_value,
        )
        audio_snapshot = self._observe_audio_safe()
        audio_policy_snapshot = self._observe_audio_policy(
            now=now,
            audio_observation=audio_snapshot.observation,
        )
        observation = SocialObservation(
            observed_at=now,
            inspected=False,
            pir_motion_detected=motion_active,
            low_motion=self._is_low_motion(now, motion_active=motion_active),
            vision=SocialVisionObservation(person_visible=False),
            audio=audio_snapshot.observation,
        )
        self._record_observation_if_changed(
            observation,
            inspected=False,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            runtime_status_value=runtime_status_value,
        )
        return ProactiveTickResult()

    def _feed_absence(
        self,
        *,
        now: float,
        motion_active: bool,
        audio_observation: SocialAudioObservation,
        room_busy_or_overlapping: bool = False,
        evaluate_trigger: bool = True,
    ) -> tuple[SocialObservation, SocialTriggerDecision | None]:
        """Build one observation for the no-camera path and optionally score it."""

        observation = SocialObservation(
            observed_at=now,
            inspected=False,
            pir_motion_detected=motion_active,
            low_motion=self._is_low_motion(now, motion_active=motion_active),
            vision=SocialVisionObservation(person_visible=False),
            audio=audio_observation,
        )
        return (
            observation,
            (
                self._safety_trigger_fusion.observe(
                    observation,
                    room_busy_or_overlapping=room_busy_or_overlapping,
                )
                if evaluate_trigger
                else None
            ),
        )  # AUDIT-FIX(#5): bypass trigger evaluation in activation-only/privacy-disabled proactive mode.

    def _observe_presence(
        self,
        *,
        now: float,
        person_visible: bool | None,
        motion_active: bool,
        audio_observation: SocialAudioObservation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> PresenceSessionSnapshot:
        """Update and publish the current voice-activation presence snapshot."""

        if self.presence_session is None:
            snapshot = PresenceSessionSnapshot(
                armed=False,
                reason="disabled",
                person_visible=bool(person_visible),
                last_speech_age_s=_round_optional_seconds(audio_observation.recent_speech_age_s),
                presence_audio_active=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active
                ),
                recent_follow_up_speech=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech
                ),
                room_busy_or_overlapping=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping
                ),
                quiet_window_open=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open
                ),
                barge_in_recent=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent
                ),
                speaker_direction_stable=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
                ),
                mute_blocks_voice_capture=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
                ),
                resume_window_open=(
                    None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
                ),
                device_runtime_mode=audio_observation.device_runtime_mode,
                transport_reason=audio_observation.transport_reason,
            )
            self.latest_presence_snapshot = snapshot
            return snapshot
        snapshot = self.presence_session.observe(
            now=now,
            person_visible=person_visible,
            motion_active=motion_active,
            speech_detected=audio_observation.speech_detected is True,
            recent_speech_age_s=audio_observation.recent_speech_age_s,
            presence_audio_active=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active
            ),
            recent_follow_up_speech=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech
            ),
            room_busy_or_overlapping=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping
            ),
            quiet_window_open=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open
            ),
            barge_in_recent=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent
            ),
            speaker_direction_stable=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
            ),
            mute_blocks_voice_capture=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
            ),
            resume_window_open=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
            ),
            device_runtime_mode=audio_observation.device_runtime_mode,
            transport_reason=audio_observation.transport_reason,
        )
        self.latest_presence_snapshot = snapshot
        self._record_presence_if_changed(snapshot)
        return snapshot

    def _handle_decision(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        inspected: bool,
        presence_snapshot: PresenceSessionSnapshot,
        review: ProactiveVisionReview | None = None,
    ) -> ProactiveTickResult:
        """Dispatch one reviewed proactive trigger and update session-side state."""

        self._record_trigger_detected(decision, observation=observation, review=review)
        try:  # AUDIT-FIX(#6): downstream trigger actions are external code and must not crash the monitor loop.
            handled = bool(self.trigger_handler(decision))
        except Exception as exc:
            self._record_fault(
                event="proactive_trigger_handler_failed",
                message="Proactive trigger handler failed.",
                error=exc,
                data={"trigger": decision.trigger_id},
            )
            handled = False
        presence_session_id = getattr(presence_snapshot, "session_id", None)
        if handled:
            self._emit(f"proactive_trigger={decision.trigger_id}")  # AUDIT-FIX(#1): safe emit wrapper keeps trigger dispatch resilient even when emit sinks break.
            if (
                decision.trigger_id == "possible_fall"
                and self.config.proactive_possible_fall_once_per_presence_session
                and presence_snapshot.armed
                and presence_session_id is not None
            ):
                self._last_possible_fall_prompted_session_id = presence_session_id
        return ProactiveTickResult(
            decision=decision if handled else None,
            inspected=inspected,
            person_visible=observation.vision.person_visible,
        )

    def _review_trigger(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
    ) -> tuple[SocialTriggerDecision | None, ProactiveVisionReview | None]:
        """Run buffered vision review for one trigger when configured."""

        if self.vision_reviewer is None:
            return decision, None
        if not is_reviewable_image_trigger(decision.trigger_id):
            return decision, None
        try:
            review = self.vision_reviewer.review(decision, observation=observation)
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_review_failed",
                message="Buffered proactive vision review failed.",
                error=exc,
                data={"trigger": decision.trigger_id},
            )
            if self._should_fail_open_without_vision_review(decision):
                self._append_ops_event(
                    event="proactive_vision_review_fail_open",
                    level="warning",
                    message="A safety-critical proactive trigger proceeded because buffered vision review failed.",
                    data={"trigger": decision.trigger_id},
                )
                return decision, None
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        if review is None:
            self._append_ops_event(
                event="proactive_vision_review_unavailable",
                message="Buffered proactive vision review was enabled but no usable result was available.",
                data={"trigger": decision.trigger_id},
            )
            if self._should_fail_open_without_vision_review(decision):
                self._append_ops_event(
                    event="proactive_vision_review_fail_open",
                    level="warning",
                    message="A safety-critical proactive trigger proceeded because buffered vision review was unavailable.",
                    data={"trigger": decision.trigger_id},
                )
                return decision, None
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        self._record_vision_review(decision, review=review)
        if not review.approved:
            self._record_trigger_skipped_vision_review(decision, review=review)
            return None, review
        return decision, review

    def _update_motion(self, now: float) -> bool:
        """Poll PIR state and return whether motion is currently active."""

        if self.pir_monitor is None:
            return False
        motion_active = False
        try:  # AUDIT-FIX(#6): sensor backend faults must not abort the entire proactive tick.
            while True:
                event = self.pir_monitor.poll(timeout=0.0)
                if event is None:
                    break
                if event.motion_detected:
                    self._last_motion_at = now
                    motion_active = True
        except Exception as exc:
            self._record_fault(
                event="pir_poll_failed",
                message="PIR event polling failed; continuing with the last known motion state.",
                error=exc,
            )
        try:
            if self.pir_monitor.motion_detected():
                self._last_motion_at = now
                motion_active = True
        except Exception as exc:
            self._record_fault(
                event="pir_read_failed",
                message="Direct PIR state read failed; continuing with queued motion state only.",
                error=exc,
            )
        return motion_active

    def _should_inspect(self, now: float, *, motion_active: bool) -> bool:
        """Return whether the next tick should trigger a camera inspection."""

        if not self._proactive_triggers_enabled():
            return False
        if self.vision_observer is None:
            return False
        if self._last_motion_at is None:
            return False
        if not motion_active and (now - self._last_motion_at) > self.config.proactive_motion_window_s:
            return False
        if self._last_capture_at is not None and (now - self._last_capture_at) < self.config.proactive_capture_interval_s:
            return False
        return True

    def _should_bootstrap_inspect_from_speech(
        self,
        now: float,
        *,
        audio_observation: SocialAudioObservation,
    ) -> bool:
        """Return whether speech should trigger one bounded local vision inspection.

        Remote voice activation is presence-gated. If PIR is quiet or absent,
        but the local microphone hears active speech while the remote voice
        path is enabled, a bounded camera inspection lets the device establish
        real presence instead of staying permanently idle behind the PIR gate.
        """

        if not bool(getattr(self.config, "voice_orchestrator_enabled", False)):
            return False
        if self.vision_observer is None:
            return False
        if self.presence_session is None:
            return False
        if audio_observation.speech_detected is not True:
            return False
        latest_presence = self.latest_presence_snapshot
        if latest_presence is not None and latest_presence.armed:
            return False
        if (
            self._last_capture_at is not None
            and (now - self._last_capture_at) < self.config.proactive_capture_interval_s
        ):
            return False
        return True

    def refresh_display_attention(self) -> bool:
        """Refresh HDMI attention-follow from the cheap local camera path.

        Eye-follow must remain responsive even when explicit hand-gesture
        recognition is heavier. This refresh therefore prefers a dedicated
        attention-only camera observation and a dedicated stabilized surface so
        gesture-path tuning cannot regress the face-follow loop.
        """

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        self._last_display_attention_fusion_debug = None

        if not display_attention_refresh_supported(
            config=self.config,
            vision_observer=self.vision_observer,
        ):
            self._record_attention_debug_tick(
                observed_at=self.clock(),
                outcome="unsupported",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        now = self.clock()
        interval_s = resolve_display_attention_refresh_interval(self.config)
        if interval_s is None:
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="no_refresh_interval",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if (
            self._last_display_attention_refresh_at is not None
            and (now - self._last_display_attention_refresh_at) < interval_s
        ):
            return False
        try:
            runtime_status_value = self.runtime.status.value
        except Exception as exc:
            self._record_fault(
                event="proactive_display_attention_runtime_status_failed",
                message="Failed to read runtime status for HDMI attention refresh.",
                error=exc,
            )
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="runtime_status_failed",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="runtime_status_blocked",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        stage_started_ns = time.monotonic_ns()
        snapshot = self._observe_vision_for_attention_refresh()
        stage_ms["vision_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        if snapshot is None:
            stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="vision_snapshot_missing",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        stage_started_ns = time.monotonic_ns()
        fused_vision = self._fuse_display_attention_camera_observation(
            observed_at=now,
            observation=snapshot.observation,
        )
        stage_ms["camera_fusion"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        self._last_display_attention_refresh_at = now
        self._record_vision_snapshot_safe(snapshot)
        stage_started_ns = time.monotonic_ns()
        camera_update = self._observe_display_attention_camera_surface(
            SocialObservation(
                observed_at=now,
                inspected=True,
                pir_motion_detected=False,
                low_motion=False,
                vision=fused_vision,
                audio=SocialAudioObservation(),
            ),
            inspected=True,
        )
        stage_ms["camera_surface"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        audio_snapshot = self._observe_audio_for_attention_refresh(now=now)
        stage_ms["audio_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        audio_policy_snapshot = self._observe_audio_policy(
            now=now,
            audio_observation=audio_snapshot.observation,
        )
        stage_ms["audio_policy"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        self._publish_display_attention_live_context(
            observed_at=now,
            vision_observation=fused_vision,
            camera_snapshot=camera_update.snapshot,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        stage_started_ns = time.monotonic_ns()
        publish_result = self._update_display_attention_follow(
            source="display_attention_refresh",
            observed_at=now,
            camera_snapshot=camera_update.snapshot,
            audio_observation=audio_snapshot.observation,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        stage_ms["publish"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        self._update_display_debug_signals(camera_update)
        stage_ms["debug_signals"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
        self._record_display_attention_follow_if_changed(
            observed_at=now,
            runtime_status_value=runtime_status_value,
            camera_snapshot=camera_update.snapshot,
            publish_result=publish_result,
        )
        self._record_attention_debug_tick(
            observed_at=now,
            outcome="published" if publish_result is not None else "no_publish_result",
            runtime_status_value=runtime_status_value,
            camera_snapshot=camera_update.snapshot,
            audio_observation=audio_snapshot.observation,
            publish_result=publish_result,
            stage_ms=stage_ms,
        )
        return True

    def refresh_display_gesture_emoji(self) -> bool:
        """Refresh HDMI gesture acknowledgements from the dedicated gesture path."""

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        if self.display_gesture_emoji_publisher is None:
            return False
        if not display_gesture_refresh_supported(
            config=self.config,
            vision_observer=self.vision_observer,
        ):
            self._record_gesture_debug_tick(
                observed_at=self.clock(),
                outcome="unsupported",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        now = self.clock()
        interval_s = resolve_display_gesture_refresh_interval(self.config)
        if interval_s is None:
            self._record_gesture_debug_tick(
                observed_at=now,
                outcome="no_refresh_interval",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if (
            self._last_display_gesture_refresh_at is not None
            and (now - self._last_display_gesture_refresh_at) < interval_s
        ):
            return False
        try:
            runtime_status_value = self.runtime.status.value
        except Exception as exc:
            self._record_fault(
                event="proactive_display_gesture_runtime_status_failed",
                message="Failed to read runtime status for HDMI gesture refresh.",
                error=exc,
            )
            self._record_gesture_debug_tick(
                observed_at=now,
                outcome="runtime_status_failed",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
            self._record_gesture_debug_tick(
                observed_at=now,
                outcome="runtime_status_blocked",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        with self._gesture_forensics.bind_refresh(
            observed_at=now,
            runtime_status_value=runtime_status_value,
            vision_mode=self._last_gesture_vision_refresh_mode,
            refresh_interval_s=interval_s,
        ):
            workflow_decision(
                msg="gesture_refresh_runtime_status_gate",
                question="Should the dedicated HDMI gesture refresh run on this tick?",
                selected={
                    "id": "allowed",
                    "summary": "The runtime state allows the dedicated gesture refresh to execute.",
                },
                options=[
                    {"id": "allowed", "summary": "Proceed with the dedicated gesture refresh."},
                    {"id": "blocked", "summary": "Skip the gesture refresh because the runtime state forbids it."},
                ],
                context={
                    "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                    "refresh_interval_s": interval_s,
                },
                confidence="forensic",
                guardrails=["display_gesture_runtime_status_gate"],
                kpi_impact_estimate={"latency": "low", "user_feedback": "high"},
            )
            with workflow_span(
                name="proactive_display_gesture_refresh",
                kind="turn",
                details={
                    "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                    "refresh_interval_s": interval_s,
                },
            ):
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_observe_vision",
                    kind="io",
                    details={"vision_observer_type": type(self.vision_observer).__name__},
                ):
                    snapshot = self._observe_vision_for_gesture_refresh()
                stage_ms["vision_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                if snapshot is None:
                    stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
                    workflow_event(
                        kind="branch",
                        msg="gesture_refresh_snapshot_missing",
                        details={"stage_ms": dict(stage_ms)},
                        reason={
                            "selected": {
                                "id": "snapshot_missing",
                                "justification": "The gesture refresh could not continue because the vision observer returned no snapshot.",
                                "expected_outcome": "Abort the current refresh without touching HDMI gesture state.",
                            },
                            "options": [
                                {"id": "snapshot_present", "summary": "Continue with the captured snapshot."},
                                {"id": "snapshot_missing", "summary": "Abort because no snapshot was returned."},
                            ],
                            "confidence": "forensic",
                            "guardrails": ["vision_snapshot_required"],
                            "kpi_impact_estimate": {"latency": "low", "display_side_effect": "none"},
                        },
                    )
                    self._record_gesture_debug_tick(
                        observed_at=now,
                        outcome="vision_snapshot_missing",
                        runtime_status_value=runtime_status_value,
                        stage_ms=stage_ms,
                    )
                    return False
                self._last_display_gesture_refresh_at = now
                self._record_vision_snapshot_safe(snapshot)
                self._remember_display_attention_camera_semantics(
                    observed_at=now,
                    observation=snapshot.observation,
                    source="gesture",
                )
                workflow_event(
                    kind="io",
                    msg="gesture_refresh_observation",
                    details=self._gesture_observation_trace_details(snapshot.observation),
                )
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_ack_lane_observe",
                    kind="decision",
                ):
                    decision = self._gesture_ack_lane.observe(
                        observed_at=now,
                        observation=snapshot.observation,
                    )
                stage_ms["ack_lane"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                self._trace_gesture_ack_lane_decision(
                    observation=snapshot.observation,
                    decision=decision,
                )
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_wakeup_lane_observe",
                    kind="decision",
                ):
                    wakeup_decision = self._gesture_wakeup_lane.observe(
                        observed_at=now,
                        observation=snapshot.observation,
                    )
                stage_ms["wakeup_lane"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                self._trace_gesture_wakeup_lane_decision(decision=wakeup_decision)
                if wakeup_decision.active:
                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_wakeup_priority",
                        kind="decision",
                    ):
                        wakeup_priority = decide_gesture_wakeup_priority(
                            runtime_status_value=runtime_status_value,
                            voice_path_enabled=bool(getattr(self.config, "voice_orchestrator_enabled", False)),
                            presence_snapshot=self.latest_presence_snapshot,
                            recent_speech_guard_s=_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
                        )
                    if not wakeup_priority.allow:
                        wakeup_decision = replace(
                            wakeup_decision,
                            active=False,
                            reason=wakeup_priority.reason,
                        )
                    stage_ms["wakeup_priority"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                    workflow_decision(
                        msg="gesture_wakeup_priority_gate",
                        question="Should the active gesture wake candidate be allowed to start the voice path now?",
                        selected={
                            "id": "allow" if wakeup_priority.allow else wakeup_priority.reason,
                            "summary": (
                                "Allow the wake request to proceed."
                                if wakeup_priority.allow
                                else "Suppress the wake request because the current runtime priority forbids it."
                            ),
                        },
                        options=[
                            {"id": "allow", "summary": "Allow the wake request to proceed."},
                            {"id": "block", "summary": "Block the wake request due to current runtime policy."},
                        ],
                        context={
                            "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                            "voice_path_enabled": bool(getattr(self.config, "voice_orchestrator_enabled", False)),
                            "wakeup_reason": wakeup_decision.reason,
                        },
                        confidence="forensic",
                        guardrails=["gesture_wakeup_priority"],
                        kpi_impact_estimate={"voice_turn": "high"},
                    )
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_publish",
                    kind="mutation",
                    details={"decision_reason": decision.reason, "decision_active": decision.active},
                ):
                    publish_result = self._publish_display_gesture_decision(decision)
                stage_ms["publish"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_wakeup_handler",
                    kind="mutation",
                    details={"wakeup_reason": wakeup_decision.reason, "wakeup_active": wakeup_decision.active},
                ):
                    wakeup_handled = self._dispatch_gesture_wakeup_with_fresh_context(
                        observed_at=now,
                        vision_snapshot=snapshot,
                        decision=wakeup_decision,
                    )
                stage_ms["wakeup_handler"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
                self._trace_gesture_publish_decision(
                    decision=decision,
                    publish_result=publish_result,
                    wakeup_decision=wakeup_decision,
                    wakeup_handled=wakeup_handled,
                )
                workflow_event(
                    kind="metric",
                    msg="gesture_refresh_stage_metrics",
                    details={"stage_ms": dict(stage_ms)},
                    kpi={f"stage_{key}_ms": value for key, value in stage_ms.items()},
                )
                self._record_gesture_debug_tick(
                    observed_at=now,
                    outcome=("publish_failed" if publish_result is None else publish_result.action),
                    runtime_status_value=runtime_status_value,
                    observation=snapshot.observation,
                    decision=decision,
                    publish_result=publish_result,
                    wakeup_decision=wakeup_decision,
                    wakeup_handled=wakeup_handled,
                    stage_ms=stage_ms,
                )
                return True

    def _dispatch_gesture_wakeup_with_fresh_context(
        self,
        *,
        observed_at: float,
        vision_snapshot,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Prime current gesture facts before dispatching an accepted wakeup."""

        return service_gesture_helpers.dispatch_gesture_wakeup_with_fresh_context(
            self,
            observed_at=observed_at,
            vision_snapshot=vision_snapshot,
            decision=decision,
        )

    def _prime_gesture_wakeup_sensor_context(
        self,
        *,
        observed_at: float,
        vision_snapshot,
    ) -> None:
        """Export one fresh sensor/person-state payload from the active gesture tick."""

        service_gesture_helpers.prime_gesture_wakeup_sensor_context(
            self,
            observed_at=observed_at,
            vision_snapshot=vision_snapshot,
        )

    def _publish_display_attention_live_context(
        self,
        *,
        observed_at: float,
        vision_observation: SocialVisionObservation,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_snapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> None:
        """Export authoritative HDMI-attention facts to the live voice context."""

        service_attention_helpers.publish_display_attention_live_context(
            self,
            observed_at=observed_at,
            vision_observation=vision_observation,
            camera_snapshot=camera_snapshot,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )

    def _is_low_motion(self, now: float, *, motion_active: bool) -> bool:
        """Return whether recent PIR history qualifies as low motion."""

        if motion_active:
            return False
        if self._last_motion_at is None:
            return False  # AUDIT-FIX(#4): do not infer "low motion" before any real motion history exists, especially when PIR hardware is missing or has not fired yet.
        return (now - self._last_motion_at) >= self.config.proactive_low_motion_after_s

    def _note_audio_observer_runtime_context(
        self,
        *,
        now: float,
        motion_active: bool,
        inspect_requested: bool,
        runtime_status_value: str,
    ) -> None:
        """Forward current runtime context to schedulable audio observers."""

        callback = getattr(self.audio_observer, "note_runtime_context", None)
        if not callable(callback):
            return
        note_runtime_context = cast(Callable[..., None], callback)
        presence_snapshot = self.latest_presence_snapshot
        note_runtime_context(  # pylint: disable=not-callable
            observed_at=now,
            motion_active=motion_active,
            inspect_requested=inspect_requested,
            presence_session_armed=bool(
                presence_snapshot is not None and getattr(presence_snapshot, "armed", False)
            ),
            assistant_output_active=runtime_status_value == "answering",
        )

    def _observe_audio_policy(
        self,
        *,
        now: float,
        audio_observation: SocialAudioObservation,
    ) -> ReSpeakerAudioPolicySnapshot:
        """Derive one conservative ReSpeaker policy snapshot for this tick."""

        snapshot = self.audio_policy_tracker.observe(now=now, audio=audio_observation)
        self.latest_audio_policy_snapshot = snapshot
        return snapshot

    def _record_observation_if_changed(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        vision_snapshot=None,
        audio_snapshot=None,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None = None,
        presence_snapshot: PresenceSessionSnapshot | None = None,
        runtime_status_value: str | None = None,
    ) -> None:
        """Append one observation event only when the visible state changed."""

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        observation_key = (
            inspected,
            runtime_status_value,
            observation.pir_motion_detected,
            observation.low_motion,
            observation.vision.person_visible,
            observation.vision.looking_toward_device,
            observation.vision.body_pose.value,
            observation.vision.smiling,
            observation.vision.hand_or_object_near_camera,
            observation.audio.speech_detected,
            observation.audio.distress_detected,
            observation.audio.assistant_output_active,
            observation.audio.device_runtime_mode,
            observation.audio.host_control_ready,
            observation.audio.transport_reason,
            observation.audio.non_speech_audio_likely,
            observation.audio.background_media_likely,
            observation.audio.signal_source,
            observation.audio.direction_confidence,
            observation.audio.speech_overlap_likely,
            observation.audio.barge_in_detected,
            observation.audio.mute_active,
            None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active,
            None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech,
            None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping,
            None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open,
            None if audio_policy_snapshot is None else audio_policy_snapshot.non_speech_audio_likely,
            None if audio_policy_snapshot is None else audio_policy_snapshot.background_media_likely,
            None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent,
            None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable,
            None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture,
            None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open,
            None if audio_policy_snapshot is None else audio_policy_snapshot.initiative_block_reason,
            None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason,
            None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code,
            None if presence_snapshot is None else presence_snapshot.armed,
            None if presence_snapshot is None else presence_snapshot.reason,
            presence_session_id,
            None if self.latest_speaker_association_snapshot is None else self.latest_speaker_association_snapshot.state,
            None if self.latest_speaker_association_snapshot is None else self.latest_speaker_association_snapshot.associated,
            None if self.latest_speaker_association_snapshot is None else self.latest_speaker_association_snapshot.confidence,
            None if self.latest_multimodal_initiative_snapshot is None else self.latest_multimodal_initiative_snapshot.ready,
            None if self.latest_multimodal_initiative_snapshot is None else self.latest_multimodal_initiative_snapshot.confidence,
            None if self.latest_multimodal_initiative_snapshot is None else self.latest_multimodal_initiative_snapshot.block_reason,
            None if self.latest_ambiguous_room_guard_snapshot is None else self.latest_ambiguous_room_guard_snapshot.guard_active,
            None if self.latest_ambiguous_room_guard_snapshot is None else self.latest_ambiguous_room_guard_snapshot.reason,
            None if self.latest_identity_fusion_snapshot is None else self.latest_identity_fusion_snapshot.state,
            None if self.latest_identity_fusion_snapshot is None else self.latest_identity_fusion_snapshot.matches_main_user,
            None if self.latest_identity_fusion_snapshot is None else self.latest_identity_fusion_snapshot.claim.confidence,
            None if self.latest_portrait_match_snapshot is None else self.latest_portrait_match_snapshot.state,
            None if self.latest_portrait_match_snapshot is None else self.latest_portrait_match_snapshot.matches_reference_user,
            None if self.latest_portrait_match_snapshot is None else self.latest_portrait_match_snapshot.claim.confidence,
            None if self.latest_known_user_hint_snapshot is None else self.latest_known_user_hint_snapshot.state,
            None if self.latest_known_user_hint_snapshot is None else self.latest_known_user_hint_snapshot.matches_main_user,
            None if self.latest_known_user_hint_snapshot is None else self.latest_known_user_hint_snapshot.claim.confidence,
            None if self.latest_affect_proxy_snapshot is None else self.latest_affect_proxy_snapshot.state,
            None if self.latest_affect_proxy_snapshot is None else self.latest_affect_proxy_snapshot.claim.confidence,
            None if self.latest_attention_target_snapshot is None else self.latest_attention_target_snapshot.state,
            None if self.latest_attention_target_snapshot is None else self.latest_attention_target_snapshot.target_horizontal,
            None if self.latest_attention_target_snapshot is None else self.latest_attention_target_snapshot.focus_source,
        )
        if observation_key == self._last_observation_key:
            return
        if not inspected and self._last_observation_key is None:
            self._last_observation_key = observation_key
            return
        self._last_observation_key = observation_key
        if audio_policy_snapshot is not None:
            self._record_respeaker_runtime_alert_if_changed(
                observation.audio,
                audio_policy_snapshot=audio_policy_snapshot,
            )
        indicator_state = resolve_respeaker_indicator_state(
            runtime_status=runtime_status_value,
            runtime_alert_code=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
            ),
            mute_active=observation.audio.mute_active,
        )
        data = {
            "inspected": inspected,
            "runtime_status": runtime_status_value,
            "pir_motion_detected": observation.pir_motion_detected,
            "low_motion": observation.low_motion,
            "person_visible": observation.vision.person_visible,
            "camera_person_count": observation.vision.person_count,
            "camera_primary_person_zone": observation.vision.primary_person_zone.value,
            "camera_primary_person_center_x": _round_optional_ratio(observation.vision.primary_person_center_x),
            "camera_primary_person_center_y": _round_optional_ratio(observation.vision.primary_person_center_y),
            "looking_toward_device": observation.vision.looking_toward_device,
            "camera_person_near_device": observation.vision.person_near_device,
            "camera_engaged_with_device": observation.vision.engaged_with_device,
            "camera_visual_attention_score": _round_optional_ratio(observation.vision.visual_attention_score),
            "body_pose": observation.vision.body_pose.value,
            "camera_pose_confidence": _round_optional_ratio(observation.vision.pose_confidence),
            "camera_motion_state": observation.vision.motion_state.value,
            "camera_motion_confidence": _round_optional_ratio(observation.vision.motion_confidence),
            "smiling": observation.vision.smiling,
            "hand_or_object_near_camera": observation.vision.hand_or_object_near_camera,
            "camera_gesture_event": observation.vision.gesture_event.value,
            "camera_gesture_confidence": _round_optional_ratio(observation.vision.gesture_confidence),
            "camera_fine_hand_gesture": observation.vision.fine_hand_gesture.value,
            "camera_fine_hand_gesture_confidence": _round_optional_ratio(
                observation.vision.fine_hand_gesture_confidence
            ),
            "camera_online": observation.vision.camera_online,
            "camera_ready": observation.vision.camera_ready,
            "camera_ai_ready": observation.vision.camera_ai_ready,
            "camera_error": observation.vision.camera_error,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
            "audio_room_quiet": observation.audio.room_quiet,
            "audio_recent_speech_age_s": _round_optional_seconds(observation.audio.recent_speech_age_s),
            "audio_assistant_output_active": observation.audio.assistant_output_active,
            "audio_azimuth_deg": observation.audio.azimuth_deg,
            "audio_direction_confidence": observation.audio.direction_confidence,
            "audio_signal_source": observation.audio.signal_source,
            "audio_device_runtime_mode": observation.audio.device_runtime_mode,
            "audio_host_control_ready": observation.audio.host_control_ready,
            "audio_transport_reason": observation.audio.transport_reason,
            "audio_non_speech_audio_likely": observation.audio.non_speech_audio_likely,
            "audio_background_media_likely": observation.audio.background_media_likely,
            "audio_speech_overlap_likely": observation.audio.speech_overlap_likely,
            "audio_barge_in_detected": observation.audio.barge_in_detected,
            "audio_mute_active": observation.audio.mute_active,
            "audio_indicator_mode": indicator_state.mode,
            "audio_indicator_semantics": indicator_state.semantics,
        }
        if self.latest_speaker_association_snapshot is not None:
            data.update(self.latest_speaker_association_snapshot.event_data())
        if self.latest_multimodal_initiative_snapshot is not None:
            data.update(self.latest_multimodal_initiative_snapshot.event_data())
        if self.latest_ambiguous_room_guard_snapshot is not None:
            data.update(self.latest_ambiguous_room_guard_snapshot.event_data())
        if self.latest_identity_fusion_snapshot is not None:
            data.update(self.latest_identity_fusion_snapshot.event_data())
        if self.latest_portrait_match_snapshot is not None:
            data.update(self.latest_portrait_match_snapshot.event_data())
        if self.latest_known_user_hint_snapshot is not None:
            data.update(self.latest_known_user_hint_snapshot.event_data())
        if self.latest_affect_proxy_snapshot is not None:
            data.update(self.latest_affect_proxy_snapshot.event_data())
        if self.latest_attention_target_snapshot is not None:
            data.update(self.latest_attention_target_snapshot.event_data())
        if self.latest_person_state_snapshot is not None:
            data.update(self.latest_person_state_snapshot.event_data())
        if audio_policy_snapshot is not None:
            data.update(
                {
                    "presence_audio_active": audio_policy_snapshot.presence_audio_active,
                    "recent_follow_up_speech": audio_policy_snapshot.recent_follow_up_speech,
                    "room_busy_or_overlapping": audio_policy_snapshot.room_busy_or_overlapping,
                    "quiet_window_open": audio_policy_snapshot.quiet_window_open,
                    "non_speech_audio_likely": audio_policy_snapshot.non_speech_audio_likely,
                    "background_media_likely": audio_policy_snapshot.background_media_likely,
                    "barge_in_recent": audio_policy_snapshot.barge_in_recent,
                    "speaker_direction_stable": audio_policy_snapshot.speaker_direction_stable,
                    "mute_blocks_voice_capture": audio_policy_snapshot.mute_blocks_voice_capture,
                    "resume_window_open": audio_policy_snapshot.resume_window_open,
                    "audio_initiative_block_reason": audio_policy_snapshot.initiative_block_reason,
                    "audio_speech_delivery_defer_reason": audio_policy_snapshot.speech_delivery_defer_reason,
                    "respeaker_runtime_alert_code": audio_policy_snapshot.runtime_alert_code,
                }
            )
        if presence_snapshot is not None:
            data.update(
                {
                    "voice_activation_armed": presence_snapshot.armed,
                    "voice_activation_presence_reason": presence_snapshot.reason,
                    "voice_activation_presence_session_id": presence_session_id,
                    "voice_activation_presence_audio_active": presence_snapshot.presence_audio_active,
                    "voice_activation_recent_follow_up_speech": presence_snapshot.recent_follow_up_speech,
                    "voice_activation_room_busy_or_overlapping": presence_snapshot.room_busy_or_overlapping,
                    "voice_activation_quiet_window_open": presence_snapshot.quiet_window_open,
                    "voice_activation_barge_in_recent": presence_snapshot.barge_in_recent,
                    "voice_activation_speaker_direction_stable": presence_snapshot.speaker_direction_stable,
                    "voice_activation_mute_blocks_voice_capture": presence_snapshot.mute_blocks_voice_capture,
                    "voice_activation_resume_window_open": presence_snapshot.resume_window_open,
                }
            )
        if vision_snapshot is not None:
            data.update(
                {
                    "vision_model": vision_snapshot.model,
                    "vision_request_id": vision_snapshot.request_id,
                    "vision_response_id": vision_snapshot.response_id,
                }
            )
        if audio_snapshot is not None and audio_snapshot.sample is not None:
            data.update(
                {
                    "audio_average_rms": audio_snapshot.sample.average_rms,
                    "audio_peak_rms": audio_snapshot.sample.peak_rms,
                    "audio_active_ratio": audio_snapshot.sample.active_ratio,
                    "audio_active_chunks": audio_snapshot.sample.active_chunk_count,
                    "audio_chunk_count": audio_snapshot.sample.chunk_count,
                }
            )
        if audio_snapshot is not None and audio_snapshot.signal_snapshot is not None:
            data.update(
                {
                    "audio_requires_elevated_permissions": audio_snapshot.signal_snapshot.requires_elevated_permissions,
                    "audio_firmware_version": _format_firmware_version(audio_snapshot.signal_snapshot.firmware_version),
                    "audio_gpo_logic_levels": audio_snapshot.signal_snapshot.gpo_logic_levels,
                }
            )
        top_evaluation = self._safety_trigger_fusion.best_evaluation
        if top_evaluation is not None and top_evaluation.score > 0.0:
            data.update(
                {
                    "top_trigger": top_evaluation.trigger_id,
                    "top_score": top_evaluation.score,
                    "top_threshold": top_evaluation.threshold,
                    "top_trigger_passed": top_evaluation.passed,
                }
            )
            if top_evaluation.blocked_reason is not None:
                data["top_blocked_reason"] = top_evaluation.blocked_reason
            if top_evaluation.passed and audio_policy_snapshot is not None:
                data["top_audio_policy_block_reason"] = audio_policy_snapshot.initiative_block_reason
        self._append_ops_event(
            event="proactive_observation",
            message="Proactive monitor recorded a changed observation.",
            data=data,
        )

    def _record_presence_if_changed(self, snapshot: PresenceSessionSnapshot) -> None:
        """Append one ops event when the presence-session state changes."""

        session_id = getattr(snapshot, "session_id", None)
        key = (
            snapshot.armed,
            snapshot.reason,
            session_id,
            snapshot.presence_audio_active,
            snapshot.recent_follow_up_speech,
            snapshot.room_busy_or_overlapping,
            snapshot.quiet_window_open,
            snapshot.barge_in_recent,
            snapshot.speaker_direction_stable,
            snapshot.mute_blocks_voice_capture,
            snapshot.resume_window_open,
            snapshot.device_runtime_mode,
            snapshot.transport_reason,
        )
        if key == self._last_presence_key:
            return
        self._last_presence_key = key
        self._append_ops_event(
            event="voice_activation_presence_changed",
            message="Voice-activation presence session changed.",
            data={
                "armed": snapshot.armed,
                "reason": snapshot.reason,
                "session_id": session_id,
                "person_visible": snapshot.person_visible,
                "last_person_seen_age_s": snapshot.last_person_seen_age_s,
                "last_motion_age_s": snapshot.last_motion_age_s,
                "last_speech_age_s": snapshot.last_speech_age_s,
                "presence_audio_active": snapshot.presence_audio_active,
                "recent_follow_up_speech": snapshot.recent_follow_up_speech,
                "room_busy_or_overlapping": snapshot.room_busy_or_overlapping,
                "quiet_window_open": snapshot.quiet_window_open,
                "barge_in_recent": snapshot.barge_in_recent,
                "speaker_direction_stable": snapshot.speaker_direction_stable,
                "mute_blocks_voice_capture": snapshot.mute_blocks_voice_capture,
                "resume_window_open": snapshot.resume_window_open,
                "device_runtime_mode": snapshot.device_runtime_mode,
                "transport_reason": snapshot.transport_reason,
            },
        )

    def _record_respeaker_runtime_alert_if_changed(
        self,
        audio: SocialAudioObservation,
        *,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot,
    ) -> None:
        """Append one explicit operator-readable ReSpeaker runtime alert on change."""

        alert_code = audio_policy_snapshot.runtime_alert_code
        if alert_code is None:
            return
        if alert_code == self._last_respeaker_runtime_alert_code:
            return
        self._last_respeaker_runtime_alert_code = alert_code
        level = "warning" if alert_code != "ready" else "info"
        message = audio_policy_snapshot.runtime_alert_message or "ReSpeaker runtime state changed."
        self._emit(f"respeaker_runtime_alert={alert_code}")
        self._append_ops_event(
            event="respeaker_runtime_alert",
            level=level,
            message=message,
            data={
                "alert_code": alert_code,
                "device_runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
                "transport_reason": audio.transport_reason,
                "mute_active": audio.mute_active,
            },
        )
        self._record_respeaker_runtime_blocker_if_changed(
            alert_code=alert_code,
            message=message,
            audio=audio,
        )

    def _record_respeaker_runtime_blocker_if_changed(
        self,
        *,
        alert_code: str,
        message: str,
        audio: SocialAudioObservation,
    ) -> None:
        """Emit explicit hard-block lifecycle events for DFU runtime states."""

        if is_respeaker_runtime_hard_block(alert_code):
            if alert_code == self._last_respeaker_runtime_blocker_code:
                return
            self._last_respeaker_runtime_blocker_code = alert_code
            self._emit(f"respeaker_runtime_blocker={alert_code}")
            self._append_ops_event(
                event="respeaker_runtime_blocker",
                level="error",
                message=message,
                data={
                    "alert_code": alert_code,
                    "device_runtime_mode": audio.device_runtime_mode,
                    "host_control_ready": audio.host_control_ready,
                    "transport_reason": audio.transport_reason,
                    "mute_active": audio.mute_active,
                },
            )
            return

        if self._last_respeaker_runtime_blocker_code is None:
            return
        cleared_code = self._last_respeaker_runtime_blocker_code
        self._last_respeaker_runtime_blocker_code = None
        self._emit(f"respeaker_runtime_blocker_cleared={cleared_code}")
        self._append_ops_event(
            event="respeaker_runtime_blocker_cleared",
            message="ReSpeaker hard runtime blocker cleared and capture is usable again.",
            data={
                "cleared_alert_code": cleared_code,
                "current_alert_code": alert_code,
                "device_runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
            },
        )

    def _block_respeaker_dead_capture(self, error: AudioCaptureReadinessError) -> None:
        """Fail closed on unreadable ReSpeaker capture during live monitoring."""

        self.audio_observer = self._null_audio_observer
        self._audio_observer_fallback_factory = None
        if self._last_respeaker_runtime_blocker_code == "dead_capture":
            self._last_respeaker_runtime_alert_code = "capture_unknown"
            return
        _record_respeaker_dead_capture_blocker(
            runtime=self.runtime,
            emit=self.emit,
            probe=error.probe,
            stage="runtime",
            signal=None,
        )
        self._last_respeaker_runtime_alert_code = "capture_unknown"
        self._last_respeaker_runtime_blocker_code = "dead_capture"

    def _record_trigger_detected(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        review: ProactiveVisionReview | None = None,
    ) -> None:
        """Record one proactive trigger that reached dispatch evaluation."""

        data = {
            "trigger": decision.trigger_id,
            "reason": decision.reason,
            "priority": int(decision.priority),
            "prompt": decision.prompt,
            "score": decision.score,
            "threshold": decision.threshold,
            "evidence": [item.to_dict() for item in decision.evidence],
            "person_visible": observation.vision.person_visible,
            "body_pose": observation.vision.body_pose.value,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
            "low_motion": observation.low_motion,
            "trigger_source": self._safety_trigger_fusion.last_selected_source,
        }
        fused_claim = self._safety_trigger_fusion.last_selected_claim
        if fused_claim is not None:
            data.update(
                {
                    "fused_claim_state": fused_claim.state,
                    "fused_claim_confidence": fused_claim.confidence,
                    "fused_claim_action_level": fused_claim.action_level.value,
                    "fused_claim_supporting_audio_events": list(fused_claim.supporting_audio_events),
                    "fused_claim_supporting_vision_events": list(fused_claim.supporting_vision_events),
                    "fused_claim_blocked_by": list(fused_claim.blocked_by),
                }
            )
        if review is not None:
            data.update(
                {
                    "vision_review_decision": review.decision,
                    "vision_review_confidence": review.confidence,
                    "vision_review_reason": review.reason,
                    "vision_review_scene": review.scene,
                    "vision_review_frame_count": review.frame_count,
                    "vision_review_response_id": review.response_id,
                    "vision_review_request_id": review.request_id,
                    "vision_review_model": review.model,
                }
            )
        self._append_ops_event(
            event="proactive_trigger_detected",
            message="Proactive trigger conditions were met.",
            data=data,
        )

    def _record_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review: ProactiveVisionReview,
    ) -> None:
        """Record one buffered vision review result."""

        self._append_ops_event(
            event="proactive_vision_reviewed",
            message="Buffered proactive camera frames were reviewed before speaking.",
            data={
                "trigger": decision.trigger_id,
                "approved": review.approved,
                "decision": review.decision,
                "confidence": review.confidence,
                "reason": review.reason,
                "scene": review.scene,
                "frame_count": review.frame_count,
                "response_id": review.response_id,
                "request_id": review.request_id,
                "model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review: ProactiveVisionReview,
    ) -> None:
        """Record that buffered vision review rejected one trigger."""

        self._emit("social_trigger_skipped=vision_review_rejected")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review rejected it.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_rejected",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
                "vision_review_decision": review.decision,
                "vision_review_confidence": review.confidence,
                "vision_review_reason": review.reason,
                "vision_review_scene": review.scene,
                "vision_review_frame_count": review.frame_count,
                "vision_review_response_id": review.response_id,
                "vision_review_request_id": review.request_id,
                "vision_review_model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review_unavailable(
        self,
        decision: SocialTriggerDecision,
    ) -> None:
        """Record that buffered vision review was unavailable for one trigger."""

        self._emit("social_trigger_skipped=vision_review_unavailable")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review was unavailable.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_unavailable",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _presence_session_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> str | None:
        """Return a presence-session block reason for one trigger if any."""

        session_id = getattr(presence_snapshot, "session_id", None)
        if decision.trigger_id != "possible_fall":
            return None
        if not self.config.proactive_possible_fall_once_per_presence_session:
            return None
        if not presence_snapshot.armed or session_id is None:
            return None
        if self._last_possible_fall_prompted_session_id != session_id:
            return None
        return "already_prompted_this_presence_session"

    def _record_trigger_skipped_presence_session(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
        reason: str,
    ) -> None:
        """Record that a trigger was skipped by per-session suppression."""

        session_id = getattr(presence_snapshot, "session_id", None)
        self._emit(f"social_trigger_skipped={reason}")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because it already fired in the current presence session.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _audio_policy_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> str | None:
        """Return a conservative ReSpeaker-driven suppression reason for one trigger."""

        del presence_snapshot
        if decision.trigger_id in _VISION_REVIEW_FAIL_OPEN_TRIGGERS:
            return None
        if audio_policy_snapshot is None:
            return None
        return audio_policy_snapshot.initiative_block_reason

    def _record_trigger_skipped_audio_policy(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        reason: str,
    ) -> None:
        """Record that a trigger was skipped by conservative ReSpeaker policy hooks."""

        session_id = getattr(presence_snapshot, "session_id", None)
        self._emit(f"social_trigger_skipped={reason}")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped by conservative ReSpeaker audio policy.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
                "presence_audio_active": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active
                ),
                "recent_follow_up_speech": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech
                ),
                "room_busy_or_overlapping": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping
                ),
                "quiet_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open
                ),
                "non_speech_audio_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.non_speech_audio_likely
                ),
                "background_media_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.background_media_likely
                ),
                "barge_in_recent": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent
                ),
                "resume_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
                ),
                "mute_blocks_voice_capture": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
                ),
                "speech_delivery_defer_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason
                ),
                "runtime_alert_code": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                ),
            },
        )

    def _dispatch_automation_observation(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_snapshot=None,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot: PresenceSessionSnapshot | None,
        detected_trigger_id: str | None = None,
    ) -> None:
        """Publish one normalized observation to the automation observer hook."""

        try:
            self._remember_display_attention_camera_semantics(
                observed_at=observation.observed_at,
                observation=observation.vision,
                source="full",
            )
            camera_update = self._observe_camera_surface(observation, inspected=inspected)
            self._update_display_debug_signals(
                camera_update,
                detected_trigger_ids=(() if detected_trigger_id is None else (detected_trigger_id,)),
            )
            if not display_gesture_refresh_supported(
                config=self.config,
                vision_observer=self.vision_observer,
            ):
                self._update_display_gesture_emoji_ack(camera_update)
            publish_result = self._update_display_attention_follow(
                source="automation_observation",
                observed_at=observation.observed_at,
                camera_snapshot=camera_update.snapshot,
                audio_observation=observation.audio,
                audio_policy_snapshot=audio_policy_snapshot,
            )
            self._record_display_attention_follow_if_changed(
                observed_at=observation.observed_at,
                runtime_status_value=getattr(getattr(self.runtime, "status", None), "value", None),
                camera_snapshot=camera_update.snapshot,
                publish_result=publish_result,
            )
            if self.observation_handler is None:
                return
            facts = self._build_automation_facts(
                observation,
                inspected=inspected,
                audio_snapshot=audio_snapshot,
                camera_snapshot=camera_update.snapshot,
                audio_policy_snapshot=audio_policy_snapshot,
                presence_snapshot=presence_snapshot,
            )
            event_names = self._derive_sensor_events(facts, camera_event_names=camera_update.event_names)
            self.observation_handler(facts, event_names)
        except Exception as exc:
            self._record_fault(
                event="proactive_observation_handler_failed",
                message="Automation observation dispatch failed.",
                error=exc,
            )

    def _update_display_debug_signals(
        self,
        camera_update: ProactiveCameraSurfaceUpdate,
        *,
        detected_trigger_ids: tuple[str, ...] = (),
    ) -> None:
        """Persist HDMI debug pills from current camera state and recent triggers."""

        service_attention_helpers.update_display_debug_signals(
            self,
            camera_update,
            detected_trigger_ids=detected_trigger_ids,
        )

    def _update_display_attention_follow(
        self,
        *,
        source: str,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_observation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> DisplayAttentionCuePublishResult | None:
        """Update the HDMI face and body-follow servo from the current attention target."""

        return service_attention_helpers.update_display_attention_follow(
            self,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            audio_observation=audio_observation,
            audio_policy_snapshot=audio_policy_snapshot,
        )

    def _update_attention_servo_follow(
        self,
        *,
        source: str,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        attention_target: MultimodalAttentionTargetSnapshot | None,
        attention_target_debug: dict[str, object] | None = None,
    ) -> None:
        """Update the optional body-orientation servo from the current attention target."""

        service_attention_helpers.update_attention_servo_follow(
            self,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
            attention_target_debug=attention_target_debug,
        )

    def _record_attention_servo_forensic_tick(
        self,
        *,
        source: str,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        attention_target: MultimodalAttentionTargetSnapshot | None,
        attention_target_debug: dict[str, object] | None,
        decision: AttentionServoDecision,
    ) -> None:
        """Record one per-tick forensic servo ledger when scoped instrumentation is enabled."""

        service_attention_helpers.record_attention_servo_forensic_tick(
            self,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
            attention_target_debug=attention_target_debug,
            decision=decision,
        )

    def _summarize_visible_persons(self, camera_snapshot: ProactiveCameraSnapshot) -> list[dict[str, object]]:
        """Return one bounded summary of current visible-person anchors."""

        return service_attention_helpers.summarize_visible_persons(camera_snapshot)

    def _build_attention_servo_decision_ledger(
        self,
        *,
        source: str,
        camera_snapshot: ProactiveCameraSnapshot,
        attention_target: MultimodalAttentionTargetSnapshot | None,
        controller_debug: dict[str, object] | None,
        decision: AttentionServoDecision,
    ) -> dict[str, object]:
        """Build one compact decision-ledger payload for forensic servo debugging."""

        return service_attention_helpers.build_attention_servo_decision_ledger(
            self,
            source=source,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
            controller_debug=controller_debug,
            decision=decision,
        )

    def _attention_servo_source_is_authoritative(self, *, source: str) -> bool:
        """Prefer the dedicated HDMI attention-refresh path over automation snapshots for physical servo control."""

        return service_attention_helpers.attention_servo_source_is_authoritative(
            self,
            source=source,
        )

    def _record_attention_follow_pipeline_if_changed(
        self,
        *,
        source: str,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        attention_target: MultimodalAttentionTargetSnapshot | None,
    ) -> None:
        """Record one changed runtime attention-follow pipeline state before servo gating."""

        service_attention_helpers.record_attention_follow_pipeline_if_changed(
            self,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
        )

    def _record_attention_servo_follow_if_changed(
        self,
        *,
        source: str,
        observed_at: float,
        attention_target: MultimodalAttentionTargetSnapshot | None,
        decision: AttentionServoDecision,
    ) -> None:
        """Record only materially changed servo-follow decisions for Pi root-cause tracing."""

        service_attention_helpers.record_attention_servo_follow_if_changed(
            self,
            source=source,
            observed_at=observed_at,
            attention_target=attention_target,
            decision=decision,
        )

    def _update_display_gesture_emoji_ack(
        self,
        camera_update: ProactiveCameraSurfaceUpdate,
    ) -> None:
        """Mirror clear stabilized user gestures into the HDMI emoji reserve area."""

        service_gesture_helpers.update_display_gesture_emoji_ack(self, camera_update)

    def _publish_display_gesture_decision(
        self,
        decision: DisplayGestureEmojiDecision,
    ) -> DisplayGestureEmojiPublishResult | None:
        """Persist one direct gesture-ack decision through the emoji publisher."""

        return service_gesture_helpers.publish_display_gesture_decision(self, decision)

    def _maybe_publish_display_ambient_impulse(
        self,
        *,
        observed_at: float,
        runtime_status_value: str,
        tick_result: ProactiveTickResult,
        presence_active: bool,
    ) -> DisplayAmbientImpulsePublishResult | None:
        """Persist one calm ambient reserve-card impulse when the room is idle."""

        publisher = self.display_ambient_impulse_publisher
        if publisher is None:
            return None
        if tick_result.decision is not None:
            return None
        try:
            return publisher.publish_if_due(
                config=self.config,
                monotonic_now=observed_at,
                runtime_status=runtime_status_value,
                presence_active=presence_active,
            )
        except Exception as exc:
            self._record_fault(
                event="proactive_display_ambient_impulse_failed",
                message="Failed to update the HDMI ambient impulse reserve cue.",
                error=exc,
                data={
                    "runtime_status": runtime_status_value,
                    "presence_active": presence_active,
                },
            )
            return None

    def open_background_lanes(self) -> None:
        """Re-enable background runtime helpers after monitor startup."""

        self._gesture_wakeup_dispatcher.open()

    def close_background_lanes(self, *, timeout_s: float = 0.25) -> bool:
        """Stop background helpers while shutting the monitor down.

        Args:
            timeout_s: Join budget for any in-flight visual wakeup worker.

        Returns:
            True when no visual wakeup worker remains active after the join
            window, False when a worker is still running in the background.
        """

        return self._gesture_wakeup_dispatcher.close(timeout_s=timeout_s)

    def _handle_gesture_wakeup_decision(
        self,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Dispatch one accepted visual wake decision without blocking refresh."""

        return service_gesture_helpers.handle_gesture_wakeup_decision(self, decision)

    def _run_gesture_wakeup_handler(
        self,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Run one visual wakeup handler on the dedicated dispatcher thread."""

        return service_gesture_helpers.run_gesture_wakeup_handler(self, decision)

    def _record_display_attention_follow_if_changed(
        self,
        *,
        observed_at: float,
        runtime_status_value: object,
        camera_snapshot: ProactiveCameraSnapshot,
        publish_result: DisplayAttentionCuePublishResult | None,
    ) -> None:
        """Persist one bounded changed-only trace of the HDMI attention-follow path."""

        service_attention_helpers.record_display_attention_follow_if_changed(
            self,
            observed_at=observed_at,
            runtime_status_value=runtime_status_value,
            camera_snapshot=camera_snapshot,
            publish_result=publish_result,
        )

    def _record_attention_debug_tick(
        self,
        *,
        observed_at: float,
        outcome: str,
        runtime_status_value: object,
        stage_ms: dict[str, float],
        camera_snapshot: ProactiveCameraSnapshot | None = None,
        audio_observation: SocialAudioObservation | None = None,
        publish_result: DisplayAttentionCuePublishResult | None = None,
    ) -> None:
        """Append one continuous bounded attention-debug tick."""

        service_attention_helpers.record_attention_debug_tick(
            self,
            observed_at=observed_at,
            outcome=outcome,
            runtime_status_value=runtime_status_value,
            stage_ms=stage_ms,
            camera_snapshot=camera_snapshot,
            audio_observation=audio_observation,
            publish_result=publish_result,
        )

    def _record_gesture_debug_tick(
        self,
        *,
        observed_at: float,
        outcome: str,
        runtime_status_value: object,
        stage_ms: dict[str, float],
        observation: SocialVisionObservation | None = None,
        decision: DisplayGestureEmojiDecision | None = None,
        publish_result: DisplayGestureEmojiPublishResult | None = None,
        wakeup_decision: GestureWakeupDecision | None = None,
        wakeup_handled: bool | None = None,
    ) -> None:
        """Append one continuous bounded gesture-debug tick."""

        service_gesture_helpers.record_gesture_debug_tick(
            self,
            observed_at=observed_at,
            outcome=outcome,
            runtime_status_value=runtime_status_value,
            stage_ms=stage_ms,
            observation=observation,
            decision=decision,
            publish_result=publish_result,
            wakeup_decision=wakeup_decision,
            wakeup_handled=wakeup_handled,
        )

    def _gesture_observation_trace_details(
        self,
        observation: SocialVisionObservation,
    ) -> dict[str, object]:
        """Return one bounded trace summary for the current gesture observation."""

        return service_gesture_helpers.gesture_observation_trace_details(observation)

    def _trace_gesture_ack_lane_decision(
        self,
        *,
        observation: SocialVisionObservation,
        decision: DisplayGestureEmojiDecision,
    ) -> None:
        """Emit one decision ledger entry for the HDMI ack lane result."""

        service_gesture_helpers.trace_gesture_ack_lane_decision(observation, decision)

    def _trace_gesture_wakeup_lane_decision(
        self,
        *,
        decision: GestureWakeupDecision,
    ) -> None:
        """Emit one decision ledger entry for the visual wake lane."""

        service_gesture_helpers.trace_gesture_wakeup_lane_decision(decision)

    def _trace_gesture_publish_decision(
        self,
        *,
        decision: DisplayGestureEmojiDecision,
        publish_result: DisplayGestureEmojiPublishResult | None,
        wakeup_decision: GestureWakeupDecision,
        wakeup_handled: bool,
    ) -> None:
        """Emit one decision ledger entry for the final publish/dispatch outcome."""

        service_gesture_helpers.trace_gesture_publish_decision(
            decision=decision,
            publish_result=publish_result,
            wakeup_decision=wakeup_decision,
            wakeup_handled=wakeup_handled,
        )

    def _observe_camera_surface(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
    ) -> ProactiveCameraSurfaceUpdate:
        """Project one raw vision observation onto the stabilized camera surface."""

        return self._camera_surface.observe(
            inspected=inspected,
            observed_at=observation.observed_at,
            observation=observation.vision,
        )

    def _observe_display_attention_camera_surface(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
    ) -> ProactiveCameraSurfaceUpdate:
        """Project one raw eye-follow observation onto the attention-only surface."""

        return self._display_attention_camera_surface.observe(
            inspected=inspected,
            observed_at=observation.observed_at,
            observation=observation.vision,
        )

    def _remember_display_attention_camera_semantics(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
        source: str,
    ) -> None:
        """Keep recent rich camera semantics warm for the HDMI attention lane."""

        if source == "gesture":
            self._display_attention_camera_fusion.remember_gesture(
                observed_at=observed_at,
                observation=observation,
            )
            return
        self._display_attention_camera_fusion.remember_full(
            observed_at=observed_at,
            observation=observation,
        )

    def _fuse_display_attention_camera_observation(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> SocialVisionObservation:
        """Fuse richer recent camera semantics into one fast attention sample."""

        result = self._display_attention_camera_fusion.fuse_attention(
            observed_at=observed_at,
            observation=observation,
        )
        self._last_display_attention_fusion_debug = dict(result.debug_details)
        return result.observation

    def _observe_vision_for_gesture_refresh(self):
        """Capture one gesture-only vision snapshot for HDMI emoji acknowledgement."""

        if self.vision_observer is None:
            self._last_gesture_vision_refresh_mode = "missing"
            return None
        observe_gesture = getattr(self.vision_observer, "observe_gesture", None)
        if callable(observe_gesture):
            self._last_gesture_vision_refresh_mode = "gesture_fast"
            return self._observe_vision_with_method(observe_gesture)
        self._last_gesture_vision_refresh_mode = "full_fallback"
        return self._observe_vision_safe()

    def _build_automation_facts(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_snapshot=None,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot: PresenceSessionSnapshot | None,
    ) -> dict[str, Any]:
        """Build the automation-facing fact payload for one observation."""

        now = observation.observed_at
        audio = observation.audio

        speech_detected = audio.speech_detected is True
        quiet = audio.speech_detected is False
        self._speech_detected_since = self._next_since(speech_detected, self._speech_detected_since, now)
        self._quiet_since = self._next_since(quiet, self._quiet_since, now)

        no_motion_for_s = 0.0
        if not observation.pir_motion_detected and self._last_motion_at is not None:
            no_motion_for_s = max(0.0, now - self._last_motion_at)

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        signal_snapshot = None if audio_snapshot is None else getattr(audio_snapshot, "signal_snapshot", None)
        respeaker_claim_contract = build_respeaker_claim_payloads(
            signal_snapshot=signal_snapshot,
            session_id=presence_session_id,
            non_speech_audio_likely=audio.non_speech_audio_likely,
            background_media_likely=audio.background_media_likely,
        )

        facts = {
            "sensor": {
                "inspected": inspected,
                "observed_at": now,
                "captured_at": now,
                "presence_session_id": presence_session_id,
                "voice_activation_armed": None if presence_snapshot is None else presence_snapshot.armed,
                "voice_activation_presence_reason": None if presence_snapshot is None else presence_snapshot.reason,
            },
            "pir": {
                "motion_detected": observation.pir_motion_detected,
                "low_motion": observation.low_motion,
                "no_motion_for_s": round(no_motion_for_s, 3),
            },
            "camera": camera_snapshot.to_automation_facts(),
            "vad": {
                "speech_detected": speech_detected,
                "speech_detected_for_s": round(self._duration_since(self._speech_detected_since, now), 3),
                "quiet": quiet,
                "quiet_for_s": round(self._duration_since(self._quiet_since, now), 3),
                "distress_detected": audio.distress_detected is True,
                "room_quiet": audio.room_quiet,
                "recent_speech_age_s": _round_optional_seconds(audio.recent_speech_age_s),
                "assistant_output_active": audio.assistant_output_active,
                "signal_source": audio.signal_source,
            },
            "respeaker": {
                "runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
                "transport_reason": audio.transport_reason,
                "azimuth_deg": audio.azimuth_deg,
                "direction_confidence": audio.direction_confidence,
                "non_speech_audio_likely": audio.non_speech_audio_likely,
                "background_media_likely": audio.background_media_likely,
                "speech_overlap_likely": audio.speech_overlap_likely,
                "barge_in_detected": audio.barge_in_detected,
                "mute_active": audio.mute_active,
                **resolve_respeaker_indicator_state(
                    runtime_status=getattr(getattr(self.runtime, "status", None), "value", None),
                    runtime_alert_code=(
                        None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                    ),
                    mute_active=audio.mute_active,
                ).event_data(),
                "claim_contract": respeaker_claim_contract,
            },
            "audio_policy": {
                "presence_audio_active": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active
                ),
                "recent_follow_up_speech": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech
                ),
                "room_busy_or_overlapping": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping
                ),
                "quiet_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open
                ),
                "non_speech_audio_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.non_speech_audio_likely
                ),
                "background_media_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.background_media_likely
                ),
                "barge_in_recent": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent
                ),
                "speaker_direction_stable": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
                ),
                "mute_blocks_voice_capture": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
                ),
                "resume_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
                ),
                "initiative_block_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.initiative_block_reason
                ),
                "speech_delivery_defer_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason
                ),
                "runtime_alert_code": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                ),
            },
        }
        speaker_association = derive_respeaker_speaker_association(
            observed_at=now,
            live_facts=facts,
        )
        multimodal_initiative = derive_respeaker_multimodal_initiative(
            observed_at=now,
            live_facts=facts,
            speaker_association=speaker_association,
        )
        self.latest_speaker_association_snapshot = speaker_association
        self.latest_multimodal_initiative_snapshot = multimodal_initiative
        ambiguous_room_guard = derive_ambiguous_room_guard(
            observed_at=now,
            live_facts=facts,
        )
        portrait_match = derive_portrait_match(
            observed_at=now,
            live_facts=facts,
            provider=self.portrait_match_provider,
            ambiguous_room_guard=ambiguous_room_guard,
            now_monotonic=now,
        )
        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        identity_fusion = self.identity_fusion_tracker.observe(
            observed_at=now,
            live_facts=facts,
            voice_status=getattr(self.runtime, "user_voice_status", None),
            voice_confidence=getattr(self.runtime, "user_voice_confidence", None),
            voice_checked_at=getattr(self.runtime, "user_voice_checked_at", None),
            voice_matched_user_id=getattr(self.runtime, "user_voice_user_id", None),
            voice_matched_user_display_name=getattr(self.runtime, "user_voice_user_display_name", None),
            voice_match_source=getattr(self.runtime, "user_voice_match_source", None),
            max_voice_age_s=int(getattr(self.config, "voice_assessment_max_age_s", 120) or 120),
            presence_session_id=presence_session_id,
            ambiguous_room_guard=ambiguous_room_guard,
            speaker_association=speaker_association,
            portrait_match=portrait_match,
        )
        known_user_hint = derive_known_user_hint(
            observed_at=now,
            live_facts=facts,
            voice_status=getattr(self.runtime, "user_voice_status", None),
            voice_confidence=getattr(self.runtime, "user_voice_confidence", None),
            voice_checked_at=getattr(self.runtime, "user_voice_checked_at", None),
            max_voice_age_s=int(getattr(self.config, "voice_assessment_max_age_s", 120) or 120),
            ambiguous_room_guard=ambiguous_room_guard,
            speaker_association=speaker_association,
            portrait_match=portrait_match,
            identity_fusion=identity_fusion,
        )
        affect_proxy = derive_affect_proxy(
            observed_at=now,
            live_facts=facts,
            ambiguous_room_guard=ambiguous_room_guard,
        )
        attention_target = self.attention_target_tracker.observe(
            observed_at=now,
            live_facts=facts,
            runtime_status=getattr(getattr(self.runtime, "status", None), "value", None),
            presence_session_id=presence_session_id,
            speaker_association=speaker_association,
            identity_fusion=identity_fusion,
        )
        person_state = derive_person_state(
            observed_at=now,
            live_facts={
                **facts,
                "speaker_association": speaker_association.to_automation_facts(),
                "multimodal_initiative": multimodal_initiative.to_automation_facts(),
                "ambiguous_room_guard": ambiguous_room_guard.to_automation_facts(),
                "identity_fusion": identity_fusion.to_automation_facts(),
                "portrait_match": portrait_match.to_automation_facts(),
                "known_user_hint": known_user_hint.to_automation_facts(),
                "affect_proxy": affect_proxy.to_automation_facts(),
                "attention_target": attention_target.to_automation_facts(),
            },
        )
        self.latest_ambiguous_room_guard_snapshot = ambiguous_room_guard
        self.latest_identity_fusion_snapshot = identity_fusion
        self.latest_portrait_match_snapshot = portrait_match
        self.latest_known_user_hint_snapshot = known_user_hint
        self.latest_affect_proxy_snapshot = affect_proxy
        self.latest_attention_target_snapshot = attention_target
        self.latest_person_state_snapshot = person_state
        facts["speaker_association"] = speaker_association.to_automation_facts()
        facts["multimodal_initiative"] = multimodal_initiative.to_automation_facts()
        facts["ambiguous_room_guard"] = ambiguous_room_guard.to_automation_facts()
        facts["identity_fusion"] = identity_fusion.to_automation_facts()
        facts["portrait_match"] = portrait_match.to_automation_facts()
        facts["known_user_hint"] = known_user_hint.to_automation_facts()
        facts["affect_proxy"] = affect_proxy.to_automation_facts()
        facts["attention_target"] = attention_target.to_automation_facts()
        facts["person_state"] = person_state.to_automation_facts()
        return facts

    def _derive_sensor_events(
        self,
        facts: dict[str, Any],
        *,
        camera_event_names: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return rising-edge event names derived from the latest fact payload."""

        current_flags = {
            "pir.motion_detected": bool(facts["pir"]["motion_detected"]),
            "vad.speech_detected": bool(facts["vad"]["speech_detected"]),
            "audio_policy.presence_audio_active": bool(facts["audio_policy"]["presence_audio_active"]),
            "audio_policy.quiet_window_open": bool(facts["audio_policy"]["quiet_window_open"]),
            "audio_policy.resume_window_open": bool(facts["audio_policy"]["resume_window_open"]),
            "audio_policy.room_busy_or_overlapping": bool(facts["audio_policy"]["room_busy_or_overlapping"]),
            "audio_policy.barge_in_recent": bool(facts["audio_policy"]["barge_in_recent"]),
            "speaker_association.associated": bool(facts["speaker_association"]["associated"]),
            "multimodal_initiative.ready": bool(facts["multimodal_initiative"]["ready"]),
            "ambiguous_room_guard.guard_active": bool(facts["ambiguous_room_guard"]["guard_active"]),
            "identity_fusion.matches_main_user": bool(facts["identity_fusion"]["matches_main_user"]),
            "portrait_match.matches_reference_user": bool(facts["portrait_match"]["matches_reference_user"]),
            "known_user_hint.matches_main_user": bool(facts["known_user_hint"]["matches_main_user"]),
            "affect_proxy.concern_cue": facts["affect_proxy"]["state"] == "concern_cue",
            "attention_target.session_focus_active": bool(facts["attention_target"]["session_focus_active"]),
            "person_state.interaction_ready": bool(facts["person_state"]["interaction_ready"]),
            "person_state.safety_concern_active": bool(facts["person_state"]["safety_concern_active"]),
            "person_state.calm_personalization_allowed": bool(facts["person_state"]["calm_personalization_allowed"]),
        }
        event_names: list[str] = list(camera_event_names)
        for key, value in current_flags.items():
            previous = self._last_sensor_flags.get(key)
            if value and previous is not True:
                event_names.append(key)
        self._last_sensor_flags = current_flags
        return tuple(event_names)

    def _next_since(self, active: bool, since: float | None, now: float) -> float | None:
        """Advance or clear one duration anchor depending on activity."""

        if active:
            return now if since is None else since
        return None

    def _duration_since(self, since: float | None, now: float) -> float:
        """Return the elapsed duration for one optional activity anchor."""

        if since is None:
            return 0.0
        return max(0.0, now - since)


class ProactiveMonitorService:
    """Run the proactive coordinator in a bounded background worker."""

    def __init__(
        self,
        coordinator: ProactiveCoordinator,
        *,
        poll_interval_s: float,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize one monitor service around a configured coordinator."""

        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.emit = emit or (lambda _line: None)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lifecycle_lock = Lock()  # AUDIT-FIX(#3): serialize open/close so concurrent lifecycle calls cannot double-start or half-close the worker.
        self._resources_open = False
        self._close_join_timeout_s = _DEFAULT_CLOSE_JOIN_TIMEOUT_S

    # AUDIT-FIX(#1): Safe telemetry wrappers for service lifecycle and worker faults.
    def _emit(self, line: str) -> None:
        """Emit one service-local telemetry line safely."""

        _safe_emit(self.emit, line)

    # AUDIT-FIX(#1): Safe telemetry wrappers for service lifecycle and worker faults.
    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, Any],
        level: str | None = None,
    ) -> None:
        """Append one service-local ops event safely."""

        _append_ops_event(
            self.coordinator.runtime,
            event=event,
            message=message,
            data=data,
            level=level,
            emit=self.emit,
        )

    # AUDIT-FIX(#3): Close optional resources defensively because shutdown cleanup must not raise while handling a stop request.
    def _safe_close_resource(self, resource: Any, *, name: str) -> None:
        """Close one optional resource while suppressing cleanup failures."""

        if resource is None:
            return
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            self._append_ops_event(
                event="proactive_resource_close_failed",
                level="error",
                message="Failed to close a proactive monitor resource cleanly.",
                data={
                    "resource": name,
                    "error": _exception_text(exc),
                },
            )

    # AUDIT-FIX(#3): Open hardware resources under the lifecycle lock so partial startups are unwound immediately.
    def _open_resources_locked(self) -> None:
        """Open hardware resources under the lifecycle lock."""

        if self._resources_open:
            return
        opened_pir = False
        try:
            if self.coordinator.pir_monitor is not None:
                pir_open_result = open_pir_monitor_with_busy_retry(self.coordinator.pir_monitor)
                if pir_open_result.busy_retry_count > 0:
                    self._append_ops_event(
                        event="proactive_pir_open_retried",
                        message="PIR startup waited for a transient busy GPIO line to clear.",
                        data={
                            "attempt_count": pir_open_result.attempt_count,
                            "busy_retry_count": pir_open_result.busy_retry_count,
                        },
                    )
                opened_pir = True
            self._resources_open = True
        except Exception as exc:
            if opened_pir:
                self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
            self._resources_open = False
            self._append_ops_event(
                event="proactive_monitor_start_failed",
                level="error",
                message="Failed to open proactive monitor resources.",
                data={"error": _exception_text(exc)},
            )
            raise

    # AUDIT-FIX(#3): Closing resources before join helps unblock hardware/audio backends during shutdown.
    def _close_resources_locked(self) -> None:
        """Close any opened hardware resources under the lifecycle lock."""

        if not self._resources_open:
            return
        self._safe_close_resource(self.coordinator.audio_observer, name="audio_observer")
        self._safe_close_resource(
            self.coordinator.attention_servo_controller,
            name="attention_servo_controller",
        )
        self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
        self._resources_open = False

    def open(self) -> "ProactiveMonitorService":
        """Open resources and start the background proactive worker."""

        with self._lifecycle_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._thread is not None:
                return self
            self._open_resources_locked()
            self.coordinator.open_background_lanes()
            self._stop_event.clear()
            thread = Thread(target=self._run, daemon=True, name="twinr-proactive")
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                self._thread = None
                self.coordinator.close_background_lanes(timeout_s=0.05)
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_start_failed",
                    level="error",
                    message="Failed to start the proactive monitor worker thread.",
                    data={"error": _exception_text(exc)},
                )
                raise
            self._append_ops_event(
                event="proactive_monitor_started",
                message="Proactive monitor started.",
                data={"poll_interval_s": self.poll_interval_s},
            )
            self._emit("proactive_monitor=started")
            return self

    def close(self) -> None:
        """Request worker shutdown and close monitor resources."""

        thread_to_join: Thread | None = None
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None and not self._resources_open:
                return
            self._stop_event.set()
            self.coordinator.close_background_lanes(timeout_s=min(self._close_join_timeout_s, 0.25))
            self._close_resources_locked()
            if thread is current_thread():
                self._append_ops_event(
                    event="proactive_monitor_stop_requested",
                    message="Proactive monitor stop was requested from the worker thread.",
                    data={},
                )
                self._emit("proactive_monitor=stopping")
                return
            thread_to_join = thread
        if thread_to_join is not None:
            thread_to_join.join(timeout=self._close_join_timeout_s)
            if thread_to_join.is_alive():
                self._append_ops_event(
                    event="proactive_monitor_stop_timeout",
                    level="error",
                    message="Proactive monitor worker did not stop within the shutdown budget.",
                    data={"join_timeout_s": self._close_join_timeout_s},
                )
                self._emit("proactive_monitor=stop_timeout")

    def __enter__(self) -> "ProactiveMonitorService":
        """Enter the monitor context by starting the service."""

        return self.open()  # AUDIT-FIX(#3): centralize all startup/open logic in one path so resource handling stays consistent.

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the monitor context by stopping the service."""

        self.close()

    def _run(self) -> None:
        """Run the proactive tick loop until stopped."""

        next_tick_at = 0.0
        next_attention_refresh_at = 0.0
        next_gesture_refresh_at = 0.0
        try:
            while not self._stop_event.is_set():
                did_work = False
                now = time.monotonic()
                if now >= next_attention_refresh_at:
                    try:
                        if self.coordinator.refresh_display_attention():
                            did_work = True
                    except Exception as exc:
                        error_text = _exception_text(exc)
                        self._emit(f"proactive_error={error_text}")
                        self._append_ops_event(
                            event="proactive_error",
                            level="error",
                            message="Display attention refresh failed.",
                            data={"error": error_text},
                        )
                    interval_s = resolve_display_attention_refresh_interval(self.coordinator.config)
                    next_attention_refresh_at = (
                        time.monotonic() + interval_s
                        if interval_s is not None
                        else time.monotonic() + self.poll_interval_s
                    )
                now = time.monotonic()
                if now >= next_gesture_refresh_at:
                    try:
                        if self.coordinator.refresh_display_gesture_emoji():
                            did_work = True
                    except Exception as exc:
                        error_text = _exception_text(exc)
                        self._emit(f"proactive_error={error_text}")
                        self._append_ops_event(
                            event="proactive_error",
                            level="error",
                            message="Display gesture refresh failed.",
                            data={"error": error_text},
                        )
                    interval_s = resolve_display_gesture_refresh_interval(self.coordinator.config)
                    next_gesture_refresh_at = (
                        time.monotonic() + interval_s
                        if interval_s is not None
                        else time.monotonic() + self.poll_interval_s
                    )
                now = time.monotonic()
                if now >= next_tick_at:
                    try:
                        self.coordinator.tick()
                        did_work = True
                    except Exception as exc:
                        error_text = _exception_text(exc)
                        self._emit(f"proactive_error={error_text}")
                        self._append_ops_event(
                            event="proactive_error",
                            level="error",
                            message="Proactive monitor tick failed.",
                            data={"error": error_text},
                        )
                    next_tick_at = time.monotonic() + self.poll_interval_s
                wait_s = max(0.02, min(0.05, max(0.0, next_tick_at - time.monotonic())))
                if did_work:
                    wait_s = 0.02
                if self._stop_event.wait(wait_s):
                    return
        finally:
            with self._lifecycle_lock:
                if self._thread is current_thread():
                    self._thread = None
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_stopped",
                    message="Proactive monitor stopped.",
                    data={},
                )
                self._emit("proactive_monitor=stopped")


def build_default_proactive_monitor(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    camera: V4L2StillCamera,
    camera_lock: Lock | None,
    audio_lock: Lock | None,
    trigger_handler: Callable[[SocialTriggerDecision], bool],
    gesture_wakeup_handler: Callable[[GestureWakeupDecision], bool] | None = None,
    idle_predicate: Callable[[], bool] | None = None,
    observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
    live_context_handler: Callable[[dict[str, Any]], None] | None = None,
    emit: Callable[[str], None] | None = None,
) -> ProactiveMonitorService | None:
    """Build the default proactive monitor stack from Twinr runtime services."""

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
        engine: SocialTriggerEngine | _NullSocialTriggerEngine = (
            SocialTriggerEngine.from_config(config)
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
            pir_monitor = configured_pir_monitor(config)
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
        presence_session = PresenceSessionController(
            presence_grace_s=config.voice_orchestrator_follow_up_timeout_s,
            motion_grace_s=config.voice_orchestrator_follow_up_timeout_s,
            speech_grace_s=config.voice_orchestrator_follow_up_timeout_s,
        )

    audio_observer = NullAudioObservationProvider()
    audio_observer_fallback_factory: Callable[[], Any] | None = None
    distress_enabled = bool(
        config.proactive_enabled and config.proactive_audio_distress_enabled
    )  # AUDIT-FIX(#5): do not run proactive distress classification when proactive mode is disabled.
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
            audio_observer = NullAudioObservationProvider()
        else:
            try:
                sampler = AmbientAudioSampler.from_config(config)
                audio_observer = AmbientAudioObservationProvider(
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
                audio_observer = NullAudioObservationProvider()

    respeaker_targeted = config_targets_respeaker(
        getattr(config, "audio_input_device", None),
        getattr(config, "proactive_audio_input_device", None),
    )
    if respeaker_targeted and config.proactive_audio_enabled:
        try:
            base_signal_provider = ReSpeakerSignalProvider(
                sensor_window_ms=config.proactive_audio_sample_ms,
                assistant_output_active_predicate=lambda: _assistant_output_active(runtime),
            )
            signal_provider = ScheduledReSpeakerSignalProvider.from_config(
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
                        sampler=AmbientAudioSampler.from_config(config),
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
            audio_observer = ReSpeakerAudioObservationProvider(
                signal_provider=signal_provider,
                fallback_observer=base_audio_observer,
            )
            previous_factory = audio_observer_fallback_factory
            if previous_factory is not None:
                def _fallback_respeaker_audio_observer_factory() -> Any:
                    fallback_observer = previous_factory()
                    return ReSpeakerAudioObservationProvider(
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
                audio_observer = NullAudioObservationProvider()
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
                vision_observer = OpenAIVisionObservationProvider(
                    backend=backend,
                    camera=camera,
                    camera_lock=camera_lock,
                )
            elif provider_name == "aideck_openai":
                vision_observer = AIDeckOpenAIVisionObservationProvider.from_config(
                    config,
                    backend=backend,
                    camera=camera,
                    camera_lock=camera_lock,
                )
            elif provider_name == "remote_proxy":
                vision_observer = RemoteAICameraObservationProvider.from_config(config)
            elif provider_name == "remote_frame":
                vision_observer = RemoteFrameAICameraObservationProvider.from_config(config)
            else:
                vision_observer = LocalAICameraObservationProvider.from_config(config)
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
            vision_reviewer = OpenAIProactiveVisionReviewer(
                backend=backend,
                frame_buffer=ProactiveVisionFrameBuffer(
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
            portrait_match_provider = PortraitMatchProvider.from_config(
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
        and isinstance(audio_observer, NullAudioObservationProvider)
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
        return None  # AUDIT-FIX(#4): avoid starting an inert monitor that gives a false sense of protection.

    coordinator = ProactiveCoordinator(
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
    return ProactiveMonitorService(
        coordinator,
        poll_interval_s=config.proactive_poll_interval_s,
        emit=emit,
    )


__all__ = [
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
