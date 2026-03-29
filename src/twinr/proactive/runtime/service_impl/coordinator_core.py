"""Core proactive coordinator state and tick orchestration.

Purpose: own dependency wiring, observer access, presence updates, trigger
review, and the main proactive tick loop without the monitor lifecycle wrapper.

Invariants: public coordinator behavior, emitted events, and fail-open/fail-
closed decisions must remain compatible with the legacy
``twinr.proactive.runtime.service`` module.
"""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast
import time
import uuid

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AudioCaptureReadinessError
from twinr.hardware.camera_ai.gesture_forensics import GestureForensics
from twinr.hardware.pir import GpioPirMonitor
from twinr.hardware.servo_follow import AttentionServoController
from twinr.hardware.respeaker import config_targets_respeaker
from twinr.hardware.portrait_match import PortraitMatchProvider

from twinr.proactive.runtime.service_impl.compat import (
    _ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
    _DEFAULT_PROJECT_ROOT,
    _VISION_REVIEW_FAIL_OPEN_TRIGGERS,
    _append_ops_event,
    _exception_text,
    _round_optional_seconds,
    _safe_emit,
)

from ...social.camera_surface import ProactiveCameraSurface
from ...social.engine import (
    SocialAudioObservation,
    SocialObservation,
    SocialTriggerDecision,
    SocialTriggerEngine,
    SocialVisionObservation,
)
from ...social.observers import AmbientAudioObservationProvider, NullAudioObservationProvider
from ...social.observers import ProactiveAudioSnapshot
from ...social.vision_review import (
    OpenAIProactiveVisionReviewer,
    ProactiveVisionReview,
    is_reviewable_image_trigger,
)
from ..affect_proxy import AffectProxySnapshot
from ..ambiguous_room_guard import AmbiguousRoomGuardSnapshot
from ..attention_debug_stream import AttentionDebugStream
from ..attention_targeting import (
    MultimodalAttentionTargetSnapshot,
)
from ..audio_policy import ReSpeakerAudioPolicySnapshot, ReSpeakerAudioPolicyTracker
from ..display_ambient_impulses import DisplayAmbientImpulsePublisher
from ..display_attention import DisplayAttentionCuePublisher
from ..display_attention_camera_fusion import DisplayAttentionCameraFusion
from ..display_debug_signals import DisplayDebugSignalPublisher
from ..display_gesture_emoji import DisplayGestureEmojiPublisher
from ..gesture_debug_stream import GestureDebugStream
from ..gesture_wakeup_dispatcher import GestureWakeupDispatcher
from ..gesture_wakeup_lane import GestureWakeupDecision
from ..identity_fusion import (
    MultimodalIdentityFusionSnapshot,
    TemporalIdentityFusionTracker,
)
from ..known_user_hint import KnownUserHintSnapshot
from ..multimodal_initiative import ReSpeakerMultimodalInitiativeSnapshot
from ..perception_orchestrator import PerceptionRuntimeSnapshot, PerceptionStreamOrchestrator
from ..person_state import PersonStateSnapshot
from ..portrait_match import PortraitMatchSnapshot
from ..presence import PresenceSessionController, PresenceSessionSnapshot
from ..safety_trigger_fusion import SafetyTriggerFusionBridge
from ..speaker_association import ReSpeakerSpeakerAssociationSnapshot

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime.runtime import TwinrRuntime


@dataclass(slots=True)
class ProactiveTickResult:
    """Describe the externally relevant outcome of one monitor tick."""

    decision: SocialTriggerDecision | None = None
    inspected: bool = False
    person_visible: bool = False


class _NullSocialTriggerEngine:
    """Provide a no-op trigger engine when proactive triggers are disabled."""

    best_evaluation = None

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Return no proactive decision for the given observation."""

        del observation
        return None


class ProactiveCoordinatorCoreMixin:
    """Hold dependency wiring and the main proactive tick sequencing."""

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
        self._null_audio_observer = NullAudioObservationProvider()
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
        project_root = Path(getattr(config, "project_root", _DEFAULT_PROJECT_ROOT)).expanduser()
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
        self._display_perception_cycle = None
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
        self.perception_orchestrator = PerceptionStreamOrchestrator.from_config(config)
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
        self.latest_perception_runtime_snapshot: PerceptionRuntimeSnapshot | None = None
        self.latest_person_state_snapshot: PersonStateSnapshot | None = None
        self.audio_policy_tracker = ReSpeakerAudioPolicyTracker.from_config(config)
        self.identity_fusion_tracker = TemporalIdentityFusionTracker.from_config(config)

    def _emit(self, line: str) -> None:
        """Emit one coordinator-local telemetry line safely."""

        _safe_emit(self.emit, line)

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
        """Return a low-latency audio snapshot for HDMI attention refresh."""

        fast_observe = getattr(self.audio_observer, "observe_signal_only", None)
        if callable(fast_observe):
            try:
                snapshot = cast(Callable[[], ProactiveAudioSnapshot], fast_observe)()  # pylint: disable=not-callable
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

        shared_snapshot = self._shared_display_perception_snapshot(consumer="attention")
        if shared_snapshot is not None:
            return shared_snapshot
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

    def _proactive_triggers_enabled(self) -> bool:
        """Return whether camera-driven proactive triggers may run."""

        return bool(self.config.proactive_enabled)

    def _should_fail_open_without_vision_review(self, decision: SocialTriggerDecision) -> bool:
        """Return whether missing review must not block this safety trigger."""

        return decision.trigger_id in _VISION_REVIEW_FAIL_OPEN_TRIGGERS

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

        del now
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
        )
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
        self._emit(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")
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
        )

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
        try:
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
            self._emit(f"proactive_trigger={decision.trigger_id}")
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
        try:
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
        """Return whether speech should trigger one bounded local vision inspection."""

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
