"""Display and camera-surface helpers for proactive observation dispatch.

Purpose: keep HDMI/display bridges, gesture helpers, and camera-surface
projection separate from ops recording and automation-fact assembly.

Invariants: HDMI cue publishing, gesture-ack behavior, camera-surface state,
and automation-dispatch side effects must remain compatible with the legacy
observation mixin.
"""

# mypy: ignore-errors

from __future__ import annotations

from twinr.hardware.servo_follow import AttentionServoDecision

from ...social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from ...social.engine import SocialAudioObservation, SocialObservation, SocialVisionObservation
from ..attention_targeting import MultimodalAttentionTargetSnapshot
from ..audio_policy import ReSpeakerAudioPolicySnapshot
from ..display_ambient_impulses import DisplayAmbientImpulsePublishResult
from ..display_attention import DisplayAttentionCuePublishResult
from ..display_gesture_emoji import (
    DisplayGestureEmojiDecision,
    DisplayGestureEmojiPublishResult,
    display_gesture_refresh_supported,
)
from ..gesture_wakeup_lane import GestureWakeupDecision
from .. import service_attention_helpers
from .. import service_gesture_helpers


class ProactiveCoordinatorObservationDisplayMixin:
    """Bridge observation dispatch into display and camera helper modules."""

    def _dispatch_automation_observation(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_snapshot=None,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot,
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
        """Prefer the dedicated HDMI attention-refresh path over automation snapshots."""

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
        tick_result,
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
        """Stop background helpers while shutting the monitor down."""

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

        shared_snapshot = self._shared_display_perception_snapshot(consumer="gesture")
        if shared_snapshot is not None:
            return shared_snapshot
        if self.vision_observer is None:
            self._last_gesture_vision_refresh_mode = "missing"
            return None
        observe_gesture = getattr(self.vision_observer, "observe_gesture", None)
        if callable(observe_gesture):
            self._last_gesture_vision_refresh_mode = "gesture_fast"
            return self._observe_vision_with_method(observe_gesture)
        self._last_gesture_vision_refresh_mode = "full_fallback"
        return self._observe_vision_safe()
