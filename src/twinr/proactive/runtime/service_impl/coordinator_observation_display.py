# CHANGELOG: 2026-03-29
# BUG-1: Entkoppelt optionale HDMI-/Gesture-/Attention-Side-Effects vom Automation-Dispatch; Fehler in Display/Servo/Debug unterdruecken observation_handler() nicht mehr.
# BUG-2: Fault-Logging ist jetzt fail-safe; Fehler in _record_fault() propagieren nicht mehr in den Observation-Pfad.
# BUG-3: HDMI-Gesture-Refresh bevorzugt wieder den dedizierten Low-Latency observe_gesture()-Pfad statt zuerst den langsameren shared perception_stream Snapshot zu uebernehmen.
# SEC-1: detected_trigger_id und Fault-Payloads werden vor Display-/Log-Emission sanitisiert und begrenzt, um Overlay-/Log-Injection und Log-Amplification zu verhindern.
# IMP-1: Optionale Display-Stages nutzen adaptives Fault-Backoff/Circuit-Breaking, damit ein defekter HDMI-/Servo-Pfad den Pi nicht mit Exceptions und I/O flutet.
# IMP-2: Dispatch erhaelt strukturierte, bounded Stage-Timings/Outcome-Diagnostik; Gesture shared-snapshot refresh mode wird korrekt markiert.

"""Display and camera-surface helpers for proactive observation dispatch.

Purpose: keep HDMI/display bridges, gesture helpers, and camera-surface
projection separate from ops recording and automation-fact assembly.

Invariants: HDMI cue publishing, gesture-ack behavior, camera-surface state,
and automation-dispatch side effects must remain compatible with the legacy
observation mixin.
"""


from __future__ import annotations

import logging
import math
import time
from typing import Callable, TypeVar, cast

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

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


class ProactiveCoordinatorObservationDisplayMixin:
    """Bridge observation dispatch into display and camera helper modules."""

    _OPTIONAL_STAGE_BACKOFF_BASE_S = 0.5
    _OPTIONAL_STAGE_BACKOFF_MAX_S = 30.0
    _MAX_DETECTED_TRIGGER_IDS = 8
    _MAX_DEBUG_TRIGGER_ID_LEN = 96
    _MAX_FAULT_TEXT_LEN = 240
    _MAX_FAULT_DATA_DEPTH = 3
    _MAX_FAULT_DATA_ITEMS = 24
    _MONOTONIC_EPOCH_THRESHOLD_S = 100_000_000.0

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

        stage_ms: dict[str, float] = {}
        sanitized_trigger_ids = self._sanitize_detected_trigger_ids(
            () if detected_trigger_id is None else (detected_trigger_id,)
        )

        stage_started = time.perf_counter()
        self._run_optional_display_stage(
            stage="remember_display_attention_camera_semantics",
            call=lambda: self._remember_display_attention_camera_semantics(
                observed_at=observation.observed_at,
                observation=observation.vision,
                source="full",
            ),
            fault_event="proactive_display_attention_camera_semantics_failed",
            fault_message="Failed to retain rich camera semantics for the HDMI attention lane.",
            fault_data={"source": "full"},
        )
        stage_ms["remember_display_attention_camera_semantics"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        try:
            camera_update = self._observe_camera_surface(observation, inspected=inspected)
        except Exception as exc:
            stage_ms["observe_camera_surface"] = self._elapsed_ms(stage_started)
            self._remember_observation_dispatch_diagnostics(
                observed_at=observation.observed_at,
                outcome="failed",
                stage_ms=stage_ms,
                fault_stage="observe_camera_surface",
            )
            self._safe_record_fault(
                event="proactive_camera_surface_observation_failed",
                message="Automation observation camera-surface projection failed.",
                error=exc,
                data={
                    "inspected": inspected,
                    "detected_trigger_ids": sanitized_trigger_ids,
                    "stage_ms": stage_ms,
                },
            )
            return
        stage_ms["observe_camera_surface"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        self._run_optional_display_stage(
            stage="update_display_debug_signals",
            call=lambda: self._update_display_debug_signals(
                camera_update,
                detected_trigger_ids=sanitized_trigger_ids,
            ),
            fault_event="proactive_display_debug_signals_failed",
            fault_message="Failed to update HDMI debug pills from the latest camera state.",
            fault_data={"detected_trigger_ids": sanitized_trigger_ids},
        )
        stage_ms["update_display_debug_signals"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        gesture_refresh_supported = self._display_gesture_refresh_supported_safe()
        stage_ms["display_gesture_refresh_supported"] = self._elapsed_ms(stage_started)

        if not gesture_refresh_supported:
            stage_started = time.perf_counter()
            self._run_optional_display_stage(
                stage="update_display_gesture_emoji_ack",
                call=lambda: self._update_display_gesture_emoji_ack(camera_update),
                fault_event="proactive_display_gesture_emoji_ack_failed",
                fault_message="Failed to mirror the stabilized gesture into the HDMI emoji reserve area.",
            )
            stage_ms["update_display_gesture_emoji_ack"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        publish_result = self._run_optional_display_stage(
            stage="update_display_attention_follow",
            call=lambda: self._update_display_attention_follow(
                source="automation_observation",
                observed_at=observation.observed_at,
                camera_snapshot=camera_update.snapshot,
                audio_observation=observation.audio,
                audio_policy_snapshot=audio_policy_snapshot,
            ),
            fault_event="proactive_display_attention_follow_failed",
            fault_message="Failed to update the HDMI attention-follow cue.",
            fault_data={"source": "automation_observation"},
            default=None,
        )
        stage_ms["update_display_attention_follow"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        self._run_optional_display_stage(
            stage="record_display_attention_follow_if_changed",
            call=lambda: self._record_display_attention_follow_if_changed(
                observed_at=observation.observed_at,
                runtime_status_value=getattr(getattr(self.runtime, "status", None), "value", None),
                camera_snapshot=camera_update.snapshot,
                publish_result=publish_result,
            ),
            fault_event="proactive_display_attention_follow_record_failed",
            fault_message="Failed to persist the changed-only HDMI attention-follow trace.",
        )
        stage_ms["record_display_attention_follow_if_changed"] = self._elapsed_ms(stage_started)

        if self.observation_handler is None:
            self._remember_observation_dispatch_diagnostics(
                observed_at=observation.observed_at,
                outcome="display_only",
                stage_ms=stage_ms,
                fault_stage=None,
            )
            return

        stage_started = time.perf_counter()
        try:
            facts = self._build_automation_facts(
                observation,
                inspected=inspected,
                audio_snapshot=audio_snapshot,
                camera_snapshot=camera_update.snapshot,
                audio_policy_snapshot=audio_policy_snapshot,
                presence_snapshot=presence_snapshot,
            )
        except Exception as exc:
            stage_ms["build_automation_facts"] = self._elapsed_ms(stage_started)
            self._remember_observation_dispatch_diagnostics(
                observed_at=observation.observed_at,
                outcome="failed",
                stage_ms=stage_ms,
                fault_stage="build_automation_facts",
            )
            self._safe_record_fault(
                event="proactive_automation_facts_failed",
                message="Failed to build automation facts from the normalized observation.",
                error=exc,
                data={
                    "inspected": inspected,
                    "detected_trigger_ids": sanitized_trigger_ids,
                    "stage_ms": stage_ms,
                },
            )
            return
        stage_ms["build_automation_facts"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        try:
            event_names = self._derive_sensor_events(
                facts,
                camera_event_names=camera_update.event_names,
            )
        except Exception as exc:
            stage_ms["derive_sensor_events"] = self._elapsed_ms(stage_started)
            self._remember_observation_dispatch_diagnostics(
                observed_at=observation.observed_at,
                outcome="failed",
                stage_ms=stage_ms,
                fault_stage="derive_sensor_events",
            )
            self._safe_record_fault(
                event="proactive_sensor_event_derivation_failed",
                message="Failed to derive sensor events from automation facts.",
                error=exc,
                data={
                    "detected_trigger_ids": sanitized_trigger_ids,
                    "stage_ms": stage_ms,
                },
            )
            return
        stage_ms["derive_sensor_events"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        try:
            self.observation_handler(facts, event_names)
        except Exception as exc:
            stage_ms["observation_handler"] = self._elapsed_ms(stage_started)
            self._remember_observation_dispatch_diagnostics(
                observed_at=observation.observed_at,
                outcome="failed",
                stage_ms=stage_ms,
                fault_stage="observation_handler",
            )
            self._safe_record_fault(
                event="proactive_observation_handler_failed",
                message="Automation observation dispatch failed.",
                error=exc,
                data={
                    "detected_trigger_ids": sanitized_trigger_ids,
                    "stage_ms": stage_ms,
                },
            )
            return
        stage_ms["observation_handler"] = self._elapsed_ms(stage_started)

        self._remember_observation_dispatch_diagnostics(
            observed_at=observation.observed_at,
            outcome="completed",
            stage_ms=stage_ms,
            fault_stage=None,
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
            detected_trigger_ids=self._sanitize_detected_trigger_ids(detected_trigger_ids),
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
        return self._run_optional_display_stage(
            stage="display_ambient_impulse_publish",
            call=lambda: publisher.publish_if_due(
                config=self.config,
                monotonic_now=self._scheduler_monotonic_now(observed_at),
                runtime_status=runtime_status_value,
                presence_active=presence_active,
            ),
            fault_event="proactive_display_ambient_impulse_failed",
            fault_message="Failed to update the HDMI ambient impulse reserve cue.",
            fault_data={
                "runtime_status": runtime_status_value,
                "presence_active": presence_active,
            },
            default=None,
        )

    def open_background_lanes(self) -> None:
        """Re-enable background runtime helpers after monitor startup."""

        self._run_optional_display_stage(
            stage="gesture_wakeup_dispatcher_open",
            call=lambda: self._gesture_wakeup_dispatcher.open(),
            fault_event="proactive_gesture_wakeup_dispatcher_open_failed",
            fault_message="Failed to re-enable the gesture wakeup dispatcher after monitor startup.",
        )

    def close_background_lanes(self, *, timeout_s: float = 0.25) -> bool:
        """Stop background helpers while shutting the monitor down."""

        try:
            timeout_s = max(0.0, float(timeout_s))
        except (TypeError, ValueError):
            timeout_s = 0.25
        if not math.isfinite(timeout_s):
            timeout_s = 0.25
        return bool(
            self._run_optional_display_stage(
                stage="gesture_wakeup_dispatcher_close",
                call=lambda: self._gesture_wakeup_dispatcher.close(timeout_s=timeout_s),
                fault_event="proactive_gesture_wakeup_dispatcher_close_failed",
                fault_message="Failed to stop the gesture wakeup dispatcher while shutting the monitor down.",
                fault_data={"timeout_s": timeout_s},
                default=False,
            )
        )

    def _handle_gesture_wakeup_decision(
        self,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Dispatch one accepted visual wake decision without blocking refresh."""

        return bool(
            self._run_optional_display_stage(
                stage="handle_gesture_wakeup_decision",
                call=lambda: service_gesture_helpers.handle_gesture_wakeup_decision(self, decision),
                fault_event="proactive_gesture_wakeup_dispatch_failed",
                fault_message="Failed to dispatch the accepted visual wake decision.",
                default=False,
            )
        )

    def _run_gesture_wakeup_handler(
        self,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Run one visual wakeup handler on the dedicated dispatcher thread."""

        return bool(
            self._run_optional_display_stage(
                stage="run_gesture_wakeup_handler",
                call=lambda: service_gesture_helpers.run_gesture_wakeup_handler(self, decision),
                fault_event="proactive_gesture_wakeup_handler_failed",
                fault_message="Failed while running the visual wakeup handler on the dispatcher thread.",
                default=False,
            )
        )

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

        fast_snapshot = self._observe_dedicated_gesture_refresh_snapshot()
        if fast_snapshot is not None:
            return fast_snapshot
        shared_snapshot = self._shared_display_perception_snapshot(consumer="gesture")
        if shared_snapshot is not None:
            self._last_gesture_vision_refresh_mode = "perception_stream_shared"
            return shared_snapshot
        if self.vision_observer is None:
            self._last_gesture_vision_refresh_mode = "missing"
            return None
        self._last_gesture_vision_refresh_mode = "full_fallback"
        return self._observe_vision_safe()

    def _observe_dedicated_gesture_refresh_snapshot(self):
        """Prefer the dedicated low-latency gesture lane when the observer supports it."""

        vision_observer = self.vision_observer
        if vision_observer is None:
            return None
        observe_gesture = getattr(vision_observer, "observe_gesture", None)
        if not callable(observe_gesture):
            return None
        supports_gesture_refresh = getattr(vision_observer, "supports_gesture_refresh", None)
        if supports_gesture_refresh is False:
            return None
        snapshot = self._observe_vision_with_method(observe_gesture)
        if snapshot is not None:
            self._last_gesture_vision_refresh_mode = "gesture_fast"
        return snapshot

    def _display_gesture_refresh_supported_safe(self) -> bool:
        """Probe HDMI gesture-refresh capability without letting capability lookup break dispatch."""

        result = self._run_optional_display_stage(
            stage="display_gesture_refresh_supported",
            call=lambda: bool(
                display_gesture_refresh_supported(
                    config=self.config,
                    vision_observer=self.vision_observer,
                )
            ),
            fault_event="proactive_display_gesture_refresh_capability_failed",
            fault_message="Failed to query HDMI gesture-refresh capability; falling back to the legacy ack path.",
            fault_data={"vision_observer_present": self.vision_observer is not None},
            default=False,
        )
        return bool(result)

    def _remember_observation_dispatch_diagnostics(
        self,
        *,
        observed_at: float,
        outcome: str,
        stage_ms: dict[str, float],
        fault_stage: str | None,
    ) -> None:
        """Expose one bounded structured summary of the latest automation-dispatch pipeline."""

        self._last_observation_dispatch_observed_at = observed_at
        self._last_observation_dispatch_outcome = self._sanitize_freeform_text(outcome, max_len=32)
        self._last_observation_dispatch_stage_ms = {
            self._sanitize_freeform_text(name, max_len=64): float(duration_ms)
            for name, duration_ms in stage_ms.items()
        }
        self._last_observation_dispatch_fault_stage = (
            None if fault_stage is None else self._sanitize_freeform_text(fault_stage, max_len=64)
        )

    def _run_optional_display_stage(
        self,
        *,
        stage: str,
        call: Callable[[], _T],
        fault_event: str,
        fault_message: str,
        fault_data: dict[str, object] | None = None,
        default: _T | None = None,
    ) -> _T | None:
        """Run one optional display/helper stage with adaptive backoff and fail-open semantics."""

        stage_key = self._sanitize_freeform_text(stage, max_len=64)
        now_monotonic = time.monotonic()
        stage_state = self._optional_display_stage_state().get(stage_key)
        if stage_state is not None and now_monotonic < float(stage_state.get("skip_until", 0.0)):
            return default

        stage_started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:
            elapsed_ms = self._elapsed_ms(stage_started)
            prior_failures = 0 if stage_state is None else int(stage_state.get("failures", 0))
            failures = prior_failures + 1
            backoff_s = min(
                self._OPTIONAL_STAGE_BACKOFF_MAX_S,
                self._OPTIONAL_STAGE_BACKOFF_BASE_S * (2 ** min(failures - 1, 6)),
            )
            self._optional_display_stage_state()[stage_key] = {
                "failures": failures,
                "skip_until": now_monotonic + backoff_s,
            }
            data = dict(fault_data or ())
            data.update(
                {
                    "stage": stage_key,
                    "stage_elapsed_ms": elapsed_ms,
                    "stage_failures": failures,
                    "stage_backoff_s": round(backoff_s, 3),
                }
            )
            self._safe_record_fault(
                event=fault_event,
                message=fault_message,
                error=exc,
                data=data,
            )
            return default

        self._optional_display_stage_state().pop(stage_key, None)
        return result

    def _optional_display_stage_state(self) -> dict[str, dict[str, float | int]]:
        """Return the lazily created circuit-breaker state for optional display/helper stages."""

        state = getattr(self, "_optional_display_stage_fault_state", None)
        if isinstance(state, dict):
            return state
        state = {}
        self._optional_display_stage_fault_state = state
        return state

    def _safe_record_fault(
        self,
        *,
        event: str,
        message: str,
        error: Exception | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        """Record one fault without allowing the fault recorder itself to break runtime paths."""

        normalized_event = self._sanitize_freeform_text(event, max_len=96)
        normalized_message = self._sanitize_freeform_text(message, max_len=self._MAX_FAULT_TEXT_LEN)
        normalized_data = self._normalize_fault_payload(data)

        recorder = getattr(self, "_record_fault", None)
        if callable(recorder):
            try:
                self._call_fault_recorder(
                    cast(Callable[..., None], recorder),
                    event=normalized_event,
                    message=normalized_message,
                    error=error,
                    data=normalized_data,
                )
                return
            except Exception as recorder_exc:
                fallback_data = {
                    "original_event": normalized_event,
                    "original_data": normalized_data,
                    "fault_recorder_error": self._normalize_fault_payload({"error": repr(recorder_exc)}),
                }
                _LOGGER.error(
                    "Fault recorder failed while handling %s: %s | data=%r",
                    normalized_event,
                    normalized_message,
                    fallback_data,
                    exc_info=(type(recorder_exc), recorder_exc, recorder_exc.__traceback__),
                )
                return

        _LOGGER.error(
            "%s: %s | data=%r",
            normalized_event,
            normalized_message,
            normalized_data,
            exc_info=None if error is None else (type(error), error, error.__traceback__),
        )

    @staticmethod
    def _call_fault_recorder(
        recorder: Callable[..., None],
        *,
        event: str,
        message: str,
        error: Exception | None,
        data: object,
    ) -> None:
        """Invoke the injected fault recorder through one statically typed shim."""

        recorder(
            event=event,
            message=message,
            error=error,
            data=data,
        )

    def _normalize_fault_payload(self, value: object, *, _depth: int = 0) -> object:
        """Convert fault payloads into bounded, log-safe primitives."""

        if _depth >= self._MAX_FAULT_DATA_DEPTH:
            return "..."

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, bytes):
            return self._sanitize_freeform_text(value.decode("utf-8", errors="replace"))

        if isinstance(value, str):
            return self._sanitize_freeform_text(value)

        if isinstance(value, dict):
            items = list(value.items())[: self._MAX_FAULT_DATA_ITEMS]
            return {
                self._sanitize_freeform_text(str(key), max_len=64): self._normalize_fault_payload(
                    item_value,
                    _depth=_depth + 1,
                )
                for key, item_value in items
            }

        if isinstance(value, (list, tuple, set, frozenset)):
            normalized_items = [
                self._normalize_fault_payload(item, _depth=_depth + 1)
                for item in list(value)[: self._MAX_FAULT_DATA_ITEMS]
            ]
            if isinstance(value, tuple):
                return tuple(normalized_items)
            return normalized_items

        return self._sanitize_freeform_text(repr(value))

    def _sanitize_detected_trigger_ids(self, detected_trigger_ids) -> tuple[str, ...]:
        """Clamp and sanitize externally influenced trigger identifiers before display/log emission."""

        if detected_trigger_ids is None:
            return ()

        sanitized_ids: list[str] = []
        for raw_value in tuple(detected_trigger_ids)[: self._MAX_DETECTED_TRIGGER_IDS]:
            sanitized_value = self._sanitize_freeform_text(
                raw_value,
                max_len=self._MAX_DEBUG_TRIGGER_ID_LEN,
            )
            if sanitized_value:
                sanitized_ids.append(sanitized_value)
        return tuple(sanitized_ids)

    def _sanitize_freeform_text(self, value: object, *, max_len: int | None = None) -> str:
        """Remove control characters and bound free-form text for logs, debug pills, and state."""

        if max_len is None:
            max_len = self._MAX_FAULT_TEXT_LEN

        text = str(value)
        cleaned = "".join(" " if not character.isprintable() else character for character in text)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) <= max_len:
            return cleaned
        if max_len <= 3:
            return cleaned[:max_len]
        return f"{cleaned[: max_len - 3]}..."

    def _scheduler_monotonic_now(self, observed_at: float) -> float:
        """Use a monotonic scheduler clock even if observed_at is a wall-clock timestamp."""

        if isinstance(observed_at, (int, float)) and math.isfinite(observed_at):
            observed_at = float(observed_at)
            if 0.0 <= observed_at < self._MONOTONIC_EPOCH_THRESHOLD_S:
                return observed_at
        return time.monotonic()

    @staticmethod
    def _elapsed_ms(stage_started: float) -> float:
        """Return one bounded stage duration in milliseconds."""

        return round((time.perf_counter() - stage_started) * 1000.0, 3)
