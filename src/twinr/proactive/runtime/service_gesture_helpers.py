"""Focused gesture/runtime helper functions for the proactive coordinator.

These helpers keep HDMI gesture acknowledgement, visual wake dispatch, and
gesture-specific forensic tracing out of the main proactive monitor
orchestrator.
"""

from __future__ import annotations

from typing import Any

from twinr.agent.workflows.forensics import workflow_decision

from ..social.camera_surface import ProactiveCameraSurfaceUpdate
from ..social.engine import SocialObservation, SocialVisionObservation
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative
from .display_gesture_emoji import DisplayGestureEmojiDecision, DisplayGestureEmojiPublishResult
from .gesture_wakeup_lane import GestureWakeupDecision


def _round_optional_ratio(value: float | None) -> float | None:
    from . import service as service_module

    return service_module._round_optional_ratio(value)


def dispatch_gesture_wakeup_with_fresh_context(
    coordinator: Any,
    *,
    observed_at: float,
    vision_snapshot: Any,
    decision: GestureWakeupDecision,
) -> bool:
    """Prime current gesture facts before dispatching an accepted wakeup."""

    if decision.active:
        coordinator._prime_gesture_wakeup_sensor_context(
            observed_at=observed_at,
            vision_snapshot=vision_snapshot,
        )
    return coordinator._handle_gesture_wakeup_decision(decision)


def prime_gesture_wakeup_sensor_context(
    coordinator: Any,
    *,
    observed_at: float,
    vision_snapshot: Any,
) -> None:
    """Export one fresh sensor/person-state payload from the active gesture tick."""

    if coordinator.observation_handler is None:
        return
    audio_snapshot = coordinator._observe_audio_for_attention_refresh(now=observed_at)
    audio_policy_snapshot = coordinator._observe_audio_policy(
        now=observed_at,
        audio_observation=audio_snapshot.observation,
    )
    observation = SocialObservation(
        observed_at=observed_at,
        inspected=True,
        pir_motion_detected=False,
        low_motion=False,
        vision=vision_snapshot.observation,
        audio=audio_snapshot.observation,
    )
    camera_update = coordinator._observe_camera_surface(observation, inspected=True)
    presence_snapshot = coordinator._observe_presence(
        now=observed_at,
        person_visible=camera_update.snapshot.person_visible,
        motion_active=False,
        audio_observation=audio_snapshot.observation,
        audio_policy_snapshot=audio_policy_snapshot,
    )
    facts = coordinator._build_automation_facts(
        observation,
        inspected=True,
        audio_snapshot=audio_snapshot,
        camera_snapshot=camera_update.snapshot,
        audio_policy_snapshot=audio_policy_snapshot,
        presence_snapshot=presence_snapshot,
    )
    event_names = coordinator._derive_sensor_events(
        facts,
        camera_event_names=camera_update.event_names,
    )
    coordinator.observation_handler(facts, event_names)


def update_display_gesture_emoji_ack(
    coordinator: Any,
    camera_update: ProactiveCameraSurfaceUpdate,
) -> None:
    """Mirror clear stabilized user gestures into the HDMI emoji reserve area."""

    publisher = coordinator.display_gesture_emoji_publisher
    if publisher is None:
        return
    try:
        publisher.publish_update(camera_update)
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_display_gesture_emoji_failed",
            message="Failed to update the HDMI gesture-emoji acknowledgement cue.",
            error=exc,
            data={"event_names": list(camera_update.event_names)},
        )


def publish_display_gesture_decision(
    coordinator: Any,
    decision: DisplayGestureEmojiDecision,
) -> DisplayGestureEmojiPublishResult | None:
    """Persist one direct gesture-ack decision through the emoji publisher."""

    publisher = coordinator.display_gesture_emoji_publisher
    if publisher is None:
        return None
    try:
        return publisher.publish(decision)
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_display_gesture_emoji_failed",
            message="Failed to update the HDMI gesture-emoji acknowledgement cue.",
            error=exc,
            data={"reason": decision.reason},
        )
        return None


def handle_gesture_wakeup_decision(
    coordinator: Any,
    decision: GestureWakeupDecision,
) -> bool:
    """Dispatch one accepted visual wake decision without blocking refresh."""

    if not decision.active or coordinator.gesture_wakeup_handler is None:
        return False
    dispatched = coordinator._gesture_wakeup_dispatcher.submit(decision)
    if not dispatched:
        coordinator._append_ops_event(
            event="gesture_wakeup_dispatch_skipped",
            message="Visual wakeup was skipped because another visual wakeup is already active.",
            data={
                "gesture": decision.trigger_gesture.value,
                "reason": decision.reason,
            },
        )
    return dispatched


def run_gesture_wakeup_handler(
    coordinator: Any,
    decision: GestureWakeupDecision,
) -> bool:
    """Run one visual wakeup handler on the dedicated dispatcher thread."""

    def _run_handler() -> bool:
        try:
            return bool(coordinator.gesture_wakeup_handler(decision))
        except Exception as exc:
            coordinator._record_fault(
                event="gesture_wakeup_handler_failed",
                message="Gesture wakeup handler failed.",
                error=exc,
                data={"reason": decision.reason, "gesture": decision.trigger_gesture.value},
            )
            return False

    handled = _run_handler()
    if handled:
        coordinator._append_ops_event(
            event="gesture_wakeup_triggered",
            message="A configured visual wake gesture opened a hands-free conversation path.",
            data={
                "gesture": decision.trigger_gesture.value,
                "confidence": _round_optional_ratio(decision.confidence),
                "request_source": decision.request_source,
            },
        )
    return handled


def record_gesture_debug_tick(
    coordinator: Any,
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

    debug_stream = coordinator.display_gesture_debug_stream
    if debug_stream is None:
        return
    payload: dict[str, Any] = {
        "runtime_status": str(runtime_status_value or "").strip().lower() or None,
        "vision_mode": coordinator._last_gesture_vision_refresh_mode,
        "stage_ms": dict(stage_ms),
    }
    if observation is not None:
        gesture_stream_observation = gesture_stream(observation)
        payload.update(
            {
                "camera_online": observation.camera_online,
                "camera_ready": observation.camera_ready,
                "camera_ai_ready": observation.camera_ai_ready,
                "camera_error": observation.camera_error,
                "camera_person_count": observation.person_count,
                "camera_fine_hand_gesture": observation.fine_hand_gesture.value,
                "camera_fine_hand_gesture_confidence": _round_optional_ratio(
                    observation.fine_hand_gesture_confidence
                ),
                "camera_gesture_event": observation.gesture_event.value,
                "camera_gesture_confidence": _round_optional_ratio(observation.gesture_confidence),
                "camera_hand_or_object_near": observation.hand_or_object_near_camera,
                "camera_showing_intent_likely": observation.showing_intent_likely,
                "gesture_stream_authoritative": gesture_stream_authoritative(observation),
                "gesture_stream_activation_key": (
                    None if gesture_stream_observation is None else gesture_stream_observation.activation_key
                ),
                "gesture_stream_activation_token": (
                    None if gesture_stream_observation is None else gesture_stream_observation.activation_token
                ),
                "gesture_stream_activation_rising": (
                    None if gesture_stream_observation is None else gesture_stream_observation.activation_rising
                ),
            }
        )
    gesture_debug_details_getter = getattr(coordinator.vision_observer, "gesture_debug_details", None)
    if callable(gesture_debug_details_getter):
        try:
            gesture_debug_details = gesture_debug_details_getter()
        except Exception:
            gesture_debug_details = None
        if gesture_debug_details:
            payload["pipeline_debug"] = gesture_debug_details
    if decision is not None:
        payload.update(
            {
                "decision_active": decision.active,
                "decision_reason": decision.reason,
                "decision_symbol": decision.symbol.value,
                "decision_accent": decision.accent,
                "decision_hold_seconds": round(decision.hold_seconds, 3),
            }
        )
    if publish_result is not None:
        payload.update(
            {
                "publish_action": publish_result.action,
                "publish_owner": publish_result.owner,
            }
        )
    if wakeup_decision is not None:
        payload.update(
            {
                "gesture_wakeup_active": wakeup_decision.active,
                "gesture_wakeup_reason": wakeup_decision.reason,
                "gesture_wakeup_trigger_gesture": wakeup_decision.trigger_gesture.value,
                "gesture_wakeup_observed_gesture": wakeup_decision.observed_gesture.value,
                "gesture_wakeup_confidence": _round_optional_ratio(wakeup_decision.confidence),
                "gesture_wakeup_request_source": wakeup_decision.request_source,
                "gesture_wakeup_handled": wakeup_handled,
            }
        )
    try:
        debug_stream.append_tick(
            outcome=outcome,
            observed_at=observed_at,
            data=payload,
        )
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_gesture_debug_stream_failed",
            message="Failed to append the bounded gesture debug tick.",
            error=exc,
            data={"outcome": outcome},
        )


def gesture_observation_trace_details(
    observation: SocialVisionObservation,
) -> dict[str, object]:
    """Return one bounded trace summary for the current gesture observation."""

    return {
        "person_visible": observation.person_visible,
        "person_count": observation.person_count,
        "camera_online": observation.camera_online,
        "camera_ready": observation.camera_ready,
        "camera_ai_ready": observation.camera_ai_ready,
        "camera_error": observation.camera_error,
        "fine_hand_gesture": observation.fine_hand_gesture.value,
        "fine_hand_gesture_confidence": _round_optional_ratio(observation.fine_hand_gesture_confidence),
        "gesture_event": observation.gesture_event.value,
        "gesture_confidence": _round_optional_ratio(observation.gesture_confidence),
        "hand_or_object_near_camera": observation.hand_or_object_near_camera,
        "showing_intent_likely": observation.showing_intent_likely,
        "gesture_stream_authoritative": gesture_stream_authoritative(observation),
        "gesture_stream_activation_key": None if gesture_stream(observation) is None else gesture_stream(observation).activation_key,
        "gesture_stream_activation_token": None if gesture_stream(observation) is None else gesture_stream(observation).activation_token,
    }


def trace_gesture_ack_lane_decision(
    observation: SocialVisionObservation,
    decision: DisplayGestureEmojiDecision,
) -> None:
    """Emit one decision ledger entry for the HDMI ack lane result."""

    workflow_decision(
        msg="gesture_ack_lane_decision",
        question="Should the current gesture frame publish an HDMI acknowledgement?",
        selected={
            "id": decision.reason,
            "summary": (
                f"Publish {decision.symbol.value}." if decision.active else "Do not publish an HDMI acknowledgement."
            ),
        },
        options=[
            {"id": "publish_fine_hand_gesture", "summary": "Emit one fine-hand acknowledgement immediately."},
            {"id": "publish_coarse_gesture", "summary": "Emit one coarse motion acknowledgement immediately."},
            {"id": "gesture_stream_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
            {"id": "live_gesture_already_active", "summary": "Do not re-emit while the same authoritative activation token remains active."},
            {"id": "live_gesture_cooldown", "summary": "Suppress a repeated acknowledgement during cooldown."},
            {"id": "no_supported_live_gesture", "summary": "Ignore the frame because the authoritative gesture is unsupported by the HDMI path."},
        ],
        context={
            "gesture_stream_authoritative": gesture_stream_authoritative(observation),
            "observed_fine_hand_gesture": observation.fine_hand_gesture.value,
            "observed_fine_hand_confidence": _round_optional_ratio(observation.fine_hand_gesture_confidence),
            "observed_gesture_event": observation.gesture_event.value,
            "observed_gesture_confidence": _round_optional_ratio(observation.gesture_confidence),
            "stream_activation_key": None if gesture_stream(observation) is None else gesture_stream(observation).activation_key,
            "stream_activation_token": None if gesture_stream(observation) is None else gesture_stream(observation).activation_token,
        },
        confidence=_round_optional_ratio(observation.fine_hand_gesture_confidence or observation.gesture_confidence),
        guardrails=["gesture_ack_lane", "display_only"],
        kpi_impact_estimate={"latency": "low", "user_feedback": "high"},
    )


def trace_gesture_wakeup_lane_decision(
    decision: GestureWakeupDecision,
) -> None:
    """Emit one decision ledger entry for the visual wake lane."""

    workflow_decision(
        msg="gesture_wakeup_lane_decision",
        question="Should the current gesture frame request a visual wake-up?",
        selected={
            "id": decision.reason,
            "summary": (
                "Dispatch a wake request."
                if decision.active
                else "Do not dispatch a wake request from this frame."
            ),
        },
        options=[
            {"id": f"gesture_wakeup:{decision.trigger_gesture.value}", "summary": "Trigger a wake request immediately."},
            {"id": "gesture_wakeup_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
            {"id": "gesture_wakeup_already_active", "summary": "Do not re-dispatch while the same authoritative activation token remains active."},
            {"id": "gesture_wakeup_cooldown", "summary": "Suppress a repeated wake request during cooldown."},
            {"id": "no_gesture_wakeup_candidate", "summary": "Ignore the frame because the trigger gesture was not present."},
        ],
        context={
            "trigger_gesture": decision.trigger_gesture.value,
            "observed_gesture": decision.observed_gesture.value,
            "confidence": _round_optional_ratio(decision.confidence),
            "request_source": decision.request_source,
        },
        confidence=_round_optional_ratio(decision.confidence),
        guardrails=["gesture_wakeup_lane", "pi_only_voice_entry"],
        kpi_impact_estimate={"latency": "medium", "wake_request": "high"},
    )


def trace_gesture_publish_decision(
    *,
    decision: DisplayGestureEmojiDecision,
    publish_result: DisplayGestureEmojiPublishResult | None,
    wakeup_decision: GestureWakeupDecision,
    wakeup_handled: bool,
) -> None:
    """Emit one decision ledger entry for the final publish/dispatch outcome."""

    publish_action = None if publish_result is None else publish_result.action
    workflow_decision(
        msg="gesture_refresh_publish_outcome",
        question="What user-visible or wake-related action should this gesture refresh commit?",
        selected={
            "id": publish_action or "publish_failed",
            "summary": (
                f"Finish the refresh with publish action {publish_action}."
                if publish_action is not None
                else "No HDMI publish result was produced."
            ),
        },
        options=[
            {"id": "created", "summary": "Create a fresh HDMI emoji cue."},
            {"id": "updated", "summary": "Refresh an existing HDMI emoji cue."},
            {"id": "noop", "summary": "Leave the HDMI gesture cue unchanged."},
            {"id": "publish_failed", "summary": "The publish stage did not return a result."},
        ],
        context={
            "decision_reason": decision.reason,
            "decision_active": decision.active,
            "wakeup_reason": wakeup_decision.reason,
            "wakeup_active": wakeup_decision.active,
            "wakeup_handled": wakeup_handled,
            "publish_owner": None if publish_result is None else publish_result.owner,
        },
        confidence="forensic",
        guardrails=["gesture_refresh_commit"],
        kpi_impact_estimate={"latency": "medium", "display_side_effect": "high"},
    )


__all__ = [
    "dispatch_gesture_wakeup_with_fresh_context",
    "gesture_observation_trace_details",
    "handle_gesture_wakeup_decision",
    "prime_gesture_wakeup_sensor_context",
    "publish_display_gesture_decision",
    "record_gesture_debug_tick",
    "run_gesture_wakeup_handler",
    "trace_gesture_ack_lane_decision",
    "trace_gesture_publish_decision",
    "trace_gesture_wakeup_lane_decision",
    "update_display_gesture_emoji_ack",
]
