"""Focused attention/runtime helper functions for the proactive coordinator.

This module keeps HDMI attention-follow, servo tracing, and live-context
projection out of the main monitor orchestrator so `service.py` can stay
focused on lifecycle and tick sequencing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinr.hardware.servo_follow import AttentionServoDecision

from ..social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from ..social.engine import SocialAudioObservation, SocialObservation, SocialVisionObservation
from .attention_targeting import MultimodalAttentionTargetSnapshot
from .audio_policy import ReSpeakerAudioPolicySnapshot
from .display_attention import DisplayAttentionCuePublishResult, display_attention_refresh_supported
from .speaker_association import derive_respeaker_speaker_association


def _round_optional_ratio(value: float | None) -> float | None:
    from . import service as service_module

    return service_module._round_optional_ratio(value)


def _round_optional_seconds(value: float | None) -> float | None:
    from . import service as service_module

    return service_module._round_optional_seconds(value)


def _emit_key_value_line(prefix: str, /, **fields: object) -> str:
    from . import service as service_module

    return service_module._emit_key_value_line(prefix, **fields)


def _coerce_mapping(value: object) -> dict[str, object]:
    """Normalize one optional mapping-like payload into a plain dict."""

    if not isinstance(value, Mapping):
        return {}
    return {str(key): mapped_value for key, mapped_value in value.items()}


def _coerce_float(value: object) -> float | None:
    """Parse one optional float-like value conservatively."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def publish_display_attention_live_context(
    coordinator: Any,
    *,
    observed_at: float,
    vision_observation: SocialVisionObservation,
    camera_snapshot: ProactiveCameraSnapshot,
    audio_snapshot: Any,
    audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
) -> None:
    """Export authoritative HDMI-attention facts to the live voice context."""

    if coordinator.live_context_handler is None:
        return
    observation = SocialObservation(
        observed_at=observed_at,
        inspected=True,
        pir_motion_detected=False,
        low_motion=False,
        vision=vision_observation,
        audio=audio_snapshot.observation,
    )
    presence_snapshot = coordinator._observe_presence(
        now=observed_at,
        person_visible=camera_snapshot.person_visible,
        motion_active=False,
        audio_observation=audio_snapshot.observation,
        audio_policy_snapshot=audio_policy_snapshot,
    )
    facts = coordinator._build_automation_facts(
        observation,
        inspected=True,
        audio_snapshot=audio_snapshot,
        camera_snapshot=camera_snapshot,
        audio_policy_snapshot=audio_policy_snapshot,
        presence_snapshot=presence_snapshot,
    )
    try:
        coordinator.live_context_handler(facts)
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_live_context_handler_failed",
            message="HDMI attention refresh failed to publish live sensor context.",
            error=exc,
        )


def update_display_debug_signals(
    coordinator: Any,
    camera_update: ProactiveCameraSurfaceUpdate,
    *,
    detected_trigger_ids: tuple[str, ...] = (),
) -> None:
    """Persist HDMI debug pills from current camera state and recent triggers."""

    publisher = coordinator.display_debug_signal_publisher
    if publisher is None:
        return
    try:
        publisher.publish_from_camera_facts(
            camera_facts=camera_update.snapshot.to_automation_facts(),
            event_names=camera_update.event_names,
            trigger_ids=detected_trigger_ids,
        )
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_display_debug_signals_publish_failed",
            message="Failed to publish HDMI debug-signal state.",
            error=exc,
            data={
                "camera_event_names": list(camera_update.event_names),
                "trigger_ids": list(detected_trigger_ids),
            },
            level="warning",
        )


def update_display_attention_follow(
    coordinator: Any,
    *,
    source: str,
    observed_at: float,
    camera_snapshot: ProactiveCameraSnapshot,
    audio_observation: Any,
    audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
) -> DisplayAttentionCuePublishResult | None:
    """Update the HDMI face and body-follow servo from the current attention target."""

    presence_snapshot = coordinator.latest_presence_snapshot
    presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
    live_facts = {
        "camera": camera_snapshot.to_automation_facts(),
        "vad": {
            "speech_detected": getattr(audio_observation, "speech_detected", None),
        },
        "respeaker": {
            "azimuth_deg": getattr(audio_observation, "azimuth_deg", None),
            "direction_confidence": getattr(audio_observation, "direction_confidence", None),
        },
        "audio_policy": {
            "speaker_direction_stable": (
                None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
            ),
        },
    }
    speaker_association = derive_respeaker_speaker_association(
        observed_at=observed_at,
        live_facts=live_facts,
    )
    coordinator.latest_speaker_association_snapshot = speaker_association
    attention_target = coordinator.attention_target_tracker.observe(
        observed_at=observed_at,
        live_facts=live_facts,
        runtime_status=getattr(getattr(coordinator.runtime, "status", None), "value", None),
        presence_session_id=presence_session_id,
        speaker_association=speaker_association,
        identity_fusion=coordinator.latest_identity_fusion_snapshot,
    )
    attention_target_debug = coordinator.attention_target_tracker.debug_snapshot(observed_at=observed_at)
    coordinator.latest_attention_target_snapshot = attention_target
    live_facts["speaker_association"] = speaker_association.to_automation_facts()
    live_facts["attention_target"] = attention_target.to_automation_facts()
    record_attention_follow_pipeline_if_changed(
        coordinator,
        source=source,
        observed_at=observed_at,
        camera_snapshot=camera_snapshot,
        attention_target=attention_target,
    )
    update_attention_servo_follow(
        coordinator,
        source=source,
        observed_at=observed_at,
        camera_snapshot=camera_snapshot,
        attention_target=attention_target,
        attention_target_debug=attention_target_debug,
    )
    publisher = coordinator.display_attention_publisher
    if publisher is None:
        return None
    try:
        return publisher.publish_from_facts(
            config=coordinator.config,
            live_facts=live_facts,
        )
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_display_attention_follow_failed",
            message="Failed to update the HDMI face attention-follow cue.",
            error=exc,
            data={"observed_at": observed_at},
        )
        return None


def update_attention_servo_follow(
    coordinator: Any,
    *,
    source: str,
    observed_at: float,
    camera_snapshot: ProactiveCameraSnapshot,
    attention_target: MultimodalAttentionTargetSnapshot | None,
    attention_target_debug: dict[str, object] | None = None,
) -> None:
    """Update the optional body-orientation servo from the current attention target."""

    controller = coordinator.attention_servo_controller
    if controller is None:
        return
    inactive_decision = AttentionServoDecision(
        observed_at=observed_at,
        active=False,
        reason="ignored_non_authoritative_source",
        confidence=0.0 if attention_target is None else attention_target.confidence,
        target_center_x=None if attention_target is None else attention_target.target_center_x,
    )
    if not attention_servo_source_is_authoritative(coordinator, source=source):
        record_attention_servo_follow_if_changed(
            coordinator,
            source=source,
            observed_at=observed_at,
            attention_target=attention_target,
            decision=inactive_decision,
        )
        record_attention_servo_forensic_tick(
            coordinator,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
            attention_target_debug=attention_target_debug,
            decision=inactive_decision,
        )
        return
    try:
        decision = controller.update(
            observed_at=observed_at,
            active=False if attention_target is None else attention_target.active,
            target_center_x=None if attention_target is None else attention_target.target_center_x,
            confidence=0.0 if attention_target is None else attention_target.confidence,
            visible_target_present=camera_snapshot.person_visible,
            visible_target_box_left=(
                None if camera_snapshot.primary_person_box is None else camera_snapshot.primary_person_box.left
            ),
            visible_target_box_right=(
                None if camera_snapshot.primary_person_box is None else camera_snapshot.primary_person_box.right
            ),
        )
        record_attention_servo_follow_if_changed(
            coordinator,
            source=source,
            observed_at=observed_at,
            attention_target=attention_target,
            decision=decision,
        )
        record_attention_servo_forensic_tick(
            coordinator,
            source=source,
            observed_at=observed_at,
            camera_snapshot=camera_snapshot,
            attention_target=attention_target,
            attention_target_debug=attention_target_debug,
            decision=decision,
        )
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_attention_servo_follow_failed",
            message="Failed to update the attention-follow servo output.",
            error=exc,
            data={"observed_at": observed_at},
        )


def record_attention_servo_forensic_tick(
    coordinator: Any,
    *,
    source: str,
    observed_at: float,
    camera_snapshot: ProactiveCameraSnapshot,
    attention_target: MultimodalAttentionTargetSnapshot | None,
    attention_target_debug: dict[str, object] | None,
    decision: AttentionServoDecision,
) -> None:
    """Record one per-tick forensic servo ledger when scoped instrumentation is enabled."""

    if not coordinator._attention_servo_forensic_trace_enabled:
        return
    controller = coordinator.attention_servo_controller
    controller_debug = None
    if controller is not None and hasattr(controller, "debug_snapshot"):
        controller_debug = controller.debug_snapshot(observed_at=observed_at)
    coordinator._attention_servo_forensic_tick_index += 1
    coordinator._append_ops_event(
        event="proactive_attention_servo_forensic_tick",
        message="Recorded one forensic attention-servo tick for end-to-end exit-follow debugging.",
        data={
            "run_id": coordinator._attention_servo_forensic_run_id,
            "tick_index": coordinator._attention_servo_forensic_tick_index,
            "observed_at": observed_at,
            "follow_source": source,
            "camera": {
                "person_visible": camera_snapshot.person_visible,
                "person_visible_for_s": _round_optional_seconds(camera_snapshot.person_visible_for_s),
                "person_count": camera_snapshot.person_count,
                "visible_person_count": len(camera_snapshot.visible_persons),
                "primary_person_zone": camera_snapshot.primary_person_zone.value,
                "primary_person_box": (
                    None
                    if camera_snapshot.primary_person_box is None
                    else {
                        "left": _round_optional_ratio(camera_snapshot.primary_person_box.left),
                        "right": _round_optional_ratio(camera_snapshot.primary_person_box.right),
                        "top": _round_optional_ratio(camera_snapshot.primary_person_box.top),
                        "bottom": _round_optional_ratio(camera_snapshot.primary_person_box.bottom),
                    }
                ),
                "primary_person_center_x": _round_optional_ratio(camera_snapshot.primary_person_center_x),
                "primary_person_center_y": _round_optional_ratio(camera_snapshot.primary_person_center_y),
                "visible_persons": summarize_visible_persons(camera_snapshot),
            },
            "attention_target": {
                "state": None if attention_target is None else attention_target.state,
                "active": None if attention_target is None else attention_target.active,
                "track_id": None if attention_target is None else attention_target.target_track_id,
                "center_x": _round_optional_ratio(
                    None if attention_target is None else attention_target.target_center_x
                ),
                "center_y": _round_optional_ratio(
                    None if attention_target is None else attention_target.target_center_y
                ),
                "velocity_x": _round_optional_ratio(
                    None if attention_target is None else attention_target.target_velocity_x
                ),
                "focus_source": None if attention_target is None else attention_target.focus_source,
                "confidence": _round_optional_ratio(None if attention_target is None else attention_target.confidence),
            },
            "attention_tracker_debug": attention_target_debug,
            "servo_controller_debug": controller_debug,
            "decision": {
                "reason": decision.reason,
                "active": decision.active,
                "confidence": _round_optional_ratio(decision.confidence),
                "target_center_x": _round_optional_ratio(decision.target_center_x),
                "applied_center_x": _round_optional_ratio(decision.applied_center_x),
                "target_pulse_width_us": decision.target_pulse_width_us,
                "commanded_pulse_width_us": decision.commanded_pulse_width_us,
            },
            "decision_ledger": build_attention_servo_decision_ledger(
                coordinator,
                source=source,
                camera_snapshot=camera_snapshot,
                attention_target=attention_target,
                controller_debug=controller_debug,
                decision=decision,
            ),
        },
    )


def summarize_visible_persons(camera_snapshot: ProactiveCameraSnapshot) -> list[dict[str, object]]:
    """Return one bounded summary of current visible-person anchors."""

    summaries: list[dict[str, object]] = []
    for person in camera_snapshot.visible_persons[:4]:
        box = getattr(person, "box", None)
        summaries.append(
            {
                "zone": getattr(getattr(person, "zone", None), "value", None),
                "confidence": _round_optional_ratio(getattr(person, "confidence", None)),
                "center_x": _round_optional_ratio(None if box is None else getattr(box, "center_x", None)),
                "center_y": _round_optional_ratio(None if box is None else getattr(box, "center_y", None)),
                "left": _round_optional_ratio(None if box is None else getattr(box, "left", None)),
                "right": _round_optional_ratio(None if box is None else getattr(box, "right", None)),
                "area": _round_optional_ratio(None if box is None else getattr(box, "area", None)),
            }
        )
    return summaries


def build_attention_servo_decision_ledger(
    coordinator: Any,
    *,
    source: str,
    camera_snapshot: ProactiveCameraSnapshot,
    attention_target: MultimodalAttentionTargetSnapshot | None,
    controller_debug: dict[str, object] | None,
    decision: AttentionServoDecision,
) -> dict[str, object]:
    """Build one compact decision-ledger payload for forensic servo debugging."""

    controller_config = _coerce_mapping(None if controller_debug is None else controller_debug.get("config"))
    visible_edge_threshold = _coerce_float(controller_config.get("exit_visible_edge_threshold")) or 0.0
    right_threshold = _coerce_float(controller_config.get("exit_visible_box_edge_threshold")) or 1.0
    activation_delay_s = _coerce_float(controller_config.get("exit_activation_delay_s")) or 0.0
    departure_age_s = _coerce_float(
        None if controller_debug is None else controller_debug.get("visible_edge_departure_age_s")
    )
    anchor_center_x = _coerce_float(None if controller_debug is None else controller_debug.get("recent_exit_anchor_center_x"))
    anchor_box_edge = _coerce_float(None if controller_debug is None else controller_debug.get("recent_exit_box_edge"))
    waiting_constraints: list[str] = []
    if camera_snapshot.person_visible is not True:
        waiting_constraints.append("camera_not_visible")
    if anchor_center_x is None:
        waiting_constraints.append("no_recent_departure_anchor")
    elif abs(anchor_center_x - 0.5) < max(0.0, visible_edge_threshold - 0.5):
        waiting_constraints.append("departure_anchor_below_threshold")
    if anchor_box_edge is None:
        waiting_constraints.append("departure_box_not_near_edge")
    else:
        left_threshold = 1.0 - right_threshold
        if anchor_center_x is not None and anchor_center_x >= 0.5 and anchor_box_edge < right_threshold:
            waiting_constraints.append("departure_box_not_near_edge")
        if anchor_center_x is not None and anchor_center_x < 0.5 and anchor_box_edge > left_threshold:
            waiting_constraints.append("departure_box_not_near_edge")
    if departure_age_s is None:
        waiting_constraints.append("departure_timer_not_started")
    elif departure_age_s < activation_delay_s:
        waiting_constraints.append("departure_delay_not_elapsed")
    pursue_constraints: list[str] = []
    if camera_snapshot.person_visible is not True:
        pursue_constraints.append("camera_not_visible_for_visible_departure")
    if anchor_center_x is None:
        pursue_constraints.append("no_recent_departure_anchor")
    elif abs(anchor_center_x - 0.5) < max(0.0, visible_edge_threshold - 0.5):
        pursue_constraints.append("departure_anchor_below_threshold")
    if anchor_box_edge is None:
        pursue_constraints.append("departure_box_not_near_edge")
    else:
        left_threshold = 1.0 - right_threshold
        if anchor_center_x is not None and anchor_center_x >= 0.5 and anchor_box_edge < right_threshold:
            pursue_constraints.append("departure_box_not_near_edge")
        if anchor_center_x is not None and anchor_center_x < 0.5 and anchor_box_edge > left_threshold:
            pursue_constraints.append("departure_box_not_near_edge")
    if departure_age_s is None:
        pursue_constraints.append("departure_timer_not_started")
    elif departure_age_s < activation_delay_s:
        pursue_constraints.append("departure_delay_not_elapsed")
    justification_map = {
        "waiting_for_exit": "Visible target is still present but the controller has not yet confirmed a side departure strongly enough to unlock monotone pursuit.",
        "pursuing_edge_departure": "A visible side departure was confirmed, so the controller committed to a one-direction exit pursuit.",
        "pursuing_exit_direction": "The visible target was lost after a confirmed departure, so the controller is continuing the remembered exit direction.",
        "reacquired_visible_cooldown": "The target became visible again after exit pursuit, so the controller released and entered cooldown.",
        "exit_cooldown": "A recent exit pursuit already completed, so the controller is intentionally suppressing another motion burst.",
        "holding_exit_limit": "The exit limit was reached and released, so the controller is intentionally holding still at the limit.",
        "disabled": "The servo controller is disabled by configuration, so no physical motion is allowed.",
        "ignored_non_authoritative_source": "The current source is not authoritative for physical servo control, so this tick is intentionally ignored.",
    }
    return {
        "decision_id": f"{coordinator._attention_servo_forensic_run_id}:{coordinator._attention_servo_forensic_tick_index}",
        "question": "Which exit-follow state should the servo controller choose on this runtime tick?",
        "context": {
            "source": source,
            "camera_person_visible": camera_snapshot.person_visible,
            "camera_primary_person_box_left": _round_optional_ratio(
                None if camera_snapshot.primary_person_box is None else camera_snapshot.primary_person_box.left
            ),
            "camera_primary_person_box_right": _round_optional_ratio(
                None if camera_snapshot.primary_person_box is None else camera_snapshot.primary_person_box.right
            ),
            "camera_primary_person_center_x": _round_optional_ratio(camera_snapshot.primary_person_center_x),
            "camera_visible_person_count": len(camera_snapshot.visible_persons),
            "attention_target_state": None if attention_target is None else attention_target.state,
            "attention_target_track_id": None if attention_target is None else attention_target.target_track_id,
            "attention_target_center_x": _round_optional_ratio(
                None if attention_target is None else attention_target.target_center_x
            ),
            "recent_exit_anchor_center_x": anchor_center_x,
            "recent_exit_box_edge": anchor_box_edge,
            "visible_edge_departure_age_s": departure_age_s,
        },
        "options": [
            {
                "id": "waiting_for_exit",
                "summary": "Keep the servo still while a visible departure has not been confirmed strongly enough.",
                "score_components": {
                    "camera_person_visible": camera_snapshot.person_visible,
                    "recent_exit_anchor_center_x": anchor_center_x,
                    "recent_exit_box_edge": anchor_box_edge,
                    "visible_edge_departure_age_s": departure_age_s,
                },
                "constraints_violated": waiting_constraints,
            },
            {
                "id": "pursuing_edge_departure",
                "summary": "Commit to one-way exit pursuit while the target remains visibly biased to one side.",
                "score_components": {
                    "camera_person_visible": camera_snapshot.person_visible,
                    "recent_exit_anchor_center_x": anchor_center_x,
                    "recent_exit_box_edge": anchor_box_edge,
                    "visible_edge_departure_age_s": departure_age_s,
                },
                "constraints_violated": pursue_constraints,
            },
            {
                "id": "reacquired_visible_cooldown",
                "summary": "Release output after exit pursuit once the target is visibly reacquired.",
                "score_components": {
                    "camera_person_visible": camera_snapshot.person_visible,
                    "attention_target_center_x": _round_optional_ratio(
                        None if attention_target is None else attention_target.target_center_x
                    ),
                },
                "constraints_violated": [],
            },
        ],
        "selected": {
            "id": decision.reason,
            "justification": justification_map.get(
                decision.reason,
                "Selected the current servo controller state based on the bounded exit-follow state machine.",
            ),
            "expected_outcome": (
                "servo moves monotonically"
                if decision.commanded_pulse_width_us is not None
                else "servo remains still"
            ),
        },
        "counterfactuals": [
            {
                "id": "waiting_for_exit" if decision.reason != "waiting_for_exit" else "pursuing_edge_departure",
                "why_not": "Its guard conditions were not the best match for the current camera, target, and controller state.",
            }
        ],
        "confidence": "forensic",
        "guardrails": controller_config,
        "kpi_impact_estimate": {
            "latency_ms": 0.0,
            "tokens": 0,
            "cost_usd": 0.0,
        },
    }


def attention_servo_source_is_authoritative(
    coordinator: Any,
    *,
    source: str,
) -> bool:
    """Prefer HDMI attention-refresh over slower automation snapshots for servo control."""

    if not display_attention_refresh_supported(
        config=coordinator.config,
        vision_observer=coordinator.vision_observer,
    ):
        return True
    return source == "display_attention_refresh"


def record_attention_follow_pipeline_if_changed(
    coordinator: Any,
    *,
    source: str,
    observed_at: float,
    camera_snapshot: ProactiveCameraSnapshot,
    attention_target: MultimodalAttentionTargetSnapshot | None,
) -> None:
    """Record one changed runtime attention-follow pipeline state before servo gating."""

    controller = coordinator.attention_servo_controller
    controller_config = None if controller is None else getattr(controller, "config", None)
    key = (
        source,
        camera_snapshot.person_visible,
        camera_snapshot.person_count,
        camera_snapshot.camera_ready,
        camera_snapshot.camera_ai_ready,
        _round_optional_ratio(camera_snapshot.primary_person_center_x),
        None if attention_target is None else attention_target.state,
        None if attention_target is None else attention_target.active,
        None if attention_target is None else attention_target.target_track_id,
        _round_optional_ratio(None if attention_target is None else attention_target.target_center_x),
        _round_optional_ratio(None if attention_target is None else attention_target.target_velocity_x),
        None if attention_target is None else attention_target.focus_source,
        _round_optional_ratio(None if attention_target is None else attention_target.confidence),
        None if controller is None else True,
        None if controller_config is None else getattr(controller_config, "enabled", None),
        None if controller_config is None else getattr(controller_config, "follow_exit_only", None),
        None if controller_config is None else getattr(controller_config, "driver", None),
        None if controller_config is None else getattr(controller_config, "gpio", None),
    )
    if key == coordinator._last_attention_follow_pipeline_key:
        return
    coordinator._last_attention_follow_pipeline_key = key
    data = {
        "observed_at": observed_at,
        "follow_source": source,
        "camera_person_visible": camera_snapshot.person_visible,
        "camera_person_count": camera_snapshot.person_count,
        "camera_visible_person_count": len(camera_snapshot.visible_persons),
        "camera_ready": camera_snapshot.camera_ready,
        "camera_ai_ready": camera_snapshot.camera_ai_ready,
        "camera_primary_person_zone": camera_snapshot.primary_person_zone.value,
        "camera_primary_person_center_x": _round_optional_ratio(camera_snapshot.primary_person_center_x),
        "attention_target_state": None if attention_target is None else attention_target.state,
        "attention_target_active": None if attention_target is None else attention_target.active,
        "attention_target_track_id": None if attention_target is None else attention_target.target_track_id,
        "attention_target_center_x": _round_optional_ratio(
            None if attention_target is None else attention_target.target_center_x
        ),
        "attention_target_velocity_x": _round_optional_ratio(
            None if attention_target is None else attention_target.target_velocity_x
        ),
        "attention_target_focus_source": None if attention_target is None else attention_target.focus_source,
        "attention_target_confidence": _round_optional_ratio(
            None if attention_target is None else attention_target.confidence
        ),
        "servo_controller_present": controller is not None,
        "servo_config_enabled": None if controller_config is None else getattr(controller_config, "enabled", None),
        "servo_follow_exit_only": (
            None if controller_config is None else getattr(controller_config, "follow_exit_only", None)
        ),
        "servo_driver": None if controller_config is None else getattr(controller_config, "driver", None),
        "servo_gpio": None if controller_config is None else getattr(controller_config, "gpio", None),
    }
    coordinator._append_ops_event(
        event="proactive_attention_follow_pipeline",
        message="Recorded one changed attention-follow pipeline state before servo gating.",
        data=data,
    )
    coordinator._emit(
        _emit_key_value_line(
            "attention_follow_pipeline",
            source=source,
            person_visible=camera_snapshot.person_visible,
            person_count=camera_snapshot.person_count,
            camera_ready=camera_snapshot.camera_ready,
            camera_ai_ready=camera_snapshot.camera_ai_ready,
            target_state=None if attention_target is None else attention_target.state,
            target_active=None if attention_target is None else attention_target.active,
            target_center_x=_round_optional_ratio(
                None if attention_target is None else attention_target.target_center_x
            ),
            target_confidence=_round_optional_ratio(
                None if attention_target is None else attention_target.confidence
            ),
            servo_enabled=None if controller_config is None else getattr(controller_config, "enabled", None),
            follow_exit_only=None if controller_config is None else getattr(controller_config, "follow_exit_only", None),
            servo_driver=None if controller_config is None else getattr(controller_config, "driver", None),
        )
    )


def record_attention_servo_follow_if_changed(
    coordinator: Any,
    *,
    source: str,
    observed_at: float,
    attention_target: MultimodalAttentionTargetSnapshot | None,
    decision: AttentionServoDecision,
) -> None:
    """Record only materially changed servo-follow decisions for Pi root-cause tracing."""

    key = (
        source,
        decision.reason,
        decision.active,
        _round_optional_ratio(decision.target_center_x),
        _round_optional_ratio(decision.applied_center_x),
        decision.target_pulse_width_us,
        decision.commanded_pulse_width_us,
        None if attention_target is None else attention_target.state,
        None if attention_target is None else attention_target.target_track_id,
        _round_optional_ratio(None if attention_target is None else attention_target.target_center_x),
        None if attention_target is None else attention_target.focus_source,
    )
    if key == coordinator._last_attention_servo_follow_key:
        return
    coordinator._last_attention_servo_follow_key = key
    coordinator._append_ops_event(
        event="proactive_attention_servo_follow",
        message="Updated the bounded body-orientation servo trace.",
        data={
            "observed_at": observed_at,
            "follow_source": source,
            "attention_target_state": None if attention_target is None else attention_target.state,
            "attention_target_active": None if attention_target is None else attention_target.active,
            "attention_target_track_id": None if attention_target is None else attention_target.target_track_id,
            "attention_target_center_x": _round_optional_ratio(
                None if attention_target is None else attention_target.target_center_x
            ),
            "attention_target_focus_source": None if attention_target is None else attention_target.focus_source,
            "decision_reason": decision.reason,
            "decision_active": decision.active,
            "decision_confidence": _round_optional_ratio(decision.confidence),
            "decision_target_center_x": _round_optional_ratio(decision.target_center_x),
            "decision_applied_center_x": _round_optional_ratio(decision.applied_center_x),
            "decision_target_pulse_width_us": decision.target_pulse_width_us,
            "decision_commanded_pulse_width_us": decision.commanded_pulse_width_us,
        },
    )
    coordinator._emit(
        _emit_key_value_line(
            "attention_servo_decision",
            source=source,
            reason=decision.reason,
            active=decision.active,
            target_center_x=_round_optional_ratio(decision.target_center_x),
            applied_center_x=_round_optional_ratio(decision.applied_center_x),
            target_pulse_us=decision.target_pulse_width_us,
            commanded_pulse_us=decision.commanded_pulse_width_us,
        )
    )


def record_display_attention_follow_if_changed(
    coordinator: Any,
    *,
    observed_at: float,
    runtime_status_value: object,
    camera_snapshot: ProactiveCameraSnapshot,
    publish_result: DisplayAttentionCuePublishResult | None,
) -> None:
    """Persist one bounded changed-only trace of the HDMI attention-follow path."""

    decision = None if publish_result is None else publish_result.decision
    key = (
        str(runtime_status_value or "").strip().lower(),
        camera_snapshot.camera_online,
        camera_snapshot.camera_ready,
        camera_snapshot.camera_ai_ready,
        camera_snapshot.camera_error,
        camera_snapshot.person_visible,
        camera_snapshot.person_visible_unknown,
        camera_snapshot.person_count,
        camera_snapshot.person_count_unknown,
        len(camera_snapshot.visible_persons),
        camera_snapshot.primary_person_zone.value,
        _round_optional_ratio(camera_snapshot.primary_person_center_x),
        None if coordinator.latest_attention_target_snapshot is None else coordinator.latest_attention_target_snapshot.state,
        None
        if coordinator.latest_attention_target_snapshot is None
        else coordinator.latest_attention_target_snapshot.target_horizontal,
        None
        if coordinator.latest_attention_target_snapshot is None
        else coordinator.latest_attention_target_snapshot.target_track_id,
        _round_optional_ratio(
            None
            if coordinator.latest_attention_target_snapshot is None
            else coordinator.latest_attention_target_snapshot.target_center_x
        ),
        None if publish_result is None else publish_result.action,
        None if decision is None else decision.reason,
        None if decision is None else decision.gaze.value,
        None if decision is None else decision.cue_gaze_x,
        None if decision is None else decision.cue_gaze_y,
        None if decision is None else decision.head_dx,
        None if decision is None else decision.speaker_locked,
    )
    if key == coordinator._last_display_attention_follow_key:
        return
    coordinator._last_display_attention_follow_key = key
    coordinator._append_ops_event(
        event="proactive_display_attention_follow",
        message="Updated the bounded HDMI attention-follow trace.",
        data={
            "observed_at": observed_at,
            "runtime_status": str(runtime_status_value or "").strip().lower() or None,
            "camera_online": camera_snapshot.camera_online,
            "camera_ready": camera_snapshot.camera_ready,
            "camera_ai_ready": camera_snapshot.camera_ai_ready,
            "camera_error": camera_snapshot.camera_error,
            "person_visible": camera_snapshot.person_visible,
            "person_visible_unknown": camera_snapshot.person_visible_unknown,
            "camera_person_count": camera_snapshot.person_count,
            "camera_person_count_unknown": camera_snapshot.person_count_unknown,
            "camera_visible_person_count": len(camera_snapshot.visible_persons),
            "camera_visible_persons_unknown": camera_snapshot.visible_persons_unknown,
            "camera_primary_person_zone": camera_snapshot.primary_person_zone.value,
            "camera_primary_person_center_x": _round_optional_ratio(camera_snapshot.primary_person_center_x),
            "camera_primary_person_center_y": _round_optional_ratio(camera_snapshot.primary_person_center_y),
            "camera_frame_age_s": _round_optional_seconds(
                None
                if camera_snapshot.last_camera_frame_at is None
                else max(0.0, observed_at - float(camera_snapshot.last_camera_frame_at))
            ),
            "attention_target_state": (
                None if coordinator.latest_attention_target_snapshot is None else coordinator.latest_attention_target_snapshot.state
            ),
            "attention_target_active": (
                None if coordinator.latest_attention_target_snapshot is None else coordinator.latest_attention_target_snapshot.active
            ),
            "attention_target_horizontal": (
                None
                if coordinator.latest_attention_target_snapshot is None
                else coordinator.latest_attention_target_snapshot.target_horizontal
            ),
            "attention_target_track_id": (
                None
                if coordinator.latest_attention_target_snapshot is None
                else coordinator.latest_attention_target_snapshot.target_track_id
            ),
            "attention_target_center_x": _round_optional_ratio(
                None
                if coordinator.latest_attention_target_snapshot is None
                else coordinator.latest_attention_target_snapshot.target_center_x
            ),
            "attention_target_velocity_x": _round_optional_ratio(
                None
                if coordinator.latest_attention_target_snapshot is None
                else coordinator.latest_attention_target_snapshot.target_velocity_x
            ),
            "attention_target_focus_source": (
                None
                if coordinator.latest_attention_target_snapshot is None
                else coordinator.latest_attention_target_snapshot.focus_source
            ),
            "publish_action": None if publish_result is None else publish_result.action,
            "publish_owner": None if publish_result is None else publish_result.owner,
            "decision_reason": None if decision is None else decision.reason,
            "decision_gaze": None if decision is None else decision.gaze.value,
            "decision_cue_gaze_x": None if decision is None else decision.cue_gaze_x,
            "decision_cue_gaze_y": None if decision is None else decision.cue_gaze_y,
            "decision_head_dx": None if decision is None else decision.head_dx,
            "decision_head_dy": None if decision is None else decision.head_dy,
            "decision_speaker_locked": None if decision is None else decision.speaker_locked,
        },
    )


def record_attention_debug_tick(
    coordinator: Any,
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

    stream = coordinator.display_attention_debug_stream
    if stream is None:
        return
    decision = None if publish_result is None else publish_result.decision
    payload: dict[str, Any] = {
        "runtime_status": str(runtime_status_value or "").strip().lower() or None,
        "vision_mode": coordinator._last_attention_vision_refresh_mode,
        "audio_mode": coordinator._last_attention_audio_refresh_mode,
        "stage_ms": dict(stage_ms),
    }
    if camera_snapshot is not None:
        payload.update(
            {
                "camera_online": camera_snapshot.camera_online,
                "camera_ready": camera_snapshot.camera_ready,
                "camera_ai_ready": camera_snapshot.camera_ai_ready,
                "camera_error": camera_snapshot.camera_error,
                "person_visible": camera_snapshot.person_visible,
                "person_visible_unknown": camera_snapshot.person_visible_unknown,
                "camera_person_count": camera_snapshot.person_count,
                "camera_visible_person_count": len(camera_snapshot.visible_persons),
                "camera_primary_person_zone": camera_snapshot.primary_person_zone.value,
                "camera_primary_person_center_x": _round_optional_ratio(camera_snapshot.primary_person_center_x),
                "camera_primary_person_center_y": _round_optional_ratio(camera_snapshot.primary_person_center_y),
                "camera_looking_signal_state": camera_snapshot.looking_signal_state,
                "camera_looking_signal_source": camera_snapshot.looking_signal_source,
                "camera_frame_age_s": _round_optional_seconds(
                    None
                    if camera_snapshot.last_camera_frame_at is None
                    else max(0.0, observed_at - float(camera_snapshot.last_camera_frame_at))
                ),
            }
        )
    if coordinator.latest_attention_target_snapshot is not None:
        payload.update(
            {
                "attention_target_state": coordinator.latest_attention_target_snapshot.state,
                "attention_target_active": coordinator.latest_attention_target_snapshot.active,
                "attention_target_horizontal": coordinator.latest_attention_target_snapshot.target_horizontal,
                "attention_target_vertical": coordinator.latest_attention_target_snapshot.target_vertical,
                "attention_target_track_id": coordinator.latest_attention_target_snapshot.target_track_id,
                "attention_target_center_x": _round_optional_ratio(
                    coordinator.latest_attention_target_snapshot.target_center_x
                ),
                "attention_target_center_y": _round_optional_ratio(
                    coordinator.latest_attention_target_snapshot.target_center_y
                ),
                "attention_target_focus_source": coordinator.latest_attention_target_snapshot.focus_source,
                "attention_target_speaker_locked": coordinator.latest_attention_target_snapshot.speaker_locked,
            }
        )
    if audio_observation is not None:
        payload.update(
            {
                "audio_speech_detected": audio_observation.speech_detected,
                "audio_azimuth_deg": audio_observation.azimuth_deg,
                "audio_direction_confidence": _round_optional_ratio(audio_observation.direction_confidence),
                "audio_signal_source": audio_observation.signal_source,
            }
        )
    if publish_result is not None:
        payload.update(
            {
                "publish_action": publish_result.action,
                "publish_owner": publish_result.owner,
            }
        )
    if decision is not None:
        payload.update(
            {
                "decision_reason": decision.reason,
                "decision_gaze": decision.gaze.value,
                "decision_cue_gaze_x": decision.cue_gaze_x,
                "decision_cue_gaze_y": decision.cue_gaze_y,
                "decision_head_dx": decision.head_dx,
                "decision_head_dy": decision.head_dy,
                "decision_speaker_locked": decision.speaker_locked,
            }
        )
    attention_debug_details_getter = getattr(coordinator.vision_observer, "attention_debug_details", None)
    if callable(attention_debug_details_getter):
        try:
            attention_debug_details = attention_debug_details_getter()
        except Exception:
            attention_debug_details = None
        if attention_debug_details:
            payload["pipeline_debug"] = attention_debug_details
    if coordinator._last_display_attention_fusion_debug:
        payload["display_camera_fusion"] = dict(coordinator._last_display_attention_fusion_debug)
    try:
        stream.append_tick(
            outcome=outcome,
            observed_at=observed_at,
            data=payload,
        )
    except Exception as exc:
        coordinator._record_fault(
            event="proactive_attention_debug_stream_failed",
            message="Failed to append the bounded attention debug tick.",
            error=exc,
            data={"outcome": outcome},
        )


__all__ = [
    "attention_servo_source_is_authoritative",
    "build_attention_servo_decision_ledger",
    "publish_display_attention_live_context",
    "record_attention_debug_tick",
    "record_attention_follow_pipeline_if_changed",
    "record_attention_servo_follow_if_changed",
    "record_attention_servo_forensic_tick",
    "record_display_attention_follow_if_changed",
    "summarize_visible_persons",
    "update_attention_servo_follow",
    "update_display_attention_follow",
    "update_display_debug_signals",
]
