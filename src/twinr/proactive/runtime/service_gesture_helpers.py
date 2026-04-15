"""Focused gesture/runtime helper functions for the proactive coordinator.

These helpers keep HDMI gesture acknowledgement, visual wake dispatch, and
gesture-specific forensic tracing out of the main proactive monitor
orchestrator.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Gesture wake dispatch no longer fails closed when priming sensor context,
# BUG-2: forensic tracing or debug serialization raises; those side effects are
#        now isolated so wake/refresh control flow keeps running.
# BUG-3: Transcript-first voice runtimes now fail close the dedicated HDMI
#        gesture lane through the shared refresh helper contract, so the
#        proactive monitor never arms the heavy lane on the streaming PID.
# SEC-1: Arbitrary pipeline debug payloads are no longer emitted directly.
#        Telemetry is now bounded and serialization-safe while preserving the
#        stable gesture-stream contract fields expected by local debug tooling.
# IMP-1: Observation/gesture stream extraction is centralized and memoized per
#        call to remove repeated hot-path lookups and keep fields consistent.
# IMP-2: The helper now degrades gracefully on partial coordinator/dispatcher
#        failures, including synchronous wake fallback when the dispatcher is
#        unavailable.

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from twinr.agent.workflows.forensics import workflow_decision as _workflow_decision_impl
except Exception:  # pragma: no cover - optional diagnostics dependency
    _workflow_decision_impl = None

from ..social.camera_surface import ProactiveCameraSurfaceUpdate
from ..social.engine import SocialObservation, SocialVisionObservation
from ..social.perception_stream import gesture_stream, gesture_stream_authoritative
from .display_gesture_emoji import DisplayGestureEmojiDecision, DisplayGestureEmojiPublishResult
from .gesture_wakeup_lane import GestureWakeupDecision

_LOGGER = logging.getLogger(__name__)

_MAX_TELEMETRY_DEPTH = 4
_MAX_TELEMETRY_ITEMS = 16
_MAX_TELEMETRY_STRING_LENGTH = 160
_SENSITIVE_TELEMETRY_KEY_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "credential",
    "cookie",
    "auth",
    "session",
    "image",
    "frame",
    "jpeg",
    "png",
    "embedding",
    "descriptor",
    "feature",
    "biometric",
)


def _round_optional_ratio(value: float | None) -> float | None:
    from . import service as service_module

    return service_module._round_optional_ratio(value)


def _truncate_text(value: str, *, limit: int = _MAX_TELEMETRY_STRING_LENGTH) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return f"{value[: limit - 1]}…"


def _safe_enum_value(value: object) -> object:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return _truncate_text(raw)
    if isinstance(raw, (bool, int)):
        return raw
    if isinstance(raw, float):
        return raw if math.isfinite(raw) else None
    return _truncate_text(str(raw))


def _serialize_exception(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": _truncate_text(str(error)),
    }


def _is_sensitive_telemetry_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_TELEMETRY_KEY_FRAGMENTS)


def _sanitize_for_telemetry(value: object, *, depth: int = 0) -> object:
    if depth >= _MAX_TELEMETRY_DEPTH:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__}:{len(value)} bytes>"
    if isinstance(value, BaseException):
        return _serialize_exception(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        items = list(value.items())
        for idx, (key, item_value) in enumerate(items):
            if idx >= _MAX_TELEMETRY_ITEMS:
                sanitized["__truncated_keys__"] = max(len(items) - _MAX_TELEMETRY_ITEMS, 0)
                break
            key_text = _truncate_text(str(key))
            if _is_sensitive_telemetry_key(key_text):
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = _sanitize_for_telemetry(item_value, depth=depth + 1)
        return sanitized
    if isinstance(value, set):
        limited = []
        for idx, item in enumerate(value):
            if idx >= _MAX_TELEMETRY_ITEMS:
                break
            limited.append(_sanitize_for_telemetry(item, depth=depth + 1))
        if len(value) > _MAX_TELEMETRY_ITEMS:
            limited.append(f"<+{len(value) - _MAX_TELEMETRY_ITEMS} more>")
        return limited
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        limited = [
            _sanitize_for_telemetry(item, depth=depth + 1)
            for item in value[:_MAX_TELEMETRY_ITEMS]
        ]
        if len(value) > _MAX_TELEMETRY_ITEMS:
            limited.append(f"<+{len(value) - _MAX_TELEMETRY_ITEMS} more>")
        return limited
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _sanitize_for_telemetry(enum_value, depth=depth + 1)
    return _truncate_text(repr(value))


def _record_fault_safe(
    coordinator: Any | None,
    *,
    event: str,
    message: str,
    error: BaseException | None = None,
    data: Mapping[str, object] | None = None,
) -> None:
    payload = None if data is None else dict(_sanitize_for_telemetry(dict(data)))
    recorder = None if coordinator is None else getattr(coordinator, "_record_fault", None)
    if callable(recorder):
        try:
            recorder(event=event, message=message, error=error, data=payload)
            return
        except Exception:
            _LOGGER.exception("Failed to record coordinator fault %s", event)
    if error is None:
        _LOGGER.error("%s [%s] data=%s", message, event, payload)
    else:
        _LOGGER.exception(
            "%s [%s] error=%s data=%s",
            message,
            event,
            _serialize_exception(error),
            payload,
        )


def _append_ops_event_safe(
    coordinator: Any | None,
    *,
    event: str,
    message: str,
    data: Mapping[str, object] | None = None,
) -> None:
    payload = None if data is None else dict(_sanitize_for_telemetry(dict(data)))
    appender = None if coordinator is None else getattr(coordinator, "_append_ops_event", None)
    if callable(appender):
        try:
            appender(event=event, message=message, data=payload)
            return
        except Exception as exc:
            _record_fault_safe(
                coordinator,
                event="proactive_ops_event_failed",
                message="Failed to append proactive ops event.",
                error=exc,
                data={"source_event": event, "source_data": payload},
            )
            return
    _LOGGER.info("%s [%s] data=%s", message, event, payload)


def _workflow_decision_safe(**kwargs: object) -> None:
    sanitized_kwargs = {
        key: _sanitize_for_telemetry(value)
        for key, value in kwargs.items()
    }
    if _workflow_decision_impl is None:
        return
    try:
        _workflow_decision_impl(**sanitized_kwargs)
    except Exception:
        _LOGGER.exception(
            "Failed to emit workflow_decision: %s",
            sanitized_kwargs.get("msg"),
        )


def _extract_gesture_stream_state(
    observation: SocialVisionObservation,
) -> tuple[object | None, bool | None]:
    try:
        stream = gesture_stream(observation)
    except Exception:
        stream = None
    try:
        authoritative = gesture_stream_authoritative(observation)
    except Exception:
        authoritative = None
    return stream, authoritative


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _coerce_optional_non_negative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced < 0:
        return None
    return coerced


def _gesture_stream_fields(
    observation: SocialVisionObservation,
) -> dict[str, object]:
    stream, authoritative = _extract_gesture_stream_state(observation)
    return {
        "gesture_stream_authoritative": _coerce_optional_bool(authoritative),
        "gesture_stream_activation_key": (
            None if stream is None else _safe_enum_value(getattr(stream, "activation_key", None))
        ),
        "gesture_stream_activation_token": (
            None if stream is None else _coerce_optional_non_negative_int(getattr(stream, "activation_token", None))
        ),
        "gesture_stream_activation_rising": (
            None if stream is None else _coerce_optional_bool(getattr(stream, "activation_rising", None))
        ),
    }


def _observation_fields(
    observation: SocialVisionObservation,
) -> dict[str, object]:
    return {
        "person_visible": observation.person_visible,
        "person_count": observation.person_count,
        "camera_online": observation.camera_online,
        "camera_ready": observation.camera_ready,
        "camera_ai_ready": observation.camera_ai_ready,
        "camera_error": _safe_enum_value(observation.camera_error),
        "fine_hand_gesture": _safe_enum_value(observation.fine_hand_gesture),
        "fine_hand_gesture_confidence": _round_optional_ratio(observation.fine_hand_gesture_confidence),
        "gesture_event": _safe_enum_value(observation.gesture_event),
        "gesture_confidence": _round_optional_ratio(observation.gesture_confidence),
        "hand_or_object_near_camera": observation.hand_or_object_near_camera,
        "showing_intent_likely": observation.showing_intent_likely,
        **_gesture_stream_fields(observation),
    }


def _decision_fields(decision: DisplayGestureEmojiDecision) -> dict[str, object]:
    return {
        "decision_active": decision.active,
        "decision_reason": decision.reason,
        "decision_symbol": _safe_enum_value(decision.symbol),
        "decision_accent": _sanitize_for_telemetry(decision.accent),
        "decision_hold_seconds": round(decision.hold_seconds, 3),
    }


def _publish_result_fields(
    publish_result: DisplayGestureEmojiPublishResult,
) -> dict[str, object]:
    return {
        "publish_action": _sanitize_for_telemetry(publish_result.action),
        "publish_owner": _sanitize_for_telemetry(publish_result.owner),
    }


def _wakeup_fields(
    wakeup_decision: GestureWakeupDecision,
    *,
    wakeup_handled: bool | None = None,
) -> dict[str, object]:
    return {
        "gesture_wakeup_active": wakeup_decision.active,
        "gesture_wakeup_reason": wakeup_decision.reason,
        "gesture_wakeup_trigger_gesture": _safe_enum_value(wakeup_decision.trigger_gesture),
        "gesture_wakeup_observed_gesture": _safe_enum_value(wakeup_decision.observed_gesture),
        "gesture_wakeup_confidence": _round_optional_ratio(wakeup_decision.confidence),
        "gesture_wakeup_request_source": _sanitize_for_telemetry(wakeup_decision.request_source),
        "gesture_wakeup_handled": wakeup_handled,
    }


def _normalized_stage_ms(stage_ms: Mapping[str, float]) -> dict[str, float | int | None]:
    normalized: dict[str, float | int | None] = {}
    for index, (name, value) in enumerate(stage_ms.items()):
        if index >= _MAX_TELEMETRY_ITEMS:
            normalized["__truncated_keys__"] = float(max(len(stage_ms) - _MAX_TELEMETRY_ITEMS, 0))
            break
        try:
            numeric = float(value)
        except Exception:
            normalized[_truncate_text(str(name))] = None
            continue
        normalized[_truncate_text(str(name))] = round(numeric, 3) if math.isfinite(numeric) else None
    return normalized


def dispatch_gesture_wakeup_with_fresh_context(
    coordinator: Any,
    *,
    observed_at: float,
    vision_snapshot: Any,
    decision: GestureWakeupDecision,
) -> bool:
    """Prime current gesture facts before dispatching an accepted wakeup."""

    if decision.active:
        prime_context = getattr(coordinator, "_prime_gesture_wakeup_sensor_context", None)
        if callable(prime_context):
            prime_context(
                observed_at=observed_at,
                vision_snapshot=vision_snapshot,
            )
        else:
            prime_gesture_wakeup_sensor_context(
                coordinator,
                observed_at=observed_at,
                vision_snapshot=vision_snapshot,
            )
    wake_handler = getattr(coordinator, "_handle_gesture_wakeup_decision", None)
    if callable(wake_handler):
        return bool(wake_handler(decision))
    return handle_gesture_wakeup_decision(coordinator, decision)


def prime_gesture_wakeup_sensor_context(
    coordinator: Any,
    *,
    observed_at: float,
    vision_snapshot: Any,
) -> None:
    """Export one fresh sensor/person-state payload from the active gesture tick."""

    observation_handler = getattr(coordinator, "observation_handler", None)
    if observation_handler is None:
        return
    try:
        vision_observation = getattr(vision_snapshot, "observation", None)
        if vision_observation is None:
            raise AttributeError("vision_snapshot.observation is required")
        audio_snapshot = coordinator._observe_audio_for_attention_refresh(now=observed_at)
        audio_observation = getattr(audio_snapshot, "observation", None)
        if audio_observation is None:
            raise AttributeError("audio_snapshot.observation is required")
        audio_policy_snapshot = coordinator._observe_audio_policy(
            now=observed_at,
            audio_observation=audio_observation,
        )
        observation = SocialObservation(
            observed_at=observed_at,
            inspected=True,
            pir_motion_detected=False,
            low_motion=False,
            vision=vision_observation,
            audio=audio_observation,
        )
        camera_update = coordinator._observe_camera_surface(observation, inspected=True)
        camera_snapshot = getattr(camera_update, "snapshot", None)
        if camera_snapshot is None:
            raise AttributeError("camera_update.snapshot is required")
        presence_snapshot = coordinator._observe_presence(
            now=observed_at,
            person_visible=camera_snapshot.person_visible,
            motion_active=False,
            audio_observation=audio_observation,
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
        event_names = coordinator._derive_sensor_events(
            facts,
            camera_event_names=camera_update.event_names,
        )
        observation_handler(facts, event_names)
    except Exception as exc:
        _record_fault_safe(
            coordinator,
            event="proactive_gesture_wakeup_context_prime_failed",
            message="Failed to export fresh gesture wakeup sensor context; continuing with wake dispatch.",
            error=exc,
            data={"observed_at": observed_at},
        )


def update_display_gesture_emoji_ack(
    coordinator: Any,
    camera_update: ProactiveCameraSurfaceUpdate,
) -> None:
    """Mirror clear stabilized user gestures into the HDMI emoji reserve area."""

    publisher = getattr(coordinator, "display_gesture_emoji_publisher", None)
    if publisher is None:
        return
    try:
        publisher.publish_update(camera_update)
    except Exception as exc:
        _record_fault_safe(
            coordinator,
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

    publisher = getattr(coordinator, "display_gesture_emoji_publisher", None)
    if publisher is None:
        return None
    try:
        return publisher.publish(decision)
    except Exception as exc:
        _record_fault_safe(
            coordinator,
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

    gesture_wakeup_handler = getattr(coordinator, "gesture_wakeup_handler", None)
    if not decision.active or gesture_wakeup_handler is None:
        return False

    dispatcher = getattr(coordinator, "_gesture_wakeup_dispatcher", None)
    if dispatcher is None:
        _record_fault_safe(
            coordinator,
            event="gesture_wakeup_dispatcher_missing",
            message="Gesture wakeup dispatcher is unavailable; running the handler inline.",
            data={
                "gesture": _safe_enum_value(decision.trigger_gesture),
                "reason": decision.reason,
            },
        )
        return run_gesture_wakeup_handler(coordinator, decision)

    try:
        dispatched = bool(dispatcher.submit(decision))
    except Exception as exc:
        _record_fault_safe(
            coordinator,
            event="gesture_wakeup_dispatch_failed",
            message="Gesture wakeup dispatch failed; running the handler inline.",
            error=exc,
            data={
                "gesture": _safe_enum_value(decision.trigger_gesture),
                "reason": decision.reason,
            },
        )
        return run_gesture_wakeup_handler(coordinator, decision)

    if not dispatched:
        _append_ops_event_safe(
            coordinator,
            event="gesture_wakeup_dispatch_skipped",
            message="Visual wakeup was skipped because another visual wakeup is already active.",
            data={
                "gesture": _safe_enum_value(decision.trigger_gesture),
                "reason": decision.reason,
            },
        )
    return dispatched


def run_gesture_wakeup_handler(
    coordinator: Any,
    decision: GestureWakeupDecision,
) -> bool:
    """Run one visual wakeup handler on the dedicated dispatcher thread."""

    gesture_wakeup_handler = getattr(coordinator, "gesture_wakeup_handler", None)
    if gesture_wakeup_handler is None:
        return False

    def _run_handler() -> bool:
        try:
            return bool(gesture_wakeup_handler(decision))
        except Exception as exc:
            _record_fault_safe(
                coordinator,
                event="gesture_wakeup_handler_failed",
                message="Gesture wakeup handler failed.",
                error=exc,
                data={"reason": decision.reason, "gesture": _safe_enum_value(decision.trigger_gesture)},
            )
            return False

    handled = _run_handler()
    if handled:
        _append_ops_event_safe(
            coordinator,
            event="gesture_wakeup_triggered",
            message="A configured visual wake gesture opened a hands-free conversation path.",
            data={
                "gesture": _safe_enum_value(decision.trigger_gesture),
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

    debug_stream = getattr(coordinator, "display_gesture_debug_stream", None)
    if debug_stream is None:
        return

    try:
        payload: dict[str, object] = {
            "runtime_status": _sanitize_for_telemetry(str(runtime_status_value or "").strip().lower() or None),
            "vision_mode": _sanitize_for_telemetry(getattr(coordinator, "_last_gesture_vision_refresh_mode", None)),
            "stage_ms": _normalized_stage_ms(stage_ms),
        }

        if observation is not None:
            payload.update(_observation_fields(observation))

        vision_observer = getattr(coordinator, "vision_observer", None)
        gesture_debug_details_getter = getattr(vision_observer, "gesture_debug_details", None)
        if callable(gesture_debug_details_getter):
            try:
                gesture_debug_details = gesture_debug_details_getter()
            except Exception as exc:
                gesture_debug_details = {"error": _serialize_exception(exc)}
            if gesture_debug_details:
                payload["pipeline_debug"] = _sanitize_for_telemetry(gesture_debug_details)

        if decision is not None:
            payload.update(_decision_fields(decision))
        if publish_result is not None:
            payload.update(_publish_result_fields(publish_result))
        if wakeup_decision is not None:
            payload.update(_wakeup_fields(wakeup_decision, wakeup_handled=wakeup_handled))

        debug_stream.append_tick(
            outcome=_truncate_text(outcome),
            observed_at=observed_at,
            # The payload is sanitized incrementally when each section is built.
            # Avoid re-sanitizing the top-level mapping here because that would
            # truncate stable debug-contract keys such as gesture stream state.
            data=dict(payload),
        )
    except Exception as exc:
        _record_fault_safe(
            coordinator,
            event="proactive_gesture_debug_stream_failed",
            message="Failed to append the bounded gesture debug tick.",
            error=exc,
            data={"outcome": outcome},
        )


def gesture_observation_trace_details(
    observation: SocialVisionObservation,
) -> dict[str, object]:
    """Return one bounded trace summary for the current gesture observation."""

    try:
        return _observation_fields(observation)
    except Exception as exc:
        _LOGGER.exception("Failed to build gesture observation trace details")
        return {"trace_error": _serialize_exception(exc)}


def trace_gesture_ack_lane_decision(
    observation: SocialVisionObservation,
    decision: DisplayGestureEmojiDecision,
) -> None:
    """Emit one decision ledger entry for the HDMI ack lane result."""

    try:
        observation_fields = _observation_fields(observation)
    except Exception:
        _LOGGER.exception("Failed to build gesture ack trace context")
        return

    _workflow_decision_safe(
        msg="gesture_ack_lane_decision",
        question="Should the current gesture frame publish an HDMI acknowledgement?",
        selected={
            "id": decision.reason,
            "summary": (
                f"Publish {_safe_enum_value(decision.symbol)}."
                if decision.active
                else "Do not publish an HDMI acknowledgement."
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
            "gesture_stream_authoritative": observation_fields["gesture_stream_authoritative"],
            "observed_fine_hand_gesture": observation_fields["fine_hand_gesture"],
            "observed_fine_hand_confidence": observation_fields["fine_hand_gesture_confidence"],
            "observed_gesture_event": observation_fields["gesture_event"],
            "observed_gesture_confidence": observation_fields["gesture_confidence"],
            "stream_activation_key": observation_fields["gesture_stream_activation_key"],
            "stream_activation_token": observation_fields["gesture_stream_activation_token"],
        },
        confidence=_round_optional_ratio(observation.fine_hand_gesture_confidence or observation.gesture_confidence),
        guardrails=["gesture_ack_lane", "display_only"],
        kpi_impact_estimate={"latency": "low", "user_feedback": "high"},
    )


def trace_gesture_wakeup_lane_decision(
    decision: GestureWakeupDecision,
) -> None:
    """Emit one decision ledger entry for the visual wake lane."""

    _workflow_decision_safe(
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
            {"id": f"gesture_wakeup:{_safe_enum_value(decision.trigger_gesture)}", "summary": "Trigger a wake request immediately."},
            {"id": "gesture_wakeup_not_authoritative", "summary": "Ignore the frame because no authoritative gesture stream was attached."},
            {"id": "gesture_wakeup_already_active", "summary": "Do not re-dispatch while the same authoritative activation token remains active."},
            {"id": "gesture_wakeup_cooldown", "summary": "Suppress a repeated wake request during cooldown."},
            {"id": "no_gesture_wakeup_candidate", "summary": "Ignore the frame because the trigger gesture was not present."},
        ],
        context={
            "trigger_gesture": _safe_enum_value(decision.trigger_gesture),
            "observed_gesture": _safe_enum_value(decision.observed_gesture),
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
    _workflow_decision_safe(
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
