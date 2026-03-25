"""Handle portrait-identity tool calls for realtime Twinr sessions.

This module exposes synchronous handlers that let the model capture live camera
frames, persist local portrait references, inspect the current stored profile
state, and clear that on-device profile again. The handler returns structured
quality and guidance metadata so the model can choose natural next prompts
instead of relying on hardcoded dialog branches.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from twinr.hardware.portrait_match import PortraitMatchProvider

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import (
    ArgumentValidationError,
    SensitiveActionConfirmationRequired,
    require_sensitive_voice_confirmation,
)

LOGGER = logging.getLogger(__name__)
_PORTRAIT_IDENTITY_LOCK = threading.RLock()
_PRIMARY_PROFILE_USER_ID = "main_user"
_SUCCESS_ENROLLMENT_STATUSES = frozenset({"enrolled", "duplicate_reference"})
_RETRYABLE_CAPTURE_STATUSES = frozenset(
    {
        "no_face_detected",
        "ambiguous_face_count",
        "decode_failed",
        "embedding_failed",
        "unknown_face",
        "uncertain_match",
        "ambiguous_identity",
    }
)


def _emit_safe(owner: Any, message: str) -> None:
    emit_best_effort(
        owner,
        message,
        logger=LOGGER,
        failure_message="Portrait-identity telemetry emit failed.",
    )


def _record_event_safe(owner: Any, event_name: str, message: str, **fields: object) -> None:
    record_event_best_effort(
        owner,
        event_name,
        message,
        dict(fields),
        logger=LOGGER,
        failure_message="Portrait-identity event recording failed.",
    )


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _portrait_identity_error(
    owner: Any,
    *,
    status: str,
    event_name: str,
    message: str,
    detail: str,
) -> dict[str, object]:
    _emit_safe(owner, "portrait_identity_tool_call=true")
    _record_event_safe(owner, event_name, message, status=status)
    return {
        "status": status,
        "detail": detail,
        "stores_locally": True,
        "profile_kind": "local_portrait_identity",
        "capture_source": "live_camera",
        "requires_confirmation": False,
    }


def _confirmation_result(owner: Any, arguments: dict[str, object], *, action_label: str) -> dict[str, object] | None:
    try:
        require_sensitive_voice_confirmation(owner, arguments, action_label=action_label)
    except ArgumentValidationError as exc:
        return _portrait_identity_error(
            owner,
            status="error",
            event_name="portrait_identity_argument_error",
            message="Portrait-identity tool arguments were invalid.",
            detail=str(exc),
        )
    except SensitiveActionConfirmationRequired as exc:
        return {
            "status": "confirmation_required",
            "detail": str(exc),
            "stores_locally": True,
            "profile_kind": "local_portrait_identity",
            "capture_source": "live_camera",
            "requires_confirmation": True,
        }
    return None


def _get_portrait_provider(owner: Any) -> PortraitMatchProvider:
    provider = getattr(owner, "_portrait_identity_tool_provider", None)
    if provider is not None:
        return provider

    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("Twinr configuration is unavailable.")
    if not bool(getattr(config, "portrait_match_enabled", True)):
        raise RuntimeError("Local portrait identity is disabled on this device.")

    camera = getattr(owner, "camera", None)
    if camera is None:
        raise RuntimeError("The camera is unavailable right now.")

    provider = PortraitMatchProvider.from_config(
        config,
        camera=camera,
        camera_lock=getattr(owner, "_camera_lock", None),
    )
    setattr(owner, "_portrait_identity_tool_provider", provider)
    return provider


def _coverage_state(reference_image_count: int) -> str:
    if reference_image_count <= 0:
        return "empty"
    if reference_image_count == 1:
        return "single_reference"
    if reference_image_count == 2:
        return "baseline_multi_reference"
    return "multi_reference_ready"


def _quality_state(status: str, *, reference_image_count: int) -> str:
    if status in _SUCCESS_ENROLLMENT_STATUSES:
        return "usable" if reference_image_count >= 2 else "baseline_only"
    if status in {"profile_unavailable", "reference_image_unavailable"}:
        return "not_enrolled"
    if status in {"capture_unavailable", "backend_unavailable", "model_unavailable", "disabled", "unavailable"}:
        return "unavailable"
    if status in _RETRYABLE_CAPTURE_STATUSES:
        return "needs_retry"
    return "unknown"


def _detail_for_status(status: str) -> str:
    messages = {
        "enrolled": "The current camera view was added to Twinr's local face profile.",
        "duplicate_reference": "That camera view is already represented in Twinr's local face profile.",
        "capture_unavailable": "Twinr could not capture a camera image right now.",
        "no_face_detected": "The camera did not capture one clear face.",
        "ambiguous_face_count": "The camera saw multiple faces or could not isolate exactly one face.",
        "decode_failed": "The captured image was not clear enough for local face extraction.",
        "embedding_failed": "Twinr saw a face but could not turn it into a stable local face reference.",
        "backend_unavailable": "Twinr's local face-matching backend is unavailable right now.",
        "model_unavailable": "Twinr's local face-matching model is unavailable right now.",
        "reference_image_unavailable": "No local face profile is saved yet.",
        "profile_unavailable": "No local face profile is saved yet.",
        "cleared": "Twinr deleted the local face profile.",
        "disabled": "Local face matching is disabled on this device.",
        "unknown_face": "Twinr sees a face but it does not match the saved local profile confidently.",
        "uncertain_match": "Twinr sees a face but the local match is still uncertain.",
        "ambiguous_identity": "Twinr sees a face, but the local identity match is ambiguous.",
        "likely_reference_user": "Twinr currently sees a face that matches the saved local profile.",
        "known_other_user": "Twinr currently sees a different enrolled local face profile.",
    }
    return messages.get(status, "Twinr could not complete the local face-profile action.")


def _guidance_hints_for_status(status: str, *, reference_image_count: int) -> tuple[str, ...]:
    hints: list[str] = []
    if status == "no_face_detected":
        hints.extend(("single_face_in_frame", "face_camera", "steady_pose", "better_light"))
    elif status == "ambiguous_face_count":
        hints.extend(("single_face_in_frame", "remove_other_faces_from_view", "steady_pose"))
    elif status in {"decode_failed", "embedding_failed"}:
        hints.extend(("face_camera", "steady_pose", "better_light", "check_camera_view"))
    elif status in {"unknown_face", "uncertain_match", "ambiguous_identity"}:
        hints.extend(("steady_pose", "face_camera", "check_camera_view"))
        if reference_image_count < 2:
            hints.append("capture_more_angle_variation")
    elif status == "capture_unavailable":
        hints.append("retry_later")

    if status in _SUCCESS_ENROLLMENT_STATUSES and reference_image_count < 2:
        hints.extend(("capture_more_angle_variation", "slight_angle_variation"))

    deduplicated: list[str] = []
    for hint in hints:
        if hint not in deduplicated:
            deduplicated.append(hint)
    return tuple(deduplicated)


def _recommended_next_step(status: str, *, reference_image_count: int) -> str:
    if status in _SUCCESS_ENROLLMENT_STATUSES:
        return "capture_more_variation" if reference_image_count < 2 else "done"
    if status in _RETRYABLE_CAPTURE_STATUSES:
        return "retry_capture"
    if status in {"capture_unavailable", "backend_unavailable", "model_unavailable", "unavailable"}:
        return "check_system"
    if status in {"profile_unavailable", "reference_image_unavailable"}:
        return "enroll_now"
    return "none"


def _suggested_follow_up_tool(status: str) -> str | None:
    if status in _RETRYABLE_CAPTURE_STATUSES:
        return "inspect_camera"
    return None


def _suggested_inspection_question(status: str) -> str | None:
    if status not in _RETRYABLE_CAPTURE_STATUSES:
        return None
    return "Check whether exactly one face is centered, clear, and well lit enough for local portrait enrollment."


def _build_enrollment_payload(result: Any) -> dict[str, object]:
    reference_image_count = max(0, int(getattr(result, "reference_image_count", 0) or 0))
    status = str(getattr(result, "status", "error") or "error")
    payload: dict[str, object] = {
        "status": status,
        "detail": _detail_for_status(status),
        "stores_locally": True,
        "profile_kind": "local_portrait_identity",
        "capture_source": "live_camera",
        "requires_confirmation": False,
        "saved": status in _SUCCESS_ENROLLMENT_STATUSES,
        "user_id": str(getattr(result, "user_id", _PRIMARY_PROFILE_USER_ID) or _PRIMARY_PROFILE_USER_ID),
        "display_name": _normalize_optional_text(getattr(result, "display_name", None)),
        "reference_id": _normalize_optional_text(getattr(result, "reference_id", None)),
        "reference_image_count": reference_image_count,
        "coverage_state": _coverage_state(reference_image_count),
        "quality_state": _quality_state(status, reference_image_count=reference_image_count),
        "guidance_hints": list(_guidance_hints_for_status(status, reference_image_count=reference_image_count)),
        "recommended_next_step": _recommended_next_step(status, reference_image_count=reference_image_count),
    }
    follow_up_tool = _suggested_follow_up_tool(status)
    if follow_up_tool:
        payload["suggested_follow_up_tool"] = follow_up_tool
    inspection_question = _suggested_inspection_question(status)
    if inspection_question:
        payload["suggested_inspection_question"] = inspection_question
    return payload


def _build_status_payload(summary: Any, observation: Any | None) -> dict[str, object]:
    reference_image_count = max(0, int(getattr(summary, "reference_image_count", 0) or 0))
    current_signal = None if observation is None else str(getattr(observation, "state", "") or "") or None
    current_match_user_id = None if observation is None else _normalize_optional_text(getattr(observation, "matched_user_id", None))
    current_match_display_name = None if observation is None else _normalize_optional_text(getattr(observation, "matched_user_display_name", None))
    if current_signal == "ambiguous_identity":
        current_match_user_id = None
        current_match_display_name = None

    current_confidence = None
    if observation is not None:
        fused_confidence = getattr(observation, "fused_confidence", None)
        current_confidence = fused_confidence if fused_confidence is not None else getattr(observation, "confidence", None)

    recommended_next_step = "enroll_now" if not bool(getattr(summary, "enrolled", False)) else "done"
    if current_signal:
        recommended_next_step = _recommended_next_step(current_signal, reference_image_count=reference_image_count)
    elif bool(getattr(summary, "enrolled", False)) and reference_image_count < 2:
        recommended_next_step = "capture_more_variation"

    if not bool(getattr(summary, "enrolled", False)):
        guidance_hints: tuple[str, ...] = ("single_face_in_frame", "face_camera", "steady_pose")
    else:
        guidance_hints = _guidance_hints_for_status(current_signal or "enrolled", reference_image_count=reference_image_count)
        if reference_image_count < 2 and "capture_more_angle_variation" not in guidance_hints:
            guidance_hints = tuple(guidance_hints) + ("capture_more_angle_variation",)

    payload: dict[str, object] = {
        "status": "ok",
        "detail": _detail_for_status(current_signal or ("enrolled" if bool(getattr(summary, "enrolled", False)) else "profile_unavailable")),
        "stores_locally": True,
        "profile_kind": "local_portrait_identity",
        "capture_source": "live_camera",
        "requires_confirmation": False,
        "enrolled": bool(getattr(summary, "enrolled", False)),
        "user_id": str(getattr(summary, "user_id", _PRIMARY_PROFILE_USER_ID) or _PRIMARY_PROFILE_USER_ID),
        "display_name": _normalize_optional_text(getattr(summary, "display_name", None)),
        "reference_image_count": reference_image_count,
        "coverage_state": _coverage_state(reference_image_count),
        "quality_state": _quality_state(current_signal or ("enrolled" if bool(getattr(summary, "enrolled", False)) else "profile_unavailable"), reference_image_count=reference_image_count),
        "recommended_next_step": recommended_next_step,
        "guidance_hints": list(guidance_hints),
    }
    if observation is not None:
        payload.update(
            {
                "current_signal": current_signal,
                "current_confidence": current_confidence,
                "temporal_state": _normalize_optional_text(getattr(observation, "temporal_state", None)),
                "temporal_observation_count": getattr(observation, "temporal_observation_count", None),
                "matched_user_id": current_match_user_id,
                "matched_user_display_name": current_match_display_name,
                "candidate_user_count": getattr(observation, "candidate_user_count", None),
                "capture_source_device": _normalize_optional_text(getattr(observation, "capture_source_device", None)),
            }
        )
        follow_up_tool = _suggested_follow_up_tool(current_signal or "")
        if follow_up_tool:
            payload["suggested_follow_up_tool"] = follow_up_tool
        inspection_question = _suggested_inspection_question(current_signal or "")
        if inspection_question:
            payload["suggested_inspection_question"] = inspection_question
    return payload


def handle_enroll_portrait_identity(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Capture one live camera view and add it to Twinr's local portrait store."""

    with _PORTRAIT_IDENTITY_LOCK:
        _emit_safe(owner, "portrait_identity_tool_call=true")
        confirmation_result = _confirmation_result(
            owner,
            arguments,
            action_label="save or update the local face profile",
        )
        if confirmation_result is not None:
            return confirmation_result

        try:
            provider = _get_portrait_provider(owner)
        except Exception:
            LOGGER.exception("Portrait-identity provider setup failed during enrollment.")
            return _portrait_identity_error(
                owner,
                status="unavailable",
                event_name="portrait_identity_enroll_failed",
                message="Portrait-identity enrollment could not start.",
                detail="Twinr could not access the local camera-based face profile right now.",
            )

        display_name = _normalize_optional_text(arguments.get("display_name")) if isinstance(arguments, dict) else None
        try:
            result = provider.capture_and_enroll_reference(
                user_id=_PRIMARY_PROFILE_USER_ID,
                display_name=display_name,
                source="tool_camera_capture",
            )
        except Exception:
            LOGGER.exception("Portrait-identity enrollment failed.")
            return _portrait_identity_error(
                owner,
                status="error",
                event_name="portrait_identity_enroll_failed",
                message="Portrait-identity enrollment failed.",
                detail="Twinr could not save the local face profile right now.",
            )

        payload = _build_enrollment_payload(result)
        _record_event_safe(
            owner,
            "portrait_identity_enrolled",
            "Portrait-identity tool captured or refreshed the local face profile.",
            status=payload["status"],
            reference_image_count=payload["reference_image_count"],
            recommended_next_step=payload["recommended_next_step"],
        )
        return payload


def handle_get_portrait_identity_status(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Read the current local portrait profile state and live camera match signal."""

    with _PORTRAIT_IDENTITY_LOCK:
        _emit_safe(owner, "portrait_identity_tool_call=true")
        try:
            provider = _get_portrait_provider(owner)
        except Exception:
            LOGGER.exception("Portrait-identity provider setup failed during status read.")
            return _portrait_identity_error(
                owner,
                status="unavailable",
                event_name="portrait_identity_status_failed",
                message="Portrait-identity status could not be read.",
                detail="Twinr could not access the local camera-based face profile right now.",
            )

        try:
            summary = provider.summary(user_id=_PRIMARY_PROFILE_USER_ID)
        except Exception:
            LOGGER.exception("Portrait-identity summary read failed.")
            return _portrait_identity_error(
                owner,
                status="error",
                event_name="portrait_identity_status_failed",
                message="Portrait-identity summary read failed.",
                detail="Twinr could not read the local face profile right now.",
            )

        if bool(getattr(summary, "enrolled", False)):
            confirmation_result = _confirmation_result(
                owner,
                arguments,
                action_label="read the local face profile status",
            )
            if confirmation_result is not None:
                confirmation_result["enrolled"] = True
                confirmation_result["reference_image_count"] = int(getattr(summary, "reference_image_count", 0) or 0)
                return confirmation_result

        observation = None
        if bool(getattr(summary, "enrolled", False)):
            try:
                observation = provider.observe()
            except Exception:
                LOGGER.exception("Portrait-identity live observation failed during status read.")
                observation = None

        payload = _build_status_payload(summary, observation)
        _record_event_safe(
            owner,
            "portrait_identity_status_read",
            "Portrait-identity tool read the local face profile status.",
            enrolled=payload["enrolled"],
            reference_image_count=payload["reference_image_count"],
            current_signal=payload.get("current_signal"),
            recommended_next_step=payload["recommended_next_step"],
        )
        return payload


def handle_reset_portrait_identity(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Delete Twinr's saved on-device portrait identity profile."""

    with _PORTRAIT_IDENTITY_LOCK:
        _emit_safe(owner, "portrait_identity_tool_call=true")
        try:
            provider = _get_portrait_provider(owner)
        except Exception:
            LOGGER.exception("Portrait-identity provider setup failed during reset.")
            return _portrait_identity_error(
                owner,
                status="unavailable",
                event_name="portrait_identity_reset_failed",
                message="Portrait-identity reset could not start.",
                detail="Twinr could not access the local camera-based face profile right now.",
            )

        try:
            summary = provider.summary(user_id=_PRIMARY_PROFILE_USER_ID)
        except Exception:
            LOGGER.exception("Portrait-identity summary read failed before reset.")
            return _portrait_identity_error(
                owner,
                status="error",
                event_name="portrait_identity_reset_failed",
                message="Portrait-identity reset could not read the existing profile.",
                detail="Twinr could not read the local face profile right now.",
            )

        if bool(getattr(summary, "enrolled", False)):
            confirmation_result = _confirmation_result(
                owner,
                arguments,
                action_label="delete the local face profile",
            )
            if confirmation_result is not None:
                confirmation_result["enrolled"] = True
                confirmation_result["reference_image_count"] = int(getattr(summary, "reference_image_count", 0) or 0)
                return confirmation_result

        try:
            result = provider.clear_identity_profile(user_id=_PRIMARY_PROFILE_USER_ID)
        except Exception:
            LOGGER.exception("Portrait-identity reset failed.")
            return _portrait_identity_error(
                owner,
                status="error",
                event_name="portrait_identity_reset_failed",
                message="Portrait-identity reset failed.",
                detail="Twinr could not delete the local face profile right now.",
            )

        deleted_reference_count = int(getattr(summary, "reference_image_count", 0) or 0)
        result_reference_count = int(getattr(result, "reference_image_count", 0) or 0)
        status = str(getattr(result, "status", "error") or "error")
        payload = {
            "status": status,
            "detail": _detail_for_status(status),
            "stores_locally": True,
            "profile_kind": "local_portrait_identity",
            "capture_source": "live_camera",
            "requires_confirmation": False,
            "enrolled": False,
            "user_id": str(getattr(result, "user_id", _PRIMARY_PROFILE_USER_ID) or _PRIMARY_PROFILE_USER_ID),
            "display_name": _normalize_optional_text(getattr(summary, "display_name", None)),
            "reference_image_count": result_reference_count,
            "deleted_reference_count": deleted_reference_count,
            "coverage_state": _coverage_state(result_reference_count),
            "quality_state": _quality_state(status, reference_image_count=result_reference_count),
            "recommended_next_step": "enroll_now",
            "guidance_hints": ["single_face_in_frame", "face_camera", "steady_pose"],
        }
        _record_event_safe(
            owner,
            "portrait_identity_reset",
            "Portrait-identity tool deleted the local face profile.",
            status=status,
            deleted_reference_count=deleted_reference_count,
        )
        return payload
