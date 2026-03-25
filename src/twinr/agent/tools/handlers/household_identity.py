"""Handle shared household-identity tool calls for Twinr sessions.

This handler exposes one common local-identity tool surface for face
enrollment, voice enrollment, live status checks, and explicit confirm/deny
feedback. It keeps the model on one coherent tool instead of spreading
household identity logic across separate face and voice flows.
"""

from __future__ import annotations

import logging
import threading
from numbers import Integral
from typing import Any

from twinr.hardware.household_identity import (
    HouseholdIdentityManager,
    HouseholdIdentityMemberStatus,
    HouseholdIdentityObservation,
)

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import (
    ArgumentValidationError,
    SensitiveActionConfirmationRequired,
    require_current_turn_audio,
    require_sensitive_voice_confirmation,
)


LOGGER = logging.getLogger(__name__)
_HOUSEHOLD_IDENTITY_LOCK = threading.RLock()
_ALLOWED_ACTIONS = frozenset(
    {
        "status",
        "enroll_face",
        "enroll_voice",
        "confirm_identity",
        "deny_identity",
    }
)


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_optional_text(value: object | None) -> str | None:
    text = _normalize_text(value).strip()
    return text or None


def _normalize_action(arguments: dict[str, object]) -> str:
    action = _normalize_text(arguments.get("action")).strip().lower()
    if action in _ALLOWED_ACTIONS:
        return action
    raise ArgumentValidationError(
        "action must be one of: status, enroll_face, enroll_voice, confirm_identity, deny_identity"
    )


def _emit_safe(owner: Any, message: str) -> None:
    emit_best_effort(
        owner,
        message,
        logger=LOGGER,
        failure_message="Household-identity telemetry emit failed.",
    )


def _record_event_safe(owner: Any, event_name: str, message: str, **fields: object) -> None:
    record_event_best_effort(
        owner,
        event_name,
        message,
        dict(fields),
        logger=LOGGER,
        failure_message="Household-identity event recording failed.",
    )


def _error_payload(
    owner: Any,
    *,
    action: str,
    status: str,
    event_name: str,
    message: str,
    detail: str,
) -> dict[str, object]:
    _emit_safe(owner, "household_identity_tool_call=true")
    _record_event_safe(owner, event_name, message, action=action, status=status)
    return {
        "status": status,
        "action": action,
        "detail": detail,
        "stores_locally": True,
        "requires_confirmation": False,
        "profile_kind": "local_household_identity",
    }


def _confirmation_result(
    owner: Any,
    arguments: dict[str, object],
    *,
    action: str,
    action_label: str,
) -> dict[str, object] | None:
    try:
        require_sensitive_voice_confirmation(owner, arguments, action_label=action_label)
    except ArgumentValidationError as exc:
        return _error_payload(
            owner,
            action=action,
            status="error",
            event_name="household_identity_argument_error",
            message="Household-identity tool arguments were invalid.",
            detail=str(exc),
        )
    except SensitiveActionConfirmationRequired as exc:
        return {
            "status": "confirmation_required",
            "action": action,
            "detail": str(exc),
            "stores_locally": True,
            "requires_confirmation": True,
            "profile_kind": "local_household_identity",
        }
    return None


def _get_manager(owner: Any) -> HouseholdIdentityManager:
    manager = getattr(owner, "household_identity_manager", None)
    if manager is not None:
        return manager
    manager = getattr(owner, "_household_identity_tool_manager", None)
    if manager is not None:
        return manager
    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("Twinr configuration is unavailable.")
    camera = getattr(owner, "camera", None)
    if camera is None:
        raise RuntimeError("The camera is unavailable right now.")
    manager = HouseholdIdentityManager.from_config(
        config,
        camera=camera,
        camera_lock=getattr(owner, "_camera_lock", None),
    )
    setattr(owner, "_household_identity_tool_manager", manager)
    return manager


def _audio_context(owner: Any) -> tuple[bytes | None, int | None, int | None]:
    audio_pcm = getattr(owner, "_current_turn_audio_pcm", None)
    if not audio_pcm:
        return None, None, None
    sample_rate = getattr(owner, "_current_turn_audio_sample_rate", None)
    channels = getattr(getattr(owner, "config", None), "audio_channels", None)
    if isinstance(sample_rate, Integral) and not isinstance(sample_rate, bool):
        normalized_sample_rate = int(sample_rate)
    elif isinstance(sample_rate, str):
        try:
            normalized_sample_rate = int(sample_rate)
        except ValueError:
            normalized_sample_rate = None
    else:
        normalized_sample_rate = None
    if isinstance(channels, Integral) and not isinstance(channels, bool):
        normalized_channels = int(channels)
    elif isinstance(channels, str):
        try:
            normalized_channels = int(channels)
        except ValueError:
            normalized_channels = None
    else:
        normalized_channels = None
    return bytes(audio_pcm), normalized_sample_rate, normalized_channels


def _member_payload(member: HouseholdIdentityMemberStatus) -> dict[str, object]:
    return {
        "user_id": member.user_id,
        "display_name": member.display_name,
        "primary_user": member.primary_user,
        "portrait_reference_count": member.portrait_reference_count,
        "voice_sample_count": member.voice_sample_count,
        "confirm_count": member.confirm_count,
        "deny_count": member.deny_count,
        "quality_score": member.quality.score,
        "quality_state": member.quality.state,
        "recommended_next_step": member.quality.recommended_next_step,
        "guidance_hints": list(member.quality.guidance_hints),
    }


def _observation_payload(observation: HouseholdIdentityObservation | None) -> dict[str, object] | None:
    if observation is None:
        return None
    payload: dict[str, object] = {
        "state": observation.state,
        "matched_user_id": observation.matched_user_id,
        "matched_user_display_name": observation.matched_user_display_name,
        "confidence": observation.confidence,
        "modalities": list(observation.modalities),
        "temporal_state": observation.temporal_state,
        "session_support_ratio": observation.session_support_ratio,
        "session_observation_count": observation.session_observation_count,
        "policy_recommendation": observation.policy_recommendation,
        "block_reason": observation.block_reason,
    }
    if observation.voice_assessment is not None:
        payload["voice_signal"] = {
            "status": observation.voice_assessment.status,
            "confidence": observation.voice_assessment.confidence,
            "matched_user_id": observation.voice_assessment.matched_user_id,
            "matched_user_display_name": observation.voice_assessment.matched_user_display_name,
        }
    if observation.portrait_observation is not None:
        payload["face_signal"] = {
            "status": observation.portrait_observation.state,
            "confidence": (
                observation.portrait_observation.fused_confidence
                if observation.portrait_observation.fused_confidence is not None
                else observation.portrait_observation.confidence
            ),
            "matched_user_id": observation.portrait_observation.matched_user_id,
            "matched_user_display_name": observation.portrait_observation.matched_user_display_name,
        }
    return payload


def _selected_user_id(arguments: dict[str, object], *, default: str | None = None) -> str | None:
    user_id = _normalize_optional_text(arguments.get("user_id"))
    if user_id is not None:
        return user_id
    display_name = _normalize_optional_text(arguments.get("display_name"))
    if display_name is None:
        return default
    return display_name


def handle_manage_household_identity(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Run one shared local household-identity action."""

    with _HOUSEHOLD_IDENTITY_LOCK:
        _emit_safe(owner, "household_identity_tool_call=true")
        try:
            action = _normalize_action(arguments)
        except ArgumentValidationError as exc:
            return _error_payload(
                owner,
                action="invalid",
                status="error",
                event_name="household_identity_argument_error",
                message="Household-identity tool arguments were invalid.",
                detail=str(exc),
            )

        try:
            manager = _get_manager(owner)
        except Exception:
            LOGGER.exception("Household-identity manager setup failed.")
            return _error_payload(
                owner,
                action=action,
                status="unavailable",
                event_name="household_identity_unavailable",
                message="Household-identity manager could not be created.",
                detail="Twinr could not access the local household identity system right now.",
            )

        if action == "status":
            audio_pcm, sample_rate, channels = _audio_context(owner)
            try:
                status = manager.status(
                    audio_pcm=audio_pcm,
                    sample_rate=sample_rate,
                    channels=channels,
                )
            except Exception:
                LOGGER.exception("Household-identity status failed.")
                return _error_payload(
                    owner,
                    action=action,
                    status="error",
                    event_name="household_identity_status_failed",
                    message="Household-identity status read failed.",
                    detail="Twinr could not read the local household identity status right now.",
                )
            result = {
                "status": "ok",
                "action": action,
                "stores_locally": True,
                "requires_confirmation": False,
                "profile_kind": "local_household_identity",
                "primary_user_id": status.primary_user_id,
                "member_count": len(status.members),
                "members": [_member_payload(member) for member in status.members],
                "current_observation": _observation_payload(status.current_observation),
            }
            _record_event_safe(
                owner,
                "household_identity_status_read",
                "Household identity status was read.",
                action=action,
                member_count=len(status.members),
                observation_state=None if status.current_observation is None else status.current_observation.state,
            )
            return result

        if action == "enroll_face":
            confirmation_result = _confirmation_result(
                owner,
                arguments,
                action=action,
                action_label="save or update a local household face identity",
            )
            if confirmation_result is not None:
                return confirmation_result
            user_id = _selected_user_id(arguments, default=manager.primary_user_id)
            display_name = _normalize_optional_text(arguments.get("display_name"))
            try:
                enrollment, member = manager.enroll_face(
                    user_id=user_id,
                    display_name=display_name,
                )
            except Exception:
                LOGGER.exception("Household face enrollment failed.")
                return _error_payload(
                    owner,
                    action=action,
                    status="error",
                    event_name="household_identity_enroll_face_failed",
                    message="Household face enrollment failed.",
                    detail="Twinr could not save the local household face identity right now.",
                )
            hints: list[str] = []
            if enrollment.status in {"no_face_detected", "ambiguous_face_count"}:
                hints.extend(("single_face_in_frame", "face_camera", "steady_pose"))
            elif enrollment.status in {"capture_unavailable", "decode_failed", "embedding_failed"}:
                hints.extend(("check_camera_view", "better_light", "retry_later"))
            elif member is not None:
                hints.extend(member.quality.guidance_hints)
            result = {
                "status": enrollment.status,
                "action": action,
                "detail": enrollment.detail or "Twinr updated the local household face identity state.",
                "stores_locally": True,
                "requires_confirmation": False,
                "profile_kind": "local_household_identity",
                "saved": enrollment.status in {"enrolled", "duplicate_reference"},
                "user_id": enrollment.user_id,
                "display_name": enrollment.display_name,
                "reference_id": enrollment.reference_id,
                "portrait_reference_count": enrollment.reference_image_count,
                "member": None if member is None else _member_payload(member),
                "recommended_next_step": (
                    "capture_more_face_angles"
                    if member is not None and member.quality.recommended_next_step == "capture_more_face_angles"
                    else ("retry_capture" if enrollment.status not in {"enrolled", "duplicate_reference"} else "done")
                ),
                "guidance_hints": list(dict.fromkeys(hints)),
            }
            if enrollment.status in {"no_face_detected", "ambiguous_face_count", "decode_failed", "embedding_failed"}:
                result["suggested_follow_up_tool"] = "inspect_camera"
                result["suggested_inspection_question"] = (
                    "Check whether exactly one face is centered, clear, and well lit for local household identity enrollment."
                )
            _record_event_safe(
                owner,
                "household_identity_enroll_face",
                "Household face enrollment finished.",
                action=action,
                status=enrollment.status,
                user_id=enrollment.user_id,
            )
            return result

        if action == "enroll_voice":
            confirmation_result = _confirmation_result(
                owner,
                arguments,
                action=action,
                action_label="save or update a local household voice identity",
            )
            if confirmation_result is not None:
                return confirmation_result
            try:
                audio_pcm = require_current_turn_audio(owner)
            except Exception as exc:
                return _error_payload(
                    owner,
                    action=action,
                    status="error",
                    event_name="household_identity_enroll_voice_failed",
                    message="Household voice enrollment lacked current-turn audio.",
                    detail=str(exc),
                )
            _audio_pcm, sample_rate, channels = _audio_context(owner)
            if sample_rate is None or channels is None:
                return _error_payload(
                    owner,
                    action=action,
                    status="error",
                    event_name="household_identity_enroll_voice_failed",
                    message="Household voice enrollment lacked audio metadata.",
                    detail="Twinr could not determine the current audio format for local voice enrollment.",
                )
            user_id = _selected_user_id(arguments, default=manager.primary_user_id)
            display_name = _normalize_optional_text(arguments.get("display_name"))
            try:
                summary, member = manager.enroll_voice(
                    bytes(audio_pcm),
                    sample_rate=sample_rate,
                    channels=channels,
                    user_id=user_id,
                    display_name=display_name,
                )
            except Exception as exc:
                LOGGER.exception("Household voice enrollment failed.")
                return _error_payload(
                    owner,
                    action=action,
                    status="error",
                    event_name="household_identity_enroll_voice_failed",
                    message="Household voice enrollment failed.",
                    detail=str(exc),
                )
            hints = []
            if member is not None:
                hints.extend(member.quality.guidance_hints)
            if not hints:
                hints.extend(("speak_clear_sentence", "quiet_room", "one_speaker_only"))
            result = {
                "status": "enrolled",
                "action": action,
                "detail": "Twinr saved the current spoken turn as part of the local household voice identity.",
                "stores_locally": True,
                "requires_confirmation": False,
                "profile_kind": "local_household_identity",
                "saved": True,
                "user_id": summary.user_id,
                "display_name": summary.display_name,
                "voice_sample_count": summary.sample_count,
                "average_duration_ms": summary.average_duration_ms,
                "member": None if member is None else _member_payload(member),
                "recommended_next_step": None if member is None else member.quality.recommended_next_step,
                "guidance_hints": list(dict.fromkeys(hints)),
            }
            _record_event_safe(
                owner,
                "household_identity_enroll_voice",
                "Household voice enrollment finished.",
                action=action,
                user_id=summary.user_id,
                voice_sample_count=summary.sample_count,
            )
            return result

        feedback_outcome = "confirm" if action == "confirm_identity" else "deny"
        user_id = _selected_user_id(arguments)
        display_name = _normalize_optional_text(arguments.get("display_name"))
        try:
            event, member = manager.record_feedback(
                outcome=feedback_outcome,
                user_id=user_id,
                display_name=display_name,
            )
        except Exception as exc:
            LOGGER.exception("Household identity feedback failed.")
            return _error_payload(
                owner,
                action=action,
                status="error",
                event_name="household_identity_feedback_failed",
                message="Household identity feedback failed.",
                detail=str(exc),
            )
        result = {
            "status": "updated",
            "action": action,
            "detail": (
                "Twinr recorded that the current local identity match was correct."
                if feedback_outcome == "confirm"
                else "Twinr recorded that the current local identity match was incorrect."
            ),
            "stores_locally": True,
            "requires_confirmation": False,
            "profile_kind": "local_household_identity",
            "feedback": feedback_outcome,
            "user_id": event.user_id,
            "display_name": event.display_name,
            "modalities": list(event.modalities),
            "member": None if member is None else _member_payload(member),
        }
        _record_event_safe(
            owner,
            "household_identity_feedback",
            "Household identity feedback was recorded.",
            action=action,
            user_id=event.user_id,
            feedback=feedback_outcome,
        )
        return result
