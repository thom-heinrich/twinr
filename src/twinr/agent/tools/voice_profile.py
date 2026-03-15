from __future__ import annotations

import logging  # AUDIT-FIX(#2,#3,#6): Add safe server-side diagnostics for degraded follow-up steps, config fallbacks, and response normalization.
import math
import threading  # AUDIT-FIX(#4): Serialize stateful voice-profile operations across concurrent callers.
from datetime import datetime, timezone  # AUDIT-FIX(#6): Normalize datetime values before returning them.
from numbers import Integral
from typing import Any

from twinr.agent.tools.support import require_current_turn_audio, require_sensitive_voice_confirmation

LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#2): Secondary telemetry failures must not turn a committed change into an apparent hard failure.
_DEFAULT_INPUT_SAMPLE_RATE = 24000  # AUDIT-FIX(#3): Invalid .env/runtime sample-rate values should fall back safely on-device.
_DEFAULT_AUDIO_CHANNELS = 1  # AUDIT-FIX(#3): Invalid channel config must not crash voice-profile flows.
_VOICE_PROFILE_LOCK = threading.RLock()  # AUDIT-FIX(#4): Prevent concurrent enroll/reset/status races against shared local state.


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0


def _coerce_positive_int(value: Any, *, default: int, field_name: str) -> int:
    if _is_positive_int(value):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = int(stripped, 10)
            except ValueError:
                parsed = 0
            if parsed > 0:
                return parsed
    LOGGER.warning("%s=%r is invalid; falling back to %s.", field_name, value, default)  # AUDIT-FIX(#3): Reject bool/garbage config without crashing the device.
    return default


def _coerce_non_negative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = int(stripped, 10)
            except ValueError:
                return default
            return max(0, parsed)
    return default


def _current_turn_audio_sample_rate(owner: Any) -> int:
    sample_rate = getattr(owner, "_current_turn_audio_sample_rate", None)
    if _is_positive_int(sample_rate):
        return int(sample_rate)
    configured = getattr(getattr(owner, "config", None), "openai_realtime_input_sample_rate", None)
    return _coerce_positive_int(
        configured,
        default=_DEFAULT_INPUT_SAMPLE_RATE,
        field_name="openai_realtime_input_sample_rate",
    )


def _audio_channels(owner: Any) -> int:
    configured = getattr(getattr(owner, "config", None), "audio_channels", None)
    return _coerce_positive_int(
        configured,
        default=_DEFAULT_AUDIO_CHANNELS,
        field_name="audio_channels",
    )


def _coerce_pcm16(audio_pcm: Any) -> bytes:
    if isinstance(audio_pcm, bytes):
        pcm = audio_pcm
    elif isinstance(audio_pcm, bytearray):
        pcm = bytes(audio_pcm)  # AUDIT-FIX(#5): Normalize mutable audio buffers before storage/assessment.
    elif isinstance(audio_pcm, memoryview):
        pcm = audio_pcm.tobytes()  # AUDIT-FIX(#5): Accept memoryview safely without leaking a non-bytes buffer downstream.
    else:
        raise TypeError("Current-turn audio must be PCM16 bytes.")
    if not pcm:
        raise ValueError("Current-turn audio is empty.")  # AUDIT-FIX(#5): Reject zero-length captures instead of enrolling garbage data.
    return pcm


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value  # AUDIT-FIX(#6): Preserve already-serialized timestamps unchanged.
    if not isinstance(value, datetime):
        return str(value)
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")  # AUDIT-FIX(#6): Return aware datetimes in unambiguous UTC ISO-8601 form.
    return value.isoformat()  # AUDIT-FIX(#6): Force naive datetimes to strings so response serialization cannot fail.


def _coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def _json_scalar(value: Any) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    numeric = _coerce_finite_float(value)
    if numeric is not None:
        return numeric
    if isinstance(value, datetime):
        return _serialize_datetime(value)
    return str(value)


def _safe_emit(owner: Any, message: str) -> None:
    try:
        owner.emit(message)
    except Exception:
        LOGGER.warning("Voice-profile telemetry emit failed.", exc_info=True)  # AUDIT-FIX(#2): Telemetry must never turn a successful state change into an apparent failure.


def _safe_record_event(owner: Any, event_name: str, message: str, **fields: object) -> None:
    safe_fields = {key: _json_scalar(value) for key, value in fields.items()}  # AUDIT-FIX(#6): Event fields should be scalar/serialized before they reach generic log sinks.
    try:
        owner._record_event(event_name, message, **safe_fields)
    except Exception:
        LOGGER.warning("Voice-profile event recording failed.", exc_info=True)  # AUDIT-FIX(#2): Audit logging is best-effort after the primary action commits.


def _append_warning(result: dict[str, object], warning: str) -> None:
    warnings = result.setdefault("warnings", [])  # AUDIT-FIX(#2): Preserve primary success while surfacing degraded follow-up behavior to the caller.
    if isinstance(warnings, list):
        warnings.append(warning)


def _current_runtime_snapshot(owner: Any) -> tuple[object, float | None]:
    runtime = getattr(owner, "runtime", None)
    status = _json_scalar(getattr(runtime, "user_voice_status", None))
    confidence = _coerce_finite_float(getattr(runtime, "user_voice_confidence", None))
    return status, confidence


def _clear_runtime_assessment(owner: Any) -> None:
    runtime = getattr(owner, "runtime", None)
    if runtime is None:
        raise AttributeError("owner.runtime is not available.")
    runtime.update_user_voice_assessment(
        status=None,
        confidence=None,
        checked_at=None,
    )  # AUDIT-FIX(#7): Centralize clearing so stale live-assessment state does not survive failed or skipped persistence.


def _update_runtime_assessment(owner: Any, assessment: Any) -> None:
    runtime = getattr(owner, "runtime", None)
    if runtime is None:
        raise AttributeError("owner.runtime is not available.")
    if getattr(assessment, "should_persist", False):
        runtime.update_user_voice_assessment(
            status=getattr(assessment, "status", None),
            confidence=getattr(assessment, "confidence", None),
            checked_at=getattr(assessment, "checked_at", None),
        )
        return
    _clear_runtime_assessment(owner)  # AUDIT-FIX(#7): A non-persistent assessment must clear the old live snapshot rather than leave stale values behind.


def _voice_profile_error(owner: Any, *, event_name: str, message: str, detail: str) -> dict[str, object]:
    _safe_emit(owner, "voice_profile_tool_call=true")  # AUDIT-FIX(#2): Failed attempts still need observability without depending on exception bubbling.
    _safe_record_event(owner, event_name, message)
    return {
        "status": "error",
        "detail": detail,
    }


def handle_enroll_voice_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    with _VOICE_PROFILE_LOCK:  # AUDIT-FIX(#4): Summary/check/enroll/assessment/runtime update must execute as one serialized critical section.
        try:
            summary = owner.voice_profile_monitor.summary()
        except Exception:
            LOGGER.exception("Voice-profile summary read failed before enrollment.")
            return _voice_profile_error(
                owner,
                event_name="voice_profile_enroll_failed",
                message="Realtime tool could not read the local voice-profile state before enrollment.",
                detail="The device could not check the voice profile right now. Please try again.",
            )

        action_label = "replace the saved voice profile" if summary.enrolled else "save a new voice profile"
        require_sensitive_voice_confirmation(owner, arguments, action_label=action_label)  # AUDIT-FIX(#1): First-time enrollment is sensitive too and must not happen without explicit confirmation.
        audio_pcm = require_current_turn_audio(owner)

        try:
            audio_pcm_bytes = _coerce_pcm16(audio_pcm)
        except (TypeError, ValueError):
            LOGGER.exception("Invalid current-turn audio for voice-profile enrollment.")
            return _voice_profile_error(
                owner,
                event_name="voice_profile_enroll_failed",
                message="Realtime tool rejected invalid current-turn audio for voice-profile enrollment.",
                detail="The device did not capture enough clear speech to save the voice profile. Please speak again.",
            )

        sample_rate = _current_turn_audio_sample_rate(owner)
        channels = _audio_channels(owner)

        try:
            template = owner.voice_profile_monitor.enroll_pcm16(
                audio_pcm_bytes,
                sample_rate=sample_rate,
                channels=channels,
            )
        except Exception:
            LOGGER.exception("Voice-profile enrollment failed.")
            return _voice_profile_error(
                owner,
                event_name="voice_profile_enroll_failed",
                message="Realtime tool failed to store the local voice profile.",
                detail="The device could not save the voice profile right now. Please try again.",
            )

        sample_count = _coerce_non_negative_int(getattr(template, "sample_count", 0))
        average_duration_ms = _coerce_non_negative_int(getattr(template, "average_duration_ms", 0))
        result: dict[str, object] = {
            "status": "enrolled",
            "sample_count": sample_count,
            "average_duration_ms": average_duration_ms,
            "detail": "Local voice profile stored from the current spoken turn.",
        }

        try:
            assessment = owner.voice_profile_monitor.assess_pcm16(
                audio_pcm_bytes,
                sample_rate=sample_rate,
                channels=channels,
            )
        except Exception:
            LOGGER.exception("Voice-profile assessment failed after enrollment.")
            try:
                _clear_runtime_assessment(owner)
            except Exception:
                LOGGER.exception("Runtime voice assessment clear failed after assessment failure.")
            _append_warning(
                result,
                "The voice profile was saved, but the live voice check could not be completed.",
            )  # AUDIT-FIX(#2): Preserve success when follow-up assessment fails after the profile is already stored.
        else:
            result["current_signal"] = _json_scalar(getattr(assessment, "status", None))
            result["current_confidence"] = _coerce_finite_float(getattr(assessment, "confidence", None))
            try:
                _update_runtime_assessment(owner, assessment)
            except Exception:
                LOGGER.exception("Runtime voice assessment update failed after enrollment.")
                _append_warning(
                    result,
                    "The voice profile was saved, but the device could not update its live voice status.",
                )  # AUDIT-FIX(#2): Do not surface a hard failure after the profile has already changed.

    _safe_emit(owner, "voice_profile_tool_call=true")
    _safe_emit(owner, f"voice_profile_samples={sample_count}")
    _safe_record_event(
        owner,
        "voice_profile_enrolled",
        "Realtime tool stored or refreshed the local voice profile.",
        sample_count=sample_count,
        average_duration_ms=average_duration_ms,
        current_signal=result.get("current_signal"),
        current_confidence=result.get("current_confidence"),
    )
    return result


def handle_get_voice_profile_status(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    del arguments
    with _VOICE_PROFILE_LOCK:  # AUDIT-FIX(#4): Status must observe a consistent snapshot relative to concurrent enroll/reset operations.
        try:
            summary = owner.voice_profile_monitor.summary()
        except Exception:
            LOGGER.exception("Voice-profile status read failed.")
            return _voice_profile_error(
                owner,
                event_name="voice_profile_status_failed",
                message="Realtime tool could not read the local voice-profile status.",
                detail="The device could not read the voice profile status right now. Please try again.",
            )

        current_signal, current_confidence = _current_runtime_snapshot(owner)
        enrolled = bool(getattr(summary, "enrolled", False))
        sample_count = _coerce_non_negative_int(getattr(summary, "sample_count", 0))
        average_duration_ms = _coerce_non_negative_int(getattr(summary, "average_duration_ms", 0))
        updated_at = _serialize_datetime(getattr(summary, "updated_at", None))

    _safe_emit(owner, "voice_profile_tool_call=true")
    _safe_record_event(
        owner,
        "voice_profile_status_read",
        "Realtime tool read the local voice-profile status.",
        enrolled=enrolled,
        sample_count=sample_count,
        updated_at=updated_at,
    )
    return {
        "status": "ok",
        "enrolled": enrolled,
        "sample_count": sample_count,
        "updated_at": updated_at,  # AUDIT-FIX(#6): Return a JSON-safe, unambiguous timestamp payload.
        "average_duration_ms": average_duration_ms,
        "current_signal": current_signal,
        "current_confidence": current_confidence,
    }


def handle_reset_voice_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    with _VOICE_PROFILE_LOCK:  # AUDIT-FIX(#4): Reset and runtime clear must not interleave with concurrent enrollment/status reads.
        require_sensitive_voice_confirmation(owner, arguments, action_label="delete the saved voice profile")

        try:
            summary = owner.voice_profile_monitor.reset()
        except Exception:
            LOGGER.exception("Voice-profile reset failed.")
            return _voice_profile_error(
                owner,
                event_name="voice_profile_reset_failed",
                message="Realtime tool failed to delete the local voice profile.",
                detail="The device could not delete the saved voice profile right now. Please try again.",
            )

        enrolled = bool(getattr(summary, "enrolled", False))
        sample_count = _coerce_non_negative_int(getattr(summary, "sample_count", 0))
        result: dict[str, object] = {
            "status": "reset",
            "enrolled": enrolled,
            "sample_count": sample_count,
        }

        try:
            _clear_runtime_assessment(owner)
        except Exception:
            LOGGER.exception("Runtime voice assessment clear failed after voice-profile reset.")
            _append_warning(
                result,
                "The saved voice profile was deleted, but the device could not clear its live voice status.",
            )  # AUDIT-FIX(#2): Reset success must survive secondary cleanup failures.

    _safe_emit(owner, "voice_profile_tool_call=true")
    _safe_record_event(
        owner,
        "voice_profile_reset",
        "Realtime tool deleted the local voice profile.",
        enrolled=enrolled,
        sample_count=sample_count,
    )
    return result