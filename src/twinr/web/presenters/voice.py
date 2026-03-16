"""Build voice-profile page context and guarded capture helpers.

This module keeps the voice-profile web flow thin by converting runtime state,
voice assessments, and capture failures into stable template-ready data.
"""

from __future__ import annotations

import logging  # AUDIT-FIX(#2,#3,#5,#6,#7): Log degraded states, invalid inputs, and hardware failures without exposing raw internals.
import math  # AUDIT-FIX(#4,#6): Validate numeric inputs defensively.
import threading  # AUDIT-FIX(#1): Serialize in-process microphone capture attempts.

from twinr.agent.base_agent import RuntimeSnapshot, TwinrConfig
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.hardware.voice_profile import VoiceAssessment, VoiceProfileMonitor
from twinr.ops import loop_lock_owner

_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#2,#3,#5,#6,#7): Centralize diagnostic logging for degraded paths.
_CAPTURE_LOCK = threading.Lock()  # AUDIT-FIX(#1): Prevent concurrent in-process voice capture on shared audio hardware.

_DEFAULT_SPEECH_PAUSE_MS = 1200  # AUDIT-FIX(#4): Safe fallback if .env/config contains an invalid pause value.
_MIN_SPEECH_PAUSE_MS = 250  # AUDIT-FIX(#4): Avoid cutting off users because of unrealistically small pause windows.
_MAX_SPEECH_PAUSE_MS = 15000  # AUDIT-FIX(#4): Avoid effectively unbounded blocking capture from huge pause windows.


def _clean_text(value: object, *, default: str) -> str:
    """Return a trimmed string or the provided fallback text."""

    # AUDIT-FIX(#6): Normalize persisted or malformed fields before rendering them into UI strings.
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalized_confidence(value: object) -> float | None:
    """Return a finite confidence score in the inclusive ``0.0`` to ``1.0`` range."""

    # AUDIT-FIX(#6): Accept only finite confidence scores in the expected 0.0-1.0 range.
    if value is None or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        _LOGGER.warning("Ignoring non-numeric confidence value: %r", value)
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        _LOGGER.warning("Ignoring out-of-range confidence value: %r", value)
        return None
    return confidence


def _validated_speech_pause_ms(value: object) -> int:
    """Clamp speech-pause configuration into safe recorder bounds."""

    # AUDIT-FIX(#4): Clamp configuration-derived pause values to safe operational bounds.
    if value is None or isinstance(value, bool):
        _LOGGER.warning(
            "speech_pause_ms is missing or invalid (%r); falling back to %d ms.",
            value,
            _DEFAULT_SPEECH_PAUSE_MS,
        )
        return _DEFAULT_SPEECH_PAUSE_MS
    try:
        pause_ms = int(value)
    except (TypeError, ValueError, OverflowError):
        _LOGGER.warning(
            "speech_pause_ms=%r is not an integer; falling back to %d ms.",
            value,
            _DEFAULT_SPEECH_PAUSE_MS,
        )
        return _DEFAULT_SPEECH_PAUSE_MS
    if pause_ms < _MIN_SPEECH_PAUSE_MS:
        _LOGGER.warning(
            "speech_pause_ms=%r is below the safe minimum; clamping to %d ms.",
            value,
            _MIN_SPEECH_PAUSE_MS,
        )
        return _MIN_SPEECH_PAUSE_MS
    if pause_ms > _MAX_SPEECH_PAUSE_MS:
        _LOGGER.warning(
            "speech_pause_ms=%r is above the safe maximum; clamping to %d ms.",
            value,
            _MAX_SPEECH_PAUSE_MS,
        )
        return _MAX_SPEECH_PAUSE_MS
    return pause_ms


def _human_join(items: list[str]) -> str:
    """Join short phrases into a human-readable list."""

    # AUDIT-FIX(#7): Keep user-visible block reasons human-readable instead of surfacing implementation jargon.
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _coerce_audio_bytes(payload: object) -> bytes:
    """Convert recorder output into raw bytes or raise a stable runtime error."""

    # AUDIT-FIX(#4): Fail fast on empty or invalid recorder payloads instead of passing corrupted data downstream.
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, memoryview):
        return payload.tobytes()
    try:
        return bytes(payload)
    except (TypeError, ValueError) as exc:
        _LOGGER.error("Recorder returned an unexpected payload type: %s", type(payload).__name__)
        raise RuntimeError("Voice sample capture returned an invalid audio payload.") from exc


def _close_if_possible(resource: object) -> None:
    """Close a resource when it exposes a callable ``close`` method."""

    # AUDIT-FIX(#5): Release recorder resources deterministically when a close() method is available.
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        _LOGGER.warning("Failed to close %s cleanly.", type(resource).__name__, exc_info=True)


def _voice_profile_page_context(
    config: TwinrConfig,
    snapshot: RuntimeSnapshot,
    *,
    action_result: dict[str, str] | None = None,
    action_error: str | None = None,
) -> dict[str, object]:
    """Build the template context for the voice-profile page.

    Args:
        config: Active Twinr configuration.
        snapshot: Runtime snapshot used for live voice status.
        action_result: Optional result payload from enroll, verify, or reset.
        action_error: Optional user-facing action error message.

    Returns:
        Dictionary containing voice profile summary data, live status labels,
        capture availability, and any action feedback for the template.
    """

    action_error_value = action_error  # AUDIT-FIX(#2,#3): Preserve upstream errors while allowing graceful local degradation.
    profile_summary: object | None = None
    try:
        monitor = VoiceProfileMonitor.from_config(config)  # AUDIT-FIX(#2): Guard monitor construction against config/state failures.
        profile_summary = monitor.summary()  # AUDIT-FIX(#2): Guard summary generation so page rendering degrades instead of crashing.
    except Exception:
        _LOGGER.exception("Failed to load voice profile summary.")
        if action_error_value is None:
            action_error_value = "Voice profile information is temporarily unavailable."

    try:
        capture_block_reason = _voice_profile_capture_block_reason(config)  # AUDIT-FIX(#3): Do not let loop-state inspection errors break page rendering.
    except Exception:
        _LOGGER.exception("Failed to determine whether voice capture is blocked.")
        capture_block_reason = "Voice capture is temporarily unavailable."
        if action_error_value is None:
            action_error_value = "Voice capture is temporarily unavailable."

    return {
        "profile_summary": profile_summary,
        "snapshot": snapshot,
        "voice_snapshot_label": _voice_snapshot_label(snapshot),
        "capture_block_reason": capture_block_reason,
        "action_result": action_result,
        "action_error": action_error_value,
    }


def _voice_snapshot_label(snapshot: RuntimeSnapshot) -> str:
    """Return a compact live voice-status label for the page header."""

    status = _clean_text(  # AUDIT-FIX(#6): Tolerate malformed or legacy snapshot values loaded from file-backed state.
        getattr(snapshot, "user_voice_status", None),
        default="",
    )
    if not status:
        return "No recent live voice check."
    label = status.replace("_", " ")
    confidence = _normalized_confidence(getattr(snapshot, "user_voice_confidence", None))  # AUDIT-FIX(#6): Avoid rendering invalid percentages.
    if confidence is None:
        return label
    return f"{label} ({confidence * 100:.0f}%)"


def _voice_profile_capture_block_reason(config: TwinrConfig) -> str | None:
    """Return why voice capture is currently blocked, if it is blocked."""

    busy: list[str] = []
    for loop_name, label in (("hardware-loop", "hardware loop"), ("realtime-loop", "realtime loop")):
        try:
            owner = loop_lock_owner(config, loop_name)  # AUDIT-FIX(#3): Fail closed if lock-owner inspection itself breaks.
        except Exception:
            _LOGGER.exception("Failed to inspect lock owner for %s.", loop_name)
            return "Voice capture is temporarily unavailable."
        if owner is not None:
            _LOGGER.warning(  # AUDIT-FIX(#7): Keep PIDs in logs, not in user-facing strings.
                "Voice capture blocked by %s (owner pid %s).",
                loop_name,
                owner,
            )
            busy.append(label)
    if not busy:
        return None
    joined = _human_join(busy)  # AUDIT-FIX(#7): Produce a readable block reason without PID jargon.
    return f"Stop the active {joined} before capturing a voice profile sample."


def _capture_voice_profile_sample(config: TwinrConfig) -> bytes:
    """Capture one guarded voice-profile sample from the configured microphone.

    Raises:
        RuntimeError: If capture is blocked, the recorder fails, or the
            resulting payload is empty or invalid.
    """

    blocked_reason = _voice_profile_capture_block_reason(config)
    if blocked_reason:
        raise RuntimeError(blocked_reason)

    if not _CAPTURE_LOCK.acquire(blocking=False):  # AUDIT-FIX(#1): Reject concurrent in-process captures on shared hardware.
        raise RuntimeError("A voice profile capture is already in progress. Please wait and try again.")

    recorder: object | None = None
    try:
        blocked_reason = _voice_profile_capture_block_reason(config)  # AUDIT-FIX(#1): Re-check after taking the local capture lock to reduce TOCTOU exposure.
        if blocked_reason:
            raise RuntimeError(blocked_reason)

        pause_ms = _validated_speech_pause_ms(getattr(config, "speech_pause_ms", None))  # AUDIT-FIX(#4): Sanitize config before passing it into hardware code.

        try:
            recorder = SilenceDetectedRecorder.from_config(config)  # AUDIT-FIX(#5): Keep hardware initialization inside a guarded block.
            audio = recorder.record_until_pause(pause_ms=pause_ms)  # AUDIT-FIX(#5): Convert low-level recorder failures into stable caller-facing errors.
        except RuntimeError:
            raise
        except Exception as exc:
            _LOGGER.exception("Voice profile capture failed.")
            raise RuntimeError("Voice sample capture failed. Please try again.") from exc

        audio_bytes = _coerce_audio_bytes(audio)  # AUDIT-FIX(#4): Validate recorder output type before downstream use.
        if not audio_bytes:
            raise RuntimeError("Voice sample capture returned no audio data.")
        return audio_bytes
    finally:
        _close_if_possible(recorder)
        _CAPTURE_LOCK.release()


def _voice_action_result(assessment: VoiceAssessment) -> dict[str, str]:
    """Convert a voice assessment into the standard web action-result payload."""

    assessment_status = _clean_text(getattr(assessment, "status", None), default="").lower()  # AUDIT-FIX(#6): Normalize status values before mapping them.
    status = "warn"
    if assessment_status == "likely_user":
        status = "ok"
    elif assessment_status in {"disabled", "not_enrolled"}:
        status = "muted"

    detail = _clean_text(getattr(assessment, "detail", None), default="Voice assessment completed.")  # AUDIT-FIX(#6): Avoid leaking None or blank details into UI.
    confidence = _normalized_confidence(getattr(assessment, "confidence", None))  # AUDIT-FIX(#6): Ignore malformed confidence values safely.
    if confidence is not None:
        detail = f"{detail} Confidence {confidence * 100:.0f}%."

    return {
        "status": status,
        "title": _clean_text(getattr(assessment, "label", None), default="Voice assessment"),  # AUDIT-FIX(#6): Guarantee a stable title string.
        "detail": detail,
    }
