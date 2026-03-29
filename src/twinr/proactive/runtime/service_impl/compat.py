"""Compatibility helpers for the proactive runtime service refactor.

Purpose: keep stable helper functions, constants, and telemetry formatting in a
small shared module so the public ``service.py`` wrapper can re-export the
historic helper surface unchanged.

Invariants: helper behavior, ops-event keys, and fail-closed/runtime-warning
semantics must stay identical to the legacy implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import logging
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import (
    AudioCaptureReadinessProbe,
    capture_device_identity,
    resolve_capture_device,
)
from twinr.ops.locks import loop_lock_owner

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime.runtime import TwinrRuntime

_VISION_REVIEW_FAIL_OPEN_TRIGGERS = frozenset(
    {"possible_fall", "floor_stillness", "distress_possible"}
)
_DEFAULT_CLOSE_JOIN_TIMEOUT_S = 5.0
_DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES = frozenset({"waiting", "listening", "processing", "answering"})
_DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES = frozenset({"error"})
_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S = 2.0
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[4]

_LOGGER = logging.getLogger("twinr.proactive.runtime.service")


def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    """Emit one telemetry line while suppressing sink failures."""

    if emit is None:
        return
    try:
        emit(line)
    except Exception:
        _LOGGER.warning("Proactive emit sink failed.", exc_info=True)
        return


def _exception_text(error: BaseException | object, *, limit: int = 240) -> str:
    """Normalize one exception payload into bounded log-safe text."""

    raw = str(error) if not isinstance(error, BaseException) else (str(error) or error.__class__.__name__)
    text = " ".join(raw.split())
    if not text:
        text = "unknown_error"
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _emit_token(value: object, *, limit: int = 96) -> str:
    """Render one bounded telemetry token for journal-friendly key=value lines."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value).lower()
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    text = " ".join(str(value).split())
    if not text:
        return "none"
    safe_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"_", "-", ".", ":", "/"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars) or "none"
    if len(safe) <= limit:
        return safe
    if limit <= 3:
        return safe[:limit]
    return safe[: limit - 3] + "..."


def _emit_key_value_line(prefix: str, /, **fields: object) -> str:
    """Build one stable key=value telemetry line for changed-only journal tracing."""

    parts = [prefix]
    for key, value in fields.items():
        parts.append(f"{key}={_emit_token(value)}")
    return " ".join(parts)


def _append_ops_event(
    runtime: TwinrRuntime,
    *,
    event: str,
    message: str,
    data: dict[str, Any],
    emit: Callable[[str], None] | None = None,
    level: str | None = None,
) -> None:
    """Append one ops event without letting persistence failures escape."""

    kwargs: dict[str, Any] = {
        "event": event,
        "message": message,
        "data": data,
    }
    if level is not None:
        kwargs["level"] = level
    try:
        runtime.ops_events.append(**kwargs)
    except Exception:
        _safe_emit(emit, f"ops_event_append_failed={event}")


def _normalize_optional_text(*values: Any) -> str:
    """Return the first non-blank text value from a config-like list."""

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _proactive_audio_capture_device(config: TwinrConfig) -> str:
    """Return the device used by proactive ambient PCM sampling."""

    return resolve_capture_device(
        getattr(config, "proactive_audio_input_device", None),
        getattr(config, "audio_input_device", None),
    )


def _voice_orchestrator_capture_device(config: TwinrConfig) -> str:
    """Return the device used by the long-lived voice-orchestrator capture."""

    return resolve_capture_device(
        getattr(config, "voice_orchestrator_audio_device", None),
        getattr(config, "proactive_audio_input_device", None),
        getattr(config, "audio_input_device", None),
    )


def _proactive_pcm_capture_conflicts_with_voice_orchestrator(
    config: TwinrConfig,
    *,
    require_active_owner: bool = False,
) -> bool:
    """Return whether proactive PCM fallback would fight a shared voice capture.

    Production monitor assembly uses the shared-device check alone because the
    proactive monitor and voice orchestrator run inside the same long-lived
    process. Standalone diagnostics and self-tests can opt into
    ``require_active_owner`` so a merely configured voice orchestrator does not
    block bounded PCM probes after the runtime supervisor was stopped.
    """

    if not bool(getattr(config, "voice_orchestrator_enabled", False)):
        return False
    if not bool(getattr(config, "proactive_audio_enabled", False)):
        return False
    proactive_device = _proactive_audio_capture_device(config)
    voice_device = _voice_orchestrator_capture_device(config)
    if not proactive_device or not voice_device:
        return False
    if capture_device_identity(proactive_device) != capture_device_identity(voice_device):
        return False
    if not require_active_owner:
        return True
    for loop_name in ("runtime-supervisor", "streaming-loop"):
        try:
            owner = loop_lock_owner(config, loop_name)
        except OSError:
            _LOGGER.warning(
                "Could not inspect %s lock owner while deciding standalone proactive PCM ownership; failing closed.",
                loop_name,
                exc_info=True,
            )
            return True
        if owner is not None:
            return True
    return False


def _normalize_text_tuple(values: Any) -> tuple[str, ...]:
    """Normalize sequence-like config input to one tuple of non-blank strings."""

    if values is None:
        return ()
    if isinstance(values, (str, Path)):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _round_optional_seconds(value: float | None) -> float | None:
    """Round one optional duration to milliseconds for ops-safe payloads."""

    if value is None:
        return None
    return round(max(0.0, float(value)), 3)


def _round_optional_ratio(value: float | None) -> float | None:
    """Round one optional bounded ratio or score for ops-safe payloads."""

    if value is None:
        return None
    return round(float(value), 4)


def _format_firmware_version(version: tuple[int, int, int] | None) -> str | None:
    """Format one optional firmware tuple for ops/event payloads."""

    if version is None:
        return None
    return ".".join(str(int(part)) for part in version)


def _respeaker_capture_probe_duration_ms(config: TwinrConfig) -> int:
    """Return a short bounded capture probe window for ReSpeaker startup checks."""

    chunk_ms = max(20, int(getattr(config, "audio_chunk_ms", 100) or 100))
    requested_ms = max(chunk_ms, int(getattr(config, "proactive_audio_sample_ms", chunk_ms) or chunk_ms))
    return min(requested_ms, max(250, chunk_ms * 3))


def _display_attention_refresh_allowed_runtime_status(runtime_status_value: object) -> bool:
    """Return whether bounded local HDMI eye-follow may refresh in this runtime state."""

    normalized = str(runtime_status_value or "").strip().lower()
    if normalized in _DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES:
        return True
    return normalized in _DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES


def _preserve_local_attention_on_audio_block(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    detail: str,
) -> None:
    """Keep local HDMI camera follow alive while marking the runtime as blocked."""

    _record_component_warning(
        runtime=runtime,
        emit=emit,
        reason="respeaker_camera_follow_only",
        detail=(
            "ReSpeaker startup is blocked, so Twinr preserved only the local "
            f"HDMI camera-follow path: {detail}"
        ),
    )
    try:
        runtime.fail(detail)
    except Exception:
        _safe_emit(emit, f"runtime_fail_failed={_exception_text(detail)}")


def _respeaker_dead_capture_payload(
    *,
    probe: AudioCaptureReadinessProbe,
    stage: str,
    signal: object | None = None,
) -> dict[str, Any]:
    """Build one ops-safe payload for unreadable ReSpeaker capture failures."""

    payload: dict[str, Any] = {
        "stage": stage,
        "capture_device": probe.device,
        "capture_sample_rate": probe.sample_rate,
        "capture_channels": probe.channels,
        "capture_chunk_ms": probe.chunk_ms,
        "capture_probe_duration_ms": probe.duration_ms,
        "capture_probe_target_chunk_count": probe.target_chunk_count,
        "capture_probe_chunk_count": probe.captured_chunk_count,
        "capture_probe_bytes": probe.captured_bytes,
        "capture_probe_ready": probe.ready,
        "capture_probe_failure_reason": probe.failure_reason,
        "capture_probe_detail": probe.detail,
        "transport_reason": "capture_unreadable",
    }
    if signal is not None:
        payload.update(
            {
                "device_runtime_mode": getattr(signal, "device_runtime_mode", None),
                "host_control_ready": getattr(signal, "host_control_ready", None),
                "transport_reason": getattr(signal, "transport_reason", None) or "capture_unreadable",
                "firmware_version": _format_firmware_version(getattr(signal, "firmware_version", None)),
            }
        )
    return payload


def _record_respeaker_dead_capture_blocker(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    probe: AudioCaptureReadinessProbe,
    stage: str,
    signal: object | None = None,
) -> str:
    """Emit explicit alert/blocker events when ReSpeaker capture yields no frames."""

    payload = _respeaker_dead_capture_payload(
        probe=probe,
        stage=stage,
        signal=signal,
    )
    detail = (
        "ReSpeaker XVF3800 is enumerated, but the configured capture path yielded no readable "
        f"audio frames. {probe.detail or 'Twinr refuses to treat this path as ready.'}"
    )
    _safe_emit(emit, "respeaker_runtime_alert=capture_unknown")
    _append_ops_event(
        runtime,
        event="respeaker_runtime_alert",
        level="error",
        message=detail,
        data={
            **payload,
            "alert_code": "capture_unknown",
        },
        emit=emit,
    )
    _safe_emit(emit, "respeaker_runtime_blocker=dead_capture")
    _append_ops_event(
        runtime,
        event="respeaker_runtime_blocker",
        level="error",
        message="ReSpeaker capture is unreadable even though the device still enumerates.",
        data={
            **payload,
            "alert_code": "dead_capture",
            "blocker_code": "dead_capture",
        },
        emit=emit,
    )
    _append_ops_event(
        runtime,
        event="proactive_component_blocked",
        level="error",
        message="ReSpeaker unreadable capture blocked the proactive audio path.",
        data={
            **payload,
            "reason": (
                "respeaker_dead_capture_blocked"
                if stage == "startup"
                else "respeaker_dead_capture_runtime_blocked"
            ),
            "detail": detail,
            "blocker_code": "dead_capture",
        },
        emit=emit,
    )
    return detail


def _assistant_output_active(runtime: TwinrRuntime) -> bool:
    """Return whether Twinr is actively speaking right now."""

    try:
        return getattr(runtime.status, "value", None) == "answering"
    except Exception:
        return False


def _record_component_warning(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    reason: str,
    detail: str,
) -> None:
    """Record one degraded-mode warning during proactive monitor setup."""

    _safe_emit(emit, f"proactive_component_warning={reason}")
    _append_ops_event(
        runtime,
        event="proactive_component_warning",
        level="warning",
        message="Proactive monitor is running in degraded mode.",
        data={
            "reason": reason,
            "detail": detail,
        },
        emit=emit,
    )


__all__ = [
    "_append_ops_event",
    "_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S",
    "_DEFAULT_CLOSE_JOIN_TIMEOUT_S",
    "_DEFAULT_PROJECT_ROOT",
    "_DISPLAY_ATTENTION_ACTIVE_RUNTIME_STATES",
    "_DISPLAY_ATTENTION_CUE_ONLY_RUNTIME_STATES",
    "_LOGGER",
    "_VISION_REVIEW_FAIL_OPEN_TRIGGERS",
    "_assistant_output_active",
    "_display_attention_refresh_allowed_runtime_status",
    "_emit_key_value_line",
    "_emit_token",
    "_exception_text",
    "_format_firmware_version",
    "_normalize_optional_text",
    "_normalize_text_tuple",
    "_preserve_local_attention_on_audio_block",
    "_proactive_audio_capture_device",
    "_proactive_pcm_capture_conflicts_with_voice_orchestrator",
    "_record_component_warning",
    "_record_respeaker_dead_capture_blocker",
    "_respeaker_capture_probe_duration_ms",
    "_respeaker_dead_capture_payload",
    "_round_optional_ratio",
    "_round_optional_seconds",
    "_safe_emit",
    "_voice_orchestrator_capture_device",
]
