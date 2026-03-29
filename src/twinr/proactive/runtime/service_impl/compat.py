# CHANGELOG: 2026-03-29
# BUG-1: _round_optional_seconds/_round_optional_ratio could emit NaN/Inf or raise on malformed values,
#        which can poison JSON/OTel exporters and silently drop ops events; both now fail safe to None.
# BUG-2: _respeaker_capture_probe_duration_ms claimed to be short and bounded but could block startup for
#        arbitrarily long windows on bad config; it now uses safe parsing and a hard upper bound.
# BUG-3: _normalize_text_tuple crashed on scalar non-iterables (for example misconfigured numeric/env-backed
#        values); it now accepts singleton scalars and normalizes them safely.
# SEC-1: _append_ops_event forwarded arbitrary nested objects/strings directly into ops persistence, creating
#        practical log/parser-poisoning and memory-amplification risk from config/device/error text; payloads
#        are now recursively copied into bounded JSON-safe primitives before persistence.
# IMP-1: _safe_emit now rate-limits sink-failure warnings so a broken telemetry sink cannot trigger a Pi-wide
#        log storm and CPU churn during hot loops.
# IMP-2: shared-capture Linux backends (PipeWire / ALSA dsnoop / Pulse compatibility) are now treated as
#        non-conflicting when both paths intentionally use a multiplexing capture source.
# IMP-3: project-root discovery now searches sentinel files before falling back, avoiding brittle import-time
#        failures from fixed parent-depth assumptions while staying drop-in compatible.

"""Compatibility helpers for the proactive runtime service refactor.

Purpose: keep stable helper functions, constants, and telemetry formatting in a
small shared module so the public ``service.py`` wrapper can re-export the
historic helper surface unchanged.

Invariants: helper behavior, ops-event keys, and fail-closed/runtime-warning
semantics must stay identical to the legacy implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable
import logging
import math
import time

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
_PROJECT_ROOT_SENTINELS = (
    "pyproject.toml",
    ".git",
    "setup.cfg",
    "setup.py",
    "README.md",
)
_DEFAULT_PROJECT_ROOT_FALLBACK_DEPTH = 4
_EMIT_SINK_FAILURE_LOG_INTERVAL_S = 30.0
_MAX_EMIT_FAILURE_LINE_CHARS = 160
_OPS_EVENT_MESSAGE_MAX_CHARS = 512
_OPS_EVENT_STRING_MAX_CHARS = 512
_OPS_EVENT_MAX_CONTAINER_ITEMS = 64
_OPS_EVENT_MAX_DEPTH = 6
_RESPEAKER_CAPTURE_PROBE_MIN_MS = 250
_RESPEAKER_CAPTURE_PROBE_MAX_MS = 2000
_DEFAULT_PROJECT_ROOT = None  # populated below after helper definitions

_LOGGER = logging.getLogger("twinr.proactive.runtime.service")
_EMIT_SINK_FAILURE_STATE_LOCK = Lock()
_EMIT_SINK_FAILURE_STATE = {
    "last_warning_monotonic": 0.0,
    "suppressed_count": 0,
}


JSONScalar = None | bool | int | float | str
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _collapse_whitespace(value: object, *, limit: int | None = None) -> str:
    """Return one printable single-line representation with optional truncation."""

    text = " ".join(str(value).split())
    if not text:
        text = "none"
    if limit is not None and limit >= 0 and len(text) > limit:
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3] + "..."
    return text


def _discover_project_root(anchor: Path) -> Path:
    """Discover the repository/project root without assuming a fixed depth."""

    resolved = anchor.resolve()
    for candidate in resolved.parents:
        for sentinel in _PROJECT_ROOT_SENTINELS:
            if (candidate / sentinel).exists():
                return candidate
    parents = resolved.parents
    if not parents:
        return resolved.parent
    fallback_index = min(_DEFAULT_PROJECT_ROOT_FALLBACK_DEPTH, len(parents) - 1)
    return parents[fallback_index]


_DEFAULT_PROJECT_ROOT = _discover_project_root(Path(__file__))


def _coerce_int(value: object, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse one int-like config value with bounds and a safe fallback."""

    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if minimum is not None and parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _finite_float_or_none(value: object | None) -> float | None:
    """Convert a numeric-like value to a finite float or return ``None``."""

    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _ops_json_safe(value: object, *, depth: int = 0) -> JSONValue:
    """Convert arbitrary event payload data into bounded JSON-safe primitives."""

    if depth >= _OPS_EVENT_MAX_DEPTH:
        return _collapse_whitespace(value, limit=_OPS_EVENT_STRING_MAX_CHARS)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return _collapse_whitespace(value, limit=16)
        return value
    if isinstance(value, (str, Path)):
        return _collapse_whitespace(value, limit=_OPS_EVENT_STRING_MAX_CHARS)
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", errors="replace")
        except Exception:
            decoded = repr(value)
        return _collapse_whitespace(decoded, limit=_OPS_EVENT_STRING_MAX_CHARS)
    if isinstance(value, Mapping):
        safe_items: dict[str, JSONValue] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= _OPS_EVENT_MAX_CONTAINER_ITEMS:
                safe_items["_truncated"] = True
                safe_items["_truncated_count"] = len(value) - _OPS_EVENT_MAX_CONTAINER_ITEMS
                break
            safe_key = _collapse_whitespace(raw_key, limit=64)
            safe_items[safe_key] = _ops_json_safe(raw_value, depth=depth + 1)
        return safe_items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        safe_values: list[JSONValue] = []
        for index, item in enumerate(value):
            if index >= _OPS_EVENT_MAX_CONTAINER_ITEMS:
                safe_values.append("...")
                break
            safe_values.append(_ops_json_safe(item, depth=depth + 1))
        return safe_values
    return _collapse_whitespace(value, limit=_OPS_EVENT_STRING_MAX_CHARS)


def _capture_device_supports_shared_readers(device: str) -> bool:
    """Return whether the configured capture device is a known multiplexing source."""

    normalized = str(device or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"pipewire", "pulse"}:
        return True
    if normalized.startswith("pipewire:") or normalized.startswith("pulse:"):
        return True
    tokenized = normalized.replace(",", ":").replace(";", ":").split(":")
    return any(token in {"pipewire", "pulse", "dsnoop"} for token in tokenized)


def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    """Emit one telemetry line while suppressing sink failures."""

    if emit is None:
        return
    try:
        emit(line)
    except Exception:
        now = time.monotonic()
        with _EMIT_SINK_FAILURE_STATE_LOCK:
            last_warning = float(_EMIT_SINK_FAILURE_STATE["last_warning_monotonic"])
            elapsed = now - last_warning
            if elapsed < _EMIT_SINK_FAILURE_LOG_INTERVAL_S:
                _EMIT_SINK_FAILURE_STATE["suppressed_count"] = int(_EMIT_SINK_FAILURE_STATE["suppressed_count"]) + 1
                return
            suppressed = int(_EMIT_SINK_FAILURE_STATE["suppressed_count"])
            _EMIT_SINK_FAILURE_STATE["last_warning_monotonic"] = now
            _EMIT_SINK_FAILURE_STATE["suppressed_count"] = 0
        if suppressed:
            _LOGGER.warning(
                "Proactive emit sink failed repeatedly; %s additional failures were suppressed. line=%s",
                suppressed,
                _collapse_whitespace(line, limit=_MAX_EMIT_FAILURE_LINE_CHARS),
                exc_info=True,
            )
        else:
            _LOGGER.warning(
                "Proactive emit sink failed. line=%s",
                _collapse_whitespace(line, limit=_MAX_EMIT_FAILURE_LINE_CHARS),
                exc_info=True,
            )
        return


def _exception_text(error: BaseException | object, *, limit: int = 240) -> str:
    """Normalize one exception payload into bounded log-safe text."""

    raw = str(error) if not isinstance(error, BaseException) else (str(error) or error.__class__.__name__)
    text = " ".join(raw.split())
    if not text:
        text = "unknown_error"
    if len(text) > limit:
        if limit <= 3:
            return text[:limit]
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
        "event": _collapse_whitespace(event, limit=96),
        "message": _collapse_whitespace(message, limit=_OPS_EVENT_MESSAGE_MAX_CHARS),
        "data": _ops_json_safe(dict(data)),
    }
    if level is not None:
        kwargs["level"] = _collapse_whitespace(level, limit=16)
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
    if (
        _capture_device_supports_shared_readers(proactive_device)
        and _capture_device_supports_shared_readers(voice_device)
    ):
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
    if isinstance(values, (str, bytes, bytearray, Path)):
        values = (values,)
    elif isinstance(values, Mapping):
        values = tuple(values.values())
    elif not isinstance(values, Iterable):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _round_optional_seconds(value: float | None) -> float | None:
    """Round one optional duration to milliseconds for ops-safe payloads."""

    parsed = _finite_float_or_none(value)
    if parsed is None:
        return None
    return round(max(0.0, parsed), 3)


def _round_optional_ratio(value: float | None) -> float | None:
    """Round one optional bounded ratio or score for ops-safe payloads."""

    parsed = _finite_float_or_none(value)
    if parsed is None:
        return None
    return round(parsed, 4)


def _format_firmware_version(version: tuple[int, int, int] | None) -> str | None:
    """Format one optional firmware tuple for ops/event payloads."""

    if version is None:
        return None
    try:
        return ".".join(str(int(part)) for part in version)
    except (TypeError, ValueError, OverflowError):
        return _collapse_whitespace(version, limit=32)


def _respeaker_capture_probe_duration_ms(config: TwinrConfig) -> int:
    """Return a short bounded capture probe window for ReSpeaker startup checks."""

    chunk_ms = _coerce_int(
        getattr(config, "audio_chunk_ms", 100),
        default=100,
        minimum=20,
        maximum=_RESPEAKER_CAPTURE_PROBE_MAX_MS,
    )
    requested_ms = _coerce_int(
        getattr(config, "proactive_audio_sample_ms", chunk_ms),
        default=chunk_ms,
        minimum=chunk_ms,
        maximum=_RESPEAKER_CAPTURE_PROBE_MAX_MS,
    )
    bounded_ms = min(requested_ms, max(_RESPEAKER_CAPTURE_PROBE_MIN_MS, chunk_ms * 3))
    return max(_RESPEAKER_CAPTURE_PROBE_MIN_MS, min(bounded_ms, _RESPEAKER_CAPTURE_PROBE_MAX_MS))


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
        return str(getattr(runtime.status, "value", "")).strip().lower() == "answering"
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