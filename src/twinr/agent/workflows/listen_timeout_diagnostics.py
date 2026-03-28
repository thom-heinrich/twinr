"""Emit bounded capture diagnostics for listen-timeout workflow events."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: diagnostics_from_exception now unwraps __cause__, __context__, and
#        ExceptionGroup instances so timeout diagnostics are not silently lost
#        in wrapped async / task-group failures.
# SEC-1: emitted string fields are now sanitized and length-bounded to prevent
#        practical log injection / parser corruption via crafted prefix or
#        device strings.
# IMP-1: added diagnostics_as_attributes() and emit_listen_timeout_event() for
#        structured, OpenTelemetry-friendly diagnostics without adding a hard
#        telemetry dependency.
# IMP-2: centralized normalization keeps diagnostics primitive, bounded, and
#        finite so the same schema can safely feed logs, spans, and metrics.

from collections.abc import Callable, Iterator
import math
import re

from twinr.hardware.audio import ListenTimeoutCaptureDiagnostics, SpeechStartTimeoutError

try:  # Python 3.11+
    _BASE_EXCEPTION_GROUP_TYPE = BaseExceptionGroup
except NameError:  # pragma: no cover - older runtimes
    _BASE_EXCEPTION_GROUP_TYPE = None  # type: ignore[assignment]

__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "diagnostics_from_exception",
    "diagnostics_as_details",
    "diagnostics_as_attributes",
    "emit_listen_timeout_event",
    "emit_listen_timeout_diagnostics",
]

DIAGNOSTICS_SCHEMA_VERSION = 2

_DEFAULT_LEGACY_PREFIX = "listen_timeout"
_DEFAULT_ATTRIBUTE_NAMESPACE = "twinr.audio.listen_timeout"
_UNKNOWN_DEVICE = "unknown"

_MAX_PREFIX_LEN = 64
_MAX_NAMESPACE_LEN = 96
_MAX_DEVICE_LEN = 128

_LEGACY_PREFIX_SANITIZER = re.compile(r"[^0-9A-Za-z_-]+")
_ATTRIBUTE_NAMESPACE_SANITIZER = re.compile(r"[^0-9a-z_.]+")
_MULTI_UNDERSCORE = re.compile(r"_+")
_MULTI_DOT = re.compile(r"\.+")


def diagnostics_from_exception(exc: BaseException) -> ListenTimeoutCaptureDiagnostics | None:
    """Return listen-timeout diagnostics when the exception or a wrapped child carries them."""

    for candidate in _iter_exception_graph(exc):
        diagnostics = _direct_diagnostics_from_exception(candidate)
        if diagnostics is not None:
            return diagnostics
    return None


def diagnostics_as_details(
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
) -> dict[str, object]:
    """Convert capture diagnostics into bounded structured details."""

    if diagnostics is None:
        return {}

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "capture_device": _sanitize_text(diagnostics.device, max_len=_MAX_DEVICE_LEN),
        "sample_rate": _normalize_positive_int(diagnostics.sample_rate),
        "channels": _normalize_positive_int(diagnostics.channels),
        "chunk_ms": _normalize_positive_int(diagnostics.chunk_ms),
        "speech_threshold": _normalize_float(diagnostics.speech_threshold),
        "chunk_count": _normalize_non_negative_int(diagnostics.chunk_count),
        "active_chunk_count": _normalize_non_negative_int(diagnostics.active_chunk_count),
        "average_rms": _normalize_float(diagnostics.average_rms),
        "peak_rms": _normalize_float(diagnostics.peak_rms),
        "active_ratio": _normalize_float(diagnostics.active_ratio, digits=3),
        "listened_ms": _normalize_non_negative_int(diagnostics.listened_ms),
    }


def diagnostics_as_attributes(
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
    *,
    namespace: str = _DEFAULT_ATTRIBUTE_NAMESPACE,
) -> dict[str, object]:
    """Return a primitive-only, OTel-friendly flat attribute map."""

    if diagnostics is None:
        return {}

    safe_namespace = _sanitize_attribute_namespace(namespace)
    details = diagnostics_as_details(diagnostics)

    attributes: dict[str, object] = {
        f"{safe_namespace}.schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        f"{safe_namespace}.capture.device": details["capture_device"],
    }

    _maybe_set_attribute(attributes, f"{safe_namespace}.capture.sample_rate_hz", details["sample_rate"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.capture.channels", details["channels"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.capture.chunk_ms", details["chunk_ms"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.vad.speech_threshold", details["speech_threshold"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.vad.chunk_count", details["chunk_count"])
    _maybe_set_attribute(
        attributes,
        f"{safe_namespace}.vad.active_chunk_count",
        details["active_chunk_count"],
    )
    _maybe_set_attribute(attributes, f"{safe_namespace}.vad.average_rms", details["average_rms"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.vad.peak_rms", details["peak_rms"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.vad.active_ratio", details["active_ratio"])
    _maybe_set_attribute(attributes, f"{safe_namespace}.timeout.listened_ms", details["listened_ms"])
    return attributes


def emit_listen_timeout_event(
    emit_event: Callable[[str, dict[str, object]], None],
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
    *,
    event_name: str = _DEFAULT_ATTRIBUTE_NAMESPACE,
    namespace: str = _DEFAULT_ATTRIBUTE_NAMESPACE,
) -> None:
    """Emit a single structured event using an ``(event_name, attributes)`` callback."""

    if diagnostics is None:
        return

    emit_event(
        _sanitize_attribute_namespace(event_name),
        diagnostics_as_attributes(diagnostics, namespace=namespace),
    )


def emit_listen_timeout_diagnostics(
    emit: Callable[[str], None],
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
    *,
    prefix: str = _DEFAULT_LEGACY_PREFIX,
) -> None:
    """Emit one bounded set of scalar diagnostics for a no-speech timeout."""

    if diagnostics is None:
        return

    details = diagnostics_as_details(diagnostics)

    # BREAKING: custom prefixes are now canonicalized to a safe token to prevent
    # log-injection and parser ambiguity. The default prefix remains unchanged.
    safe_prefix = _sanitize_legacy_prefix(prefix)

    emit(f"{safe_prefix}_capture_device={details['capture_device']}")

    speech_threshold = details.get("speech_threshold")
    if speech_threshold is not None:
        emit(f"{safe_prefix}_speech_threshold={speech_threshold}")

    chunk_count = details.get("chunk_count")
    if chunk_count is not None:
        emit(f"{safe_prefix}_chunk_count={chunk_count}")

    active_chunk_count = details.get("active_chunk_count")
    if active_chunk_count is not None:
        emit(f"{safe_prefix}_active_chunk_count={active_chunk_count}")

    average_rms = details.get("average_rms")
    if average_rms is not None:
        emit(f"{safe_prefix}_average_rms={average_rms}")

    peak_rms = details.get("peak_rms")
    if peak_rms is not None:
        emit(f"{safe_prefix}_peak_rms={peak_rms}")

    active_ratio = details.get("active_ratio")
    if isinstance(active_ratio, (int, float)):
        emit(f"{safe_prefix}_active_ratio={active_ratio:.2f}")

    listened_ms = details.get("listened_ms")
    if listened_ms is not None:
        emit(f"{safe_prefix}_listened_ms={listened_ms}")


def _direct_diagnostics_from_exception(
    exc: BaseException,
) -> ListenTimeoutCaptureDiagnostics | None:
    if isinstance(exc, SpeechStartTimeoutError):
        return exc.diagnostics

    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, ListenTimeoutCaptureDiagnostics):
        return diagnostics
    return None


def _iter_exception_graph(root: BaseException) -> Iterator[BaseException]:
    """Walk nested exception wrappers without looping forever."""

    stack: list[BaseException] = [root]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)

        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            stack.append(context)

        if _BASE_EXCEPTION_GROUP_TYPE is not None and isinstance(current, _BASE_EXCEPTION_GROUP_TYPE):
            for child in reversed(current.exceptions):
                if isinstance(child, BaseException):
                    stack.append(child)


def _maybe_set_attribute(
    attributes: dict[str, object],
    key: str,
    value: object,
) -> None:
    if value is not None:
        attributes[key] = value


def _normalize_positive_int(value: object) -> int | None:
    normalized = _normalize_int(value)
    if normalized is None or normalized <= 0:
        return None
    return normalized


def _normalize_non_negative_int(value: object) -> int | None:
    normalized = _normalize_int(value)
    if normalized is None or normalized < 0:
        return None
    return normalized


def _normalize_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _normalize_float(value: object, *, digits: int | None = None) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(normalized):
        return None

    if digits is not None:
        normalized = round(normalized, digits)

    if normalized == 0:
        return 0.0
    return normalized


def _sanitize_text(value: object, *, max_len: int) -> str:
    if value is None:
        return _UNKNOWN_DEVICE

    sanitized_chars: list[str] = []
    for char in str(value):
        if char == "\n":
            sanitized_chars.append(r"\n")
        elif char == "\r":
            sanitized_chars.append(r"\r")
        elif char == "\t":
            sanitized_chars.append(r"\t")
        elif char.isprintable():
            sanitized_chars.append(char)
        else:
            sanitized_chars.append("?")

    sanitized = "".join(sanitized_chars).strip()
    if not sanitized:
        sanitized = _UNKNOWN_DEVICE

    if len(sanitized) > max_len:
        return f"{sanitized[: max_len - 3]}..."
    return sanitized


def _sanitize_legacy_prefix(prefix: str) -> str:
    candidate = _sanitize_text(prefix, max_len=_MAX_PREFIX_LEN)
    candidate = _LEGACY_PREFIX_SANITIZER.sub("_", candidate)
    candidate = _MULTI_UNDERSCORE.sub("_", candidate).strip("_")
    if not candidate:
        return _DEFAULT_LEGACY_PREFIX
    return candidate


def _sanitize_attribute_namespace(namespace: str) -> str:
    candidate = _sanitize_text(namespace, max_len=_MAX_NAMESPACE_LEN).lower()
    candidate = candidate.replace("-", "_")
    candidate = _ATTRIBUTE_NAMESPACE_SANITIZER.sub(".", candidate)
    candidate = _MULTI_DOT.sub(".", candidate).strip(".")
    if not candidate:
        return _DEFAULT_ATTRIBUTE_NAMESPACE
    return candidate