"""Emit bounded capture diagnostics for listen-timeout workflow events."""

from __future__ import annotations

from collections.abc import Callable

from twinr.hardware.audio import ListenTimeoutCaptureDiagnostics, SpeechStartTimeoutError


def diagnostics_from_exception(exc: BaseException) -> ListenTimeoutCaptureDiagnostics | None:
    """Return listen-timeout diagnostics when the exception carries them."""

    if isinstance(exc, SpeechStartTimeoutError):
        return exc.diagnostics
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, ListenTimeoutCaptureDiagnostics):
        return diagnostics
    return None


def diagnostics_as_details(
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
) -> dict[str, object]:
    """Convert capture diagnostics into bounded structured details."""

    if diagnostics is None:
        return {}
    return {
        "capture_device": diagnostics.device,
        "sample_rate": diagnostics.sample_rate,
        "channels": diagnostics.channels,
        "chunk_ms": diagnostics.chunk_ms,
        "speech_threshold": diagnostics.speech_threshold,
        "chunk_count": diagnostics.chunk_count,
        "active_chunk_count": diagnostics.active_chunk_count,
        "average_rms": diagnostics.average_rms,
        "peak_rms": diagnostics.peak_rms,
        "active_ratio": round(diagnostics.active_ratio, 3),
        "listened_ms": diagnostics.listened_ms,
    }


def emit_listen_timeout_diagnostics(
    emit: Callable[[str], None],
    diagnostics: ListenTimeoutCaptureDiagnostics | None,
    *,
    prefix: str = "listen_timeout",
) -> None:
    """Emit one bounded set of scalar diagnostics for a no-speech timeout."""

    if diagnostics is None:
        return
    emit(f"{prefix}_capture_device={diagnostics.device}")
    emit(f"{prefix}_speech_threshold={diagnostics.speech_threshold}")
    emit(f"{prefix}_chunk_count={diagnostics.chunk_count}")
    emit(f"{prefix}_active_chunk_count={diagnostics.active_chunk_count}")
    emit(f"{prefix}_average_rms={diagnostics.average_rms}")
    emit(f"{prefix}_peak_rms={diagnostics.peak_rms}")
    emit(f"{prefix}_active_ratio={diagnostics.active_ratio:.2f}")
    emit(f"{prefix}_listened_ms={diagnostics.listened_ms}")
