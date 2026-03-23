"""Wait briefly for transient ReSpeaker capture loss to clear.

Twinr's XVF3800 USB microphone can briefly disappear from ALSA while the Pi
power or USB stack forces a short re-enumeration. The local listen recorder
and the edge voice orchestrator both need the same bounded recovery helper so
they do not fail a turn immediately when the configured ReSpeaker path comes
back a fraction of a second later.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import select
import subprocess
import time

from twinr.hardware.audio_env import build_audio_subprocess_env
from twinr.hardware.respeaker.probe import config_targets_respeaker, probe_respeaker_xvf3800

_SAMPLE_WIDTH_BYTES = 2
_DEFAULT_RECOVERY_WAIT_S = 2.5
_PROBE_INTERVAL_S = 0.1
_PROCESS_STOP_TIMEOUT_S = 0.25
_SELECT_TIMEOUT_S = 0.1
_READABLE_FRAME_GRACE_S = 0.35


def wait_for_transient_respeaker_capture_ready(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    max_wait_s: float = _DEFAULT_RECOVERY_WAIT_S,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Return whether a targeted ReSpeaker capture path becomes readable again.

    This helper only activates for explicit XVF3800/ReSpeaker device strings.
    All other ALSA devices continue to fail fast so unrelated capture bugs do
    not get masked behind extra waits.
    """

    normalized_device = str(device or "").strip()
    if not config_targets_respeaker(normalized_device):
        return False
    deadline_at = time.monotonic() + max(0.0, float(max_wait_s))
    while True:
        if should_stop is not None and should_stop():
            return False
        if time.monotonic() >= deadline_at:
            return False
        probe = probe_respeaker_xvf3800()
        if probe.capture_ready and _probe_readable_frame(
            device=normalized_device,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=chunk_ms,
            should_stop=should_stop,
        ):
            return True
        remaining_s = deadline_at - time.monotonic()
        if remaining_s <= 0.0:
            return False
        time.sleep(min(_PROBE_INTERVAL_S, remaining_s))


def _probe_readable_frame(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    should_stop: Callable[[], bool] | None,
) -> bool:
    process = _spawn_probe_process(
        [
            "arecord",
            "-D",
            device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            str(max(1, int(channels))),
            "-r",
            str(max(1, int(sample_rate))),
        ]
    )
    if process is None:
        return False
    try:
        if process.stdout is None:
            return False
        try:
            os.set_blocking(process.stdout.fileno(), False)
        except OSError:
            return False
        chunk_bytes = _chunk_byte_count(
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=chunk_ms,
        )
        deadline_at = time.monotonic() + max(
            _READABLE_FRAME_GRACE_S,
            (max(20, int(chunk_ms)) / 1000.0) + _READABLE_FRAME_GRACE_S,
        )
        while True:
            if should_stop is not None and should_stop():
                return False
            if time.monotonic() >= deadline_at:
                return False
            if process.poll() is not None:
                return False
            timeout_s = min(_SELECT_TIMEOUT_S, max(0.0, deadline_at - time.monotonic()))
            try:
                readable, _write_ready, _error_ready = select.select([process.stdout], [], [], timeout_s)
            except (OSError, ValueError):
                return False
            if not readable:
                continue
            try:
                chunk = os.read(process.stdout.fileno(), chunk_bytes)
            except (BlockingIOError, InterruptedError, OSError):
                return False
            if _trim_to_frame_alignment(chunk, channels=max(1, int(channels))):
                return True
    finally:
        _stop_process(process)


def _spawn_probe_process(command: list[str]) -> subprocess.Popen[bytes] | None:
    try:
        return subprocess.Popen(
            command,
            env=build_audio_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            bufsize=0,
        )
    except (FileNotFoundError, OSError, ValueError):
        return None


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                continue


def _chunk_byte_count(*, sample_rate: int, channels: int, chunk_ms: int) -> int:
    normalized_sample_rate = max(1, int(sample_rate))
    normalized_channels = max(1, int(channels))
    normalized_chunk_ms = max(20, int(chunk_ms))
    return max(
        _SAMPLE_WIDTH_BYTES * normalized_channels,
        int(
            (
                normalized_sample_rate
                * normalized_channels
                * _SAMPLE_WIDTH_BYTES
                * normalized_chunk_ms
            )
            / 1000
        ),
    )


def _trim_to_frame_alignment(payload: bytes, *, channels: int) -> bytes:
    frame_alignment = _SAMPLE_WIDTH_BYTES * max(1, int(channels))
    if frame_alignment <= 1 or not payload:
        return payload
    usable_length = len(payload) - (len(payload) % frame_alignment)
    if usable_length <= 0:
        return b""
    return payload[:usable_length]


__all__ = ["wait_for_transient_respeaker_capture_ready"]
