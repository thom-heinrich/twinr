# CHANGELOG: 2026-03-28
# BUG-1: max_wait_s<=0 now performs one real readiness probe instead of returning False before probing.
# BUG-2: Each probe attempt now respects the caller's remaining wait budget instead of letting one probe overrun max_wait_s.
# BUG-3: Add one bounded XVF3800 host-control REBOOT fallback so proven warm-reboot/capture-stall states can recover without manual unplug/reset.
# SEC-1: Hardened subprocess execution by resolving a trusted arecord path and scrubbing dangerous loader-injection environment variables.
# IMP-1: Prefer native ALSA probing via pyalsaaudio (PCM_NONBLOCK + poll descriptors) when available on Raspberry Pi.
# IMP-2: Optionally use pyudev sound-subsystem monitoring so hotplug/re-enumeration waits wake on real device events instead of blind polling alone.

"""Wait briefly for transient ReSpeaker capture loss to clear.

Twinr's XVF3800 USB microphone can briefly disappear from ALSA while the Pi
power or USB stack forces a short re-enumeration. The local listen recorder
and the edge voice orchestrator both need the same bounded recovery helper so
they do not fail a turn immediately when the configured ReSpeaker path comes
back a fraction of a second later.

2026 upgrade notes:
- Prefer native ALSA PCM probing through ``pyalsaaudio`` when available.
- Optionally use ``pyudev`` to wake waits on real ``sound`` hotplug events.
- Keep a hardened ``arecord`` fallback for environments without extra deps.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import os
from pathlib import Path
import select
import subprocess
from threading import RLock
import time

try:  # Optional frontier fast-path on Linux/ALSA systems.
    import alsaaudio  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    alsaaudio = None  # type: ignore[assignment]

try:  # Optional hotplug-assisted wait path.
    import pyudev  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pyudev = None  # type: ignore[assignment]

from twinr.hardware.audio_env import build_audio_subprocess_env_for_mode
from twinr.hardware.respeaker.probe import config_targets_respeaker, probe_respeaker_xvf3800
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport
from twinr.hardware.respeaker.write_specs import REBOOT_PARAMETER

_SAMPLE_WIDTH_BYTES = 2
_DEFAULT_RECOVERY_WAIT_S = 2.5
_POST_REBOOT_RECOVERY_WAIT_S = 8.0
_PROBE_INTERVAL_S = 0.1
_HOTPLUG_WAIT_SLICE_S = 0.5
_PROCESS_STOP_TIMEOUT_S = 0.25
_SELECT_TIMEOUT_S = 0.1
_READABLE_FRAME_GRACE_S = 0.35
_HOST_REBOOT_COOLDOWN_S = 30.0
_HOST_REBOOT_VALUE = 1
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_CHANNELS = 1
_DEFAULT_CHUNK_MS = 20
# BREAKING: the arecord fallback only trusts system executable locations instead of PATH-based lookup.
_ARECORD_PATH_CANDIDATES = (
    Path("/usr/bin/arecord"),
    Path("/bin/arecord"),
    Path("/usr/local/bin/arecord"),
)
_UNSAFE_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
)
_HOST_REBOOT_LOCK = RLock()
_LAST_HOST_REBOOT_ATTEMPT_AT: dict[str, float] = {}


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

    A call with ``max_wait_s <= 0`` still performs one immediate readiness
    attempt instead of returning ``False`` without probing the device.
    """

    normalized_device = str(device or "").strip()
    if not config_targets_respeaker(normalized_device):
        return False

    deadline_at = time.monotonic() + _coerce_non_negative_float(max_wait_s, default=0.0)
    monitor = _maybe_build_sound_monitor()

    while True:
        if should_stop is not None and should_stop():
            return False

        remaining_s = deadline_at - time.monotonic()
        probe_timeout_s = _probe_timeout_s(chunk_ms)
        if remaining_s > 0.0:
            probe_timeout_s = min(probe_timeout_s, remaining_s)
        else:
            probe_timeout_s = min(probe_timeout_s, _READABLE_FRAME_GRACE_S)

        probe = probe_respeaker_xvf3800()
        capture_ready = bool(getattr(probe, "capture_ready", False))
        if capture_ready and _probe_readable_frame(
            device=normalized_device,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=chunk_ms,
            attempt_timeout_s=probe_timeout_s,
            should_stop=should_stop,
        ):
            return True

        remaining_s = deadline_at - time.monotonic()
        if remaining_s <= 0.0:
            return False

        wait_slice_s = _PROBE_INTERVAL_S
        if monitor is not None and not capture_ready:
            wait_slice_s = _HOTPLUG_WAIT_SLICE_S
        _wait_between_attempts(
            timeout_s=min(wait_slice_s, remaining_s),
            monitor=monitor,
            should_stop=should_stop,
        )


def recover_stalled_respeaker_capture(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    max_wait_s: float = _DEFAULT_RECOVERY_WAIT_S,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Return whether a stalled XVF3800 capture path can be recovered.

    Recovery happens in two bounded stages:
    1. wait for a normal transient ALSA/udev recovery
    2. if that fails, issue the official XVF3800 host-control ``REBOOT`` write
       once per cooldown window and then wait for capture to come back
    """

    if wait_for_transient_respeaker_capture_ready(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_ms=chunk_ms,
        max_wait_s=max_wait_s,
        should_stop=should_stop,
    ):
        return True

    if should_stop is not None and should_stop():
        return False
    if not _attempt_respeaker_host_control_reboot(device=device):
        return False

    return wait_for_transient_respeaker_capture_ready(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_ms=chunk_ms,
        max_wait_s=max(_POST_REBOOT_RECOVERY_WAIT_S, _coerce_non_negative_float(max_wait_s, default=0.0)),
        should_stop=should_stop,
    )


def _attempt_respeaker_host_control_reboot(*, device: str) -> bool:
    normalized_device = str(device or "").strip()
    if not config_targets_respeaker(normalized_device):
        return False

    now = time.monotonic()
    with _HOST_REBOOT_LOCK:
        last_attempt_at = _LAST_HOST_REBOOT_ATTEMPT_AT.get(normalized_device)
        if last_attempt_at is not None and (now - last_attempt_at) < _HOST_REBOOT_COOLDOWN_S:
            return False
        _LAST_HOST_REBOOT_ATTEMPT_AT[normalized_device] = now

    probe = probe_respeaker_xvf3800()
    if not getattr(probe, "usb_visible", False):
        return False

    transport = ReSpeakerLibusbTransport()
    try:
        availability = transport.write_parameter(
            REBOOT_PARAMETER,
            (_HOST_REBOOT_VALUE,),
            probe=probe,
        )
        return bool(availability.available)
    except Exception:
        return False
    finally:
        transport.close()


def _probe_readable_frame(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    attempt_timeout_s: float,
    should_stop: Callable[[], bool] | None,
) -> bool:
    native_result = _probe_readable_frame_native(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_ms=chunk_ms,
        attempt_timeout_s=attempt_timeout_s,
        should_stop=should_stop,
    )
    if native_result is not None:
        return native_result
    return _probe_readable_frame_arecord(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_ms=chunk_ms,
        attempt_timeout_s=attempt_timeout_s,
        should_stop=should_stop,
    )


def _probe_readable_frame_native(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    attempt_timeout_s: float,
    should_stop: Callable[[], bool] | None,
) -> bool | None:
    if alsaaudio is None:
        return None

    normalized_sample_rate = _coerce_positive_int(sample_rate, minimum=1, default=_DEFAULT_SAMPLE_RATE)
    normalized_channels = _coerce_positive_int(channels, minimum=1, default=_DEFAULT_CHANNELS)
    chunk_frames = _chunk_frame_count(sample_rate=normalized_sample_rate, chunk_ms=chunk_ms)
    deadline_at = time.monotonic() + max(0.0, attempt_timeout_s)

    pcm = None
    try:
        pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NONBLOCK,
            rate=normalized_sample_rate,
            channels=normalized_channels,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=chunk_frames,
            periods=2,
            device=device,
        )
    except Exception:
        return None

    try:
        poller = _build_pcm_poller(pcm)
        while True:
            if should_stop is not None and should_stop():
                return False
            if _pcm_is_disconnected(pcm):
                return False

            try:
                frames_read, payload = pcm.read()
            except Exception:
                return False

            if isinstance(frames_read, int) and frames_read > 0:
                aligned_payload = _trim_to_frame_alignment(
                    _ensure_bytes(payload),
                    channels=normalized_channels,
                )
                if aligned_payload:
                    return True
            elif isinstance(frames_read, int) and frames_read < 0:
                return False

            remaining_s = deadline_at - time.monotonic()
            if remaining_s <= 0.0:
                return False

            _wait_for_pcm_readability(
                poller,
                timeout_s=min(_SELECT_TIMEOUT_S, remaining_s),
                should_stop=should_stop,
            )
    finally:
        _close_native_pcm(pcm)


def _build_pcm_poller(pcm: object) -> object | None:
    if not hasattr(select, "poll"):
        return None
    poll_descriptors = getattr(pcm, "polldescriptors", None)
    if poll_descriptors is None:
        return None
    try:
        descriptors = list(poll_descriptors())
    except Exception:
        return None
    if not descriptors:
        return None
    try:
        poller = select.poll()
        for fd, eventmask in descriptors:
            poller.register(fd, eventmask)
        return poller
    except Exception:
        return None


def _wait_for_pcm_readability(
    poller: object | None,
    *,
    timeout_s: float,
    should_stop: Callable[[], bool] | None,
) -> None:
    if timeout_s <= 0.0:
        return
    if should_stop is not None and should_stop():
        return
    if poller is None:
        time.sleep(timeout_s)
        return
    try:
        poller.poll(max(1, int(timeout_s * 1000)))
    except Exception:
        time.sleep(timeout_s)


def _pcm_is_disconnected(pcm: object) -> bool:
    if alsaaudio is None:
        return False
    state_fn = getattr(pcm, "state", None)
    disconnected_state = getattr(alsaaudio, "PCM_STATE_DISCONNECTED", None)
    if state_fn is None or disconnected_state is None:
        return False
    try:
        return state_fn() == disconnected_state
    except Exception:
        return False


def _close_native_pcm(pcm: object | None) -> None:
    if pcm is None:
        return
    close_fn = getattr(pcm, "close", None)
    if close_fn is None:
        return
    try:
        close_fn()
    except Exception:
        return


def _probe_readable_frame_arecord(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    attempt_timeout_s: float,
    should_stop: Callable[[], bool] | None,
) -> bool:
    process = _spawn_probe_process(
        _arecord_probe_command(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=chunk_ms,
        )
    )
    if process is None:
        return False

    deadline_at = time.monotonic() + max(0.0, attempt_timeout_s)
    try:
        while True:
            if should_stop is not None and should_stop():
                return False
            remaining_s = deadline_at - time.monotonic()
            if remaining_s <= 0.0:
                return False
            try:
                stdout, _stderr = process.communicate(timeout=min(_SELECT_TIMEOUT_S, remaining_s))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode != 0:
            return False
        return bool(
            _trim_to_frame_alignment(
                stdout,
                channels=_coerce_positive_int(channels, minimum=1, default=_DEFAULT_CHANNELS),
            )
        )
    finally:
        _stop_process(process)


def _arecord_probe_command(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
) -> list[str]:
    normalized_sample_rate = _coerce_positive_int(sample_rate, minimum=1, default=_DEFAULT_SAMPLE_RATE)
    normalized_channels = _coerce_positive_int(channels, minimum=1, default=_DEFAULT_CHANNELS)
    chunk_frames = _chunk_frame_count(sample_rate=normalized_sample_rate, chunk_ms=chunk_ms)
    return [
        _ARECORD_PATH or "",
        "-D",
        device,
        "-q",
        "-N",
        "--fatal-errors",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-c",
        str(normalized_channels),
        "-r",
        str(normalized_sample_rate),
        "-s",
        str(chunk_frames),
    ]


def _spawn_probe_process(command: list[str]) -> subprocess.Popen[bytes] | None:
    if not _ARECORD_PATH or not command or command[0] != _ARECORD_PATH:
        return None
    try:
        return subprocess.Popen(
            command,
            env=_build_probe_subprocess_env(),
            stdin=subprocess.DEVNULL,
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
                process.communicate(timeout=_PROCESS_STOP_TIMEOUT_S)
                return
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.communicate(timeout=_PROCESS_STOP_TIMEOUT_S)
                    return
                except subprocess.TimeoutExpired:
                    pass
        else:
            try:
                process.communicate(timeout=0)
            except (subprocess.TimeoutExpired, ValueError):
                pass
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                continue


def _build_probe_subprocess_env() -> dict[str, str]:
    # BREAKING: strip dynamic-loader override variables for the native helper subprocess.
    raw_env = build_audio_subprocess_env_for_mode(
        allow_root_borrowed_session_audio=True,
    )
    env = dict(os.environ if raw_env is None else raw_env)
    for key in _UNSAFE_SUBPROCESS_ENV_KEYS:
        env.pop(key, None)
    return env


def _resolve_arecord_path() -> str | None:
    for candidate in _ARECORD_PATH_CANDIDATES:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return os.fspath(candidate)
        except OSError:
            continue
    return None


def _maybe_build_sound_monitor() -> object | None:
    if pyudev is None:
        return None
    try:
        context = pyudev.Context()
        monitor = pyudev.Monitor.from_netlink(context)
        monitor.filter_by(subsystem="sound")
        return monitor
    except Exception:
        return None


def _wait_between_attempts(
    *,
    timeout_s: float,
    monitor: object | None,
    should_stop: Callable[[], bool] | None,
) -> None:
    if timeout_s <= 0.0:
        return
    if should_stop is not None and should_stop():
        return
    if monitor is None:
        time.sleep(timeout_s)
        return
    try:
        monitor.poll(timeout=timeout_s)
    except Exception:
        time.sleep(timeout_s)


def _probe_timeout_s(chunk_ms: int) -> float:
    normalized_chunk_ms = _coerce_positive_int(chunk_ms, minimum=20, default=_DEFAULT_CHUNK_MS)
    return max(
        _READABLE_FRAME_GRACE_S,
        (normalized_chunk_ms / 1000.0) + _READABLE_FRAME_GRACE_S,
    )


def _chunk_frame_count(*, sample_rate: int, chunk_ms: int) -> int:
    normalized_sample_rate = _coerce_positive_int(sample_rate, minimum=1, default=_DEFAULT_SAMPLE_RATE)
    normalized_chunk_ms = _coerce_positive_int(chunk_ms, minimum=20, default=_DEFAULT_CHUNK_MS)
    return max(
        1,
        math.ceil((normalized_sample_rate * normalized_chunk_ms) / 1000.0),
    )


def _chunk_byte_count(*, sample_rate: int, channels: int, chunk_ms: int) -> int:
    normalized_channels = _coerce_positive_int(channels, minimum=1, default=_DEFAULT_CHANNELS)
    return _chunk_frame_count(sample_rate=sample_rate, chunk_ms=chunk_ms) * _frame_alignment(normalized_channels)


def _frame_alignment(channels: int) -> int:
    return _SAMPLE_WIDTH_BYTES * _coerce_positive_int(channels, minimum=1, default=_DEFAULT_CHANNELS)


def _trim_to_frame_alignment(payload: bytes, *, channels: int) -> bytes:
    frame_alignment = _frame_alignment(channels)
    if frame_alignment <= 1 or not payload:
        return payload
    usable_length = len(payload) - (len(payload) % frame_alignment)
    if usable_length <= 0:
        return b""
    return payload[:usable_length]


def _ensure_bytes(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    return b""


def _coerce_positive_int(value: object, *, minimum: int, default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return max(minimum, int(default))
    return max(minimum, normalized)


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return max(0.0, float(default))
    if not math.isfinite(normalized):
        return max(0.0, float(default))
    return max(0.0, normalized)


_ARECORD_PATH = _resolve_arecord_path()

__all__ = ["wait_for_transient_respeaker_capture_ready"]
