# CHANGELOG: 2026-04-04
# BUG-4: Keep productive direct-guard aplay processes inside the supervisor-owned
# streaming-loop session. Starting a separate session let /dev/zero keepalive
# helpers survive loop restarts as PPID=1 orphans, leaving twinr_playback_softvol
# busy for the next foreground thinking/TTS playback.
# BUG-5: Tie direct-guard aplay helpers to the parent process lifetime via
# PR_SET_PDEATHSIG and fail closed when the playback device is already busy.
# Treating a foreign/stale owner as "externally satisfied" let orphaned guards
# mask the real fault until foreground Thinking/TTS playback crashed.
# CHANGELOG: 2026-03-28
# BUG-1: Detect ReSpeaker ALSA devices referenced as numeric/symbolic hw/plughw/sysdefault cards via /proc/asound/cards; official Seeed examples use plughw:<card>,0 and were previously missed.
# BUG-2: Prevent helper hangs and false positives by using non-blocking open, startup settling, shared ref-counted guards, and multi-rate fallback/caching.
# BUG-3: Drain stderr asynchronously so long-lived guards cannot stall on a full stderr pipe; stop logic is now process-group based and non-blocking.
# SEC-1: Resolve the absolute aplay path before exec and sanitize/bound log output to reduce PATH hijack and log-injection/flood risk in local Pi deployments.
# IMP-1: Tune ALSA guard startup for bounded helpers (non-blocking open, reduced buffer time, immediate start delay, fatal error semantics).

"""Manage short-lived XVF3800 playback guards for bounded capture helpers.

The reSpeaker XVF3800 on the Pi can expose a duplex-specific failure mode where
``arecord`` on the ReSpeaker capture path yields no readable frames until a
Twinr-owned playback stream is already active. The productive voice
orchestrator solves this with a long-lived playback coordinator keepalive. This
module provides the equivalent bounded helper for short-lived hardware probes,
ambient samplers, and self-tests that run outside that orchestrator.

For the productive voice path, the supervisor already launches each streaming
loop in its own dedicated POSIX session. The guard therefore must not detach
again into a second session, or its silent ``aplay /dev/zero`` helper can
survive a loop restart and strand the productive playback device as an orphan.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from twinr.hardware.audio_env import build_audio_subprocess_env_for_mode

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig

logger = logging.getLogger(__name__)

_DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ = 24_000
_MIN_PLAYBACK_SAMPLE_RATE_HZ = 8_000

# Keep startup and shutdown bounded for probe-style helpers.
_PROCESS_STARTUP_GRACE_S = 0.20
_PROCESS_READY_SETTLE_S = 0.06
_PROCESS_STOP_TIMEOUT_S = 0.50

# Lower than aplay's default maximum buffer (up to 500 ms) to make the guard
# engage quickly and keep helper shutdown crisp.
_APLAY_BUFFER_TIME_US = 80_000
_APLAY_PERIOD_TIME_US = 20_000

# Bound log volume and prevent multiline stderr/device strings from polluting
# service logs.
_MAX_LOG_TEXT_CHARS = 512
_STDERR_RING_MAX_LINES = 32

_PLAYBACK_DEVICES_COMPATIBLE_WITH_RESPEAKER_DUPLEX = frozenset(
    {
        "twinr_playback_hw",
        "twinr_playback_softvol",
    }
)
_RESPEAKER_DEVICE_MARKERS = (
    "card=array",
    "respeaker",
    "xvf3800",
    "seeed_studio_respeaker_xvf3800",
)

# Seeed's official Raspberry Pi examples commonly use plughw:<card>,0 /
# aplay -D plughw:<card>,0 rather than descriptive names. Detect those too.
_ALSA_CARD_TOKEN_PREFIXES = frozenset(
    {
        "hw",
        "plughw",
        "sysdefault",
        "front",
        "rear",
        "center_lfe",
        "side",
        "surround40",
        "surround41",
        "surround50",
        "surround51",
        "surround71",
        "iec958",
        "dmix",
        "dsnoop",
        "asym",
        "plug",
        "softvol",
    }
)
_ALSA_CARD_LINE_RE = re.compile(r"^\s*(?P<index>\d+)\s+\[(?P<id>[^\]]+)\]:\s*(?P<desc>.*)$")
_DEVICE_BUSY_MARKERS = (
    "device or resource busy",
    "resource busy",
    "temporarily unavailable",
)
_PR_SET_PDEATHSIG = 1
_PARENT_DEATH_SIGNAL = signal.SIGTERM

_SHARED_GUARD_LOCK = threading.Lock()
_SHARED_GUARDS: dict[str, "_SharedPlaybackState"] = {}
_SUCCESSFUL_SAMPLE_RATE_BY_DEVICE: dict[str, int] = {}


def _normalize_device_string(value: object | None) -> str:
    return str(value or "").strip().lower()


def _load_prctl_libc() -> ctypes.CDLL | None:
    if sys.platform != "linux":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None
    prctl = getattr(libc, "prctl", None)
    if prctl is None:
        return None
    prctl.restype = ctypes.c_int
    return libc


_PRCTL_LIBC = _load_prctl_libc()


def _build_parent_death_signal_preexec(expected_parent_pid: int) -> Callable[[], None] | None:
    """Return one Linux-only pre-exec hook that terminates the child when the parent dies."""

    if sys.platform != "linux" or _PRCTL_LIBC is None:
        return None

    def _preexec() -> None:
        libc = _PRCTL_LIBC
        if libc is None:  # pragma: no cover - child inherits the ready global.
            return
        ctypes.set_errno(0)
        if libc.prctl(_PR_SET_PDEATHSIG, _PARENT_DEATH_SIGNAL, 0, 0, 0) != 0:
            errno_value = ctypes.get_errno() or 0
            raise OSError(errno_value, "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != expected_parent_pid:
            os.kill(os.getpid(), _PARENT_DEATH_SIGNAL)

    return _preexec


def _sanitize_log_text(value: object | None, *, max_chars: int = _MAX_LOG_TEXT_CHARS) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _coerce_sample_rate_hz(value: object | None) -> int:
    text = str(value or "").strip()
    try:
        sample_rate_hz = int(text) if text else _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ
    except ValueError:
        sample_rate_hz = _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ
    return max(_MIN_PLAYBACK_SAMPLE_RATE_HZ, sample_rate_hz)


def _extract_alsa_card_token(normalized_device: str) -> str | None:
    if not normalized_device:
        return None
    card_marker = "card="
    marker_index = normalized_device.find(card_marker)
    if marker_index >= 0:
        tail = normalized_device[marker_index + len(card_marker) :]
        token = tail.split(",", 1)[0].split(":", 1)[0].strip()
        return token or None
    if ":" not in normalized_device:
        return None
    prefix, tail = normalized_device.split(":", 1)
    if prefix not in _ALSA_CARD_TOKEN_PREFIXES:
        return None
    token = tail.split(",", 1)[0].strip()
    return token or None


def _read_alsa_cards() -> list[dict[str, str]]:
    try:
        with open("/proc/asound/cards", "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError:
        return []

    cards: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = _ALSA_CARD_LINE_RE.match(line)
        if match:
            current = {
                "index": match.group("index").strip(),
                "id": match.group("id").strip().lower(),
                "text": " ".join(
                    part for part in (match.group("id"), match.group("desc")) if part
                ).strip().lower(),
            }
            cards.append(current)
            continue
        if current is not None and (line.startswith(" ") or line.startswith("\t")):
            current["text"] = f'{current["text"]} {line.strip().lower()}'.strip()
        else:
            current = None
    return cards


def _alsa_card_token_targets_respeaker(card_token: str | None) -> bool:
    normalized_token = _normalize_device_string(card_token)
    if not normalized_token:
        return False
    for card in _read_alsa_cards():
        searchable = f'{card["index"]} {card["id"]} {card["text"]}'
        if normalized_token != card["index"] and normalized_token != card["id"]:
            continue
        if any(marker in searchable for marker in _RESPEAKER_DEVICE_MARKERS):
            return True
    return False


def config_targets_respeaker(*device_values: str | None) -> bool:
    """Return whether any configured device string explicitly targets XVF3800."""

    for value in device_values:
        normalized = _normalize_device_string(value)
        if not normalized:
            continue
        if any(marker in normalized for marker in _RESPEAKER_DEVICE_MARKERS):
            return True
        if _alsa_card_token_targets_respeaker(_extract_alsa_card_token(normalized)):
            return True
    return False


def respeaker_duplex_playback_supported(
    *,
    capture_device: str,
    playback_device: str,
) -> bool:
    """Return whether the active capture/playback pairing needs the XVF3800 guard."""

    if not config_targets_respeaker(capture_device):
        return False
    normalized_playback = _normalize_device_string(playback_device)
    if not normalized_playback:
        return False
    if any(
        compatible_alias in normalized_playback
        for compatible_alias in _PLAYBACK_DEVICES_COMPATIBLE_WITH_RESPEAKER_DUPLEX
    ):
        return True
    return config_targets_respeaker(playback_device)


def resolve_respeaker_duplex_playback_sample_rate_hz(config: TwinrConfig) -> int:
    """Return the playback sample rate used by short-lived XVF3800 duplex guards."""

    for attr_name in (
        "audio_output_sample_rate_hz",
        "audio_output_sample_rate",
        "audio_playback_sample_rate_hz",
        "audio_playback_sample_rate",
        "playback_sample_rate_hz",
        "tts_output_sample_rate_hz",
        "tts_output_sample_rate",
    ):
        raw_value = getattr(config, attr_name, None)
        text = str(raw_value or "").strip()
        try:
            sample_rate = int(text) if text else 0
        except ValueError:
            continue
        if sample_rate >= _MIN_PLAYBACK_SAMPLE_RATE_HZ:
            return sample_rate
    try:
        fallback = int(
            getattr(
                config,
                "openai_realtime_input_sample_rate",
                _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ,
            )
        )
    except (TypeError, ValueError):
        fallback = _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ
    return max(_MIN_PLAYBACK_SAMPLE_RATE_HZ, fallback)


@dataclass
class _ProcessHandle:
    process: subprocess.Popen[bytes]
    executable_path: str
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=_STDERR_RING_MAX_LINES))
    stderr_lock: threading.Lock = field(default_factory=threading.Lock)
    stderr_thread: threading.Thread | None = None

    def diagnostics(self) -> str:
        with self.stderr_lock:
            return " | ".join(self.stderr_lines).strip()


@dataclass
class _SharedPlaybackState:
    playback_key: str
    playback_device: str
    sample_rate_hz: int
    process_handle: _ProcessHandle | None
    ref_count: int = 1

    @property
    def alive(self) -> bool:
        handle = self.process_handle
        return handle is not None and handle.process.poll() is None

    def diagnostics(self) -> str:
        handle = self.process_handle
        if handle is None:
            return ""
        return handle.diagnostics()


def _start_stderr_drain_thread(handle: _ProcessHandle) -> None:
    pipe = handle.process.stderr
    if pipe is None:
        return

    def _drain() -> None:
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    return
                line = _sanitize_log_text(chunk.decode("utf-8", errors="ignore"))
                if not line:
                    continue
                with handle.stderr_lock:
                    handle.stderr_lines.append(line)
        except (OSError, ValueError):
            return

    thread = threading.Thread(
        target=_drain,
        name="respeaker-duplex-guard-stderr",
        daemon=True,
    )
    handle.stderr_thread = thread
    thread.start()


def _join_stderr_drain_thread(handle: _ProcessHandle, timeout_s: float = 0.10) -> None:
    thread = handle.stderr_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=timeout_s)


def _resolve_aplay_executable(env: dict[str, str]) -> str:
    path_override = env.get("PATH")
    resolved = shutil.which("aplay", path=path_override) or shutil.which("aplay")
    if not resolved:
        raise RuntimeError(
            "Required ReSpeaker duplex playback guard could not start because 'aplay' is not installed"
        )
    return resolved


def _build_aplay_command(*, executable_path: str, playback_device: str, sample_rate_hz: int) -> list[str]:
    return [
        executable_path,
        "-N",
        "-D",
        playback_device,
        "-q",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(sample_rate_hz),
        "-B",
        str(_APLAY_BUFFER_TIME_US),
        "-F",
        str(_APLAY_PERIOD_TIME_US),
        "-R",
        "0",
        "--fatal-errors",
        "/dev/zero",
    ]


def _spawn_aplay_handle(
    *,
    playback_device: str,
    sample_rate_hz: int,
    env: dict[str, str],
) -> _ProcessHandle:
    executable_path = _resolve_aplay_executable(env)
    command = _build_aplay_command(
        executable_path=executable_path,
        playback_device=playback_device,
        sample_rate_hz=sample_rate_hz,
    )
    preexec_fn = _build_parent_death_signal_preexec(os.getpid())
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            close_fds=True,
            # Keep the guard inside the current session/process group so the
            # runtime supervisor can reap it together with the owning loop.
            preexec_fn=preexec_fn,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Required ReSpeaker duplex playback guard could not start on {playback_device}: {exc}"
        ) from exc

    handle = _ProcessHandle(
        process=process,
        executable_path=executable_path,
    )
    _start_stderr_drain_thread(handle)
    return handle


def _wait_for_playback_ready(handle: _ProcessHandle) -> bool:
    deadline = time.monotonic() + _PROCESS_STARTUP_GRACE_S
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            _join_stderr_drain_thread(handle)
            return False
        time.sleep(0.01)
    time.sleep(_PROCESS_READY_SETTLE_S)
    if handle.process.poll() is not None:
        _join_stderr_drain_thread(handle)
        return False
    return True


def _terminate_process_group(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except OSError:
            return


def _stop_process_handle(handle: _ProcessHandle) -> None:
    process = handle.process
    try:
        if process.poll() is None:
            _terminate_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        _join_stderr_drain_thread(handle)
        _close_pipe(process.stderr)
        _close_pipe(process.stdout)


def _playback_device_busy(stderr_text: str) -> bool:
    normalized = _normalize_device_string(stderr_text)
    return any(marker in normalized for marker in _DEVICE_BUSY_MARKERS)


def _candidate_sample_rates_hz(playback_device: str, preferred_sample_rate_hz: int) -> list[int]:
    normalized_playback = _normalize_device_string(playback_device)
    cached = _SUCCESSFUL_SAMPLE_RATE_BY_DEVICE.get(normalized_playback)
    candidates = [
        cached,
        preferred_sample_rate_hz,
        48_000,
        24_000,
        16_000,
        44_100,
        32_000,
        8_000,
    ]
    result: list[int] = []
    for raw_value in candidates:
        sample_rate_hz = _coerce_sample_rate_hz(raw_value)
        if sample_rate_hz not in result:
            result.append(sample_rate_hz)
    return result


def _acquire_shared_guard(
    *,
    playback_device: str,
    preferred_sample_rate_hz: int,
) -> tuple[_SharedPlaybackState, str]:
    playback_key = _normalize_device_string(playback_device)
    env = build_audio_subprocess_env_for_mode(
        allow_root_borrowed_session_audio=True,
    )

    with _SHARED_GUARD_LOCK:
        existing = _SHARED_GUARDS.get(playback_key)
        if existing is not None and existing.alive:
            existing.ref_count += 1
            return existing, "reused"
        if existing is not None:
            _SHARED_GUARDS.pop(playback_key, None)

        last_error_text = ""
        for sample_rate_hz in _candidate_sample_rates_hz(playback_device, preferred_sample_rate_hz):
            handle = _spawn_aplay_handle(
                playback_device=playback_device,
                sample_rate_hz=sample_rate_hz,
                env=env,
            )
            if _wait_for_playback_ready(handle):
                _SUCCESSFUL_SAMPLE_RATE_BY_DEVICE[playback_key] = sample_rate_hz
                state = _SharedPlaybackState(
                    playback_key=playback_key,
                    playback_device=playback_device,
                    sample_rate_hz=sample_rate_hz,
                    process_handle=handle,
                    ref_count=1,
                )
                _SHARED_GUARDS[playback_key] = state
                return state, "started"

            stderr_text = handle.diagnostics()
            _stop_process_handle(handle)
            if _playback_device_busy(stderr_text):
                detail = stderr_text or "device or resource busy"
                raise RuntimeError(
                    "Required ReSpeaker duplex playback guard could not acquire "
                    f"{playback_device}: {detail}"
                )
            last_error_text = stderr_text or last_error_text

        raise RuntimeError(
            "Required ReSpeaker duplex playback guard exited immediately"
            if not last_error_text
            else f"Required ReSpeaker duplex playback guard exited immediately: {last_error_text}"
        )


def _release_shared_guard(state: _SharedPlaybackState) -> tuple[str, str]:
    with _SHARED_GUARD_LOCK:
        current = _SHARED_GUARDS.get(state.playback_key)
        if current is not state:
            return "stale", state.diagnostics()

        if current.ref_count > 1:
            current.ref_count -= 1
            return "released_shared", current.diagnostics()

        _SHARED_GUARDS.pop(state.playback_key, None)

    if state.process_handle is None:
        return "released_external", state.diagnostics()

    _stop_process_handle(state.process_handle)
    return "stopped", state.diagnostics()


class ReSpeakerDuplexPlaybackGuard:
    """Keep one silent playback stream open while a short ReSpeaker capture runs."""

    def __init__(
        self,
        *,
        capture_device: str,
        playback_device: str | None,
        sample_rate_hz: int = _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ,
    ) -> None:
        self._capture_device = str(capture_device or "").strip()
        self._playback_device = str(playback_device or "").strip()
        self._requested_sample_rate_hz = _coerce_sample_rate_hz(sample_rate_hz)
        self._state: _SharedPlaybackState | None = None
        self._last_diagnostics = ""

    @property
    def active(self) -> bool:
        """Return whether the guard currently has a running playback process."""

        state = self._state
        return state is not None and state.alive

    @property
    def sample_rate_hz(self) -> int:
        """Return the effective playback sample rate for the active guard."""

        state = self._state
        if state is None:
            return self._requested_sample_rate_hz
        return state.sample_rate_hz

    @property
    def diagnostics(self) -> str:
        """Return bounded diagnostic text collected from the guard process."""

        state = self._state
        if state is not None:
            diagnostics = state.diagnostics()
            if diagnostics:
                return diagnostics
        return self._last_diagnostics

    def __enter__(self) -> "ReSpeakerDuplexPlaybackGuard":
        if self._state is not None:
            return self
        if not respeaker_duplex_playback_supported(
            capture_device=self._capture_device,
            playback_device=self._playback_device,
        ):
            return self

        state, action = _acquire_shared_guard(
            playback_device=self._playback_device,
            preferred_sample_rate_hz=self._requested_sample_rate_hz,
        )
        self._state = state

        if action == "started":
            logger.info(
                "Started temporary ReSpeaker duplex playback guard | capture_device=%s | playback_device=%s | sample_rate_hz=%d",
                _sanitize_log_text(self._capture_device),
                _sanitize_log_text(self._playback_device),
                state.sample_rate_hz,
            )
        elif action == "reused":
            logger.info(
                "Reused temporary ReSpeaker duplex playback guard | capture_device=%s | playback_device=%s | sample_rate_hz=%d",
                _sanitize_log_text(self._capture_device),
                _sanitize_log_text(self._playback_device),
                state.sample_rate_hz,
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Stop or release the temporary playback stream when it is active."""

        state = self._state
        self._state = None
        if state is None:
            return

        action, diagnostics = _release_shared_guard(state)
        self._last_diagnostics = diagnostics

        if action == "released_shared":
            logger.info(
                "Released shared ReSpeaker duplex playback guard | capture_device=%s | playback_device=%s | sample_rate_hz=%d",
                _sanitize_log_text(self._capture_device),
                _sanitize_log_text(self._playback_device),
                state.sample_rate_hz,
            )
            return

        if action == "stopped":
            if diagnostics:
                logger.info(
                    "Stopped temporary ReSpeaker duplex playback guard | capture_device=%s | playback_device=%s | sample_rate_hz=%d | stderr=%s",
                    _sanitize_log_text(self._capture_device),
                    _sanitize_log_text(self._playback_device),
                    state.sample_rate_hz,
                    _sanitize_log_text(diagnostics),
                )
            else:
                logger.info(
                    "Stopped temporary ReSpeaker duplex playback guard | capture_device=%s | playback_device=%s | sample_rate_hz=%d",
                    _sanitize_log_text(self._capture_device),
                    _sanitize_log_text(self._playback_device),
                    state.sample_rate_hz,
                )
            return

        logger.debug(
            "Ignored stale ReSpeaker duplex playback guard release | capture_device=%s | playback_device=%s",
            _sanitize_log_text(self._capture_device),
            _sanitize_log_text(self._playback_device),
        )


def maybe_open_respeaker_duplex_playback_guard(
    *,
    capture_device: str,
    playback_device: str | None,
    sample_rate_hz: int = _DEFAULT_DUPLEX_PLAYBACK_SAMPLE_RATE_HZ,
) -> ReSpeakerDuplexPlaybackGuard:
    """Return one temporary playback guard for short ReSpeaker capture helpers."""

    return ReSpeakerDuplexPlaybackGuard(
        capture_device=capture_device,
        playback_device=playback_device,
        sample_rate_hz=sample_rate_hz,
    )


def _close_pipe(pipe: object | None) -> None:
    if pipe is None or not hasattr(pipe, "close"):
        return
    try:
        pipe.close()  # type: ignore[call-arg]
    except OSError:
        return


__all__ = [
    "ReSpeakerDuplexPlaybackGuard",
    "config_targets_respeaker",
    "maybe_open_respeaker_duplex_playback_guard",
    "respeaker_duplex_playback_supported",
    "resolve_respeaker_duplex_playback_sample_rate_hz",
]
