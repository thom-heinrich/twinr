"""Keep one Twinr-owned playback stream open for XVF3800 duplex stability.

The productive Pi can reproduce a ReSpeaker XVF3800 failure mode where opening
capture without an already-active Twinr playback stream causes immediate
``arecord`` ``Input/output error`` failures. On the productive
``twinr_playback_softvol`` path, feeding endless silence through the normal
``WaveAudioPlayer`` stream can falsely time out while waiting for the pipe to
become writable again even though a direct ``aplay /dev/zero`` guard stays
healthy. This helper therefore prefers the direct guard on the softvol path and
keeps the older coordinator-backed silent PCM fallback for other Twinr-owned
playback devices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import Condition, Event, Lock, Thread, current_thread
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackActivityEvent, PlaybackCoordinator
from twinr.hardware.respeaker.probe import config_targets_respeaker
from twinr.hardware.respeaker_duplex_playback import maybe_open_respeaker_duplex_playback_guard

_KEEPALIVE_OWNER = "respeaker_duplex_keepalive"
_KEEPALIVE_PRIORITY = 0
_DEFAULT_SAMPLE_RATE_HZ = 24000
_DEFAULT_CHUNK_MS = 200
_DIRECT_GUARD_POLL_S = 0.05
_DIRECT_GUARD_RELEASE_TIMEOUT_S = 1.0
_DIRECT_GUARD_RESUME_GRACE_S = 0.35
_KEEPALIVE_ERROR_MAX_CHARS = 180
_RESTART_DELAY_S = 0.05
_PLAYBACK_DEVICES_COMPATIBLE_WITH_RESPEAKER_DUPLEX = frozenset(
    {
        "twinr_playback_hw",
        "twinr_playback_softvol",
    }
)
_DIRECT_GUARD_KEEPALIVE_PLAYBACK_DEVICES = frozenset({"twinr_playback_softvol"})


def build_respeaker_duplex_keepalive(
    *,
    config: TwinrConfig,
    capture_device: str,
    playback_coordinator: PlaybackCoordinator | None,
    emit: Callable[[str], None] | None = None,
) -> "ReSpeakerDuplexKeepalive | None":
    """Return one keepalive helper when the current Pi audio path needs it."""

    playback_device = str(getattr(config, "audio_output_device", "") or "").strip()
    if not respeaker_duplex_keepalive_supported(
        capture_device=capture_device,
        playback_device=playback_device,
    ):
        return None
    if not _prefers_direct_guard_keepalive(playback_device) and playback_coordinator is None:
        return None
    return ReSpeakerDuplexKeepalive(
        capture_device=capture_device,
        playback_device=playback_device,
        playback_coordinator=playback_coordinator,
        sample_rate_hz=_resolve_playback_sample_rate_hz(config),
        emit=emit,
    )


def respeaker_duplex_keepalive_supported(
    *,
    capture_device: str,
    playback_device: str,
) -> bool:
    """Return whether the active capture/playback path matches the Pi workaround."""

    if not config_targets_respeaker(capture_device):
        return False
    normalized_playback = str(playback_device or "").strip().lower()
    if not normalized_playback:
        return False
    if normalized_playback in _PLAYBACK_DEVICES_COMPATIBLE_WITH_RESPEAKER_DUPLEX:
        return True
    return config_targets_respeaker(playback_device)


def _resolve_playback_sample_rate_hz(config: TwinrConfig) -> int:
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
        try:
            sample_rate = int(raw_value)
        except (TypeError, ValueError):
            continue
        if sample_rate >= 8000:
            return sample_rate
    try:
        fallback = int(getattr(config, "openai_realtime_input_sample_rate", _DEFAULT_SAMPLE_RATE_HZ))
    except (TypeError, ValueError):
        fallback = _DEFAULT_SAMPLE_RATE_HZ
    return max(8000, fallback)


def _prefers_direct_guard_keepalive(playback_device: str) -> bool:
    normalized = str(playback_device or "").strip().lower()
    return normalized in _DIRECT_GUARD_KEEPALIVE_PLAYBACK_DEVICES


class ReSpeakerDuplexKeepalive:
    """Own one restartable silent playback request for XVF3800 capture stability."""

    def __init__(
        self,
        *,
        capture_device: str,
        playback_device: str,
        playback_coordinator: PlaybackCoordinator | None,
        sample_rate_hz: int,
        emit: Callable[[str], None] | None = None,
        chunk_ms: int = _DEFAULT_CHUNK_MS,
    ) -> None:
        self._capture_device = str(capture_device or "").strip()
        self._playback_device = str(playback_device or "").strip()
        self._playback_coordinator = playback_coordinator
        self._prefers_direct_guard = _prefers_direct_guard_keepalive(self._playback_device)
        self._sample_rate_hz = max(8000, int(sample_rate_hz))
        self._chunk_ms = max(20, int(chunk_ms))
        self._emit = emit
        self._stop_event = Event()
        self._thread_lock = Lock()
        self._thread: Thread | None = None
        self._direct_guard_state = Condition(Lock())
        self._direct_guard_active = False
        self._direct_guard_transitioning = False
        self._foreground_playback_depth = 0
        self._resume_guard_not_before_monotonic = 0.0

    def open(self) -> "ReSpeakerDuplexKeepalive":
        """Start the background keepalive loop once."""

        with self._thread_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return self
            self._stop_event.clear()
            with self._direct_guard_state:
                self._foreground_playback_depth = 0
                self._direct_guard_active = False
                self._direct_guard_transitioning = False
                self._resume_guard_not_before_monotonic = 0.0
                self._direct_guard_state.notify_all()
            thread = Thread(
                target=self._worker_main,
                name="twinr-respeaker-duplex-keepalive",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        self._safe_emit("voice_orchestrator_duplex_keepalive=started")
        return self

    def close(self) -> None:
        """Stop the keepalive loop and cancel the queued/running silent stream."""

        with self._thread_lock:
            thread = self._thread
            self._stop_event.set()
        with self._direct_guard_state:
            self._foreground_playback_depth = 0
            self._direct_guard_active = False
            self._direct_guard_transitioning = False
            self._direct_guard_state.notify_all()
        coordinator = self._playback_coordinator
        if coordinator is not None:
            coordinator.stop_owner(_KEEPALIVE_OWNER)
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1.5)
        with self._thread_lock:
            if self._thread is thread:
                self._thread = None
        self._safe_emit("voice_orchestrator_duplex_keepalive=stopped")

    def handle_playback_activity(self, event: PlaybackActivityEvent) -> None:
        """Suspend the direct guard while a real playback request owns the speaker."""

        if not self._prefers_direct_guard:
            return
        if event.owner == _KEEPALIVE_OWNER:
            return
        if event.phase == "starting":
            self._pause_direct_guard_for_foreground_playback(owner=event.owner)
            return
        if event.phase == "finished":
            self._resume_direct_guard_after_foreground_playback()

    def _worker_main(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    if self._prefers_direct_guard:
                        self._run_direct_guard_keepalive()
                    else:
                        self._run_coordinator_keepalive()
                except Exception as exc:  # Defensive containment: keepalive failures must not kill runtime threads.
                    self._safe_emit(
                        "voice_orchestrator_duplex_keepalive_error="
                        f"{type(exc).__name__}:{self._format_error_detail(exc)}"
                    )
                    if self._stop_event.wait(0.25):
                        break
                if self._stop_event.wait(_RESTART_DELAY_S):
                    break
        finally:
            with self._thread_lock:
                if self._thread is current_thread():
                    self._thread = None

    def _run_direct_guard_keepalive(self) -> None:
        while not self._stop_event.is_set():
            with self._direct_guard_state:
                while not self._stop_event.is_set():
                    if self._foreground_playback_depth > 0:
                        self._direct_guard_state.wait(timeout=_DIRECT_GUARD_POLL_S)
                        continue
                    resume_guard_not_before = self._resume_guard_not_before_monotonic
                    if resume_guard_not_before <= 0.0:
                        break
                    # Avoid reclaiming the softvol device in the tiny gap between
                    # processing feedback and the next real TTS/playback request.
                    remaining_s = resume_guard_not_before - time.monotonic()
                    if remaining_s <= 0.0:
                        self._resume_guard_not_before_monotonic = 0.0
                        break
                    self._direct_guard_state.wait(timeout=min(_DIRECT_GUARD_POLL_S, remaining_s))
                if self._stop_event.is_set():
                    return

            self._set_direct_guard_state(active=False, transitioning=True)
            try:
                guard = maybe_open_respeaker_duplex_playback_guard(
                    capture_device=self._capture_device,
                    playback_device=self._playback_device,
                    sample_rate_hz=self._sample_rate_hz,
                )
                with guard:
                    self._set_direct_guard_state(
                        active=bool(getattr(guard, "active", False)),
                        transitioning=False,
                    )
                    while not self._stop_event.wait(_DIRECT_GUARD_POLL_S):
                        with self._direct_guard_state:
                            if self._foreground_playback_depth > 0:
                                break
            finally:
                self._set_direct_guard_state(active=False, transitioning=False)

    def _run_coordinator_keepalive(self) -> None:
        coordinator = self._playback_coordinator
        if coordinator is None:
            raise RuntimeError("playback coordinator unavailable for duplex keepalive fallback")
        handle = coordinator.submit_pcm16_chunks(
            owner=_KEEPALIVE_OWNER,
            priority=_KEEPALIVE_PRIORITY,
            chunks=self._iter_silence_chunks(),
            sample_rate=self._sample_rate_hz,
            channels=1,
            should_stop=self._stop_event.is_set,
            supersede_pending_owner=True,
        )
        handle.wait()

    def _iter_silence_chunks(self) -> Iterator[bytes]:
        frames_per_chunk = max(1, int(round(self._sample_rate_hz * (self._chunk_ms / 1000.0))))
        chunk = b"\x00" * (frames_per_chunk * 2)
        while not self._stop_event.is_set():
            yield chunk

    def _pause_direct_guard_for_foreground_playback(self, *, owner: str) -> None:
        self._safe_emit(
            "voice_orchestrator_duplex_keepalive="
            f"paused_for_playback:{self._format_text(owner)}"
        )
        with self._direct_guard_state:
            self._foreground_playback_depth += 1
            self._direct_guard_state.notify_all()
            deadline = time.monotonic() + _DIRECT_GUARD_RELEASE_TIMEOUT_S
            while (
                (self._direct_guard_active or self._direct_guard_transitioning)
                and time.monotonic() < deadline
            ):
                self._direct_guard_state.wait(timeout=max(0.01, deadline - time.monotonic()))
            timed_out = self._direct_guard_active or self._direct_guard_transitioning
        if timed_out:
            self._safe_emit(
                "voice_orchestrator_duplex_keepalive_pause_timeout="
                f"{self._format_text(owner)}"
            )

    def _resume_direct_guard_after_foreground_playback(self) -> None:
        self._safe_emit("voice_orchestrator_duplex_keepalive=resume_grace_after_playback")
        with self._direct_guard_state:
            if self._foreground_playback_depth > 0:
                self._foreground_playback_depth -= 1
            if self._foreground_playback_depth == 0:
                self._resume_guard_not_before_monotonic = time.monotonic() + _DIRECT_GUARD_RESUME_GRACE_S
            self._direct_guard_state.notify_all()

    def _set_direct_guard_state(self, *, active: bool, transitioning: bool) -> None:
        with self._direct_guard_state:
            self._direct_guard_active = bool(active)
            self._direct_guard_transitioning = bool(transitioning)
            self._direct_guard_state.notify_all()

    @staticmethod
    def _format_error_detail(exc: Exception) -> str:
        text = " ".join(str(exc).split())
        if not text:
            return "unknown"
        if len(text) > _KEEPALIVE_ERROR_MAX_CHARS:
            return text[: _KEEPALIVE_ERROR_MAX_CHARS - 1].rstrip() + "…"
        return text

    @staticmethod
    def _format_text(value: object) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return "unknown"
        if len(text) > _KEEPALIVE_ERROR_MAX_CHARS:
            return text[: _KEEPALIVE_ERROR_MAX_CHARS - 1].rstrip() + "…"
        return text

    def _safe_emit(self, message: str) -> None:
        callback = self._emit
        if not callable(callback):
            return
        try:
            callback(message)
        except Exception:
            return


__all__ = [
    "ReSpeakerDuplexKeepalive",
    "build_respeaker_duplex_keepalive",
    "respeaker_duplex_keepalive_supported",
]
