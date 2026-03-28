"""Keep one Twinr-owned playback stream open for XVF3800 duplex stability.

The productive Pi can reproduce a ReSpeaker XVF3800 failure mode where opening
capture without an already-active Twinr playback stream causes immediate
``arecord`` ``Input/output error`` failures. This helper keeps a low-priority
silent PCM stream alive behind the existing playback coordinator so real speech,
beeps, and feedback can still preempt it cleanly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import Event, Lock, Thread, current_thread

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator
from twinr.hardware.respeaker.probe import config_targets_respeaker

_KEEPALIVE_OWNER = "respeaker_duplex_keepalive"
_KEEPALIVE_PRIORITY = 0
_DEFAULT_SAMPLE_RATE_HZ = 24000
_DEFAULT_CHUNK_MS = 200
_RESTART_DELAY_S = 0.05
_PLAYBACK_DEVICES_COMPATIBLE_WITH_RESPEAKER_DUPLEX = frozenset(
    {
        "twinr_playback_hw",
        "twinr_playback_softvol",
    }
)


def build_respeaker_duplex_keepalive(
    *,
    config: TwinrConfig,
    capture_device: str,
    playback_coordinator: PlaybackCoordinator | None,
    emit: Callable[[str], None] | None = None,
) -> "ReSpeakerDuplexKeepalive | None":
    """Return one keepalive helper when the current Pi audio path needs it."""

    if playback_coordinator is None:
        return None
    playback_device = str(getattr(config, "audio_output_device", "") or "").strip()
    if not respeaker_duplex_keepalive_supported(
        capture_device=capture_device,
        playback_device=playback_device,
    ):
        return None
    return ReSpeakerDuplexKeepalive(
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


class ReSpeakerDuplexKeepalive:
    """Own one restartable silent playback request for XVF3800 capture stability."""

    def __init__(
        self,
        *,
        playback_coordinator: PlaybackCoordinator,
        sample_rate_hz: int,
        emit: Callable[[str], None] | None = None,
        chunk_ms: int = _DEFAULT_CHUNK_MS,
    ) -> None:
        self._playback_coordinator = playback_coordinator
        self._sample_rate_hz = max(8000, int(sample_rate_hz))
        self._chunk_ms = max(20, int(chunk_ms))
        self._emit = emit
        self._stop_event = Event()
        self._thread_lock = Lock()
        self._thread: Thread | None = None

    def open(self) -> "ReSpeakerDuplexKeepalive":
        """Start the background keepalive loop once."""

        with self._thread_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return self
            self._stop_event.clear()
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
        self._playback_coordinator.stop_owner(_KEEPALIVE_OWNER)
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1.5)
        with self._thread_lock:
            if self._thread is thread:
                self._thread = None
        self._safe_emit("voice_orchestrator_duplex_keepalive=stopped")

    def _worker_main(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    handle = self._playback_coordinator.submit_pcm16_chunks(
                        owner=_KEEPALIVE_OWNER,
                        priority=_KEEPALIVE_PRIORITY,
                        chunks=self._iter_silence_chunks(),
                        sample_rate=self._sample_rate_hz,
                        channels=1,
                        should_stop=self._stop_event.is_set,
                        supersede_pending_owner=True,
                    )
                    handle.wait()
                except Exception as exc:  # Defensive containment: keepalive failures must not kill runtime threads.
                    self._safe_emit(f"voice_orchestrator_duplex_keepalive_error={type(exc).__name__}")
                    if self._stop_event.wait(0.25):
                        break
                if self._stop_event.wait(_RESTART_DELAY_S):
                    break
        finally:
            with self._thread_lock:
                if self._thread is current_thread():
                    self._thread = None

    def _iter_silence_chunks(self) -> Iterator[bytes]:
        frames_per_chunk = max(1, int(round(self._sample_rate_hz * (self._chunk_ms / 1000.0))))
        chunk = b"\x00" * (frames_per_chunk * 2)
        while not self._stop_event.is_set():
            yield chunk

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
