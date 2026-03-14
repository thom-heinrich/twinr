from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
import math
import os
import select
import subprocess
import time

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.openwakeword_spotter import WakewordOpenWakeWordFrameSpotter
from twinr.proactive.presence import PresenceSessionSnapshot
from twinr.proactive.wakeword import WakewordMatch


@dataclass(frozen=True, slots=True)
class WakewordStreamDetection:
    match: WakewordMatch
    presence_snapshot: PresenceSessionSnapshot
    capture_window: AmbientAudioCaptureWindow | None = None


class OpenWakeWordStreamingMonitor:
    def __init__(
        self,
        *,
        device: str,
        sample_rate: int,
        channels: int,
        spotter: WakewordOpenWakeWordFrameSpotter,
        attempt_cooldown_s: float,
        speech_threshold: int = 700,
        history_ms: int = 30_000,
        emit=None,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.spotter = spotter
        self.attempt_cooldown_s = max(0.0, float(attempt_cooldown_s))
        self.speech_threshold = int(speech_threshold)
        self.chunk_ms = max(20, int((self.spotter.frame_bytes * 1000) / (self.sample_rate * self.channels * 2)))
        self._history_frames = max(1, math.ceil(max(self.chunk_ms, history_ms) / self.chunk_ms))
        self.emit = emit or (lambda _line: None)
        self._presence_snapshot = PresenceSessionSnapshot(armed=False, reason="idle")
        self._presence_lock = Lock()
        self._history_lock = Lock()
        self._recent_frames: deque[tuple[bytes, int]] = deque(maxlen=self._history_frames)
        self._detections: Queue[WakewordStreamDetection] = Queue()
        self._errors: Queue[str] = Queue()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_detection_at: float | None = None
        self._was_armed = False

    def open(self) -> "OpenWakeWordStreamingMonitor":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="twinr-openwakeword")
        self._thread.start()
        self.emit("wakeword_stream=started")
        return self

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=2.0)
        self._thread = None
        self.emit("wakeword_stream=stopped")

    def __enter__(self) -> "OpenWakeWordStreamingMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update_presence(self, snapshot: PresenceSessionSnapshot) -> None:
        with self._presence_lock:
            self._presence_snapshot = snapshot
        if self._was_armed and not snapshot.armed:
            self.spotter.reset()
        self._was_armed = snapshot.armed

    def poll_detection(self) -> WakewordStreamDetection | None:
        try:
            return self._detections.get_nowait()
        except Empty:
            return None

    def poll_error(self) -> str | None:
        try:
            return self._errors.get_nowait()
        except Empty:
            return None

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        target_duration_ms = max(self.chunk_ms, duration_ms or self.chunk_ms)
        target_frames = max(1, math.ceil(target_duration_ms / self.chunk_ms))
        with self._history_lock:
            frames = list(self._recent_frames)[-target_frames:]
        if not frames:
            frames = [(b"", 0)]
        pcm_fragments = [chunk for chunk, _rms in frames]
        rms_values = [rms for _chunk, rms in frames]
        active_chunk_count = sum(1 for rms in rms_values if rms >= self.speech_threshold)
        sample = AmbientAudioLevelSample(
            duration_ms=len(frames) * self.chunk_ms,
            chunk_count=len(frames),
            active_chunk_count=active_chunk_count,
            average_rms=int(sum(rms_values) / len(rms_values)),
            peak_rms=max(rms_values),
            active_ratio=active_chunk_count / max(1, len(frames)),
        )
        return AmbientAudioCaptureWindow(
            sample=sample,
            pcm_bytes=b"".join(pcm_fragments),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def sample_levels(self, *, duration_ms: int | None = None) -> AmbientAudioLevelSample:
        return self.sample_window(duration_ms=duration_ms).sample

    def _run(self) -> None:
        command = [
            "arecord",
            "-D",
            self.device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("openWakeWord stream did not expose stdout")
            while not self._stop_event.is_set():
                if process.poll() is not None:
                    raise RuntimeError(self._process_error_message(process))
                ready, _write_ready, _error_ready = select.select([process.stdout], [], [], 0.2)
                if not ready:
                    continue
                pcm_bytes = os.read(process.stdout.fileno(), self.spotter.frame_bytes)
                if not pcm_bytes:
                    continue
                rms = _pcm16_rms(pcm_bytes)
                with self._history_lock:
                    self._recent_frames.append((pcm_bytes, rms))
                with self._presence_lock:
                    presence_snapshot = self._presence_snapshot
                if not presence_snapshot.armed:
                    continue
                match = self.spotter.process_pcm_bytes(pcm_bytes, channels=self.channels)
                if match is None:
                    continue
                now = time.monotonic()
                if self._last_detection_at is not None and (now - self._last_detection_at) < self.attempt_cooldown_s:
                    continue
                self._last_detection_at = now
                self._detections.put(
                    WakewordStreamDetection(
                        match=match,
                        presence_snapshot=presence_snapshot,
                        capture_window=self.sample_window(duration_ms=2500),
                    )
                )
        except Exception as exc:
            self._errors.put(str(exc))
        finally:
            self._stop_process(process)

    def _process_error_message(self, process: subprocess.Popen[bytes]) -> str:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        if stderr:
            return stderr.decode("utf-8", errors="ignore")
        return f"openWakeWord stream exited with code {process.returncode}"

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    import math
    import sys
    from array import array

    pcm_samples = array("h")
    pcm_samples.frombytes(samples)
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(math.sqrt(mean_square))


__all__ = [
    "OpenWakeWordStreamingMonitor",
    "WakewordStreamDetection",
]
