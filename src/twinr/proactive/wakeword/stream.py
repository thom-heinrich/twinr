"""Monitor live microphone PCM for wakeword detections with bounded lifecycle.

This module runs a bounded ``arecord`` worker, feeds exact frames into the
stream spotter, keeps short recent audio history for capture windows, and
surfaces detections and errors without letting subprocess failures escape the
runtime loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
import math
import os
import select
import shutil
import subprocess
import time
from typing import Callable, Protocol

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample

from ..runtime.presence import PresenceSessionSnapshot
from .matching import WakewordMatch


class WakewordFrameSpotter(Protocol):
    """Describe the streaming detector surface used by the monitor."""

    @property
    def frame_bytes(self) -> int:
        """Return the default frame size in bytes."""
        ...

    def frame_bytes_for_channels(self, channels: int) -> int:
        """Return the exact frame size for the supplied channel count."""
        ...

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        """Feed PCM bytes and optionally return a wakeword match."""
        ...

    def reset(self) -> None:
        """Reset detector state after disarm or runtime restart."""
        ...


@dataclass(frozen=True, slots=True)
class WakewordStreamDetection:
    """Describe one streaming wakeword detection with presence context."""

    match: WakewordMatch
    presence_snapshot: PresenceSessionSnapshot
    capture_window: AmbientAudioCaptureWindow | None = None


class OpenWakeWordStreamingMonitor:
    """Monitor live microphone audio and emit wakeword stream detections.

    The monitor owns exactly one worker thread plus one ``arecord``
    subprocess, keeps bounded detection and error queues, and resets detector
    state when the presence session disarms.
    """

    _SELECT_TIMEOUT_S = 0.2
    _STOP_WAIT_TIMEOUT_S = 1.0
    _STOP_JOIN_TIMEOUT_S = 3.0
    _MAX_STDERR_BYTES = 8_192
    _MAX_PENDING_DETECTIONS = 8
    _MAX_PENDING_ERRORS = 16

    def __init__(
        self,
        *,
        device: str,
        sample_rate: int,
        channels: int,
        spotter: WakewordFrameSpotter,
        attempt_cooldown_s: float,
        speech_threshold: int = 700,
        history_ms: int = 30_000,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        # AUDIT-FIX(#3): Validate constructor inputs early so bad .env or hardware config fails fast with a clear message.
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if channels <= 0:
            raise ValueError("channels must be greater than 0")
        if speech_threshold < 0:
            raise ValueError("speech_threshold must be greater than or equal to 0")
        frame_bytes = int(getattr(spotter, "frame_bytes", 0))
        if frame_bytes <= 0:
            raise ValueError("spotter.frame_bytes must be greater than 0")

        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.spotter = spotter
        self.attempt_cooldown_s = max(0.0, float(attempt_cooldown_s))
        self.speech_threshold = int(speech_threshold)
        self._frame_bytes = frame_bytes
        self.chunk_ms = max(20, int((self._frame_bytes * 1000) / (self.sample_rate * self.channels * 2)))
        self._history_frames = max(1, math.ceil(max(self.chunk_ms, history_ms) / self.chunk_ms))
        self.emit = emit or (lambda _line: None)
        self._presence_snapshot = PresenceSessionSnapshot(armed=False, reason="idle")
        self._presence_lock = Lock()
        self._history_lock = Lock()
        self._lifecycle_lock = Lock()
        self._spotter_lock = Lock()
        self._state_lock = Lock()
        self._recent_frames: deque[tuple[bytes, int]] = deque(maxlen=self._history_frames)
        # AUDIT-FIX(#9): Bound queues so a stalled consumer cannot grow memory without limit on a Raspberry Pi.
        self._detections: Queue[WakewordStreamDetection] = Queue(maxsize=self._MAX_PENDING_DETECTIONS)
        # AUDIT-FIX(#9): Bound error backlog as well and retain only the newest diagnostics.
        self._errors: Queue[str] = Queue(maxsize=self._MAX_PENDING_ERRORS)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._last_detection_at: float | None = None
        self._was_armed = False
        self._stderr_tail = bytearray()

    def open(self) -> "OpenWakeWordStreamingMonitor":
        """Start the worker thread and reset runtime state if needed."""

        with self._lifecycle_lock:
            # AUDIT-FIX(#1): Recover from stale dead worker threads instead of treating the monitor as still running.
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
                self._process = None
            if self._thread is not None:
                return self

            # AUDIT-FIX(#10): Reset runtime buffers so a fresh session cannot emit stale detections, errors, history, or cooldown state.
            self._reset_runtime_state()
            self._stop_event.clear()

            # AUDIT-FIX(#5): Start every session from a known detector state and serialize reset with frame processing.
            with self._spotter_lock:
                self.spotter.reset()

            thread = Thread(target=self._run, daemon=True, name="twinr-wakeword")
            self._thread = thread
            thread.start()
        return self

    def close(self) -> None:
        """Stop the worker thread and its child process."""

        with self._lifecycle_lock:
            thread = self._thread
            process = self._process
            if thread is None:
                return
            self._stop_event.set()

        # AUDIT-FIX(#7): Actively stop arecord during close so join does not wait on a child process that is still blocking the worker.
        if process is not None:
            self._stop_process(process)

        if thread is current_thread():
            return

        thread.join(timeout=self._STOP_JOIN_TIMEOUT_S)
        if thread.is_alive():
            # AUDIT-FIX(#1): Keep the live thread reference in place on stop timeout so a second open() cannot spawn a duplicate worker.
            self._queue_error("RuntimeError: wakeword stream worker did not stop cleanly before timeout")

    def __enter__(self) -> "OpenWakeWordStreamingMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update_presence(self, snapshot: PresenceSessionSnapshot) -> None:
        """Update the current presence snapshot and reset on disarm."""

        should_reset = False
        with self._presence_lock:
            self._presence_snapshot = snapshot
            should_reset = self._was_armed and not snapshot.armed
            self._was_armed = snapshot.armed

            if should_reset:
                # AUDIT-FIX(#5): Hold the presence lock while taking the spotter lock so disarm/reset is ordered against in-flight inference.
                with self._spotter_lock:
                    self.spotter.reset()

        if should_reset:
            # AUDIT-FIX(#10): Clear cooldown on disarm so the next arming cycle is not throttled by stale timing state.
            with self._state_lock:
                self._last_detection_at = None

    def poll_detection(self) -> WakewordStreamDetection | None:
        """Return the next queued wakeword detection if available."""

        try:
            return self._detections.get_nowait()
        except Empty:
            return None

    def poll_error(self) -> str | None:
        """Return the next queued streaming error if available."""

        try:
            return self._errors.get_nowait()
        except Empty:
            return None

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        """Return a recent audio window assembled from buffered frames."""

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
        """Return audio-level stats for the requested recent window."""

        return self.sample_window(duration_ms=duration_ms).sample

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        started = False
        pending_pcm = bytearray()

        try:
            command = self._build_command()
            # AUDIT-FIX(#4): Create the subprocess inside the guarded path so arecord startup failures are surfaced through the monitor error channel.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            if process.stdout is None:
                raise RuntimeError("Wakeword stream did not expose stdout")
            if process.stderr is None:
                raise RuntimeError("Wakeword stream did not expose stderr")

            # AUDIT-FIX(#6): Make both pipes non-blocking so stdout/stderr can be drained incrementally without deadlocking the worker.
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stderr.fileno(), False)

            with self._lifecycle_lock:
                self._process = process

            # AUDIT-FIX(#1): Abort cleanly if close() raced with startup so we do not publish a ghost "started" state.
            if self._stop_event.is_set():
                return

            started = True
            # AUDIT-FIX(#4): Emit "started" only after arecord and both pipes are actually ready.
            self._safe_emit("wakeword_stream=started")

            stdout_fd = process.stdout.fileno()
            stderr_fd = process.stderr.fileno()

            while not self._stop_event.is_set():
                if process.poll() is not None and not pending_pcm:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError(self._process_error_message(process))

                ready, _write_ready, _error_ready = select.select(
                    [stdout_fd, stderr_fd],
                    [],
                    [],
                    self._SELECT_TIMEOUT_S,
                )

                if stderr_fd in ready:
                    self._drain_stderr(process)

                if stdout_fd not in ready:
                    continue

                # AUDIT-FIX(#2): Reassemble the PCM stream into exact wakeword frames before RMS and model inference.
                pcm_chunk = self._read_stdout_chunk(stdout_fd, self._frame_bytes - len(pending_pcm))
                if not pcm_chunk:
                    if self._stop_event.is_set():
                        break
                    if process.poll() is None:
                        time.sleep(0.01)
                        continue
                    raise RuntimeError(self._process_error_message(process))

                pending_pcm.extend(pcm_chunk)

                while len(pending_pcm) >= self._frame_bytes:
                    pcm_bytes = bytes(pending_pcm[: self._frame_bytes])
                    del pending_pcm[: self._frame_bytes]
                    self._process_frame(pcm_bytes)

            self._drain_stderr(process)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._queue_error(self._format_exception(exc))
        finally:
            if process is not None:
                self._stop_process(process)

            with self._lifecycle_lock:
                if self._process is process:
                    self._process = None
                if self._thread is current_thread():
                    self._thread = None

            if started:
                # AUDIT-FIX(#1): Emit "stopped" from worker teardown so lifecycle telemetry matches the real stream state.
                self._safe_emit("wakeword_stream=stopped")

    def _process_frame(self, pcm_bytes: bytes) -> None:
        rms = _pcm16_rms(pcm_bytes)
        with self._history_lock:
            self._recent_frames.append((pcm_bytes, rms))

        with self._presence_lock:
            presence_snapshot = self._presence_snapshot
            if not presence_snapshot.armed:
                return
            # AUDIT-FIX(#5): Guard model access so reset() and process_pcm_bytes() cannot mutate detector state concurrently.
            with self._spotter_lock:
                match = self.spotter.process_pcm_bytes(pcm_bytes, channels=self.channels)

        if match is None:
            return

        now = time.monotonic()
        with self._state_lock:
            if self._last_detection_at is not None and (now - self._last_detection_at) < self.attempt_cooldown_s:
                return
            self._last_detection_at = now

        self._queue_put_latest(
            self._detections,
            WakewordStreamDetection(
                match=match,
                presence_snapshot=presence_snapshot,
                capture_window=self.sample_window(duration_ms=2500),
            ),
        )

    def _build_command(self) -> list[str]:
        arecord_path = shutil.which("arecord")
        if arecord_path is None:
            raise RuntimeError("arecord executable not found")
        return [
            arecord_path,
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

    def _process_error_message(self, process: subprocess.Popen[bytes]) -> str:
        self._drain_stderr(process)
        stderr = bytes(self._stderr_tail).strip()
        if stderr:
            return stderr.decode("utf-8", errors="ignore")
        return f"openWakeWord stream exited with code {process.returncode}"

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stderr = process.stderr
        if stderr is None:
            return

        while True:
            try:
                chunk = os.read(stderr.fileno(), 4096)
            except BlockingIOError:
                return
            except InterruptedError:
                continue
            except OSError:
                return

            if not chunk:
                return

            # AUDIT-FIX(#6): Keep only a bounded stderr tail so diagnostics survive without unbounded memory growth.
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > self._MAX_STDERR_BYTES:
                del self._stderr_tail[:-self._MAX_STDERR_BYTES]

    def _read_stdout_chunk(self, stdout_fd: int, target_size: int) -> bytes:
        read_size = max(1, target_size)
        try:
            return os.read(stdout_fd, read_size)
        except BlockingIOError:
            return b""
        except InterruptedError:
            return b""

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self._STOP_WAIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=self._STOP_WAIT_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            # AUDIT-FIX(#7): Always close child pipes so repeated open/close cycles do not leak file descriptors.
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass

    def _reset_runtime_state(self) -> None:
        with self._history_lock:
            self._recent_frames.clear()
        with self._state_lock:
            self._last_detection_at = None
        self._clear_queue(self._detections)
        self._clear_queue(self._errors)
        self._stderr_tail.clear()

    def _queue_error(self, message: str) -> None:
        normalized = message.strip() or "RuntimeError: wakeword stream failed without an error message"
        self._queue_put_latest(self._errors, normalized)
        self._safe_emit(f"wakeword_stream=error detail={normalized.replace(chr(10), ' | ')}")

    def _safe_emit(self, line: str) -> None:
        try:
            # AUDIT-FIX(#8): Never let telemetry/log emitters crash lifecycle or worker paths.
            self.emit(line)
        except Exception as exc:
            fallback = f"RuntimeError: emit failed with {self._format_exception(exc)}"
            self._queue_put_latest(self._errors, fallback)

    @staticmethod
    def _clear_queue(queue: Queue[object]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    @staticmethod
    def _queue_put_latest(queue: Queue[object], item: object) -> None:
        try:
            queue.put_nowait(item)
            return
        except Full:
            pass

        try:
            queue.get_nowait()
        except Empty:
            pass

        try:
            queue.put_nowait(item)
        except Full:
            pass

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        message = str(exc).strip()
        if not message:
            return exc.__class__.__name__
        return f"{exc.__class__.__name__}: {message}"


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    import math
    import sys
    from array import array

    # AUDIT-FIX(#2): Trim odd trailing bytes defensively so truncated PCM cannot raise array.frombytes().
    usable_length = len(samples) - (len(samples) % 2)
    if usable_length <= 0:
        return 0

    pcm_samples = array("h")
    pcm_samples.frombytes(samples[:usable_length])
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(math.sqrt(mean_square))


__all__ = [
    "OpenWakeWordStreamingMonitor",
    "WakewordFrameSpotter",
    "WakewordStreamDetection",
]
