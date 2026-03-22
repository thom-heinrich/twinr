"""Own the edge-side streaming voice bridge for the orchestrator path.

This module keeps `realtime_runner.py` orchestration-focused by handling the
bounded `arecord` lifecycle, websocket transport, and decision dispatch for the
Alexa-like hybrid voice path. It does not run turns itself; it only forwards
server-side wake/follow-up/barge-in decisions back into the realtime loop.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
from threading import Event, Lock, Thread, current_thread
from typing import Callable
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive import WakewordMatch
from twinr.orchestrator import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceFollowUpCaptureRequestedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
    OrchestratorVoiceWakeConfirmedEvent,
    OrchestratorVoiceWebSocketClient,
)


class EdgeVoiceOrchestrator:
    """Stream bounded PCM frames to the remote voice orchestrator service."""

    _SELECT_TIMEOUT_S = 0.2
    _STOP_WAIT_TIMEOUT_S = 1.0
    _STOP_JOIN_TIMEOUT_S = 3.0
    _MAX_STDERR_BYTES = 8_192

    def __init__(
        self,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None],
        on_wakeword_match: Callable[[WakewordMatch], bool],
        on_follow_up_capture_requested: Callable[[], bool],
        on_barge_in_interrupt: Callable[[], bool],
    ) -> None:
        self.config = config
        self.emit = emit
        self._on_wakeword_match = on_wakeword_match
        self._on_follow_up_capture_requested = on_follow_up_capture_requested
        self._on_barge_in_interrupt = on_barge_in_interrupt
        self._device = (
            str(config.voice_orchestrator_audio_device or config.proactive_audio_input_device or config.audio_input_device)
            .strip()
            or "default"
        )
        self._sample_rate = int(config.audio_sample_rate)
        self._channels = int(config.audio_channels)
        self._chunk_ms = max(20, int(config.audio_chunk_ms))
        self._frame_bytes = max(320, int(round(self._sample_rate * (self._chunk_ms / 1000.0))) * self._channels * 2)
        self._client = OrchestratorVoiceWebSocketClient(
            config.voice_orchestrator_ws_url,
            shared_secret=config.voice_orchestrator_shared_secret,
            on_event=self._handle_server_event,
            require_tls=False,
        )
        self._session_id = f"voice-{uuid4().hex[:12]}"
        self._stop_event = Event()
        self._paused = Event()
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()
        self._thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0
        self._stderr_tail = bytearray()
        self._connected = False

    def open(self) -> "EdgeVoiceOrchestrator":
        """Connect the websocket and start the bounded capture worker."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._paused.clear()
            self._sequence = 0
            self._stderr_tail.clear()
            try:
                self._connect_client()
            except Exception as exc:
                self._connected = False
                self.emit(f"voice_orchestrator_unavailable={type(exc).__name__}")
                return self
            thread = Thread(target=self._capture_loop, daemon=True, name="twinr-voice-orchestrator")
            self._thread = thread
            thread.start()
        return self

    def close(self) -> None:
        """Stop capture and close the websocket transport."""

        with self._lifecycle_lock:
            thread = self._thread
            process = self._process
            self._stop_event.set()
        if process is not None:
            self._stop_process(process)
        if thread is not None and thread is not current_thread():
            thread.join(timeout=self._STOP_JOIN_TIMEOUT_S)
        self._client.close()
        with self._lifecycle_lock:
            self._thread = None
            self._process = None
            self._connected = False

    def __enter__(self) -> "EdgeVoiceOrchestrator":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def pause_capture(self, *, reason: str) -> None:
        """Pause the long-lived microphone stream before a local bounded capture."""

        self._paused.set()
        self.emit(f"voice_orchestrator_capture_paused={reason}")
        process = self._process
        if process is not None:
            self._stop_process(process)

    def resume_capture(self, *, reason: str) -> None:
        """Resume microphone streaming after a local bounded capture."""

        self.emit(f"voice_orchestrator_capture_resumed={reason}")
        self._paused.clear()

    def notify_runtime_state(self, *, state: str, detail: str | None = None, follow_up_allowed: bool = False) -> None:
        """Send the current edge runtime state to the server."""

        if not self._connected:
            return
        try:
            self._client.send_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state=state,
                    detail=detail,
                    follow_up_allowed=follow_up_allowed,
                )
            )
        except Exception as exc:
            self.emit(f"voice_orchestrator_state_failed={type(exc).__name__}")
            self._connected = False

    def _connect_client(self) -> None:
        self._client.open()
        self._client.send_hello(
            OrchestratorVoiceHelloRequest(
                session_id=self._session_id,
                sample_rate=self._sample_rate,
                channels=self._channels,
                chunk_ms=self._chunk_ms,
                initial_state="waiting",
            )
        )
        self._connected = True

    def _capture_loop(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        pending_pcm = bytearray()
        started = False
        try:
            while not self._stop_event.is_set():
                if self._paused.is_set():
                    if process is not None:
                        self._stop_process(process)
                        process = None
                        with self._lifecycle_lock:
                            if self._process is process:
                                self._process = None
                    self._stop_event.wait(0.05)
                    continue
                if process is None:
                    process = self._start_process()
                    started = True
                    pending_pcm.clear()
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("Voice orchestrator capture did not expose stdout/stderr")
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()
                ready, _write_ready, _error_ready = select.select(
                    [stdout_fd, stderr_fd],
                    [],
                    [],
                    self._SELECT_TIMEOUT_S,
                )
                if stderr_fd in ready:
                    self._drain_stderr(process)
                if stdout_fd not in ready:
                    if process.poll() is not None:
                        raise RuntimeError(self._process_error_message(process))
                    continue
                pcm_chunk = self._read_stdout_chunk(stdout_fd, self._frame_bytes - len(pending_pcm))
                if not pcm_chunk:
                    if process.poll() is not None:
                        raise RuntimeError(self._process_error_message(process))
                    continue
                pending_pcm.extend(pcm_chunk)
                while len(pending_pcm) >= self._frame_bytes:
                    frame_bytes = bytes(pending_pcm[: self._frame_bytes])
                    del pending_pcm[: self._frame_bytes]
                    self._send_frame(frame_bytes)
            self._drain_stderr(process)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.emit(f"voice_orchestrator_capture_failed={type(exc).__name__}")
        finally:
            if process is not None:
                self._stop_process(process)
            with self._lifecycle_lock:
                if self._process is process:
                    self._process = None
                if self._thread is current_thread():
                    self._thread = None
            if started:
                self.emit("voice_orchestrator_capture=stopped")

    def _send_frame(self, pcm_bytes: bytes) -> None:
        if not self._connected or not pcm_bytes:
            return
        try:
            self._client.send_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=self._sequence, pcm_bytes=pcm_bytes)
            )
            self._sequence += 1
        except Exception as exc:
            self._connected = False
            self.emit(f"voice_orchestrator_send_failed={type(exc).__name__}")

    def _handle_server_event(self, event) -> None:
        if isinstance(event, OrchestratorVoiceReadyEvent):
            self.emit(f"voice_orchestrator_ready={event.backend}")
            self._connected = True
            return
        if isinstance(event, OrchestratorVoiceWakeConfirmedEvent):
            self.emit(f"voice_orchestrator_wake_confirmed={event.matched_phrase or 'unknown'}")
            self._on_wakeword_match(
                WakewordMatch(
                    detected=True,
                    transcript="",
                    matched_phrase=event.matched_phrase,
                    remaining_text=event.remaining_text,
                    normalized_transcript="",
                    backend=event.backend,
                    detector_label=event.detector_label,
                    score=event.score,
                )
            )
            return
        if isinstance(event, OrchestratorVoiceFollowUpCaptureRequestedEvent):
            self.emit("voice_orchestrator_follow_up_capture_requested=true")
            self._on_follow_up_capture_requested()
            return
        if isinstance(event, OrchestratorVoiceBargeInInterruptEvent):
            self.emit("voice_orchestrator_barge_in_interrupt=true")
            self._on_barge_in_interrupt()
            return
        if isinstance(event, OrchestratorVoiceErrorEvent):
            self.emit(f"voice_orchestrator_error={event.error}")
            self._connected = False
            return
        self.emit(f"voice_orchestrator_event={type(event).__name__}")

    def _start_process(self) -> subprocess.Popen[bytes]:
        arecord_path = shutil.which("arecord")
        if arecord_path is None:
            raise RuntimeError("arecord executable not found")
        process = subprocess.Popen(
            [
                arecord_path,
                "-D",
                self._device,
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                str(self._channels),
                "-r",
                str(self._sample_rate),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Voice orchestrator capture process did not expose stdout/stderr")
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        with self._lifecycle_lock:
            self._process = process
        self.emit("voice_orchestrator_capture=started")
        return process

    def _read_stdout_chunk(self, stdout_fd: int, read_size: int) -> bytes:
        try:
            return os.read(stdout_fd, max(1, read_size))
        except (BlockingIOError, InterruptedError):
            return b""

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        while True:
            try:
                chunk = os.read(stderr.fileno(), 4096)
            except (BlockingIOError, InterruptedError, OSError):
                return
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > self._MAX_STDERR_BYTES:
                del self._stderr_tail[:-self._MAX_STDERR_BYTES]

    def _process_error_message(self, process: subprocess.Popen[bytes]) -> str:
        self._drain_stderr(process)
        stderr = bytes(self._stderr_tail).strip()
        if stderr:
            return stderr.decode("utf-8", errors="ignore")
        return f"Voice orchestrator capture exited with code {process.returncode}"

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
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


__all__ = ["EdgeVoiceOrchestrator"]
