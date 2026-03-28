# CHANGELOG: 2026-03-28
# BUG-1: close() now flushes pending text before shutdown, preventing final-segment loss and close()/wait_until_idle hangs.
# BUG-2: abort() now preempts active/queued speech without killing the worker, so the instance remains reusable after an interrupt.
# BUG-3: first-audio and speaking-started state now resets per answer instead of once per object lifetime.
# BUG-4: playback arbitration now waits for real first audio on every path, so stalled TTS no longer grabs the playback lock early.
# SEC-1: bounded TTS-chunk buffering plus queued/pending text caps prevent practical memory-exhaustion denial of service on Raspberry Pi 4.
# SEC-2: submit_* after close() or worker failure now fails fast instead of silently accumulating unreachable speech.
# IMP-1: bounded first-chunk and inter-chunk stall timeouts prevent wedged providers from pinning the session forever.
# IMP-2: forced bounded segmentation and queue draining on preempt keep context/backlog bounded and improve long-form streaming stability.
# BUG-5: buffer header-only WAV prefixes before signaling "first audio", preventing silent-speaking states and false 2s chunk stalls on live Pi TTS.
# BREAKING: abort() is now non-terminal and keeps the playback worker reusable; call close() to terminate the worker.
# BREAKING: first_chunk_timeout_s and chunk_stall_timeout_s now default to bounded values instead of waiting forever.
# BREAKING: submit_* after close() or worker failure now raises RuntimeError.

"""Stream interruptible text-to-speech playback for dual-lane workflows.

This module owns the low-latency spoken-output path used by the streaming
workflow. It deliberately keeps queueing, first-audio signaling, and abort
behavior separate so a stalled TTS request cannot pin the whole conversation
session in ``answering``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable, Protocol, cast
import logging
import time

from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority


class StreamingTextToSpeechProviderLike(Protocol):
    """Describe the streaming TTS interface required by speech output."""

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        ...


class WaveAudioPlayerLike(Protocol):
    """Describe the chunked WAV playback interface required by speech output."""

    def play_wav_chunks(
        self,
        chunks: object,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class _PlaybackItem:
    """Capture one queued speech segment and its cancellation generation."""

    text: str
    generation: int
    cancel_event: Event
    atomic: bool = False


_QUEUE_SENTINEL = object()
_TTS_CHUNK_POLL_TIMEOUT_SECONDS = 0.02
_TTS_CHUNK_ENQUEUE_TIMEOUT_SECONDS = 0.02
_TTS_PUMP_JOIN_TIMEOUT_SECONDS = 0.05
_INTERRUPTED_TTS_PUMP_JOIN_TIMEOUT_SECONDS = 0.01
_TTS_STREAM_CLOSE_TIMEOUT_SECONDS = 0.05
_POST_IDLE_WORKER_JOIN_TIMEOUT_SECONDS = 0.25

# Keep these aligned with the older realtime support path. The 2s defaults were
# too aggressive for production OpenAI TTS on the Pi and caused false stalls.
_DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 20.0
_DEFAULT_CHUNK_STALL_TIMEOUT_SECONDS = 15.0
_DEFAULT_MAX_BUFFERED_TTS_CHUNKS = 8
_DEFAULT_MAX_PENDING_CHARS = 480
_DEFAULT_MAX_ATOMIC_SEGMENT_CHARS = 4096
_DEFAULT_MAX_QUEUED_CHARS = 16384

_LOGGER = logging.getLogger(__name__)


def _looks_like_wav_prefix(chunk: bytes) -> bool:
    """Return whether the given bytes start with a RIFF/WAVE header."""

    return len(chunk) >= 12 and chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE"


def _wav_buffer_has_audio_frames(buffer: bytes) -> bool:
    """Return whether a RIFF/WAVE prefix already includes data-chunk payload."""

    if not _looks_like_wav_prefix(buffer):
        return True

    offset = 12
    limit = len(buffer)
    while offset + 8 <= limit:
        chunk_id = buffer[offset : offset + 4]
        chunk_size = int.from_bytes(buffer[offset + 4 : offset + 8], "little", signed=False)
        payload_start = offset + 8
        if chunk_id == b"data":
            return limit > payload_start
        payload_end = payload_start + chunk_size
        if payload_end > limit:
            return False
        offset = payload_end + (chunk_size & 1)
    return False


class _TTSChunkPump:
    """Read one streaming TTS request on a daemon thread.

    The playback worker must be able to stop waiting even when the provider has
    not yielded its first chunk yet. This pump decouples provider I/O from the
    playback iterator so cancellation can stop the playback worker immediately
    without waiting for the upstream stream to finish.
    """

    def __init__(
        self,
        *,
        tts_provider: StreamingTextToSpeechProviderLike,
        text: str,
        chunk_size: int,
        max_buffered_chunks: int,
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
    ) -> None:
        self._tts_provider = tts_provider
        self._text = text
        self._chunk_size = chunk_size
        self._trace_event = trace_event
        self._queue: Queue[bytes | object] = Queue(maxsize=max(1, int(max_buffered_chunks)))
        self._done = Event()
        self._stop = Event()
        self._stream_lock = Lock()
        self._stream: Iterable[bytes] | None = None
        self._close_lock = Lock()
        self._close_thread: Thread | None = None
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, daemon=True, name="twinr-tts-pump")

    def start(self) -> None:
        """Start the provider pump exactly once."""

        self._thread.start()
        self._trace("tts_chunk_pump_started", text_len=len(self._text), chunk_size=self._chunk_size)

    def stop(self) -> None:
        """Request pump shutdown and best-effort close any live stream."""

        self._stop.set()
        self._trace("tts_chunk_pump_stop_requested", text_len=len(self._text))
        with self._stream_lock:
            stream = self._stream
            self._stream = None
        self._request_stream_close(stream)

    def join(self, *, timeout_s: float | None = None) -> bool:
        """Wait briefly for the pump thread and report whether it stopped."""

        self._thread.join(timeout=timeout_s)
        self._trace(
            "tts_chunk_pump_join_completed",
            timeout_s=timeout_s,
            stopped=not self._thread.is_alive(),
        )
        return not self._thread.is_alive()

    def iter_chunks(
        self,
        *,
        should_stop: Callable[[], bool],
        on_first_chunk: Callable[[], None],
        prefetched_first_chunk: bytes | None = None,
        stall_timeout_s: float | None = None,
    ):
        """Yield queued chunks while polling for cancellation and stall deadlines."""

        first_chunk_seen = False
        last_progress_s = time.monotonic()

        if prefetched_first_chunk is not None:
            if should_stop():
                self.stop()
                return
            first_chunk_seen = True
            on_first_chunk()
            self._trace("tts_chunk_pump_first_chunk_delivered", text_len=len(self._text))
            last_progress_s = time.monotonic()
            yield prefetched_first_chunk

        while True:
            if should_stop():
                self.stop()
                return
            try:
                item = self._queue.get(timeout=_TTS_CHUNK_POLL_TIMEOUT_SECONDS)
            except Empty:
                if self._done.is_set():
                    if self._error is not None:
                        raise self._error
                    return
                if stall_timeout_s is not None and (time.monotonic() - last_progress_s) >= stall_timeout_s:
                    self.stop()
                    self._trace(
                        "tts_chunk_pump_chunk_timeout",
                        timeout_s=stall_timeout_s,
                        text_len=len(self._text),
                    )
                    raise TimeoutError(
                        f"Timed out waiting {stall_timeout_s:.3f}s for the next TTS audio chunk"
                    )
                continue

            if item is _QUEUE_SENTINEL:
                if self._error is not None:
                    raise self._error
                return

            if should_stop():
                self.stop()
                return

            if not first_chunk_seen:
                first_chunk_seen = True
                on_first_chunk()
                self._trace("tts_chunk_pump_first_chunk_delivered", text_len=len(self._text))

            last_progress_s = time.monotonic()
            yield cast(bytes, item)

    def await_first_chunk(
        self,
        *,
        should_stop: Callable[[], bool],
        timeout_s: float | None = None,
    ) -> bytes | None:
        """Return the first queued chunk before playback preempts other owners."""

        deadline_s = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)

        while True:
            if should_stop():
                self.stop()
                return None
            try:
                item = self._queue.get(timeout=_TTS_CHUNK_POLL_TIMEOUT_SECONDS)
            except Empty:
                if self._done.is_set():
                    if self._error is not None:
                        raise self._error
                    return None
                if deadline_s is not None and time.monotonic() >= deadline_s:
                    self.stop()
                    self._trace(
                        "tts_chunk_pump_first_chunk_timeout",
                        timeout_s=timeout_s,
                        text_len=len(self._text),
                    )
                    raise TimeoutError(
                        f"Timed out waiting {timeout_s:.3f}s for the first TTS audio chunk"
                    )
                continue

            if item is _QUEUE_SENTINEL:
                if self._error is not None:
                    raise self._error
                return None

            return cast(bytes, item)

    def await_first_playable_chunk(
        self,
        *,
        should_stop: Callable[[], bool],
        timeout_s: float | None = None,
    ) -> bytes | None:
        """Return the first chunk bundle that includes actual audible payload."""

        deadline_s = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        buffered_wav = bytearray()

        while True:
            chunk = self.await_first_chunk(
                should_stop=should_stop,
                timeout_s=None if deadline_s is None else max(0.0, deadline_s - time.monotonic()),
            )
            if chunk is None:
                if not buffered_wav or not _wav_buffer_has_audio_frames(buffered_wav):
                    return None
                return bytes(buffered_wav)
            if not buffered_wav and not _looks_like_wav_prefix(chunk):
                return chunk

            buffered_wav.extend(chunk)
            if _wav_buffer_has_audio_frames(buffered_wav):
                self._trace(
                    "tts_chunk_pump_first_playable_wav_ready",
                    text_len=len(self._text),
                    buffered_bytes=len(buffered_wav),
                )
                return bytes(buffered_wav)

            self._trace(
                "tts_chunk_pump_buffering_wav_header",
                text_len=len(self._text),
                buffered_bytes=len(buffered_wav),
            )

    def _run(self) -> None:
        stream: Iterable[bytes] | None = None
        try:
            stream = self._tts_provider.synthesize_stream(
                self._text,
                chunk_size=self._chunk_size,
            )
            with self._stream_lock:
                self._stream = stream
            for chunk in stream:
                if self._stop.is_set():
                    self._trace("tts_chunk_pump_stopped_before_chunk", text_len=len(self._text))
                    return
                if not chunk:
                    continue
                self._enqueue_chunk(bytes(chunk))
                self._trace("tts_chunk_pump_chunk_enqueued", size=len(chunk))
        except Exception as exc:  # pragma: no cover - raised through caller path
            self._error = exc
            self._trace("tts_chunk_pump_failed", error_type=type(exc).__name__)
        finally:
            with self._stream_lock:
                stream_to_close = self._stream
                self._stream = None
            self._request_stream_close(stream_to_close)
            self._done.set()
            self._enqueue_sentinel()
            self._trace("tts_chunk_pump_finished", has_error=self._error is not None)

    def _enqueue_chunk(self, chunk: bytes) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(chunk, timeout=_TTS_CHUNK_ENQUEUE_TIMEOUT_SECONDS)
                return
            except Full:
                continue

    def _enqueue_sentinel(self) -> None:
        try:
            self._queue.put_nowait(_QUEUE_SENTINEL)
        except Full:
            return

    def _request_stream_close(self, stream: object | None) -> None:
        """Close one live TTS stream without letting shutdown block indefinitely."""

        close = getattr(stream, "close", None)
        if not callable(close):
            return

        def _close_stream() -> None:
            try:
                close()
            except ValueError as exc:
                if "generator already executing" in str(exc):
                    self._trace("tts_chunk_pump_close_deferred_reentrant", text_len=len(self._text))
                    return
                _LOGGER.warning("Streaming speech output failed to close the live TTS stream.", exc_info=True)
            except Exception:
                _LOGGER.warning("Streaming speech output failed to close the live TTS stream.", exc_info=True)

        with self._close_lock:
            active_close = self._close_thread
            if active_close is not None and active_close.is_alive():
                return
            close_thread = Thread(
                target=_close_stream,
                daemon=True,
                name="twinr-tts-pump-close",
            )
            self._close_thread = close_thread
            close_thread.start()

        close_thread.join(timeout=_TTS_STREAM_CLOSE_TIMEOUT_SECONDS)
        if close_thread.is_alive():
            self._trace("tts_chunk_pump_close_timeout", text_len=len(self._text))

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self._trace_event):
            return
        try:
            self._trace_event(msg, details)
        except Exception:
            _LOGGER.warning("Streaming speech-output trace sink failed for %s.", msg, exc_info=True)


class InterruptibleSpeechOutput:
    """Queue text segments and play them with preemption-aware streaming TTS.

    Segments are emitted once a boundary callback marks them ready. Newer
    generations cancel stale playback so filler speech can be replaced safely.
    """

    def __init__(
        self,
        *,
        tts_provider: StreamingTextToSpeechProviderLike,
        player: WaveAudioPlayerLike,
        chunk_size: int,
        segment_boundary: Callable[[str], int | None],
        on_speaking_started: Callable[[], None] | None = None,
        on_first_audio: Callable[[], None] | None = None,
        on_preempt: Callable[[], None] | None = None,
        playback_coordinator: PlaybackCoordinator | None = None,
        playback_lock: Lock | None = None,
        should_stop: Callable[[], bool] | None = None,
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
        # BREAKING: startup and stall waits are now bounded by default. Pass None
        # to restore the previous infinite-wait behavior.
        first_chunk_timeout_s: float | None = _DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS,
        chunk_stall_timeout_s: float | None = _DEFAULT_CHUNK_STALL_TIMEOUT_SECONDS,
        max_buffered_tts_chunks: int = _DEFAULT_MAX_BUFFERED_TTS_CHUNKS,
        max_pending_chars: int = _DEFAULT_MAX_PENDING_CHARS,
        max_atomic_segment_chars: int = _DEFAULT_MAX_ATOMIC_SEGMENT_CHARS,
        max_queued_chars: int = _DEFAULT_MAX_QUEUED_CHARS,
    ) -> None:
        self.tts_provider = tts_provider
        self.player = player
        self.chunk_size = max(256, int(chunk_size))
        self.segment_boundary = segment_boundary
        self.on_speaking_started = on_speaking_started
        self.on_first_audio = on_first_audio
        self.on_preempt = on_preempt
        self.playback_coordinator = playback_coordinator
        self.playback_lock = playback_lock
        self.should_stop = should_stop
        self._trace_event = trace_event

        self.first_chunk_timeout_s = None if first_chunk_timeout_s is None else max(0.0, float(first_chunk_timeout_s))
        self.chunk_stall_timeout_s = None if chunk_stall_timeout_s is None else max(0.0, float(chunk_stall_timeout_s))
        self.max_buffered_tts_chunks = max(1, int(max_buffered_tts_chunks))
        self.max_pending_chars = max(64, int(max_pending_chars))
        self.max_atomic_segment_chars = max(self.max_pending_chars, int(max_atomic_segment_chars))
        self.max_queued_chars = max(self.max_atomic_segment_chars, int(max_queued_chars))

        self._queue: Queue[_PlaybackItem | None] = Queue()
        self._pending_segment = ""
        self._generation = 0
        self._cancel_event = Event()
        self._segment_lock = Lock()
        self._idle_event = Event()
        self._error_lock = Lock()
        self._first_audio_lock = Lock()
        self._first_audio_event = Event()
        self._answer_started = False
        self._first_audio_emitted = False
        self._queued_items = 0
        self._queued_chars = 0
        self._playing = False
        self._closing = False
        self._closed = False
        self._errors: list[Exception] = []
        self._worker = Thread(target=self._tts_worker, daemon=True, name="twinr-speech-output")
        self._idle_event.set()
        self._worker.start()

        self._trace(
            "speech_output_initialized",
            chunk_size=self.chunk_size,
            first_chunk_timeout_s=self.first_chunk_timeout_s,
            chunk_stall_timeout_s=self.chunk_stall_timeout_s,
            max_buffered_tts_chunks=self.max_buffered_tts_chunks,
            max_pending_chars=self.max_pending_chars,
            max_atomic_segment_chars=self.max_atomic_segment_chars,
            max_queued_chars=self.max_queued_chars,
        )

    def submit_text_delta(self, delta: str) -> None:
        self._submit_delta(delta, replace_current=False)

    def submit_lane_delta(self, delta: SpeechLaneDelta) -> None:
        self._submit_delta(
            delta.text,
            replace_current=getattr(delta, "replace_current", False),
            atomic=getattr(delta, "atomic", False),
        )

    def flush(self) -> None:
        with self._segment_lock:
            self._ensure_accepting_locked()
            self._flush_pending_locked()
            self._update_idle_locked()
        self._trace("speech_output_flush_requested", pending_len=len(self._pending_segment))

    def close(self, *, timeout_s: float | None = None) -> None:
        """Flush the worker and wait for a bounded clean shutdown."""

        close_started = time.monotonic()
        self._trace("speech_output_close_requested", timeout_s=timeout_s)

        with self._segment_lock:
            if self._closed:
                self._trace("speech_output_close_completed", timeout_s=timeout_s, already_closed=True)
                return

            self._closing = True
            try:
                self._flush_pending_locked()
                self._update_idle_locked()
            except Exception:
                self._closing = False
                self._update_idle_locked()
                raise

        self._queue.put(None)

        idle_ready = self.wait_until_idle(timeout_s=timeout_s)
        join_timeout_s = timeout_s
        if timeout_s is not None:
            elapsed_s = max(0.0, time.monotonic() - close_started)
            join_timeout_s = max(0.0, timeout_s - elapsed_s)
        if idle_ready:
            join_timeout_s = (
                _POST_IDLE_WORKER_JOIN_TIMEOUT_SECONDS
                if join_timeout_s is None
                else min(join_timeout_s, _POST_IDLE_WORKER_JOIN_TIMEOUT_SECONDS)
            )

        self._trace(
            "speech_output_close_join_started",
            timeout_s=timeout_s,
            idle_ready=idle_ready,
            join_timeout_s=join_timeout_s,
        )

        self._worker.join(timeout=join_timeout_s)
        if self._worker.is_alive():
            self._trace("speech_output_close_timeout", timeout_s=timeout_s)
            raise RuntimeError("Text-to-speech playback worker did not exit before timeout")

        with self._segment_lock:
            self._closed = True
            self._closing = False
            self._update_idle_locked()

        self._trace(
            "speech_output_close_completed",
            timeout_s=timeout_s,
            idle_ready=idle_ready,
            join_timeout_s=join_timeout_s,
        )

    # BREAKING: abort() is now non-terminal. It cancels active and queued speech
    # but keeps the worker alive so the same instance can speak again.
    def abort(self, *, timeout_s: float | None = 0.25) -> bool:
        """Cancel active and queued speech without destroying the worker."""

        callback: Callable[[], None] | None = None
        with self._segment_lock:
            if self._closed:
                return True
            self._pending_segment = ""
            if self._preempt_locked():
                callback = self.on_preempt
            self._update_idle_locked()

        if callback is not None:
            self._invoke_callback(callback, callback_name="on_preempt")

        ready = self.wait_until_idle(timeout_s=timeout_s)
        self._trace("speech_output_abort_completed", timeout_s=timeout_s, idle_ready=ready)
        return ready

    def wait_for_first_audio(self, *, timeout_s: float | None = None) -> bool:
        if self._first_audio_event.is_set():
            self._trace("speech_output_first_audio_already_ready", timeout_s=timeout_s)
            return True
        ready = self._first_audio_event.wait(timeout=timeout_s)
        self._trace("speech_output_first_audio_wait_completed", timeout_s=timeout_s, ready=ready)
        return ready

    def wait_until_idle(self, *, timeout_s: float | None = None) -> bool:
        """Wait until no queued or active speech remains for the current output."""

        if self._idle_event.is_set():
            self._trace("speech_output_idle_already_ready", timeout_s=timeout_s)
            return True
        ready = self._idle_event.wait(timeout=timeout_s)
        self._trace("speech_output_idle_wait_completed", timeout_s=timeout_s, ready=ready)
        return ready

    def raise_if_error(self) -> None:
        with self._error_lock:
            error = self._errors[0] if self._errors else None
        if error is not None:
            raise error

    def _submit_delta(self, delta: str, *, replace_current: bool, atomic: bool = False) -> None:
        cleaned = str(delta or "")
        if not cleaned:
            return

        self._trace(
            "speech_output_delta_received",
            delta_len=len(cleaned),
            replace_current=replace_current,
            atomic=atomic,
        )

        callback: Callable[[], None] | None = None
        with self._segment_lock:
            self._ensure_accepting_locked()

            if replace_current:
                had_activity = bool(self._pending_segment.strip()) or self._queued_items > 0 or self._playing
                if self._preempt_locked():
                    callback = self.on_preempt
                if not had_activity:
                    self._reset_answer_state_locked()
            elif self._is_idle_locked():
                self._reset_answer_state_locked()

            if atomic:
                self._pending_segment = ""
                self._enqueue_locked(cleaned, atomic=True)
                self._update_idle_locked()
            else:
                self._pending_segment += cleaned
                self._queue_ready_segments_locked()
                self._update_idle_locked()

        if callback is not None:
            self._invoke_callback(callback, callback_name="on_preempt")

    def _ensure_accepting_locked(self) -> None:
        # BREAKING: submit after close() or worker failure now raises explicitly
        # instead of silently queueing unreachable speech forever.
        if self._closing or self._closed:
            raise RuntimeError("Speech output is closed")

        with self._error_lock:
            if self._errors:
                raise RuntimeError("Speech output worker has failed; call raise_if_error()")

        if not self._worker.is_alive():
            raise RuntimeError("Speech output worker is not running")

    def _reset_answer_state_locked(self) -> None:
        with self._first_audio_lock:
            self._answer_started = False
            self._first_audio_emitted = False
            self._first_audio_event.clear()
        self._trace("speech_output_answer_state_reset", generation=self._generation)

    def _is_idle_locked(self) -> bool:
        return self._queued_items == 0 and not self._playing and not self._pending_segment.strip()

    def _preempt_locked(self) -> bool:
        had_activity = bool(self._pending_segment.strip()) or self._queued_items > 0 or self._playing
        self._pending_segment = ""
        self._generation += 1
        self._cancel_event.set()
        self._cancel_event = Event()

        drained_items, drained_chars = self._drain_queued_items_locked()
        self._queued_items = max(0, self._queued_items - drained_items)
        self._queued_chars = max(0, self._queued_chars - drained_chars)

        self._update_idle_locked()
        self._trace(
            "speech_output_preempted",
            generation=self._generation,
            drained_items=drained_items,
            drained_chars=drained_chars,
        )
        return had_activity or drained_items > 0

    def _drain_queued_items_locked(self) -> tuple[int, int]:
        drained_items = 0
        drained_chars = 0
        preserved_sentinel = False

        while True:
            try:
                queued = self._queue.get_nowait()
            except Empty:
                break
            if queued is None:
                preserved_sentinel = True
                break
            drained_items += 1
            drained_chars += len(queued.text)

        if preserved_sentinel:
            self._queue.put_nowait(None)

        return drained_items, drained_chars

    def _queue_ready_segments_locked(self) -> None:
        while True:
            boundary = self.segment_boundary(self._pending_segment)
            forced = False
            if boundary is None:
                boundary = self._forced_boundary(self._pending_segment)
                forced = boundary is not None
            if boundary is None:
                return

            segment = self._pending_segment[:boundary].strip()
            remainder = self._pending_segment[boundary:].lstrip()

            if not segment:
                self._pending_segment = remainder
                continue

            self._trace(
                "speech_output_segment_boundary",
                segment_len=len(segment),
                generation=self._generation,
                forced=forced,
            )

            self._enqueue_locked(segment)
            self._pending_segment = remainder

    def _forced_boundary(self, text: str) -> int | None:
        """Force bounded segments when no higher-level boundary appears.

        This keeps long unpunctuated streams from growing without bound and
        prevents pathological latency and memory growth on small edge devices.
        """

        if len(text) < self.max_pending_chars:
            return None

        window = text[: self.max_pending_chars]

        for token_group in ("\n", ".!?;:", ",)", " "):
            candidate = max(window.rfind(token) for token in token_group)
            if candidate >= max(32, self.max_pending_chars // 2):
                return candidate + 1

        return self.max_pending_chars

    def _flush_pending_locked(self) -> None:
        segment = self._pending_segment.strip()
        if not segment:
            self._pending_segment = ""
            return

        self._trace("speech_output_pending_flushed", segment_len=len(segment), generation=self._generation)
        self._enqueue_locked(segment)
        self._pending_segment = ""

    def _enqueue_locked(self, segment: str, *, atomic: bool = False) -> None:
        if atomic:
            if len(segment) > self.max_atomic_segment_chars:
                raise ValueError(
                    f"Atomic speech segment exceeds max_atomic_segment_chars={self.max_atomic_segment_chars}"
                )
            self._ensure_backlog_capacity_locked(len(segment))
            self._enqueue_one_locked(segment, atomic=True)
            return

        pieces = self._split_non_atomic_segment(segment)
        additional_chars = sum(len(piece) for piece in pieces)
        self._ensure_backlog_capacity_locked(additional_chars)
        for piece in pieces:
            self._enqueue_one_locked(piece, atomic=False)

    def _split_non_atomic_segment(self, segment: str) -> list[str]:
        segment = segment.strip()
        if not segment:
            return []

        pieces: list[str] = []
        remaining = segment

        while len(remaining) > self.max_pending_chars:
            boundary = self._forced_boundary(remaining)
            if boundary is None or boundary <= 0:
                break
            piece = remaining[:boundary].strip()
            if piece:
                pieces.append(piece)
            remaining = remaining[boundary:].lstrip()

        if remaining:
            pieces.append(remaining)

        return pieces

    def _ensure_backlog_capacity_locked(self, additional_chars: int) -> None:
        projected_chars = self._queued_chars + additional_chars
        if projected_chars <= self.max_queued_chars:
            return

        self._trace(
            "speech_output_backlog_limit_exceeded",
            additional_chars=additional_chars,
            queued_chars=self._queued_chars,
            max_queued_chars=self.max_queued_chars,
        )
        raise RuntimeError(
            f"Speech output backlog exceeded max_queued_chars={self.max_queued_chars}; "
            "refusing to buffer more speech"
        )

    def _enqueue_one_locked(self, segment: str, *, atomic: bool) -> None:
        self._queued_items += 1
        self._queued_chars += len(segment)

        self._queue.put(
            _PlaybackItem(
                text=segment,
                generation=self._generation,
                cancel_event=self._cancel_event,
                atomic=atomic,
            )
        )

        self._trace(
            "speech_output_segment_enqueued",
            segment_len=len(segment),
            generation=self._generation,
            atomic=atomic,
            queued_items=self._queued_items,
            queued_chars=self._queued_chars,
        )

    def _tts_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._trace("speech_output_worker_exit", reason="sentinel")
                return

            with self._segment_lock:
                self._queued_items = max(0, self._queued_items - 1)
                self._queued_chars = max(0, self._queued_chars - len(item.text))
                current_generation = self._generation
                self._playing = True
                self._update_idle_locked()

            if item.generation != current_generation:
                self._trace(
                    "speech_output_stale_item_skipped",
                    item_generation=item.generation,
                    current_generation=current_generation,
                )
                with self._segment_lock:
                    self._playing = False
                    self._update_idle_locked()
                continue

            try:
                self._trace(
                    "speech_output_play_item_started",
                    generation=item.generation,
                    text_len=len(item.text),
                    atomic=item.atomic,
                )
                self._play_item(item)
            except Exception as exc:  # pragma: no cover - exercised via raise_if_error
                with self._error_lock:
                    self._errors.append(exc)
                self._trace("speech_output_worker_failed", error_type=type(exc).__name__)

                with self._segment_lock:
                    self._pending_segment = ""
                    drained_items, drained_chars = self._drain_queued_items_locked()
                    self._queued_items = 0
                    self._queued_chars = 0
                    self._playing = False
                    self._update_idle_locked()

                self._trace(
                    "speech_output_worker_failed_queue_drained",
                    drained_items=drained_items,
                    drained_chars=drained_chars,
                )
                return

            with self._segment_lock:
                self._playing = False
                self._update_idle_locked()

    def _play_item(self, item: _PlaybackItem) -> None:
        pump = _TTSChunkPump(
            tts_provider=self.tts_provider,
            text=item.text,
            chunk_size=self.chunk_size,
            max_buffered_chunks=self.max_buffered_tts_chunks,
            trace_event=self._trace_event,
        )

        def stop_requested() -> bool:
            if item.cancel_event.is_set():
                self._trace("speech_output_stop_requested_cancel_event", generation=item.generation)
                return True

            if self.should_stop is None:
                return False

            try:
                should_stop = bool(self.should_stop())
            except Exception:
                _LOGGER.warning(
                    "Speech output should_stop callback failed; ignoring callback for this poll.",
                    exc_info=True,
                )
                self._trace("speech_output_stop_requested_callback_failed", generation=item.generation)
                return False

            if should_stop:
                self._trace("speech_output_stop_requested_external", generation=item.generation)
            return should_stop

        def emit_first_chunk() -> None:
            speaking_callback: Callable[[], None] | None = None
            first_audio_callback: Callable[[], None] | None = None

            with self._first_audio_lock:
                if not self._answer_started:
                    self._answer_started = True
                    speaking_callback = self.on_speaking_started
                if not self._first_audio_emitted:
                    self._first_audio_emitted = True
                    self._first_audio_event.set()
                    first_audio_callback = self.on_first_audio

            if speaking_callback is not None:
                self._invoke_callback(speaking_callback, callback_name="on_speaking_started")
                self._trace("speech_output_answering_started", generation=item.generation)

            if first_audio_callback is not None:
                self._invoke_callback(first_audio_callback, callback_name="on_first_audio")
                self._trace("speech_output_first_audio_emitted", generation=item.generation)

        pump.start()
        try:
            first_chunk = pump.await_first_playable_chunk(
                should_stop=stop_requested,
                timeout_s=self.first_chunk_timeout_s,
            )
            if first_chunk is None:
                self._trace("speech_output_play_item_completed_without_audio", generation=item.generation)
                return

            chunks = pump.iter_chunks(
                should_stop=stop_requested,
                on_first_chunk=emit_first_chunk,
                prefetched_first_chunk=first_chunk,
                stall_timeout_s=self.chunk_stall_timeout_s,
            )

            if self.playback_coordinator is not None:
                self.playback_coordinator.play_wav_chunks(
                    owner="streaming_tts",
                    priority=PlaybackPriority.SPEECH,
                    chunks=chunks,
                    should_stop=stop_requested,
                    atomic=item.atomic,
                )
                return

            if self.playback_lock is None:
                self.player.play_wav_chunks(chunks, should_stop=stop_requested)
                return

            with self.playback_lock:
                self.player.play_wav_chunks(chunks, should_stop=stop_requested)
        finally:
            pump.stop()
            pump.join(timeout_s=self._pump_join_timeout(item))
            self._trace("speech_output_play_item_completed", generation=item.generation)

    def _invoke_callback(self, callback: Callable[[], None], *, callback_name: str) -> None:
        try:
            callback()
        except Exception:
            _LOGGER.warning("Speech output callback %s failed.", callback_name, exc_info=True)
            self._trace("speech_output_callback_failed", callback_name=callback_name)

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self._trace_event):
            return
        try:
            self._trace_event(msg, details)
        except Exception:
            _LOGGER.warning(
                "Speech output trace callback failed for %s.",
                msg,
                exc_info=True,
            )

    def _pump_join_timeout(self, item: _PlaybackItem) -> float:
        """Use a very short join budget after interruption or preemption."""

        if item.cancel_event.is_set():
            return _INTERRUPTED_TTS_PUMP_JOIN_TIMEOUT_SECONDS

        if self.should_stop is not None:
            try:
                if bool(self.should_stop()):
                    return _INTERRUPTED_TTS_PUMP_JOIN_TIMEOUT_SECONDS
            except Exception:
                _LOGGER.warning(
                    "Speech output should_stop callback failed; using interrupted join timeout.",
                    exc_info=True,
                )
                return _INTERRUPTED_TTS_PUMP_JOIN_TIMEOUT_SECONDS

        return _TTS_PUMP_JOIN_TIMEOUT_SECONDS

    def _update_idle_locked(self) -> None:
        """Refresh the idle signal after queue or playback state changes."""

        has_pending_text = bool(self._pending_segment.strip())
        if self._queued_items == 0 and not self._playing and not has_pending_text:
            self._idle_event.set()
            return
        self._idle_event.clear()
