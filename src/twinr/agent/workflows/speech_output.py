"""Stream interruptible text-to-speech playback for dual-lane workflows.

This module owns the low-latency spoken-output path used by the streaming
workflow. It deliberately keeps queueing, first-audio signaling, and abort
behavior separate so a stalled TTS request cannot pin the whole conversation
session in ``answering``.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Protocol
import time

from twinr.agent.tools.runtime.dual_lane_loop import SpeechLaneDelta


class StreamingTextToSpeechProviderLike(Protocol):
    """Describe the streaming TTS interface required by speech output."""

    def synthesize_stream(self, text: str, **kwargs) -> object:
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
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
    ) -> None:
        self._tts_provider = tts_provider
        self._text = text
        self._chunk_size = chunk_size
        self._trace_event = trace_event
        self._queue: Queue[bytes | object] = Queue()
        self._done = Event()
        self._stop = Event()
        self._stream_lock = Lock()
        self._stream = None
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
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

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
    ):
        """Yield queued chunks while polling for cancellation."""

        first_chunk_seen = False
        while True:
            if should_stop():
                self.stop()
                return
            try:
                item = self._queue.get(timeout=0.05)
            except Empty:
                if self._done.is_set():
                    if self._error is not None:
                        raise self._error
                    return
                continue
            if item is _QUEUE_SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            if not first_chunk_seen:
                first_chunk_seen = True
                on_first_chunk()
                self._trace("tts_chunk_pump_first_chunk_delivered", text_len=len(self._text))
            yield item

    def _run(self) -> None:
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
                self._queue.put(bytes(chunk))
                self._trace("tts_chunk_pump_chunk_enqueued", size=len(chunk))
        except Exception as exc:  # pragma: no cover - raised through caller path
            self._error = exc
            self._trace("tts_chunk_pump_failed", error_type=type(exc).__name__)
        finally:
            with self._stream_lock:
                stream = self._stream
                self._stream = None
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._done.set()
            self._queue.put(_QUEUE_SENTINEL)
            self._trace("tts_chunk_pump_finished", has_error=self._error is not None)

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self._trace_event):
            return
        try:
            self._trace_event(msg, details)
        except Exception:
            return


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
        playback_lock: Lock | None = None,
        should_stop: Callable[[], bool] | None = None,
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
    ) -> None:
        self.tts_provider = tts_provider
        self.player = player
        self.chunk_size = max(256, int(chunk_size))
        self.segment_boundary = segment_boundary
        self.on_speaking_started = on_speaking_started
        self.on_first_audio = on_first_audio
        self.on_preempt = on_preempt
        self.playback_lock = playback_lock
        self.should_stop = should_stop
        self._trace_event = trace_event

        self._queue: Queue[_PlaybackItem | None] = Queue()
        self._pending_segment = ""
        self._generation = 0
        self._cancel_event = Event()
        self._segment_lock = Lock()
        self._error_lock = Lock()
        self._first_audio_lock = Lock()
        self._first_audio_event = Event()
        self._answer_started = False
        self._first_audio_emitted = False
        self._errors: list[Exception] = []
        self._worker = Thread(target=self._tts_worker, daemon=True)
        self._worker.start()
        self._trace("speech_output_initialized", chunk_size=self.chunk_size)

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
            self._flush_pending_locked()
        self._trace("speech_output_flush_requested", pending_len=len(self._pending_segment))

    def close(self, *, timeout_s: float | None = None) -> None:
        """Flush the worker and wait for a bounded clean shutdown."""

        self._trace("speech_output_close_requested", timeout_s=timeout_s)
        self._queue.put(None)
        self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            self._trace("speech_output_close_timeout", timeout_s=timeout_s)
            raise RuntimeError("Text-to-speech playback worker did not exit before timeout")
        self._trace("speech_output_close_completed", timeout_s=timeout_s)

    def abort(self, *, timeout_s: float | None = 0.25) -> bool:
        """Stop playback immediately without waiting for a slow provider drain.

        Returns ``True`` when the worker stopped within the timeout. A lingering
        daemon worker is acceptable during interrupt handling because the active
        generation has already been canceled and future speech must not wait for
        the stale upstream request to finish.
        """

        with self._segment_lock:
            self._pending_segment = ""
            self._preempt_locked()
        self._queue.put(None)
        self._worker.join(timeout=timeout_s)
        self._trace("speech_output_abort_completed", timeout_s=timeout_s, stopped=not self._worker.is_alive())
        return not self._worker.is_alive()

    def wait_for_first_audio(self, *, timeout_s: float | None = None) -> bool:
        if self._first_audio_event.is_set():
            self._trace("speech_output_first_audio_already_ready", timeout_s=timeout_s)
            return True
        ready = self._first_audio_event.wait(timeout=timeout_s)
        self._trace("speech_output_first_audio_wait_completed", timeout_s=timeout_s, ready=ready)
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
        with self._segment_lock:
            if replace_current:
                self._preempt_locked()
            if atomic:
                self._pending_segment = ""
                self._enqueue_locked(cleaned, atomic=True)
                return
            self._pending_segment += cleaned
            self._queue_ready_segments_locked()

    def _preempt_locked(self) -> None:
        if self._pending_segment.strip():
            self._pending_segment = ""
        self._generation += 1
        self._cancel_event.set()
        self._cancel_event = Event()
        self._trace("speech_output_preempted", generation=self._generation)
        if self.on_preempt is not None:
            self.on_preempt()

    def _queue_ready_segments_locked(self) -> None:
        while True:
            boundary = self.segment_boundary(self._pending_segment)
            if boundary is None:
                return
            segment = self._pending_segment[:boundary].strip()
            self._pending_segment = self._pending_segment[boundary:].lstrip()
            if not segment:
                continue
            self._trace("speech_output_segment_boundary", segment_len=len(segment), generation=self._generation)
            self._enqueue_locked(segment)

    def _flush_pending_locked(self) -> None:
        segment = self._pending_segment.strip()
        self._pending_segment = ""
        if not segment:
            return
        self._trace("speech_output_pending_flushed", segment_len=len(segment), generation=self._generation)
        self._enqueue_locked(segment)

    def _enqueue_locked(self, segment: str, *, atomic: bool = False) -> None:
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
        )

    def _tts_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._trace("speech_output_worker_exit", reason="sentinel")
                return
            with self._segment_lock:
                current_generation = self._generation
            if item.generation != current_generation:
                self._trace(
                    "speech_output_stale_item_skipped",
                    item_generation=item.generation,
                    current_generation=current_generation,
                )
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
                return

    def _play_item(self, item: _PlaybackItem) -> None:
        pump = _TTSChunkPump(
            tts_provider=self.tts_provider,
            text=item.text,
            chunk_size=self.chunk_size,
            trace_event=self._trace_event,
        )

        def stop_requested() -> bool:
            if item.cancel_event.is_set():
                self._trace("speech_output_stop_requested_cancel_event", generation=item.generation)
                return True
            if self.should_stop is None:
                return False
            should_stop = bool(self.should_stop())
            if should_stop:
                self._trace("speech_output_stop_requested_external", generation=item.generation)
            return should_stop

        def emit_first_chunk() -> None:
            if not self._answer_started:
                self._answer_started = True
                if self.on_speaking_started is not None:
                    self.on_speaking_started()
                self._trace("speech_output_answering_started", generation=item.generation)
            with self._first_audio_lock:
                if self._first_audio_emitted:
                    return
                self._first_audio_emitted = True
                self._first_audio_event.set()
                if self.on_first_audio is not None:
                    self.on_first_audio()
                self._trace("speech_output_first_audio_emitted", generation=item.generation)

        pump.start()
        if self.playback_lock is None:
            try:
                self.player.play_wav_chunks(
                    pump.iter_chunks(
                        should_stop=stop_requested,
                        on_first_chunk=emit_first_chunk,
                    ),
                    should_stop=stop_requested,
                )
            finally:
                pump.stop()
                pump.join(timeout_s=0.2)
                self._trace("speech_output_play_item_completed", generation=item.generation)
            return
        with self.playback_lock:
            try:
                self.player.play_wav_chunks(
                    pump.iter_chunks(
                        should_stop=stop_requested,
                        on_first_chunk=emit_first_chunk,
                    ),
                    should_stop=stop_requested,
                )
            finally:
                pump.stop()
                pump.join(timeout_s=0.2)
                self._trace("speech_output_play_item_completed", generation=item.generation)

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self._trace_event):
            return
        try:
            self._trace_event(msg, details)
        except Exception:
            return
