from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Protocol
import time

from twinr.agent.tools.dual_lane_loop import SpeechLaneDelta


class StreamingTextToSpeechProviderLike(Protocol):
    def synthesize_stream(self, text: str, **kwargs) -> object:
        ...


class WaveAudioPlayerLike(Protocol):
    def play_wav_chunks(
        self,
        chunks: object,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class _PlaybackItem:
    text: str
    generation: int
    cancel_event: Event
    atomic: bool = False


class InterruptibleSpeechOutput:
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
    ) -> None:
        self.tts_provider = tts_provider
        self.player = player
        self.chunk_size = max(256, int(chunk_size))
        self.segment_boundary = segment_boundary
        self.on_speaking_started = on_speaking_started
        self.on_first_audio = on_first_audio
        self.on_preempt = on_preempt
        self.playback_lock = playback_lock

        self._queue: Queue[_PlaybackItem | None] = Queue()
        self._pending_segment = ""
        self._generation = 0
        self._cancel_event = Event()
        self._segment_lock = Lock()
        self._error_lock = Lock()
        self._first_audio_lock = Lock()
        self._answer_started = False
        self._first_audio_emitted = False
        self._errors: list[Exception] = []
        self._worker = Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

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

    def close(self, *, timeout_s: float | None = None) -> None:
        self._queue.put(None)
        self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            raise RuntimeError("Text-to-speech playback worker did not exit before timeout")

    def raise_if_error(self) -> None:
        with self._error_lock:
            error = self._errors[0] if self._errors else None
        if error is not None:
            raise error

    def _submit_delta(self, delta: str, *, replace_current: bool, atomic: bool = False) -> None:
        cleaned = str(delta or "")
        if not cleaned:
            return
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
            self._enqueue_locked(segment)

    def _flush_pending_locked(self) -> None:
        segment = self._pending_segment.strip()
        self._pending_segment = ""
        if not segment:
            return
        self._enqueue_locked(segment)

    def _enqueue_locked(self, segment: str, *, atomic: bool = False) -> None:
        if not self._answer_started:
            self._answer_started = True
            if self.on_speaking_started is not None:
                self.on_speaking_started()
        self._queue.put(
            _PlaybackItem(
                text=segment,
                generation=self._generation,
                cancel_event=self._cancel_event,
                atomic=atomic,
            )
        )

    def _tts_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            with self._segment_lock:
                current_generation = self._generation
            if item.generation != current_generation:
                continue
            try:
                self._play_item(item)
            except Exception as exc:  # pragma: no cover - exercised via raise_if_error
                with self._error_lock:
                    self._errors.append(exc)
                return

    def _play_item(self, item: _PlaybackItem) -> None:
        def synthesize_chunks():
            for chunk in self.tts_provider.synthesize_stream(
                item.text,
                chunk_size=self.chunk_size,
            ):
                if item.cancel_event.is_set():
                    return
                if not chunk:
                    continue
                with self._first_audio_lock:
                    if not self._first_audio_emitted:
                        self._first_audio_emitted = True
                        if self.on_first_audio is not None:
                            self.on_first_audio()
                yield chunk

        if self.playback_lock is None:
            self.player.play_wav_chunks(
                synthesize_chunks(),
                should_stop=item.cancel_event.is_set,
            )
            return
        with self.playback_lock:
            self.player.play_wav_chunks(
                synthesize_chunks(),
                should_stop=item.cancel_event.is_set,
            )
