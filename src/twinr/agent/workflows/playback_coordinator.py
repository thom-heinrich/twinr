"""Serialize Twinr speaker output under one priority-aware owner.

This module centralizes listen beeps, feedback tones, and spoken TTS playback
behind one queue and one preemption policy. It does not own microphone or
ambient-audio sampling; callers may still pass a broader I/O lock so playback
stays isolated from recorder use on the Pi.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heappop, heappush
from threading import Condition, Event, Lock, Thread
import time


class PlaybackPriority(IntEnum):
    """Define Twinr playback priority bands."""

    FEEDBACK = 10
    SPEECH = 20
    BEEP = 30


@dataclass(frozen=True, slots=True)
class PlaybackRunResult:
    """Capture queue and runtime timings for one playback request."""

    owner: str
    priority: int
    queue_wait_ms: float
    runtime_ms: float
    preempted: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result view."""

        return {
            "owner": self.owner,
            "priority": self.priority,
            "queue_wait_ms": self.queue_wait_ms,
            "runtime_ms": self.runtime_ms,
            "preempted": self.preempted,
        }


@dataclass(slots=True)
class _PlaybackRequest:
    """Track one queued or active playback operation."""

    owner: str
    priority: int
    sequence: int
    action: Callable[[Callable[[], bool]], None]
    stop: Callable[[], None] | None
    external_should_stop: Callable[[], bool] | None
    atomic: bool
    enqueued_at: float
    done: Event = field(default_factory=Event)
    cancel_event: Event = field(default_factory=Event)
    error: Exception | None = None
    result: PlaybackRunResult | None = None

    def should_stop(self) -> bool:
        """Return whether playback should stop for cancellation or shutdown."""

        if self.cancel_event.is_set():
            return True
        callback = self.external_should_stop
        if not callable(callback):
            return False
        try:
            return bool(callback())
        except Exception:
            return True


class PlaybackCoordinator:
    """Run all speaker output through one priority queue and worker."""

    def __init__(
        self,
        player,
        *,
        emit: Callable[[str], None] | None = None,
        io_lock: Lock | None = None,
    ) -> None:
        self.player = player
        self.emit = emit
        self.io_lock = io_lock
        self._condition = Condition()
        self._queue: list[tuple[int, int, _PlaybackRequest]] = []
        self._current: _PlaybackRequest | None = None
        self._closed = False
        self._sequence = 0
        self._worker = Thread(target=self._worker_main, daemon=True, name="twinr-playback-coordinator")
        self._worker.start()

    def close(self, *, timeout_s: float | None = None) -> None:
        """Stop the coordinator after draining or interrupting active playback."""

        stop_callback: Callable[[], None] | None = None
        with self._condition:
            self._closed = True
            current = self._current
            if current is not None:
                current.cancel_event.set()
                stop_callback = self._build_bound_stop_callback_locked(current)
            self._condition.notify_all()
        self._safe_stop(stop_callback)
        self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            raise RuntimeError("Playback coordinator did not shut down before timeout")

    def stop_playback(self) -> None:
        """Best-effort stop for the active player process."""

        self._safe_stop(getattr(self.player, "stop_playback", None))

    def stop_owner(self, owner: str) -> None:
        """Stop the active playback only when the named owner still owns it."""

        stop_callback: Callable[[], None] | None = None
        with self._condition:
            current = self._current
            if current is None or current.owner != owner:
                return
            current.cancel_event.set()
            stop_callback = self._build_bound_stop_callback_locked(current)
        self._safe_stop(stop_callback)

    def play_tone(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> PlaybackRunResult:
        """Queue one bounded tone request."""

        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            action=lambda request_should_stop: self._play_tone(
                request_should_stop=request_should_stop,
                frequency_hz=frequency_hz,
                duration_ms=duration_ms,
                volume=volume,
                sample_rate=sample_rate,
            ),
        )

    def play_tone_sequence(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        sequence: tuple[tuple[int, int], ...],
        volume: float,
        sample_rate: int,
        gap_ms: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> PlaybackRunResult:
        """Queue one multi-tone feedback request."""

        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            action=lambda request_should_stop: self._play_tone_sequence(
                request_should_stop=request_should_stop,
                sequence=sequence,
                volume=volume,
                sample_rate=sample_rate,
                gap_ms=gap_ms,
            ),
        )

    def play_wav_chunks(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        chunks,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
    ) -> PlaybackRunResult:
        """Queue one streamed WAV playback request."""

        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            action=lambda request_should_stop: self._play_wav_chunks(
                request_should_stop=request_should_stop,
                chunks=chunks,
            ),
        )

    def play_wav_bytes(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        wav_bytes: bytes,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
    ) -> PlaybackRunResult:
        """Queue one fully-buffered WAV playback request."""

        play_wav_bytes = getattr(self.player, "play_wav_bytes", None)
        if not callable(play_wav_bytes):
            raise AttributeError("player must implement play_wav_bytes()")
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            action=lambda request_should_stop: self._play_wav_bytes(
                request_should_stop=request_should_stop,
                wav_bytes=wav_bytes,
                play_wav_bytes=play_wav_bytes,
            ),
        )

    def play_pcm16_chunks(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        chunks,
        sample_rate: int,
        channels: int = 1,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
    ) -> PlaybackRunResult:
        """Queue one streamed PCM16 playback request."""

        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            action=lambda request_should_stop: self._play_pcm16_chunks(
                request_should_stop=request_should_stop,
                chunks=chunks,
                sample_rate=sample_rate,
                channels=channels,
            ),
        )

    def _submit_player_action(
        self,
        *,
        owner: str,
        priority: int,
        action: Callable[[Callable[[], bool]], None],
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
    ) -> PlaybackRunResult:
        request = _PlaybackRequest(
            owner=owner,
            priority=priority,
            sequence=self._next_sequence(),
            action=action,
            stop=getattr(self.player, "stop_playback", None),
            external_should_stop=should_stop,
            atomic=bool(atomic),
            enqueued_at=time.monotonic(),
        )
        stop_callback: Callable[[], None] | None = None
        with self._condition:
            if self._closed:
                raise RuntimeError("Playback coordinator is closed")
            heappush(self._queue, (-priority, request.sequence, request))
            stop_callback = self._request_preemption_locked(priority)
            self._condition.notify_all()
        self._safe_stop(stop_callback)
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Playback coordinator completed without a result")
        return request.result

    def _next_sequence(self) -> int:
        with self._condition:
            self._sequence += 1
            return self._sequence

    def _request_preemption_locked(self, incoming_priority: int) -> Callable[[], None] | None:
        current = self._current
        if current is None:
            return None
        if current.atomic or incoming_priority <= current.priority:
            return None
        if current.cancel_event.is_set():
            return None
        current.cancel_event.set()
        self._emit(f"playback_preempt_requested={current.owner}")
        return self._build_bound_stop_callback_locked(current)

    def _build_bound_stop_callback_locked(
        self,
        request: _PlaybackRequest,
    ) -> Callable[[], None]:
        """Return one stop callback that only targets the original active request.

        A generic ``player.stop_playback()`` is unsafe here because the active
        player process may already have advanced to a newer request by the time
        the callback runs. That race can kill the replacement speech stream
        instead of the stale feedback request that requested preemption.
        """

        def _stop_if_still_current() -> None:
            stop_callback: Callable[[], None] | None = None
            with self._condition:
                current = self._current
                if current is request and current.cancel_event.is_set():
                    stop_callback = current.stop
            self._safe_stop(stop_callback)

        return _stop_if_still_current

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait(timeout=0.1)
                if self._closed and not self._queue:
                    return
                _, _, request = heappop(self._queue)
                self._current = request
            self._run_request(request)
            with self._condition:
                self._current = None
                self._condition.notify_all()

    def _run_request(self, request: _PlaybackRequest) -> None:
        started = time.monotonic()
        queue_wait_ms = round(max(0.0, (started - request.enqueued_at) * 1000.0), 3)
        self._emit(f"playback_started={request.owner}")
        try:
            self._run_action_with_optional_io_lock(request)
        except Exception as exc:
            request.error = exc
        runtime_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)
        request.result = PlaybackRunResult(
            owner=request.owner,
            priority=request.priority,
            queue_wait_ms=queue_wait_ms,
            runtime_ms=runtime_ms,
            preempted=request.cancel_event.is_set(),
        )
        self._emit(f"playback_completed={request.owner}")
        request.done.set()

    def _run_action_with_optional_io_lock(self, request: _PlaybackRequest) -> None:
        if request.should_stop():
            return
        if self.io_lock is None:
            request.action(request.should_stop)
            return
        with self.io_lock:
            request.action(request.should_stop)

    def _play_tone(
        self,
        *,
        request_should_stop: Callable[[], bool],
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
    ) -> None:
        if request_should_stop():
            return
        self.player.play_tone(
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            volume=volume,
            sample_rate=sample_rate,
        )

    def _play_tone_sequence(
        self,
        *,
        request_should_stop: Callable[[], bool],
        sequence: tuple[tuple[int, int], ...],
        volume: float,
        sample_rate: int,
        gap_ms: int,
    ) -> None:
        for tone_index, (frequency_hz, duration_ms) in enumerate(sequence):
            if request_should_stop():
                return
            self.player.play_tone(
                frequency_hz=frequency_hz,
                duration_ms=duration_ms,
                volume=volume,
                sample_rate=sample_rate,
            )
            if request_should_stop():
                return
            if gap_ms > 0 and tone_index + 1 < len(sequence):
                deadline = time.monotonic() + (gap_ms / 1000.0)
                while time.monotonic() < deadline:
                    if request_should_stop():
                        return
                    time.sleep(0.01)

    @staticmethod
    def _play_wav_bytes(
        *,
        request_should_stop: Callable[[], bool],
        wav_bytes: bytes,
        play_wav_bytes: Callable[[bytes], None],
    ) -> None:
        if request_should_stop():
            return
        play_wav_bytes(wav_bytes)

    def _play_wav_chunks(
        self,
        *,
        request_should_stop: Callable[[], bool],
        chunks,
    ) -> None:
        if request_should_stop():
            return
        try:
            self.player.play_wav_chunks(chunks, should_stop=request_should_stop)
        except TypeError as exc:
            if "should_stop" not in str(exc):
                raise
            self.player.play_wav_chunks(chunks)

    def _play_pcm16_chunks(
        self,
        *,
        request_should_stop: Callable[[], bool],
        chunks,
        sample_rate: int,
        channels: int,
    ) -> None:
        if request_should_stop():
            return
        try:
            self.player.play_pcm16_chunks(
                chunks,
                sample_rate=sample_rate,
                channels=channels,
                should_stop=request_should_stop,
            )
        except TypeError as exc:
            if "should_stop" not in str(exc):
                raise
            self.player.play_pcm16_chunks(
                chunks,
                sample_rate=sample_rate,
                channels=channels,
            )

    def _safe_stop(self, callback: Callable[[], None] | None) -> None:
        if not callable(callback):
            return
        try:
            callback()
        except Exception:
            return

    def _emit(self, message: str) -> None:
        if not callable(self.emit):
            return
        try:
            self.emit(message)
        except Exception:
            return
