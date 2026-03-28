# CHANGELOG: 2026-03-27
# BUG-1: Fixed a same-priority enqueue race; sequence assignment and heap insertion are now atomic.
# BUG-2: Fixed stop_playback(); it now sets the coordinator cancel token so cooperative chunk/WAV playback really stops.
# BUG-3: Fixed unsafe daemon-worker shutdown; worker is now non-daemonic and the coordinator is a context manager for clean teardown.
# SEC-1: Added bounded queueing plus eviction/expiry to prevent practical memory/availability DoS on Raspberry Pi 4 deployments.
# SEC-2: Sanitized owner values before emitting text events to prevent control-character/log-injection via emit().
# IMP-1: Added non-blocking submit_* APIs with PlaybackHandle, wait timeouts, and immediate/draining shutdown semantics.
# IMP-2: Added stale-tone expiry, owner/category supersession, queue backpressure, and cancellable waiting on io_lock.
# IMP-3: Added interruptible WAV-byte playback fallback via PCM16 chunk streaming when the backend exposes play_pcm16_chunks().
# BUG-4: Suppress expected player stop errors for cancelled/preempted requests so feedback preemption does not explode worker threads.

"""Serialize Twinr speaker output under one priority-aware owner.

This module centralizes listen beeps, feedback tones, and spoken TTS playback
behind one queue and one preemption policy. It does not own microphone or
ambient-audio sampling; callers may still pass a broader I/O lock so playback
stays isolated from recorder use on the Pi.

2026 upgrade notes:
- synchronous play_* APIs remain available and drop-in compatible
- new submit_* APIs return a PlaybackHandle for non-blocking operation
- close(immediate=True) mirrors modern queue hard-shutdown behavior
- tone/tone-sequence requests can expire instead of playing stale cues late
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heapify, heappop, heappush
from io import BytesIO
from threading import Condition, Event, Lock, Thread
import time
import wave


class PlaybackPriority(IntEnum):
    """Define Twinr playback priority bands."""

    FEEDBACK = 10
    SPEECH = 20
    BEEP = 30


class PlaybackCoordinatorClosedError(RuntimeError):
    """Raised when new work is submitted after coordinator shutdown."""


class PlaybackQueueFullError(RuntimeError):
    """Raised when the bounded queue cannot accept more work."""


class PlaybackWaitTimeoutError(TimeoutError):
    """Raised when waiting for a playback request times out."""


@dataclass(frozen=True, slots=True)
class PlaybackRunResult:
    """Capture queue and runtime timings for one playback request."""

    owner: str
    priority: int
    queue_wait_ms: float
    runtime_ms: float
    preempted: bool = False
    started: bool = True
    completed: bool = True
    cancelled: bool = False
    dropped: bool = False
    cancel_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result view."""

        return {
            "owner": self.owner,
            "priority": self.priority,
            "queue_wait_ms": self.queue_wait_ms,
            "runtime_ms": self.runtime_ms,
            "preempted": self.preempted,
            "started": self.started,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "dropped": self.dropped,
            "cancel_reason": self.cancel_reason,
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
    enqueued_at_ns: int
    expiry_deadline_ns: int | None = None
    category: str = "generic"
    done: Event = field(default_factory=Event)
    cancel_event: Event = field(default_factory=Event)
    external_cancel_event: Event = field(default_factory=Event)
    error: Exception | None = None
    result: PlaybackRunResult | None = None
    cancel_reason: str | None = None
    started: bool = False
    in_queue: bool = True

    def should_stop(self) -> bool:
        """Return whether playback should stop for cancellation or shutdown."""

        if self.cancel_event.is_set():
            return True
        callback = self.external_should_stop
        if not callable(callback):
            return False
        try:
            should_stop = bool(callback())
        except Exception:
            should_stop = True
        if should_stop:
            self.external_cancel_event.set()
        return should_stop


class PlaybackHandle:
    """A handle for non-blocking playback submission."""

    __slots__ = ("_coordinator", "_request")

    def __init__(self, coordinator: "PlaybackCoordinator", request: _PlaybackRequest) -> None:
        self._coordinator = coordinator
        self._request = request

    @property
    def owner(self) -> str:
        return self._request.owner

    @property
    def priority(self) -> int:
        return self._request.priority

    @property
    def interrupted(self) -> bool:
        result = self._request.result
        if result is not None:
            return result.preempted
        return self._request.cancel_reason == "preempted"

    def done(self) -> bool:
        return self._request.done.is_set()

    def cancel(self, *, reason: str = "cancelled", force: bool = False) -> bool:
        """Cancel the request if still queued/running."""

        return self._coordinator._cancel_request(self._request, reason=reason, force=force)

    def interrupt(self, *, force: bool = False) -> bool:
        """Interrupt the request, respecting atomic playback unless forced."""

        return self.cancel(reason="preempted", force=force)

    def wait(self, timeout_s: float | None = None) -> PlaybackRunResult:
        """Wait for completion and return the final run result."""

        if not self._request.done.wait(timeout=timeout_s):
            raise PlaybackWaitTimeoutError(
                f"playback request for owner={self._request.owner!r} did not finish before timeout"
            )
        result = self._request.result
        if self._request.error is not None:
            if result is not None and result.cancelled:
                return result
            raise self._request.error
        if result is None:
            raise RuntimeError("Playback coordinator completed without a result")
        return result

    result = wait


class PlaybackCoordinator:
    """Run all speaker output through one priority queue and worker."""

    def __init__(
        self,
        player,
        *,
        emit: Callable[[str], None] | None = None,
        io_lock: Lock | None = None,
        max_queue_size: int = 64,
        max_wav_bytes: int = 32 * 1024 * 1024,
        io_lock_poll_interval_s: float = 0.05,
        default_tone_max_queue_wait_ms: float | None = 750.0,
        default_tone_sequence_max_queue_wait_ms: float | None = 1500.0,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if max_wav_bytes <= 0:
            raise ValueError("max_wav_bytes must be > 0")
        if io_lock_poll_interval_s <= 0:
            raise ValueError("io_lock_poll_interval_s must be > 0")

        self.player = player
        self.emit = emit
        self.io_lock = io_lock
        self.max_queue_size = int(max_queue_size)
        self.max_wav_bytes = int(max_wav_bytes)
        self.io_lock_poll_interval_s = float(io_lock_poll_interval_s)
        self.default_tone_max_queue_wait_ms = default_tone_max_queue_wait_ms
        self.default_tone_sequence_max_queue_wait_ms = default_tone_sequence_max_queue_wait_ms

        self._condition = Condition()
        self._queue: list[tuple[int, int, _PlaybackRequest]] = []
        self._current: _PlaybackRequest | None = None
        self._closed = False
        self._sequence = 0

        # BREAKING: the worker is non-daemonic so ALSA/PortAudio-backed players do
        # not get torn down abruptly at interpreter exit; callers should close()
        # explicitly or use "with PlaybackCoordinator(...)".
        self._worker = Thread(
            target=self._worker_main,
            daemon=False,
            name="twinr-playback-coordinator",
        )
        self._worker.start()

    def __enter__(self) -> "PlaybackCoordinator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close(immediate=True, timeout_s=2.0)
        return False

    def close(self, *, timeout_s: float | None = None, immediate: bool = False) -> None:
        """Stop the coordinator.

        The active request is interrupted during shutdown. If immediate=False,
        pending queued work is allowed to drain afterward. If immediate=True,
        pending queued work is cancelled immediately.
        """

        stop_callback: Callable[[], None] | None = None
        with self._condition:
            if self._closed and not self._worker.is_alive():
                return

            self._closed = True
            if immediate:
                self._cancel_pending_locked(lambda request: True, reason="shutdown", dropped=True)

            current = self._current
            if current is not None and not current.done.is_set():
                current.cancel_reason = current.cancel_reason or "shutdown"
                current.cancel_event.set()
                stop_callback = self._build_bound_stop_callback_locked(current)

            self._condition.notify_all()

        self._safe_stop(stop_callback)
        self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            raise RuntimeError("Playback coordinator did not shut down before timeout")

    def stop_playback(self) -> None:
        """Best-effort stop for the active player process."""

        stop_callback: Callable[[], None] | None = None
        with self._condition:
            current = self._current
            if current is None or current.done.is_set():
                return
            current.cancel_reason = current.cancel_reason or "stop_playback"
            current.cancel_event.set()
            stop_callback = self._build_bound_stop_callback_locked(current)

        self._safe_stop(stop_callback)

    def stop_owner(self, owner: str) -> None:
        """Stop all playback owned by the named owner.

        # BREAKING: this now cancels queued requests for the same owner too, not
        # only the currently active request. That avoids stale same-owner audio
        # playing after a user barge-in or owner-level cancellation.
        """

        stop_callback: Callable[[], None] | None = None
        with self._condition:
            current = self._current
            if current is not None and current.owner == owner and not current.done.is_set():
                current.cancel_reason = current.cancel_reason or "owner_stop"
                current.cancel_event.set()
                stop_callback = self._build_bound_stop_callback_locked(current)

            self._cancel_pending_locked(
                lambda request: request.owner == owner,
                reason="owner_stop",
                dropped=True,
            )
            self._condition.notify_all()

        self._safe_stop(stop_callback)

    def submit_tone(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
        should_stop: Callable[[], bool] | None = None,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = True,
    ) -> PlaybackHandle:
        """Queue one bounded tone request and return a handle."""

        self._validate_owner(owner)
        self._validate_tone_args(
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            volume=volume,
            sample_rate=sample_rate,
        )
        effective_wait = (
            self.default_tone_max_queue_wait_ms
            if max_queue_wait_ms is None
            else max_queue_wait_ms
        )
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            max_queue_wait_ms=effective_wait,
            supersede_pending_owner=supersede_pending_owner,
            category="tone",
            action=lambda request_should_stop: self._play_tone(
                request_should_stop=request_should_stop,
                frequency_hz=frequency_hz,
                duration_ms=duration_ms,
                volume=volume,
                sample_rate=sample_rate,
            ),
        )

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
        max_queue_wait_ms: float | None = None,
        wait_timeout_s: float | None = None,
        supersede_pending_owner: bool = True,
    ) -> PlaybackRunResult:
        """Queue one bounded tone request and wait for completion."""

        return self.submit_tone(
            owner=owner,
            priority=priority,
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            volume=volume,
            sample_rate=sample_rate,
            should_stop=should_stop,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
        ).wait(wait_timeout_s)

    def submit_tone_sequence(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        sequence: tuple[tuple[int, int], ...],
        volume: float,
        sample_rate: int,
        gap_ms: int = 0,
        should_stop: Callable[[], bool] | None = None,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = True,
    ) -> PlaybackHandle:
        """Queue one multi-tone feedback request and return a handle."""

        self._validate_owner(owner)
        self._validate_tone_sequence_args(
            sequence=sequence,
            volume=volume,
            sample_rate=sample_rate,
            gap_ms=gap_ms,
        )
        effective_wait = (
            self.default_tone_sequence_max_queue_wait_ms
            if max_queue_wait_ms is None
            else max_queue_wait_ms
        )
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            max_queue_wait_ms=effective_wait,
            supersede_pending_owner=supersede_pending_owner,
            category="tone_sequence",
            action=lambda request_should_stop: self._play_tone_sequence(
                request_should_stop=request_should_stop,
                sequence=sequence,
                volume=volume,
                sample_rate=sample_rate,
                gap_ms=gap_ms,
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
        max_queue_wait_ms: float | None = None,
        wait_timeout_s: float | None = None,
        supersede_pending_owner: bool = True,
    ) -> PlaybackRunResult:
        """Queue one multi-tone feedback request and wait for completion."""

        return self.submit_tone_sequence(
            owner=owner,
            priority=priority,
            sequence=sequence,
            volume=volume,
            sample_rate=sample_rate,
            gap_ms=gap_ms,
            should_stop=should_stop,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
        ).wait(wait_timeout_s)

    def submit_wav_chunks(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        chunks,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackHandle:
        """Queue one streamed WAV playback request and return a handle."""

        self._validate_owner(owner)
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
            category="wav_chunks",
            action=lambda request_should_stop: self._play_wav_chunks(
                request_should_stop=request_should_stop,
                chunks=chunks,
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
        max_queue_wait_ms: float | None = None,
        wait_timeout_s: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackRunResult:
        """Queue one streamed WAV playback request and wait for completion."""

        return self.submit_wav_chunks(
            owner=owner,
            priority=priority,
            chunks=chunks,
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
        ).wait(wait_timeout_s)

    def submit_wav_bytes(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        wav_bytes: bytes,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackHandle:
        """Queue one fully-buffered WAV playback request and return a handle."""

        self._validate_owner(owner)
        self._validate_wav_bytes(wav_bytes)
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
            category="wav_bytes",
            action=lambda request_should_stop: self._play_wav_bytes(
                request_should_stop=request_should_stop,
                wav_bytes=wav_bytes,
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
        max_queue_wait_ms: float | None = None,
        wait_timeout_s: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackRunResult:
        """Queue one fully-buffered WAV playback request and wait for completion."""

        return self.submit_wav_bytes(
            owner=owner,
            priority=priority,
            wav_bytes=wav_bytes,
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
        ).wait(wait_timeout_s)

    def submit_pcm16_chunks(
        self,
        *,
        owner: str,
        priority: PlaybackPriority | int,
        chunks,
        sample_rate: int,
        channels: int = 1,
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackHandle:
        """Queue one streamed PCM16 playback request and return a handle."""

        self._validate_owner(owner)
        self._validate_pcm_args(sample_rate=sample_rate, channels=channels)
        return self._submit_player_action(
            owner=owner,
            priority=int(priority),
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
            category="pcm16_chunks",
            action=lambda request_should_stop: self._play_pcm16_chunks(
                request_should_stop=request_should_stop,
                chunks=chunks,
                sample_rate=sample_rate,
                channels=channels,
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
        max_queue_wait_ms: float | None = None,
        wait_timeout_s: float | None = None,
        supersede_pending_owner: bool = False,
    ) -> PlaybackRunResult:
        """Queue one streamed PCM16 playback request and wait for completion."""

        return self.submit_pcm16_chunks(
            owner=owner,
            priority=priority,
            chunks=chunks,
            sample_rate=sample_rate,
            channels=channels,
            should_stop=should_stop,
            atomic=atomic,
            max_queue_wait_ms=max_queue_wait_ms,
            supersede_pending_owner=supersede_pending_owner,
        ).wait(wait_timeout_s)

    def _submit_player_action(
        self,
        *,
        owner: str,
        priority: int,
        action: Callable[[Callable[[], bool]], None],
        should_stop: Callable[[], bool] | None = None,
        atomic: bool = False,
        max_queue_wait_ms: float | None = None,
        supersede_pending_owner: bool = False,
        category: str = "generic",
    ) -> PlaybackHandle:
        expiry_deadline_ns = self._compute_deadline_ns(max_queue_wait_ms)
        stop_callback: Callable[[], None] | None = None

        with self._condition:
            if self._closed:
                raise PlaybackCoordinatorClosedError("Playback coordinator is closed")

            self._expire_pending_locked()

            if supersede_pending_owner:
                self._cancel_pending_locked(
                    lambda request: request.owner == owner and request.category == category,
                    reason="superseded",
                    dropped=True,
                )

            if not self._ensure_queue_capacity_locked(incoming_priority=priority, owner=owner):
                raise PlaybackQueueFullError(
                    f"playback queue is full (max_queue_size={self.max_queue_size})"
                )

            # BUG-1 fix: sequence assignment and heap insertion are inside the same
            # critical section, so same-priority FIFO ordering is stable under
            # concurrent submissions.
            self._sequence += 1
            request = _PlaybackRequest(
                owner=owner,
                priority=priority,
                sequence=self._sequence,
                action=action,
                stop=getattr(self.player, "stop_playback", None),
                external_should_stop=should_stop,
                atomic=bool(atomic),
                enqueued_at_ns=time.monotonic_ns(),
                expiry_deadline_ns=expiry_deadline_ns,
                category=category,
            )
            heappush(self._queue, (-priority, request.sequence, request))
            stop_callback = self._request_preemption_locked(priority)
            self._maybe_compact_queue_locked()
            self._condition.notify_all()

        self._safe_stop(stop_callback)
        return PlaybackHandle(self, request)

    def _compute_deadline_ns(self, max_queue_wait_ms: float | None) -> int | None:
        if max_queue_wait_ms is None:
            return None
        if max_queue_wait_ms < 0:
            raise ValueError("max_queue_wait_ms must be >= 0 or None")
        return time.monotonic_ns() + int(max_queue_wait_ms * 1_000_000)

    def _ensure_queue_capacity_locked(self, *, incoming_priority: int, owner: str) -> bool:
        if self._live_pending_count_locked() < self.max_queue_size:
            return True

        candidates: list[_PlaybackRequest] = []
        for _, _, request in self._queue:
            if request.done.is_set():
                continue
            if request.atomic:
                continue
            if request.owner == owner or request.priority < incoming_priority:
                candidates.append(request)

        if not candidates:
            return False

        candidate = min(candidates, key=lambda request: (request.priority, request.sequence))
        self._finalize_pending_request_locked(candidate, reason="queue_evicted", dropped=True)
        self._emit(f"playback_dropped={self._safe_owner_for_emit(candidate.owner)}")
        self._maybe_compact_queue_locked(force=True)
        return True

    def _live_pending_count_locked(self) -> int:
        return sum(1 for _, _, request in self._queue if not request.done.is_set())

    def _expire_pending_locked(self) -> None:
        now_ns = time.monotonic_ns()
        changed = False
        for _, _, request in self._queue:
            if request.done.is_set():
                continue
            deadline_ns = request.expiry_deadline_ns
            if deadline_ns is not None and now_ns >= deadline_ns:
                self._finalize_pending_request_locked(request, reason="expired", dropped=True)
                self._emit(f"playback_dropped={self._safe_owner_for_emit(request.owner)}")
                changed = True
        if changed:
            self._maybe_compact_queue_locked(force=True)

    def _cancel_pending_locked(
        self,
        predicate: Callable[[_PlaybackRequest], bool],
        *,
        reason: str,
        dropped: bool,
    ) -> None:
        changed = False
        for _, _, request in self._queue:
            if request.done.is_set():
                continue
            if predicate(request):
                self._finalize_pending_request_locked(request, reason=reason, dropped=dropped)
                changed = True
        if changed:
            self._maybe_compact_queue_locked(force=True)

    def _cancel_request(
        self,
        request: _PlaybackRequest,
        *,
        reason: str,
        force: bool = False,
    ) -> bool:
        stop_callback: Callable[[], None] | None = None
        with self._condition:
            if request.done.is_set():
                return False
            if request.atomic and reason == "preempted" and not force:
                return False

            if request is self._current:
                request.cancel_reason = request.cancel_reason or reason
                request.cancel_event.set()
                stop_callback = self._build_bound_stop_callback_locked(request)
            else:
                self._finalize_pending_request_locked(request, reason=reason, dropped=True)
                self._maybe_compact_queue_locked(force=True)
                self._condition.notify_all()

        self._safe_stop(stop_callback)
        return True

    def _request_preemption_locked(self, incoming_priority: int) -> Callable[[], None] | None:
        current = self._current
        if current is None:
            return None
        if current.atomic or incoming_priority <= current.priority:
            return None
        if current.cancel_event.is_set():
            return None

        current.cancel_reason = current.cancel_reason or "preempted"
        current.cancel_event.set()
        self._emit(f"playback_preempt_requested={self._safe_owner_for_emit(current.owner)}")
        return self._build_bound_stop_callback_locked(current)

    def _build_bound_stop_callback_locked(
        self,
        request: _PlaybackRequest,
    ) -> Callable[[], None]:
        """Return one stop callback that only targets the original active request.

        A generic player.stop_playback() is unsafe here because the active player
        process may already have advanced to a newer request by the time the callback
        runs. That race can kill the replacement speech stream instead of the stale
        feedback request that requested preemption.
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
                request: _PlaybackRequest | None = None
                while request is None:
                    self._expire_pending_locked()
                    while self._queue:
                        _, _, candidate = heappop(self._queue)
                        if candidate.done.is_set():
                            continue
                        candidate.in_queue = False
                        request = candidate
                        break

                    if request is not None:
                        self._current = request
                        break

                    if self._closed:
                        return

                    self._condition.wait(timeout=0.1)

            self._run_request(request)

            with self._condition:
                if self._current is request:
                    self._current = None
                self._condition.notify_all()

    def _run_request(self, request: _PlaybackRequest) -> None:
        popped_at_ns = time.monotonic_ns()
        queue_wait_ms = self._ns_to_ms(popped_at_ns - request.enqueued_at_ns)

        if request.should_stop():
            with self._condition:
                if not request.done.is_set():
                    self._finalize_request_locked(
                        request,
                        queue_wait_ms=queue_wait_ms,
                        runtime_ms=0.0,
                        started=False,
                    )
            return

        self._emit(f"playback_started={self._safe_owner_for_emit(request.owner)}")
        runtime_start_ns = time.monotonic_ns()
        try:
            self._run_action_with_optional_io_lock(request)
        except Exception as exc:
            request.error = exc

        runtime_ms = self._ns_to_ms(time.monotonic_ns() - runtime_start_ns)
        with self._condition:
            if not request.done.is_set():
                self._finalize_request_locked(
                    request,
                    queue_wait_ms=queue_wait_ms,
                    runtime_ms=runtime_ms,
                    started=request.started,
                )

        self._emit(f"playback_completed={self._safe_owner_for_emit(request.owner)}")

    def _finalize_pending_request_locked(
        self,
        request: _PlaybackRequest,
        *,
        reason: str,
        dropped: bool,
    ) -> None:
        request.cancel_reason = request.cancel_reason or reason
        request.cancel_event.set()
        request.in_queue = False
        queue_wait_ms = self._ns_to_ms(max(0, time.monotonic_ns() - request.enqueued_at_ns))
        self._finalize_request_locked(
            request,
            queue_wait_ms=queue_wait_ms,
            runtime_ms=0.0,
            started=False,
            dropped=dropped,
        )

    def _finalize_request_locked(
        self,
        request: _PlaybackRequest,
        *,
        queue_wait_ms: float,
        runtime_ms: float,
        started: bool,
        dropped: bool = False,
    ) -> None:
        cancel_reason = request.cancel_reason
        if cancel_reason is None and request.external_cancel_event.is_set():
            cancel_reason = "external_stop"

        cancelled = (
            cancel_reason is not None
            or request.cancel_event.is_set()
            or request.external_cancel_event.is_set()
        )
        preempted = cancel_reason == "preempted"
        completed = request.error is None and not cancelled

        request.result = PlaybackRunResult(
            owner=request.owner,
            priority=request.priority,
            queue_wait_ms=queue_wait_ms,
            runtime_ms=runtime_ms,
            preempted=preempted,
            started=started,
            completed=completed,
            cancelled=cancelled,
            dropped=bool(dropped),
            cancel_reason=cancel_reason,
        )
        request.done.set()

    def _maybe_compact_queue_locked(self, *, force: bool = False) -> None:
        if not force and len(self._queue) <= (self.max_queue_size * 2):
            return
        live_entries = [entry for entry in self._queue if not entry[2].done.is_set()]
        if len(live_entries) == len(self._queue):
            return
        self._queue = live_entries
        heapify(self._queue)

    def _run_action_with_optional_io_lock(self, request: _PlaybackRequest) -> None:
        if request.should_stop():
            return

        if self.io_lock is None:
            request.started = True
            request.action(request.should_stop)
            return

        # Frontier upgrade: do not block forever on the broader I/O lock. Polling
        # lets stop/preempt/owner-cancel abort while the worker is waiting for the
        # microphone/recorder side to release the lock.
        while True:
            if request.should_stop():
                return
            acquired = self.io_lock.acquire(timeout=self.io_lock_poll_interval_s)
            if acquired:
                break

        try:
            if request.should_stop():
                return
            request.started = True
            request.action(request.should_stop)
        finally:
            self.io_lock.release()

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
                deadline_ns = time.monotonic_ns() + int(gap_ms * 1_000_000)
                while time.monotonic_ns() < deadline_ns:
                    if request_should_stop():
                        return
                    remaining_s = max(
                        0.0,
                        (deadline_ns - time.monotonic_ns()) / 1_000_000_000.0,
                    )
                    time.sleep(min(0.005, remaining_s))

    def _play_wav_bytes(
        self,
        *,
        request_should_stop: Callable[[], bool],
        wav_bytes: bytes,
    ) -> None:
        """Play buffered WAV bytes.

        Frontier upgrade: if the backend exposes play_pcm16_chunks(), decode PCM16
        WAV frames and stream them cooperatively so higher-priority audio and
        stop_owner()/stop_playback() can interrupt the request cleanly.
        """

        if request_should_stop():
            return

        parsed = self._try_decode_wav_bytes_for_pcm_stream(wav_bytes)
        if parsed is not None:
            chunks, sample_rate, channels = parsed
            self._play_pcm16_chunks(
                request_should_stop=request_should_stop,
                chunks=chunks,
                sample_rate=sample_rate,
                channels=channels,
            )
            return

        play_wav_bytes = getattr(self.player, "play_wav_bytes", None)
        if not callable(play_wav_bytes):
            raise AttributeError(
                "player must implement play_wav_bytes() or play_pcm16_chunks() for WAV playback"
            )
        play_wav_bytes(wav_bytes)

    def _try_decode_wav_bytes_for_pcm_stream(
        self,
        wav_bytes: bytes,
    ) -> tuple[object, int, int] | None:
        if not callable(getattr(self.player, "play_pcm16_chunks", None)):
            return None

        try:
            with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                compression = wav_file.getcomptype()
        except wave.Error:
            return None

        if sample_width != 2 or compression != "NONE":
            return None

        def _chunks():
            with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
                while True:
                    chunk = wav_file.readframes(2048)
                    if not chunk:
                        return
                    yield chunk

        return _chunks(), sample_rate, channels

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

    def _validate_owner(self, owner: str) -> None:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner must be a non-empty string")

    def _validate_tone_args(
        self,
        *,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
    ) -> None:
        if sample_rate <= 0 or sample_rate > 384000:
            raise ValueError("sample_rate must be between 1 and 384000")
        if frequency_hz <= 0 or frequency_hz > (sample_rate // 2):
            raise ValueError("frequency_hz must be > 0 and <= sample_rate/2")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be > 0")
        if not (0.0 <= volume <= 1.0):
            raise ValueError("volume must be between 0.0 and 1.0")

    def _validate_tone_sequence_args(
        self,
        *,
        sequence: tuple[tuple[int, int], ...],
        volume: float,
        sample_rate: int,
        gap_ms: int,
    ) -> None:
        if not sequence:
            raise ValueError("sequence must not be empty")
        for frequency_hz, duration_ms in sequence:
            self._validate_tone_args(
                frequency_hz=int(frequency_hz),
                duration_ms=int(duration_ms),
                volume=volume,
                sample_rate=sample_rate,
            )
        if gap_ms < 0:
            raise ValueError("gap_ms must be >= 0")

    def _validate_pcm_args(self, *, sample_rate: int, channels: int) -> None:
        if sample_rate <= 0 or sample_rate > 384000:
            raise ValueError("sample_rate must be between 1 and 384000")
        if channels <= 0 or channels > 8:
            raise ValueError("channels must be between 1 and 8")

    def _validate_wav_bytes(self, wav_bytes: bytes) -> None:
        if not isinstance(wav_bytes, bytes):
            raise TypeError("wav_bytes must be bytes")
        if not wav_bytes:
            raise ValueError("wav_bytes must not be empty")
        if len(wav_bytes) > self.max_wav_bytes:
            raise ValueError(f"wav_bytes exceeds max_wav_bytes={self.max_wav_bytes}")

    @staticmethod
    def _safe_owner_for_emit(owner: str) -> str:
        safe = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in owner)
        safe = safe.replace("=", "_")
        safe = safe.strip()
        return safe[:120] or "unknown"

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return round(max(0.0, duration_ns / 1_000_000.0), 3)

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
