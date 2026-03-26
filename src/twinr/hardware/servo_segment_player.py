"""Play one bounded servo pulse segment with an exact disable deadline.

The continuous-servo zero-return path needs physical pulse durations that are
shorter and more precise than the main runtime update cadence. This helper
owns that concern in one focused place: start one pulse immediately, disable it
at the exact segment deadline in the background, and report completion back to
the caller without forcing the higher follow controller to block or spin.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Protocol


class ServoPulseWriter(Protocol):
    """Minimal servo-writer contract needed for bounded pulse playback."""

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        """Drive one pulse width on the configured output."""

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        """Release the configured output."""


@dataclass(frozen=True, slots=True)
class ServoPulseSegmentPlayback:
    """Describe one currently playing or completed bounded pulse segment."""

    pulse_width_us: int
    duration_s: float
    started_at: float
    due_at: float


@dataclass(frozen=True, slots=True)
class ServoPulseSegmentCompletion:
    """Report one finished bounded segment plus any disable error."""

    playback: ServoPulseSegmentPlayback
    error: str | None = None


class _ThreadLike(Protocol):
    def start(self) -> None:
        """Start one background worker."""


class ServoPulseSegmentPlayer(Protocol):
    """Interface for exact-duration replay segment playback."""

    def start_segment(self, *, pulse_width_us: int, duration_s: float) -> ServoPulseSegmentPlayback:
        """Start one bounded pulse segment immediately."""

    def active_segment(self) -> ServoPulseSegmentPlayback | None:
        """Return the currently active segment, if any."""

    def consume_completion(self) -> ServoPulseSegmentCompletion | None:
        """Return and clear the latest completion snapshot, if any."""

    def cancel(self) -> ServoPulseSegmentPlayback | None:
        """Cancel the current segment, if any."""

    def close(self) -> None:
        """Release any internal resources."""


@dataclass(slots=True)
class _ActiveSegment:
    playback: ServoPulseSegmentPlayback
    cancel_event: threading.Event
    generation: int


def _default_wait_fn(cancel_event: threading.Event, timeout_s: float) -> bool:
    return cancel_event.wait(max(0.0, float(timeout_s)))


def _default_thread_factory(target: Callable[[], None]) -> _ThreadLike:
    thread = threading.Thread(target=target, daemon=True)
    return thread


class BoundedServoPulseSegmentPlayer:
    """Play one pulse immediately, then disable it at the recorded deadline."""

    def __init__(
        self,
        *,
        pulse_writer: ServoPulseWriter,
        gpio_chip: str,
        gpio: int,
        monotonic_fn: Callable[[], float] | None = None,
        wait_fn: Callable[[threading.Event, float], bool] | None = None,
        thread_factory: Callable[[Callable[[], None]], _ThreadLike] | None = None,
    ) -> None:
        self._pulse_writer = pulse_writer
        self._gpio_chip = str(gpio_chip)
        self._gpio = int(gpio)
        self._monotonic_fn = monotonic_fn or time.monotonic
        self._wait_fn = wait_fn or _default_wait_fn
        self._thread_factory = thread_factory or _default_thread_factory
        self._lock = threading.Lock()
        self._active: _ActiveSegment | None = None
        self._completion: ServoPulseSegmentCompletion | None = None
        self._generation = 0
        self._closed = False

    def start_segment(self, *, pulse_width_us: int, duration_s: float) -> ServoPulseSegmentPlayback:
        """Write one pulse now and schedule an exact disable for its duration."""

        checked_duration_s = max(0.0, float(duration_s))
        started_at = float(self._monotonic_fn())
        playback = ServoPulseSegmentPlayback(
            pulse_width_us=int(pulse_width_us),
            duration_s=checked_duration_s,
            started_at=started_at,
            due_at=started_at + checked_duration_s,
        )
        cancel_event = threading.Event()
        with self._lock:
            if self._closed:
                raise RuntimeError("servo segment player is closed")
            if self._active is not None:
                raise RuntimeError("servo segment player is already active")
            self._generation += 1
            generation = self._generation
            self._active = _ActiveSegment(
                playback=playback,
                cancel_event=cancel_event,
                generation=generation,
            )
            self._completion = None
        try:
            self._pulse_writer.write(
                gpio_chip=self._gpio_chip,
                gpio=self._gpio,
                pulse_width_us=playback.pulse_width_us,
            )
        except Exception:
            with self._lock:
                if self._active is not None and self._active.generation == generation:
                    self._active = None
            raise
        self._thread_factory(
            lambda: self._run_disable_after_segment(
                generation=generation,
                cancel_event=cancel_event,
                playback=playback,
            )
        ).start()
        return playback

    def active_segment(self) -> ServoPulseSegmentPlayback | None:
        """Return the currently active segment, if any."""

        with self._lock:
            if self._active is None:
                return None
            return self._active.playback

    def consume_completion(self) -> ServoPulseSegmentCompletion | None:
        """Return and clear the latest completion snapshot, if any."""

        with self._lock:
            completion = self._completion
            self._completion = None
            return completion

    def cancel(self) -> ServoPulseSegmentPlayback | None:
        """Cancel the active segment without reporting it as completed."""

        active_segment: _ActiveSegment | None
        with self._lock:
            active_segment = self._active
            if active_segment is None:
                return None
            self._generation += 1
            self._active = None
            self._completion = None
        active_segment.cancel_event.set()
        return active_segment.playback

    def close(self) -> None:
        """Close the player and cancel any in-flight bounded segment."""

        self.cancel()
        with self._lock:
            self._closed = True

    def _run_disable_after_segment(
        self,
        *,
        generation: int,
        cancel_event: threading.Event,
        playback: ServoPulseSegmentPlayback,
    ) -> None:
        remaining_s = max(0.0, playback.due_at - float(self._monotonic_fn()))
        cancelled = bool(self._wait_fn(cancel_event, remaining_s))
        if cancelled:
            return
        error: str | None = None
        try:
            self._pulse_writer.disable(
                gpio_chip=self._gpio_chip,
                gpio=self._gpio,
            )
        except Exception as exc:  # pragma: no cover - surfaced through completion payload.
            error = f"{exc.__class__.__name__}: {exc}"
        with self._lock:
            if self._active is None or self._active.generation != generation:
                return
            self._active = None
            self._completion = ServoPulseSegmentCompletion(
                playback=playback,
                error=error,
            )
