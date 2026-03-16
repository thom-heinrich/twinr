"""Watch required remote readiness without blocking GPIO polling.

This helper keeps the expensive remote-primary readiness checks off the main
button-poll thread. The Pi must fail closed when remote memory is unavailable,
but button feedback must not wait behind multi-second remote probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Callable
import time


@dataclass(slots=True)
class RequiredRemoteDependencyWatch:
    """Run required-remote refreshes on a dedicated worker thread.

    The refresh callback is expected to update runtime/error state itself and
    must never be invoked from the GPIO polling thread.
    """

    interval_s: float
    refresh: Callable[[bool], bool]
    emit: Callable[[str], None] | None = None
    trace_event: Callable[[str, dict[str, object] | None], None] | None = None
    _stop_event: Event = field(init=False, repr=False, default_factory=Event)
    _wake_event: Event = field(init=False, repr=False, default_factory=Event)
    _thread: Thread | None = field(init=False, repr=False, default=None)

    def start(self) -> None:
        """Start the background watch once."""

        worker = self._thread
        if worker is not None and worker.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.set()
        worker = Thread(
            target=self._worker_main,
            name="twinr-required-remote-watch",
            daemon=True,
        )
        self._thread = worker
        worker.start()
        self._trace("required_remote_watch_thread_started", interval_s=float(self.interval_s))

    def request_refresh(self) -> None:
        """Wake the worker for an immediate check."""

        self._wake_event.set()
        self._trace("required_remote_watch_refresh_requested", wake_set=True)

    def stop(self, *, timeout_s: float = 1.0) -> None:
        """Stop the worker and join briefly."""

        self._stop_event.set()
        self._wake_event.set()
        worker = self._thread
        if worker is None:
            self._trace("required_remote_watch_stop_without_worker", timeout_s=float(timeout_s))
            return
        self._trace(
            "required_remote_watch_stop_requested",
            worker_alive=worker.is_alive(),
            timeout_s=float(timeout_s),
        )
        worker.join(timeout=max(0.05, float(timeout_s)))
        if worker.is_alive() and callable(self.emit):
            try:
                self.emit("required_remote_watch_join_timeout=true")
            except Exception:
                pass
            self._trace("required_remote_watch_join_timeout", timeout_s=float(timeout_s))

    def _worker_main(self) -> None:
        force = True
        interval_s = max(0.1, float(self.interval_s))
        self._trace("required_remote_watch_worker_entered", interval_s=interval_s)
        while not self._stop_event.is_set():
            self._wake_event.clear()
            started = time.monotonic()
            self._trace("required_remote_watch_refresh_started", force=force)
            try:
                ready = self.refresh(force)
            except Exception as exc:
                if callable(self.emit):
                    try:
                        self.emit(f"required_remote_watch_error={type(exc).__name__}")
                    except Exception:
                        pass
                self._trace(
                    "required_remote_watch_refresh_failed",
                    force=force,
                    error_type=type(exc).__name__,
                    duration_ms=round((time.monotonic() - started) * 1000.0, 3),
                )
            else:
                self._trace(
                    "required_remote_watch_refresh_completed",
                    force=force,
                    ready=bool(ready),
                    duration_ms=round((time.monotonic() - started) * 1000.0, 3),
                )
            force = False
            if self._stop_event.is_set():
                self._trace("required_remote_watch_worker_exit_stop", force=force)
                return
            self._wake_event.wait(timeout=interval_s)
            force = self._wake_event.is_set()
            self._trace("required_remote_watch_wait_completed", force=force, stop=self._stop_event.is_set())

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self.trace_event):
            return
        try:
            self.trace_event(msg, details)
        except Exception:
            return
