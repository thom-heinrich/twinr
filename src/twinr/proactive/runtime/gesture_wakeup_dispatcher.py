"""Dispatch visual gesture wakeups without blocking proactive refresh ticks.

The proactive monitor worker must keep driving HDMI attention-follow while a
gesture wakeup opens a hands-free listening session. This module provides a
single-flight dispatcher that moves the potentially long-running wakeup
handler off the monitor thread while deliberately refusing concurrent visual
wake requests.
"""

from __future__ import annotations

from threading import Lock, Thread, current_thread
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .gesture_wakeup_lane import GestureWakeupDecision


class GestureWakeupDispatcher:
    """Run at most one visual wakeup handler at a time on a worker thread.

    The dispatcher intentionally does not queue follow-up wakeup requests. If a
    visual wakeup is already opening or running a listening session, newer
    requests are ignored so the proactive monitor can keep refreshing
    attention-follow instead of stacking redundant conversation launches.
    """

    def __init__(
        self,
        *,
        handle_decision: Callable[["GestureWakeupDecision"], bool],
    ) -> None:
        """Initialize the dispatcher from the wakeup handler callback.

        Args:
            handle_decision: Callback that opens one wakeup-driven conversation
                path. It runs on the worker thread and may block until the
                conversation session ends.
        """

        self._handle_decision = handle_decision
        self._lock = Lock()
        self._worker: Thread | None = None
        self._closed = False

    def open(self) -> None:
        """Allow the dispatcher to accept new wakeup requests again."""

        with self._lock:
            self._closed = False

    def submit(self, decision: "GestureWakeupDecision") -> bool:
        """Start one visual wakeup handler without blocking the caller.

        Args:
            decision: Accepted wakeup decision to dispatch.

        Returns:
            True when a background worker was started for this decision, False
            when the dispatcher is closed or a previous wakeup is still active.
        """

        with self._lock:
            if self._closed:
                return False
            worker = self._worker
            if worker is not None and worker.is_alive():
                return False
            worker = Thread(
                target=self._worker_main,
                args=(decision,),
                name="twinr-gesture-wakeup",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return True

    def close(self, *, timeout_s: float = 0.25) -> bool:
        """Stop accepting new work and wait briefly for the active worker.

        Args:
            timeout_s: Maximum time to wait for the active wakeup handler to
                finish before returning.

        Returns:
            True when no worker remains active after the join window, False
            when a background wakeup handler is still running.
        """

        with self._lock:
            self._closed = True
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout=max(0.0, float(timeout_s)))
        return not worker.is_alive()

    def _worker_main(self, decision: "GestureWakeupDecision") -> None:
        """Run one wakeup handler and clear the worker slot afterwards."""

        try:
            self._handle_decision(decision)
        finally:
            with self._lock:
                if self._worker is current_thread():
                    self._worker = None
