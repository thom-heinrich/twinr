"""Dispatch physical button presses without blocking future GPIO polling."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread
from typing import Callable


@dataclass(slots=True)
class _PendingButtons:
    """Track the latest coalesced pending button requests."""

    green: bool = False
    yellow: bool = False


class ButtonPressDispatcher:
    """Run button handlers on a worker thread and coalesce busy presses.

    The hardware loop must keep polling GPIO edges while a turn is active.
    This dispatcher moves button handling off the poll thread so a second green
    press can interrupt the active turn instead of sitting queued until the
    whole turn completes.
    """

    def __init__(
        self,
        *,
        handle_press: Callable[[str], None],
        interrupt_current: Callable[[str], bool],
        emit: Callable[[str], None] | None = None,
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
    ) -> None:
        self._handle_press = handle_press
        self._interrupt_current = interrupt_current
        self._emit = emit
        self._trace_event = trace_event
        self._lock = Lock()
        self._worker: Thread | None = None
        self._pending = _PendingButtons()
        self._closed = False

    def submit(self, button_name: str) -> None:
        """Accept one button press without blocking the caller thread."""

        normalized = str(button_name or "").strip().lower()
        if normalized not in {"green", "yellow"}:
            raise ValueError(f"Unsupported button: {button_name}")
        self._trace("button_dispatch_submit_received", button_name=normalized)

        should_interrupt = False
        with self._lock:
            if self._closed:
                self._trace("button_dispatch_submit_ignored_closed", button_name=normalized)
                return
            worker = self._worker
            if worker is None or not worker.is_alive():
                self._trace("button_dispatch_submit_starts_worker", button_name=normalized)
                self._start_worker_locked(normalized)
                return
            if normalized == "green":
                self._pending.green = True
                should_interrupt = True
            else:
                self._pending.yellow = True
            self._emit_best_effort(f"button_dispatch_deferred={normalized}")
            self._trace(
                "button_dispatch_submit_deferred",
                button_name=normalized,
                pending_green=self._pending.green,
                pending_yellow=self._pending.yellow,
            )
        if should_interrupt:
            interrupted = bool(self._interrupt_current(normalized))
            self._emit_best_effort(
                f"button_dispatch_interrupt_requested={str(interrupted).lower()}"
            )
            self._trace(
                "button_dispatch_interrupt_requested",
                button_name=normalized,
                interrupted=interrupted,
            )

    def close(self, *, timeout_s: float = 1.0) -> None:
        """Stop accepting new work and wait briefly for the active worker."""

        with self._lock:
            self._closed = True
            worker = self._worker
        self._trace(
            "button_dispatch_close_requested",
            worker_alive=bool(worker is not None and worker.is_alive()),
            timeout_s=float(timeout_s),
        )
        if worker is not None:
            worker.join(timeout=max(0.05, float(timeout_s)))
            if worker.is_alive():
                self._emit_best_effort("button_dispatch_join_timeout=true")
                self._trace("button_dispatch_close_join_timeout", timeout_s=float(timeout_s))

    def _start_worker_locked(self, button_name: str) -> None:
        worker = Thread(
            target=self._worker_main,
            args=(button_name,),
            name="twinr-button-dispatch",
            daemon=True,
        )
        self._worker = worker
        worker.start()
        self._trace("button_dispatch_worker_started", first_button=button_name, worker_name=worker.name)

    def _worker_main(self, first_button: str) -> None:
        button_name = first_button
        self._trace("button_dispatch_worker_entered", first_button=first_button)
        while True:
            self._trace("button_dispatch_worker_handling_press", button_name=button_name)
            self._handle_press(button_name)
            with self._lock:
                if self._closed:
                    self._worker = None
                    self._trace("button_dispatch_worker_exiting_closed", button_name=button_name)
                    return
                if self._pending.green:
                    self._pending.green = False
                    button_name = "green"
                    self._trace("button_dispatch_worker_consumed_pending_green", button_name=button_name)
                    continue
                if self._pending.yellow:
                    self._pending.yellow = False
                    button_name = "yellow"
                    self._trace("button_dispatch_worker_consumed_pending_yellow", button_name=button_name)
                    continue
                self._worker = None
                self._trace("button_dispatch_worker_idle_exit", button_name=button_name)
                return

    def _emit_best_effort(self, line: str) -> None:
        if not callable(self._emit):
            return
        try:
            self._emit(line)
        except Exception:
            return

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self._trace_event):
            return
        try:
            self._trace_event(msg, details)
        except Exception:
            return
