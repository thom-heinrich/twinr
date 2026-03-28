# CHANGELOG: 2026-03-27
# BUG-1: Fixed a real green-interrupt race where a deferred green press could interrupt the newly started turn instead of the currently active one.
# BUG-2: Contained handler/interrupt callback failures so one bad callback no longer kills the dispatcher or strands coalesced button presses.
# SEC-1: No practically exploitable security issue was found in this isolated dispatcher; the changes below are reliability/operability hardening.
# IMP-1: Added per-button debounce (default 30 ms) for real Raspberry Pi mechanical switches; pass debounce_s=None to restore raw behavior.
# IMP-2: Added asynchronous interrupt dispatch, context propagation, richer latency/error tracing, and graceful non-daemon worker shutdown.

"""Dispatch physical button presses without blocking future GPIO polling."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextvars import copy_context
from dataclasses import dataclass
from threading import Condition, Lock, Thread, current_thread
from typing import Callable


@dataclass(slots=True)
class _PendingButtons:
    """Track the latest coalesced pending button requests."""

    green: bool = False
    yellow: bool = False


@dataclass(slots=True)
class _PendingMeta:
    """Track the first submit time for each coalesced pending press."""

    green_submitted_ns: int | None = None
    yellow_submitted_ns: int | None = None


def _now_ns() -> int:
    return time.monotonic_ns()


def _to_ms(delta_ns: int | None) -> float | None:
    if delta_ns is None:
        return None
    return round(delta_ns / 1_000_000.0, 3)


class ButtonPressDispatcher:
    """Run button handlers off the GPIO poll thread with coalescing and safe interrupts.

    The hardware loop must keep polling GPIO edges while a turn is active.
    This dispatcher moves button handling off the poll thread so a second green
    press can interrupt the active turn instead of sitting queued until the
    whole turn completes.

    Semantics:
    - submit("green" | "yellow") never runs handle_press() inline.
    - while busy, at most one pending green and one pending yellow are retained
      (coalesced latest-intent semantics)
    - pending green always wins over pending yellow
    - close() stops accepting new work and waits briefly for the active worker
    """

    SUPPORTED_BUTTONS = frozenset({"green", "yellow"})

    def __init__(
        self,
        *,
        handle_press: Callable[[str], None],
        interrupt_current: Callable[[str], bool],
        emit: Callable[[str], None] | None = None,
        trace_event: Callable[[str, dict[str, object] | None], None] | None = None,
        debounce_s: float | Mapping[str, float | None] | None = 0.03,
    ) -> None:
        self._handle_press = handle_press
        self._interrupt_current = interrupt_current
        self._emit = emit
        self._trace_event = trace_event

        self._lock = Lock()
        self._cv = Condition(self._lock)
        self._worker: Thread | None = None
        self._pending = _PendingButtons()
        self._pending_meta = _PendingMeta()
        self._closed = False

        self._active_press_seq = 0
        self._active_button: str | None = None

        # Track interrupt requests by active press sequence so a late interrupt
        # cannot accidentally hit the next turn.
        self._interrupt_requested_for_seq = 0
        self._interrupt_inflight_seq = 0

        # BREAKING: default debounce is now 30 ms to absorb mechanical switch
        # bounce on Raspberry Pi-class hardware. Pass debounce_s=None to restore
        # the previous raw behavior.
        self._debounce_s = self._normalize_debounce(debounce_s)
        self._last_accepted_ns: dict[str, int] = {}

    def submit(self, button_name: str) -> None:
        """Accept one button press without blocking future GPIO polling."""

        normalized = self._normalize_button_name(button_name)
        submitted_ns = _now_ns()
        self._trace("button_dispatch_submit_received", button_name=normalized)

        should_schedule_interrupt = False
        interrupt_seq = 0

        with self._lock:
            if self._closed:
                self._trace("button_dispatch_submit_ignored_closed", button_name=normalized)
                return

            debounce_s = self._debounce_s[normalized]
            if debounce_s is not None:
                last_ns = self._last_accepted_ns.get(normalized)
                if last_ns is not None:
                    elapsed_ns = submitted_ns - last_ns
                    if elapsed_ns < int(debounce_s * 1_000_000_000):
                        self._emit_best_effort(f"button_dispatch_debounced={normalized}")
                        self._trace(
                            "button_dispatch_submit_debounced",
                            button_name=normalized,
                            debounce_ms=_to_ms(int(debounce_s * 1_000_000_000)),
                            since_last_accept_ms=_to_ms(elapsed_ns),
                        )
                        return
            self._last_accepted_ns[normalized] = submitted_ns

            worker = self._worker
            if worker is None or not worker.is_alive():
                self._trace(
                    "button_dispatch_submit_starts_worker",
                    button_name=normalized,
                )
                self._start_worker_locked(normalized, submitted_ns=submitted_ns)
                return

            if normalized == "green":
                if not self._pending.green:
                    self._pending.green = True
                    self._pending_meta.green_submitted_ns = submitted_ns
                should_schedule_interrupt = self._maybe_mark_interrupt_request_locked()
                if should_schedule_interrupt:
                    interrupt_seq = self._active_press_seq
            else:
                if not self._pending.yellow:
                    self._pending.yellow = True
                    self._pending_meta.yellow_submitted_ns = submitted_ns

            self._emit_best_effort(f"button_dispatch_deferred={normalized}")
            self._trace(
                "button_dispatch_submit_deferred",
                button_name=normalized,
                pending_green=self._pending.green,
                pending_yellow=self._pending.yellow,
                active_press_seq=self._active_press_seq,
                active_button=self._active_button,
            )

        if should_schedule_interrupt:
            self._start_interrupt_request(normalized, interrupt_seq)

    def close(self, *, timeout_s: float = 1.0) -> None:
        """Stop accepting new work and wait briefly for the active worker."""

        with self._lock:
            self._closed = True
            worker = self._worker
            self._cv.notify_all()

        self._trace(
            "button_dispatch_close_requested",
            worker_alive=bool(worker is not None and worker.is_alive()),
            timeout_s=float(timeout_s),
        )

        if worker is None:
            return

        if worker is current_thread():
            self._emit_best_effort("button_dispatch_close_called_from_worker=true")
            self._trace("button_dispatch_close_called_from_worker", timeout_s=float(timeout_s))
            return

        worker.join(timeout=max(0.05, float(timeout_s)))
        if worker.is_alive():
            self._emit_best_effort("button_dispatch_join_timeout=true")
            self._trace("button_dispatch_close_join_timeout", timeout_s=float(timeout_s))

    def _start_worker_locked(self, button_name: str, *, submitted_ns: int) -> None:
        context = copy_context()

        def runner() -> None:
            context.run(self._worker_main, button_name, submitted_ns)

        # BREAKING: the worker is intentionally non-daemon now. Python's own docs
        # warn that daemon threads are abruptly stopped at interpreter shutdown.
        worker = Thread(
            target=runner,
            name="twinr-button-dispatch",
            daemon=False,
        )
        self._worker = worker
        worker.start()
        self._trace(
            "button_dispatch_worker_started",
            first_button=button_name,
            worker_name=worker.name,
            daemon=worker.daemon,
        )

    def _worker_main(self, first_button: str, first_submitted_ns: int) -> None:
        button_name = first_button
        submitted_ns = first_submitted_ns
        self._trace("button_dispatch_worker_entered", first_button=first_button)

        while True:
            with self._lock:
                self._active_press_seq += 1
                press_seq = self._active_press_seq
                self._active_button = button_name

            started_ns = _now_ns()
            self._trace(
                "button_dispatch_worker_handling_press",
                button_name=button_name,
                press_seq=press_seq,
                queue_delay_ms=_to_ms(started_ns - submitted_ns),
            )

            try:
                self._handle_press(button_name)
            except Exception as exc:
                self._emit_best_effort("button_dispatch_handler_exception=true")
                self._trace(
                    "button_dispatch_handler_exception",
                    button_name=button_name,
                    press_seq=press_seq,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            finally:
                finished_ns = _now_ns()

            with self._lock:
                self._active_button = None

                while self._interrupt_inflight_seq == press_seq:
                    self._trace(
                        "button_dispatch_worker_waiting_for_interrupt",
                        press_seq=press_seq,
                        button_name=button_name,
                    )
                    self._cv.wait()

                if self._closed:
                    self._worker = None
                    self._trace(
                        "button_dispatch_worker_exiting_closed",
                        button_name=button_name,
                        press_seq=press_seq,
                        handler_duration_ms=_to_ms(finished_ns - started_ns),
                    )
                    return

                if self._pending.green:
                    self._pending.green = False
                    button_name = "green"
                    submitted_ns = self._pending_meta.green_submitted_ns or _now_ns()
                    self._pending_meta.green_submitted_ns = None
                    self._trace(
                        "button_dispatch_worker_consumed_pending_green",
                        next_button=button_name,
                        prior_press_seq=press_seq,
                        handler_duration_ms=_to_ms(finished_ns - started_ns),
                    )
                    continue

                if self._pending.yellow:
                    self._pending.yellow = False
                    button_name = "yellow"
                    submitted_ns = self._pending_meta.yellow_submitted_ns or _now_ns()
                    self._pending_meta.yellow_submitted_ns = None
                    self._trace(
                        "button_dispatch_worker_consumed_pending_yellow",
                        next_button=button_name,
                        prior_press_seq=press_seq,
                        handler_duration_ms=_to_ms(finished_ns - started_ns),
                    )
                    continue

                self._worker = None
                self._trace(
                    "button_dispatch_worker_idle_exit",
                    button_name=button_name,
                    press_seq=press_seq,
                    handler_duration_ms=_to_ms(finished_ns - started_ns),
                )
                return

    def _start_interrupt_request(self, button_name: str, press_seq: int) -> None:
        context = copy_context()

        def runner() -> None:
            context.run(self._interrupt_request_main, button_name, press_seq)

        thread = Thread(
            target=runner,
            name=f"twinr-button-interrupt-{press_seq}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self._lock:
                if self._interrupt_inflight_seq == press_seq:
                    self._interrupt_inflight_seq = 0
                if self._interrupt_requested_for_seq == press_seq:
                    self._interrupt_requested_for_seq = 0
                self._cv.notify_all()
            self._emit_best_effort("button_dispatch_interrupt_thread_start_failed=true")
            self._trace(
                "button_dispatch_interrupt_thread_start_failed",
                button_name=button_name,
                press_seq=press_seq,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return

        self._emit_best_effort("button_dispatch_interrupt_scheduled=true")
        self._trace(
            "button_dispatch_interrupt_scheduled",
            button_name=button_name,
            press_seq=press_seq,
            thread_name=thread.name,
        )

    def _interrupt_request_main(self, button_name: str, press_seq: int) -> None:
        started_ns = _now_ns()
        interrupted = False
        error_type: str | None = None
        error_message: str | None = None

        try:
            interrupted = bool(self._interrupt_current(button_name))
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            self._emit_best_effort("button_dispatch_interrupt_exception=true")
        finally:
            finished_ns = _now_ns()
            with self._lock:
                if self._interrupt_inflight_seq == press_seq:
                    self._interrupt_inflight_seq = 0
                    self._cv.notify_all()

        self._emit_best_effort(
            f"button_dispatch_interrupt_requested={str(interrupted).lower()}"
        )
        self._trace(
            "button_dispatch_interrupt_requested",
            button_name=button_name,
            press_seq=press_seq,
            interrupted=interrupted,
            duration_ms=_to_ms(finished_ns - started_ns),
            error_type=error_type,
            error=error_message,
        )

    def _maybe_mark_interrupt_request_locked(self) -> bool:
        current_seq = self._active_press_seq
        if current_seq <= 0:
            return False
        if self._interrupt_requested_for_seq == current_seq:
            return False
        self._interrupt_requested_for_seq = current_seq
        self._interrupt_inflight_seq = current_seq
        return True

    @classmethod
    def _normalize_button_name(cls, button_name: str) -> str:
        normalized = str(button_name or "").strip().lower()
        if normalized not in cls.SUPPORTED_BUTTONS:
            raise ValueError(f"Unsupported button: {button_name}")
        return normalized

    @classmethod
    def _normalize_debounce(
        cls,
        debounce_s: float | Mapping[str, float | None] | None,
    ) -> dict[str, float | None]:
        if debounce_s is None:
            return {button: None for button in cls.SUPPORTED_BUTTONS}

        if isinstance(debounce_s, Mapping):
            normalized: dict[str, float | None] = {}
            for button in cls.SUPPORTED_BUTTONS:
                value = debounce_s.get(button)
                if value is None:
                    normalized[button] = None
                    continue
                numeric = float(value)
                if numeric < 0:
                    raise ValueError(f"debounce_s[{button!r}] must be >= 0")
                normalized[button] = numeric
            return normalized

        numeric = float(debounce_s)
        if numeric < 0:
            raise ValueError("debounce_s must be >= 0")
        return {button: numeric for button in cls.SUPPORTED_BUTTONS}

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
            self._trace_event(msg, details or None)
        except Exception:
            return