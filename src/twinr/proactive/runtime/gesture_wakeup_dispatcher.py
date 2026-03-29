# CHANGELOG: 2026-03-29
# BUG-1: close() could self-join on the worker thread and raise RuntimeError.
# BUG-2: handler crashes escaped on a background thread with no dispatcher-owned
# health state, causing silent/partial failures in service mode.
# SEC-1: one stuck or maliciously prolonged wake session could monopolize the
# single-flight lane indefinitely; add cooperative cancellation + runtime cap.
# IMP-1: isolate ContextVars for the worker thread to avoid cross-session bleed.
# IMP-2: add observability (snapshot/is_active/counters/logging) and graceful
# shutdown while preserving open()/submit()/close() compatibility.

"""Dispatch visual gesture wakeups without blocking proactive refresh ticks.

The proactive monitor worker must keep driving HDMI attention-follow while a
gesture wakeup opens a hands-free listening session. This module provides a
single-flight dispatcher that moves the potentially long-running wakeup handler
off the monitor thread while deliberately refusing concurrent visual wake
requests.

Upgrade notes:
- open()/submit()/close() remain intact.
- Legacy handlers that only accept ``decision`` still work unchanged.
- If the handler accepts ``cancel_event`` and/or ``deadline_monotonic`` as
  keyword arguments, they are passed automatically for cooperative shutdown.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
from contextvars import Context
from dataclasses import dataclass
from threading import Event, Lock, Thread, Timer, current_thread
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .gesture_wakeup_lane import GestureWakeupDecision

_LOGGER = logging.getLogger(__name__)


def _accepts_kwarg(func: Callable[..., Any] | None, name: str) -> bool:
    if func is None:
        return False
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    ):
        return True
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _normalize_max_runtime_s(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("max_runtime_s must be a positive finite float or None")
    return value


def _normalize_join_timeout_s(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or value < 0.0:
        return 0.0
    return value


def _as_result_flag(value: object) -> bool | None:
    return None if value is None else bool(value)


@dataclass(frozen=True, slots=True)
class GestureWakeupDispatchSnapshot:
    closed: bool
    active: bool
    cancel_requested: bool
    generation: int
    accepted_count: int
    rejected_busy_count: int
    rejected_closed_count: int
    timeout_count: int
    exception_count: int
    active_runtime_s: float | None
    last_duration_s: float | None
    last_result: bool | None
    last_error: str | None
    last_cancel_requested: bool
    last_timed_out: bool


class GestureWakeupDispatcher:
    """Run at most one visual wakeup handler at a time on a worker thread."""

    def __init__(
        self,
        *,
        handle_decision: Callable[..., bool | None],
        request_stop: Callable[..., None] | None = None,
        logger: logging.Logger | None = None,
        worker_name: str = "twinr-gesture-wakeup",
        daemon: bool = False,
        max_runtime_s: float | None = 120.0,
    ) -> None:
        self._handle_decision = handle_decision
        self._request_stop = request_stop
        self._logger = logger or _LOGGER
        self._worker_name = worker_name

        self._lock = Lock()
        self._worker: Thread | None = None
        self._deadline_timer: Timer | None = None
        self._cancel_event: Event | None = None
        self._closed = False

        self._generation = 0
        self._active_generation = 0
        self._cancel_requested = False
        self._deadline_requested_generation: int | None = None

        self._accepted_count = 0
        self._rejected_busy_count = 0
        self._rejected_closed_count = 0
        self._timeout_count = 0
        self._exception_count = 0

        self._last_started_monotonic: float | None = None
        self._last_finished_monotonic: float | None = None
        self._last_duration_s: float | None = None
        self._last_result: bool | None = None
        self._last_error: str | None = None
        self._last_cancel_requested = False
        self._last_timed_out = False

        self._handle_accepts_cancel_event = _accepts_kwarg(
            handle_decision,
            "cancel_event",
        )
        self._handle_accepts_deadline = _accepts_kwarg(
            handle_decision,
            "deadline_monotonic",
        )
        self._request_stop_accepts_reason = _accepts_kwarg(
            request_stop,
            "reason",
        )

        # BREAKING: default worker threads are now non-daemonic so shutdown can
        # be graceful instead of abruptly tearing down active audio/session work.
        self._daemon = bool(daemon)

        # BREAKING: default max_runtime_s is now 120s to cap indefinite
        # single-flight lockout from a hung or hostile wakeup session.
        self._max_runtime_s = _normalize_max_runtime_s(max_runtime_s)

    def open(self) -> None:
        """Allow the dispatcher to accept new wakeup requests again."""
        with self._lock:
            self._closed = False

    def submit(self, decision: "GestureWakeupDecision") -> bool:
        """Start one visual wakeup handler without blocking the caller."""
        started_monotonic = time.monotonic()

        with self._lock:
            if self._closed:
                self._rejected_closed_count += 1
                return False

            worker = self._worker
            if worker is not None and worker.is_alive():
                self._rejected_busy_count += 1
                return False

            generation = self._generation + 1
            self._generation = generation
            self._active_generation = generation
            self._cancel_requested = False
            self._deadline_requested_generation = None
            self._cancel_event = Event()
            self._last_started_monotonic = started_monotonic

            deadline_monotonic: float | None = None
            timer: Timer | None = None
            if self._max_runtime_s is not None:
                deadline_monotonic = started_monotonic + self._max_runtime_s
                timer = Timer(self._max_runtime_s, self._deadline_main, args=(generation,))
                timer.daemon = True
                timer.name = f"{self._worker_name}-deadline-{generation}"

            self._deadline_timer = timer
            worker = self._make_worker(
                decision=decision,
                generation=generation,
                cancel_event=self._cancel_event,
                deadline_monotonic=deadline_monotonic,
                started_monotonic=started_monotonic,
            )
            self._worker = worker
            self._accepted_count += 1

        try:
            worker.start()
        except BaseException:
            with self._lock:
                if self._worker is worker:
                    self._worker = None
                if self._active_generation == generation:
                    self._active_generation = 0
                    self._cancel_event = None
                    self._deadline_timer = None
                    self._cancel_requested = False
                    self._deadline_requested_generation = None
            raise

        if timer is not None:
            try:
                timer.start()
            except RuntimeError:
                self._logger.exception(
                    "Failed to start gesture wakeup deadline timer.",
                )

        return True

    def close(self, *, timeout_s: float = 0.25) -> bool:
        """Stop accepting new work and wait briefly for the active worker."""
        worker = self._request_shutdown(reason="close")
        if worker is None:
            return True
        if worker is current_thread():
            return False

        try:
            worker.join(timeout=_normalize_join_timeout_s(timeout_s))
        except RuntimeError:
            self._logger.exception("Failed to join gesture wakeup worker.")
            return False
        return not worker.is_alive()

    def is_active(self) -> bool:
        with self._lock:
            worker = self._worker
            return worker is not None and worker.is_alive()

    def snapshot(self) -> GestureWakeupDispatchSnapshot:
        with self._lock:
            worker = self._worker
            active = worker is not None and worker.is_alive()
            active_runtime_s: float | None = None
            if active and self._last_started_monotonic is not None:
                active_runtime_s = max(
                    0.0,
                    time.monotonic() - self._last_started_monotonic,
                )

            return GestureWakeupDispatchSnapshot(
                closed=self._closed,
                active=active,
                cancel_requested=self._cancel_requested,
                generation=self._generation,
                accepted_count=self._accepted_count,
                rejected_busy_count=self._rejected_busy_count,
                rejected_closed_count=self._rejected_closed_count,
                timeout_count=self._timeout_count,
                exception_count=self._exception_count,
                active_runtime_s=active_runtime_s,
                last_duration_s=self._last_duration_s,
                last_result=self._last_result,
                last_error=self._last_error,
                last_cancel_requested=self._last_cancel_requested,
                last_timed_out=self._last_timed_out,
            )

    def _make_worker(
        self,
        *,
        decision: "GestureWakeupDecision",
        generation: int,
        cancel_event: Event,
        deadline_monotonic: float | None,
        started_monotonic: float,
    ) -> Thread:
        return Thread(
            target=Context().run,
            args=(
                self._worker_main,
                decision,
                generation,
                cancel_event,
                deadline_monotonic,
                started_monotonic,
            ),
            name=f"{self._worker_name}-{generation}",
            daemon=self._daemon,
        )

    def _request_shutdown(self, *, reason: str) -> Thread | None:
        callback = None
        cancel_event: Event | None = None

        with self._lock:
            self._closed = True
            worker = self._worker
            if worker is None:
                return None

            if not self._cancel_requested:
                self._cancel_requested = True
                cancel_event = self._cancel_event
                callback = self._request_stop

        if cancel_event is not None:
            cancel_event.set()
        if callback is not None:
            self._call_request_stop(callback, reason=reason)
        return worker

    def _deadline_main(self, generation: int) -> None:
        callback = None
        cancel_event: Event | None = None

        with self._lock:
            if self._active_generation != generation:
                return

            self._timeout_count += 1
            self._deadline_requested_generation = generation

            if not self._cancel_requested:
                self._cancel_requested = True
                cancel_event = self._cancel_event
                callback = self._request_stop

        if cancel_event is not None:
            cancel_event.set()
        if callback is not None:
            self._call_request_stop(callback, reason="deadline_exceeded")

        self._logger.warning(
            "Gesture wakeup worker exceeded max_runtime_s and was asked to stop "
            "cooperatively.",
        )

    def _call_request_stop(
        self,
        callback: Callable[..., None],
        *,
        reason: str,
    ) -> None:
        try:
            if self._request_stop_accepts_reason:
                callback(reason=reason)
            else:
                callback()
        except Exception:
            self._logger.exception("Gesture wakeup stop callback failed.")

    def _invoke_handler(
        self,
        decision: "GestureWakeupDecision",
        cancel_event: Event,
        deadline_monotonic: float | None,
    ) -> bool | None:
        kwargs: dict[str, object] = {}
        if self._handle_accepts_cancel_event:
            kwargs["cancel_event"] = cancel_event
        if self._handle_accepts_deadline and deadline_monotonic is not None:
            kwargs["deadline_monotonic"] = deadline_monotonic
        return self._handle_decision(decision, **kwargs)

    def _worker_main(
        self,
        decision: "GestureWakeupDecision",
        generation: int,
        cancel_event: Event,
        deadline_monotonic: float | None,
        started_monotonic: float,
    ) -> None:
        result_flag: bool | None = None
        last_error: str | None = None

        try:
            result_flag = _as_result_flag(
                self._invoke_handler(
                    decision=decision,
                    cancel_event=cancel_event,
                    deadline_monotonic=deadline_monotonic,
                )
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("Gesture wakeup handler crashed.")
        finally:
            finished_monotonic = time.monotonic()
            duration_s = max(0.0, finished_monotonic - started_monotonic)
            timer: Timer | None = None

            with self._lock:
                timed_out = self._deadline_requested_generation == generation
                cancel_requested = self._cancel_requested

                if self._worker is current_thread():
                    self._worker = None

                if self._active_generation == generation:
                    timer = self._deadline_timer
                    self._active_generation = 0
                    self._cancel_event = None
                    self._deadline_timer = None
                    self._cancel_requested = False
                    self._deadline_requested_generation = None

                self._last_finished_monotonic = finished_monotonic
                self._last_duration_s = duration_s
                self._last_result = result_flag
                self._last_error = last_error
                self._last_cancel_requested = cancel_requested
                self._last_timed_out = timed_out

                if last_error is not None:
                    self._exception_count += 1

            if timer is not None:
                timer.cancel()