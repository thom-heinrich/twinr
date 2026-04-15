# CHANGELOG: 2026-03-28
# BUG-1: Replaced the Event.clear()/wait pattern with Condition-based predicate waiting so request_refresh() cannot be lost between a timeout return and the next loop iteration.
# BUG-2: Added lifecycle locking around start()/stop() so concurrent callers cannot spawn duplicate workers or tear worker state mid-transition.
# SEC-1: Added cooperative refresh deadlines, soft-timeout stale detection, and fail-closed readiness snapshots so a slow or malicious remote endpoint cannot silently leave stale "ready" state in memory.
# IMP-1: Added monotonic-ns scheduling, a snapshot()/is_ready() health API, and stale-after semantics inspired by watchdog/deadline monitoring.
# IMP-2: Added capped exponential backoff with jitter on exception paths, async refresh support, and deduplicated readiness/staleness telemetry.

"""Watch required remote readiness without blocking GPIO polling.

This helper keeps the expensive remote-primary readiness checks off the main
button-poll thread. The Pi must fail closed when remote memory is unavailable,
but button feedback must not wait behind multi-second remote probes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
import time
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread, Timer
from typing import Any, Awaitable, Callable, Coroutine, cast

_LOGGER = logging.getLogger(__name__)

RefreshResult = bool | Awaitable[bool]
RefreshCallable = Callable[..., RefreshResult]


@dataclass(frozen=True, slots=True)
class RequiredRemoteDependencyWatchSnapshot:
    """Cheap, immutable health view for fail-closed callers."""

    running: bool
    inflight: bool
    last_ready: bool | None
    effective_ready: bool
    stale: bool
    soft_timed_out: bool
    consecutive_failures: int
    consecutive_exceptions: int
    last_error_type: str | None
    last_started_monotonic_s: float | None
    last_completed_monotonic_s: float | None
    last_duration_ms: float | None
    last_success_monotonic_s: float | None
    next_refresh_in_s: float | None


@dataclass(slots=True)
class RequiredRemoteDependencyWatch:
    """Run required-remote refreshes on a dedicated worker thread.

    Supported refresh callback shapes:

    * refresh(force) -> bool
    * refresh(force, deadline_monotonic_s) -> bool
    * async def refresh(...)

    When the callback accepts ``deadline_monotonic_s``, it should pass the
    remaining budget into its own network or IPC timeouts. This helper can
    surface stale or over-budget checks immediately, but Python cannot
    preemptively kill a non-cooperative blocking thread.
    """

    interval_s: float
    refresh: RefreshCallable
    emit: Callable[[str], None] | None = None
    trace_event: Callable[[str, dict[str, object] | None], None] | None = None
    refresh_timeout_s: float | None = None
    stale_after_s: float | None = None
    max_exception_backoff_s: float | None = None
    exception_backoff_multiplier: float = 2.0
    exception_backoff_jitter_ratio: float = 0.15
    backoff_on_false_ready: bool = False
    supports_deadline_arg: bool | None = None

    _condition: Condition = field(init=False, repr=False, default_factory=lambda: Condition(Lock()))
    _lifecycle_lock: Lock = field(init=False, repr=False, default_factory=Lock)
    _thread: Thread | None = field(init=False, repr=False, default=None)
    _stop_requested: bool = field(init=False, repr=False, default=False)
    _pending_refresh: bool = field(init=False, repr=False, default=False)
    _inflight: bool = field(init=False, repr=False, default=False)
    _last_ready: bool | None = field(init=False, repr=False, default=None)
    _last_error_type: str | None = field(init=False, repr=False, default=None)
    _last_started_ns: int | None = field(init=False, repr=False, default=None)
    _last_completed_ns: int | None = field(init=False, repr=False, default=None)
    _last_success_ns: int | None = field(init=False, repr=False, default=None)
    _last_duration_ns: int | None = field(init=False, repr=False, default=None)
    _next_due_ns: int | None = field(init=False, repr=False, default=None)
    _consecutive_failures: int = field(init=False, repr=False, default=0)
    _consecutive_exceptions: int = field(init=False, repr=False, default=0)
    _soft_timeout_active: bool = field(init=False, repr=False, default=False)
    _soft_timeout_started_ns: int | None = field(init=False, repr=False, default=None)
    _last_emitted_ready: bool | None = field(init=False, repr=False, default=None)
    _last_emitted_stale: bool | None = field(init=False, repr=False, default=None)
    _rng: random.Random = field(init=False, repr=False, default_factory=random.Random)
    _refresh_accepts_deadline: bool = field(init=False, repr=False, default=False)
    _interval_ns: int = field(init=False, repr=False, default=0)
    _stale_after_ns: int = field(init=False, repr=False, default=0)
    _refresh_timeout_ns: int | None = field(init=False, repr=False, default=None)
    _max_exception_backoff_ns: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._interval_ns = self._validated_ns(self.interval_s, field_name="interval_s", minimum_s=0.1)

        if self.refresh_timeout_s is None:
            self._refresh_timeout_ns = None
        else:
            self._refresh_timeout_ns = self._validated_ns(
                self.refresh_timeout_s,
                field_name="refresh_timeout_s",
                minimum_s=0.001,
            )

        stale_after_s = self.stale_after_s
        if stale_after_s is None:
            stale_after_s = max(float(self.interval_s) * 2.5, float(self.interval_s) + 1.0)
        self._stale_after_ns = self._validated_ns(
            stale_after_s,
            field_name="stale_after_s",
            minimum_s=0.1,
        )

        max_exception_backoff_s = self.max_exception_backoff_s
        if max_exception_backoff_s is None:
            max_exception_backoff_s = max(float(self.interval_s), min(max(float(self.interval_s) * 8.0, 1.0), 30.0))
        self._max_exception_backoff_ns = self._validated_ns(
            max_exception_backoff_s,
            field_name="max_exception_backoff_s",
            minimum_s=max(0.1, float(self.interval_s)),
        )

        if not math.isfinite(self.exception_backoff_multiplier) or self.exception_backoff_multiplier < 1.0:
            raise ValueError("exception_backoff_multiplier must be finite and >= 1.0")
        if not math.isfinite(self.exception_backoff_jitter_ratio) or not (0.0 <= self.exception_backoff_jitter_ratio <= 1.0):
            raise ValueError("exception_backoff_jitter_ratio must be finite and between 0.0 and 1.0")

        if self.supports_deadline_arg is None:
            self._refresh_accepts_deadline = self._detect_refresh_deadline_support(self.refresh)
        else:
            self._refresh_accepts_deadline = bool(self.supports_deadline_arg)

    def start(self) -> None:
        """Start the background watch once."""
        with self._lifecycle_lock:
            worker = self._thread
            if worker is not None and worker.is_alive():
                return

            with self._condition:
                self._stop_requested = False
                self._pending_refresh = False
                self._next_due_ns = None
                self._last_emitted_ready = None
                self._last_emitted_stale = None

            worker = Thread(
                target=self._worker_main,
                name="twinr-required-remote-watch",
                daemon=True,
            )
            self._thread = worker
            worker.start()

        self._trace(
            "required_remote_watch_thread_started",
            interval_s=round(self._interval_ns / 1_000_000_000.0, 6),
            stale_after_s=round(self._stale_after_ns / 1_000_000_000.0, 6),
            refresh_timeout_s=None if self._refresh_timeout_ns is None else round(self._refresh_timeout_ns / 1_000_000_000.0, 6),
        )

    def request_refresh(self) -> None:
        """Wake the worker for an immediate check."""
        with self._condition:
            self._pending_refresh = True
            self._condition.notify_all()

        self._trace("required_remote_watch_refresh_requested", wake_set=True)

    def stop(self, *, timeout_s: float = 1.0) -> None:
        """Stop the worker and join briefly."""
        timeout_s = max(0.05, float(timeout_s))

        with self._lifecycle_lock:
            worker = self._thread
            with self._condition:
                self._stop_requested = True
                self._pending_refresh = True
                self._next_due_ns = None
                self._condition.notify_all()

        if worker is None:
            self._trace("required_remote_watch_stop_without_worker", timeout_s=float(timeout_s))
            return

        self._trace(
            "required_remote_watch_stop_requested",
            worker_alive=worker.is_alive(),
            timeout_s=float(timeout_s),
        )
        worker.join(timeout=timeout_s)

        if worker.is_alive():
            if callable(self.emit):
                try:
                    self.emit("required_remote_watch_join_timeout=true")
                except Exception:
                    _LOGGER.warning("Required-remote watch failed to emit join-timeout telemetry.", exc_info=True)
            self._trace("required_remote_watch_join_timeout", timeout_s=float(timeout_s))
            return

        with self._lifecycle_lock:
            if self._thread is worker:
                self._thread = None
            with self._condition:
                self._pending_refresh = False

        self._trace("required_remote_watch_stopped", timeout_s=float(timeout_s))

    def snapshot(self) -> RequiredRemoteDependencyWatchSnapshot:
        """Return a fail-closed health snapshot for callers and telemetry."""
        now_ns = time.monotonic_ns()

        with self._condition:
            last_started_ns = self._last_started_ns
            last_completed_ns = self._last_completed_ns
            last_success_ns = self._last_success_ns
            last_duration_ns = self._last_duration_ns
            last_ready = self._last_ready
            inflight = self._inflight
            soft_timed_out = self._soft_timeout_active
            consecutive_failures = self._consecutive_failures
            consecutive_exceptions = self._consecutive_exceptions
            last_error_type = self._last_error_type
            stop_requested = self._stop_requested
            worker = self._thread
            next_due_ns = self._next_due_ns
            pending_refresh = self._pending_refresh

        running = worker is not None and worker.is_alive() and not stop_requested
        stale = self._compute_stale(now_ns, last_started_ns, last_completed_ns, soft_timed_out)
        effective_ready = bool(last_ready) and not stale and not soft_timed_out

        if not running:
            next_refresh_in_s = None
        elif inflight or pending_refresh:
            next_refresh_in_s = 0.0
        elif next_due_ns is None:
            next_refresh_in_s = None
        else:
            next_refresh_in_s = max(0.0, (next_due_ns - now_ns) / 1_000_000_000.0)

        return RequiredRemoteDependencyWatchSnapshot(
            running=running,
            inflight=inflight,
            last_ready=last_ready,
            effective_ready=effective_ready,
            stale=stale,
            soft_timed_out=soft_timed_out,
            consecutive_failures=consecutive_failures,
            consecutive_exceptions=consecutive_exceptions,
            last_error_type=last_error_type,
            last_started_monotonic_s=None if last_started_ns is None else last_started_ns / 1_000_000_000.0,
            last_completed_monotonic_s=None if last_completed_ns is None else last_completed_ns / 1_000_000_000.0,
            last_duration_ms=None if last_duration_ns is None else round(last_duration_ns / 1_000_000.0, 3),
            last_success_monotonic_s=None if last_success_ns is None else last_success_ns / 1_000_000_000.0,
            next_refresh_in_s=next_refresh_in_s,
        )

    def is_ready(self) -> bool:
        """Return effective readiness, failing closed on stale state."""
        return self.snapshot().effective_ready

    def __enter__(self) -> "RequiredRemoteDependencyWatch":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _worker_main(self) -> None:
        force = True
        self._trace(
            "required_remote_watch_worker_entered",
            interval_s=round(self._interval_ns / 1_000_000_000.0, 6),
        )

        while True:
            with self._condition:
                if self._stop_requested:
                    self._trace("required_remote_watch_worker_exit_stop", force=force)
                    return

            next_wait_ns = self._run_refresh_cycle(force=force)
            force = self._wait_for_next_cycle(next_wait_ns)

    def _run_refresh_cycle(self, *, force: bool) -> int:
        started_ns = time.monotonic_ns()
        timeout_timer = self._arm_soft_timeout_timer(started_ns=started_ns, force=force)

        with self._condition:
            self._inflight = True
            self._last_started_ns = started_ns
            self._next_due_ns = None
            self._soft_timeout_active = False
            self._soft_timeout_started_ns = None

        self._trace(
            "required_remote_watch_refresh_started",
            force=force,
            deadline_monotonic_s=None if self._refresh_timeout_ns is None else round((started_ns + self._refresh_timeout_ns) / 1_000_000_000.0, 6),
        )

        ready = False
        failed_by_exception = False

        try:
            ready = bool(self._invoke_refresh(force=force, started_ns=started_ns))
        except Exception as exc:
            failed_by_exception = True
            completed_ns = time.monotonic_ns()
            duration_ns = completed_ns - started_ns

            with self._condition:
                soft_timed_out = self._soft_timeout_active and self._soft_timeout_started_ns == started_ns
                self._inflight = False
                self._last_ready = False
                self._last_error_type = type(exc).__name__
                self._last_duration_ns = duration_ns
                self._last_completed_ns = completed_ns
                self._consecutive_failures += 1
                self._consecutive_exceptions += 1
                self._soft_timeout_active = False
                self._soft_timeout_started_ns = None

            self._emit_state_transitions()

            if callable(self.emit):
                try:
                    self.emit(f"required_remote_watch_error={type(exc).__name__}")
                except Exception:
                    _LOGGER.warning("Required-remote watch failed to emit worker error telemetry.", exc_info=True)

            self._trace(
                "required_remote_watch_refresh_failed",
                force=force,
                error_type=type(exc).__name__,
                soft_timeout=soft_timed_out,
                duration_ms=round(duration_ns / 1_000_000.0, 3),
            )
        else:
            completed_ns = time.monotonic_ns()
            duration_ns = completed_ns - started_ns

            with self._condition:
                soft_timed_out = self._soft_timeout_active and self._soft_timeout_started_ns == started_ns
                self._inflight = False
                self._last_ready = ready
                self._last_duration_ns = duration_ns
                self._last_completed_ns = completed_ns

                if ready:
                    self._last_success_ns = completed_ns
                    self._last_error_type = None
                    self._consecutive_failures = 0
                    self._consecutive_exceptions = 0
                else:
                    self._last_error_type = None
                    self._consecutive_failures += 1
                    self._consecutive_exceptions = 0

                self._soft_timeout_active = False
                self._soft_timeout_started_ns = None

            self._emit_state_transitions()

            self._trace(
                "required_remote_watch_refresh_completed",
                force=force,
                ready=ready,
                soft_timeout=soft_timed_out,
                duration_ms=round(duration_ns / 1_000_000.0, 3),
            )
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()

        next_wait_ns = self._compute_next_wait_ns(ready=ready, failed_by_exception=failed_by_exception)

        with self._condition:
            self._next_due_ns = time.monotonic_ns() + next_wait_ns

        self._trace(
            "required_remote_watch_cycle_scheduled",
            force=force,
            ready=ready,
            failed_by_exception=failed_by_exception,
            next_wait_s=round(next_wait_ns / 1_000_000_000.0, 6),
        )
        return next_wait_ns

    def _wait_for_next_cycle(self, next_wait_ns: int) -> bool:
        deadline_ns = time.monotonic_ns() + max(0, next_wait_ns)

        while True:
            with self._condition:
                if self._stop_requested:
                    self._trace("required_remote_watch_worker_exit_stop", force=False)
                    return False

                if self._pending_refresh:
                    self._pending_refresh = False
                    self._trace("required_remote_watch_wait_completed", force=True, stop=False)
                    return True

                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    self._trace("required_remote_watch_wait_completed", force=False, stop=False)
                    return False

                self._condition.wait(timeout=remaining_ns / 1_000_000_000.0)

    def _invoke_refresh(self, *, force: bool, started_ns: int) -> bool:
        deadline_s = None
        if self._refresh_timeout_ns is not None:
            deadline_s = (started_ns + self._refresh_timeout_ns) / 1_000_000_000.0

        if self._refresh_accepts_deadline:
            result = self.refresh(force, deadline_s)
        else:
            result = self.refresh(force)

        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                return bool(asyncio.run(cast(Coroutine[Any, Any, bool], result)))
            return bool(asyncio.run(self._await_refresh_result(cast(Awaitable[bool], result))))
        return bool(result)

    def _compute_next_wait_ns(self, *, ready: bool, failed_by_exception: bool) -> int:
        with self._condition:
            failures = self._consecutive_failures

        delay_ns = self._interval_ns
        if failed_by_exception or (self.backoff_on_false_ready and not ready):
            delay_ns = self._compute_capped_exception_backoff_ns(failures=failures)

            jitter_ratio = self.exception_backoff_jitter_ratio
            if jitter_ratio > 0.0 and delay_ns > 0:
                jitter_span = int(delay_ns * jitter_ratio)
                if jitter_span > 0:
                    delay_ns = max(self._interval_ns, delay_ns + self._rng.randint(-jitter_span, jitter_span))

        return max(0, delay_ns)

    def _compute_capped_exception_backoff_ns(self, *, failures: int) -> int:
        exponent = max(0, failures - 1)
        if exponent <= 0:
            return min(self._interval_ns, self._max_exception_backoff_ns)

        if self._interval_ns >= self._max_exception_backoff_ns:
            return self._max_exception_backoff_ns

        multiplier = self.exception_backoff_multiplier
        if multiplier <= 1.0:
            return self._interval_ns

        max_scale = self._max_exception_backoff_ns / self._interval_ns
        if max_scale <= 1.0:
            return self._max_exception_backoff_ns

        try:
            max_safe_exponent = math.ceil(math.log(max_scale) / math.log(multiplier))
        except (OverflowError, ValueError, ZeroDivisionError):
            return self._max_exception_backoff_ns

        if exponent >= max_safe_exponent:
            return self._max_exception_backoff_ns

        try:
            scaled_delay = self._interval_ns * (multiplier**exponent)
        except OverflowError:
            return self._max_exception_backoff_ns

        if not math.isfinite(scaled_delay):
            return self._max_exception_backoff_ns

        return min(int(scaled_delay), self._max_exception_backoff_ns)

    def _arm_soft_timeout_timer(self, *, started_ns: int, force: bool) -> Timer | None:
        if self._refresh_timeout_ns is None:
            return None

        timeout_s = self._refresh_timeout_ns / 1_000_000_000.0
        timer = Timer(
            timeout_s,
            self._handle_soft_timeout,
            kwargs={"started_ns": started_ns, "force": force, "timeout_s": timeout_s},
        )
        timer.daemon = True
        timer.start()
        return timer

    def _handle_soft_timeout(self, *, started_ns: int, force: bool, timeout_s: float) -> None:
        with self._condition:
            if not self._inflight or self._last_started_ns != started_ns:
                return
            self._soft_timeout_active = True
            self._soft_timeout_started_ns = started_ns

        if callable(self.emit):
            try:
                self.emit("required_remote_watch_refresh_soft_timeout=true")
            except Exception:
                _LOGGER.warning("Required-remote watch failed to emit soft-timeout telemetry.", exc_info=True)

        self._trace(
            "required_remote_watch_refresh_soft_timeout",
            force=force,
            timeout_s=round(timeout_s, 6),
        )
        self._emit_state_transitions()

    def _emit_state_transitions(self) -> None:
        snapshot = self.snapshot()
        ready = snapshot.effective_ready
        stale = snapshot.stale

        # BREAKING: emit(...) now receives readiness/staleness transition records
        # in addition to the legacy error/join-timeout telemetry records.
        if callable(self.emit):
            try:
                should_emit_ready = False
                should_emit_stale = False

                with self._condition:
                    if self._last_emitted_ready is None or self._last_emitted_ready != ready:
                        self._last_emitted_ready = ready
                        should_emit_ready = True
                    if self._last_emitted_stale is None or self._last_emitted_stale != stale:
                        self._last_emitted_stale = stale
                        should_emit_stale = True

                if should_emit_ready:
                    self.emit(f"required_remote_watch_ready={'true' if ready else 'false'}")
                if should_emit_stale:
                    self.emit(f"required_remote_watch_stale={'true' if stale else 'false'}")
            except Exception:
                _LOGGER.warning("Required-remote watch failed to emit state telemetry.", exc_info=True)

        self._trace(
            "required_remote_watch_state",
            ready=ready,
            stale=stale,
            inflight=snapshot.inflight,
            soft_timed_out=snapshot.soft_timed_out,
            consecutive_failures=snapshot.consecutive_failures,
            consecutive_exceptions=snapshot.consecutive_exceptions,
            last_error_type=snapshot.last_error_type,
        )

    def _compute_stale(
        self,
        now_ns: int,
        last_started_ns: int | None,
        last_completed_ns: int | None,
        soft_timed_out: bool,
    ) -> bool:
        if soft_timed_out:
            return True

        anchor_ns = last_completed_ns
        if anchor_ns is None:
            anchor_ns = last_started_ns
        if anchor_ns is None:
            return False

        return now_ns - anchor_ns > self._stale_after_ns

    @staticmethod
    def _validated_ns(value_s: float, *, field_name: str, minimum_s: float) -> int:
        value = float(value_s)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        if value < minimum_s:
            value = minimum_s
        return int(value * 1_000_000_000)

    @staticmethod
    def _detect_refresh_deadline_support(callback: RefreshCallable) -> bool:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return False

        positional_capacity = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional_capacity += 1
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                return True

        return positional_capacity >= 2

    @staticmethod
    async def _await_refresh_result(result: Awaitable[bool]) -> bool:
        return bool(await result)

    def _trace(self, msg: str, **details: object) -> None:
        if not callable(self.trace_event):
            return

        try:
            self.trace_event(msg, details)
        except Exception:
            _LOGGER.warning("Required-remote watch trace sink failed for %s.", msg, exc_info=True)
