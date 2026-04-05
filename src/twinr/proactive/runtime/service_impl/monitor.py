# CHANGELOG: 2026-03-29
# BUG-1: Startup is now transactional; a failure in open_background_lanes() no longer leaks opened hardware resources.
# BUG-2: Shutdown now waits for the worker to drain before force-closing hardware resources, removing a use-after-close race.
# BUG-3: The worker loop now guards display-cycle open/close and refresh-interval resolution so transient faults no longer kill the thread.
# BUG-4: Worker self-termination now also closes coordinator background lanes; the legacy code could leak them on unexpected worker exit.
# BUG-5: The scheduler now sleeps until the earliest deadline instead of waking every 20-50 ms, removing a permanent idle CPU tax on the Pi 4.
# SEC-1: Repeated worker faults are now rate-limited and backoff-gated to prevent practical log-flood / SD-card wear-out denial-of-service.
# SEC-2: Emitted error text is normalized to one bounded line before telemetry emission to reduce log-injection and oversized-line risk.
# IMP-1: The service now exposes an optional heartbeat callback for process supervisors such as systemd watchdog integrations.
# IMP-2: The lifecycle wrapper now rejects reopen attempts while a previous worker is still draining after a stop timeout.
# BREAKING: open() now raises RuntimeError when called while a previous worker is still alive but already stopping; the old code silently returned a wedged service object.

"""Monitor lifecycle wrapper for the proactive runtime coordinator.

Purpose: own resource open/close sequencing and the bounded background worker
thread around an already-configured ``ProactiveCoordinator``.

Invariants: lifecycle idempotency, shutdown behavior, and worker error events
must remain identical to the legacy service implementation where safe; the
worker is now additionally hardened against resource leaks, log-flooding, and
deadline jitter under real Raspberry Pi deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable, cast
import math
import time

from twinr.ops.streaming_memory_probe import StreamingMemoryProbe
from twinr.proactive.runtime.service_impl.compat import (
    _DEFAULT_CLOSE_JOIN_TIMEOUT_S,
    _append_ops_event,
    _exception_text,
    _safe_emit,
)

from .coordinator import ProactiveCoordinator
from ..display_attention import resolve_display_attention_refresh_interval
from ..display_gesture_emoji import resolve_display_gesture_refresh_interval
from ..pir_open_gate import open_pir_monitor_with_busy_retry


_NS_PER_S = 1_000_000_000
_MIN_POST_WORK_WAIT_S = 0.01
_MAX_IDLE_WAIT_S = 0.5
_MIN_REFRESH_INTERVAL_S = 0.05
_ERROR_RATE_LIMIT_S = 30.0
_FAILURE_BACKOFF_BASE_S = 0.25
_FAILURE_BACKOFF_MAX_S = 5.0
_ERROR_TEXT_MAX_LEN = 512


@dataclass(slots=True)
class _LoopErrorWindow:
    """Track repeated worker errors for one subsystem."""

    emitted_at_ns: int = 0
    suppressed_count: int = 0


class _LoopErrorGate:
    """Rate-limit repeated worker faults per subsystem."""

    def __init__(self, *, interval_s: float) -> None:
        self._interval_ns = int(interval_s * _NS_PER_S)
        self._state: dict[str, _LoopErrorWindow] = {}

    def allow(self, subsystem: str, *, now_ns: int) -> tuple[bool, int]:
        state = self._state.get(subsystem)
        if state is None or now_ns - state.emitted_at_ns >= self._interval_ns:
            suppressed_count = 0 if state is None else state.suppressed_count
            self._state[subsystem] = _LoopErrorWindow(
                emitted_at_ns=now_ns,
                suppressed_count=0,
            )
            return True, suppressed_count
        state.suppressed_count += 1
        return False, 0

    def reset(self, subsystem: str) -> None:
        self._state.pop(subsystem, None)


class ProactiveMonitorService:
    """Run the proactive coordinator in a bounded background worker."""

    def __init__(
        self,
        coordinator: ProactiveCoordinator,
        *,
        poll_interval_s: float,
        emit: Callable[[str], None] | None = None,
        heartbeat: Callable[[], None] | None = None,
        heartbeat_interval_s: float = 10.0,
    ) -> None:
        """Initialize one monitor service around a configured coordinator."""

        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, self._coerce_interval_s(poll_interval_s, fallback_s=0.2))
        self.emit = emit or (lambda _line: None)
        self.heartbeat = heartbeat
        self.heartbeat_interval_s = max(
            1.0,
            self._coerce_interval_s(heartbeat_interval_s, fallback_s=10.0),
        )
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lifecycle_lock = Lock()
        self._resources_open = False
        self._background_lanes_open = False
        self._close_join_timeout_s = _DEFAULT_CLOSE_JOIN_TIMEOUT_S
        self._loop_error_gate = _LoopErrorGate(interval_s=_ERROR_RATE_LIMIT_S)
        self._loop_failure_counts: dict[str, int] = {}
        runtime = getattr(self.coordinator, "runtime", None)
        config = getattr(runtime, "config", None)
        self._tick_memory_probe = StreamingMemoryProbe.from_config(
            config,
            label="proactive_monitor.tick",
            owner_label="proactive_monitor.tick",
            owner_detail="Proactive monitor coordinator tick is running.",
        )
        self._display_attention_memory_probe = StreamingMemoryProbe.from_config(
            config,
            label="proactive_monitor.display_attention",
            owner_label="proactive_monitor.display_attention",
            owner_detail="Proactive monitor refreshed display attention state.",
        )
        self._display_gesture_memory_probe = StreamingMemoryProbe.from_config(
            config,
            label="proactive_monitor.display_gesture",
            owner_label="proactive_monitor.display_gesture",
            owner_detail="Proactive monitor refreshed display gesture state.",
        )

    @staticmethod
    def _now_ns() -> int:
        """Return one patch-friendly monotonic timestamp in nanoseconds."""

        return int(time.monotonic() * _NS_PER_S)

    @staticmethod
    def _single_line_text(value: str, *, max_len: int = _ERROR_TEXT_MAX_LEN) -> str:
        """Normalize telemetry text to one bounded line."""

        line = " ".join(str(value).split())
        if len(line) <= max_len:
            return line
        return f"{line[: max_len - 3]}..."

    @classmethod
    def _exception_line(cls, exc: BaseException) -> str:
        """Return one bounded single-line exception summary."""

        return cls._single_line_text(_exception_text(exc))

    @staticmethod
    def _coerce_interval_s(value: Any, *, fallback_s: float) -> float:
        """Return one finite positive interval in seconds."""

        try:
            interval_s = float(value)
        except (TypeError, ValueError):
            return fallback_s
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            return fallback_s
        return interval_s

    def _interval_ns(
        self,
        value: Any,
        *,
        fallback_s: float,
        minimum_s: float,
    ) -> int:
        """Convert one interval value to a bounded nanosecond duration."""

        interval_s = max(minimum_s, self._coerce_interval_s(value, fallback_s=fallback_s))
        return int(interval_s * _NS_PER_S)

    def _emit(self, line: str) -> None:
        """Emit one service-local telemetry line safely."""

        _safe_emit(self.emit, self._single_line_text(line, max_len=1024))

    @staticmethod
    def _record_memory_probe(
        probe: StreamingMemoryProbe,
        *,
        force: bool = False,
        owner_detail: str | None = None,
    ) -> None:
        try:
            probe.maybe_record(force=force, owner_detail=owner_detail)
        except Exception:
            return

    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, Any],
        level: str | None = None,
    ) -> None:
        """Append one service-local ops event safely."""

        _append_ops_event(
            self.coordinator.runtime,
            event=event,
            message=message,
            data=data,
            level=level,
            emit=self.emit,
        )

    def _note_loop_failure(
        self,
        *,
        subsystem: str,
        message: str,
        exc: BaseException,
    ) -> None:
        """Emit one rate-limited worker fault event and update backoff state."""

        self._loop_failure_counts[subsystem] = self._loop_failure_counts.get(subsystem, 0) + 1
        now_ns = self._now_ns()
        should_emit, suppressed_count = self._loop_error_gate.allow(subsystem, now_ns=now_ns)
        if not should_emit:
            return
        error_text = self._exception_line(exc)
        if suppressed_count > 0:
            message = f"{message} Repeated errors were suppressed."
        self._emit(f"proactive_error={error_text}")
        self._append_ops_event(
            event="proactive_error",
            level="error",
            message=message,
            data={
                "subsystem": subsystem,
                "error": error_text,
                "suppressed_count": suppressed_count,
                "consecutive_failure_count": self._loop_failure_counts[subsystem],
            },
        )

    def _clear_loop_failure(self, subsystem: str) -> None:
        """Reset error-rate and backoff state after recovery."""

        self._loop_failure_counts.pop(subsystem, None)
        self._loop_error_gate.reset(subsystem)

    def _failure_delay_ns(self, subsystem: str) -> int:
        """Return one exponential-backoff delay for a failing subsystem."""

        failure_count = max(1, self._loop_failure_counts.get(subsystem, 1))
        delay_s = min(_FAILURE_BACKOFF_MAX_S, _FAILURE_BACKOFF_BASE_S * (2 ** (failure_count - 1)))
        return int(delay_s * _NS_PER_S)

    def _safe_close_resource(self, resource: Any, *, name: str) -> None:
        """Close one optional resource while suppressing cleanup failures."""

        if resource is None:
            return
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            self._append_ops_event(
                event="proactive_resource_close_failed",
                level="error",
                message="Failed to close a proactive monitor resource cleanly.",
                data={
                    "resource": name,
                    "error": self._exception_line(exc),
                },
            )

    def _open_background_lanes_locked(self) -> None:
        """Open coordinator background lanes under the lifecycle lock."""

        if self._background_lanes_open:
            return
        self.coordinator.open_background_lanes()
        self._background_lanes_open = True

    def _close_background_lanes_locked(self, *, timeout_s: float) -> None:
        """Close coordinator background lanes under the lifecycle lock."""

        if not self._background_lanes_open:
            return
        try:
            self.coordinator.close_background_lanes(timeout_s=timeout_s)
        except Exception as exc:
            self._append_ops_event(
                event="proactive_background_lanes_close_failed",
                level="error",
                message="Failed to close proactive background lanes cleanly.",
                data={"error": self._exception_line(exc)},
            )
        finally:
            # After a close attempt the wrapper must fail closed and stay recoverable.
            self._background_lanes_open = False

    def _open_resources_locked(self) -> None:
        """Open hardware resources under the lifecycle lock."""

        if self._resources_open:
            return
        opened_pir = False
        try:
            if self.coordinator.pir_monitor is not None:
                pir_open_result = open_pir_monitor_with_busy_retry(
                    cast(Any, self.coordinator.pir_monitor)
                )
                if pir_open_result.busy_retry_count > 0:
                    self._append_ops_event(
                        event="proactive_pir_open_retried",
                        message="PIR startup waited for a transient busy GPIO line to clear.",
                        data={
                            "attempt_count": pir_open_result.attempt_count,
                            "busy_retry_count": pir_open_result.busy_retry_count,
                        },
                    )
                opened_pir = True
            self._resources_open = True
        except Exception as exc:
            if opened_pir:
                self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
            self._resources_open = False
            self._append_ops_event(
                event="proactive_monitor_start_failed",
                level="error",
                message="Failed to open proactive monitor resources.",
                data={"error": self._exception_line(exc)},
            )
            raise

    def _close_resources_locked(self) -> None:
        """Close any opened hardware resources under the lifecycle lock."""

        if not self._resources_open:
            return
        self._safe_close_resource(self.coordinator.audio_observer, name="audio_observer")
        self._safe_close_resource(
            self.coordinator.attention_servo_controller,
            name="attention_servo_controller",
        )
        self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
        self._resources_open = False

    def _safe_worker_join(self, thread: Thread) -> bool:
        """Join one worker thread and return whether it fully stopped."""

        try:
            thread.join(timeout=self._close_join_timeout_s)
        except Exception as exc:
            self._append_ops_event(
                event="proactive_monitor_stop_join_failed",
                level="error",
                message="Joining the proactive monitor worker thread failed.",
                data={"error": self._exception_line(exc)},
            )
            return False
        return not thread.is_alive()

    def _resolve_attention_interval_ns(self) -> int:
        """Resolve the next display-attention refresh interval safely."""

        try:
            interval_s = resolve_display_attention_refresh_interval(self.coordinator.config)
        except Exception as exc:
            self._note_loop_failure(
                subsystem="display_attention_interval",
                message="Display attention refresh interval resolution failed.",
                exc=exc,
            )
            return self._interval_ns(
                self.poll_interval_s,
                fallback_s=self.poll_interval_s,
                minimum_s=_MIN_REFRESH_INTERVAL_S,
            )
        self._clear_loop_failure("display_attention_interval")
        return self._interval_ns(
            interval_s if interval_s is not None else self.poll_interval_s,
            fallback_s=self.poll_interval_s,
            minimum_s=_MIN_REFRESH_INTERVAL_S,
        )

    def _resolve_gesture_interval_ns(self) -> int:
        """Resolve the next display-gesture refresh interval safely."""

        try:
            interval_s = resolve_display_gesture_refresh_interval(self.coordinator.config)
        except Exception as exc:
            self._note_loop_failure(
                subsystem="display_gesture_interval",
                message="Display gesture refresh interval resolution failed.",
                exc=exc,
            )
            return self._interval_ns(
                self.poll_interval_s,
                fallback_s=self.poll_interval_s,
                minimum_s=_MIN_REFRESH_INTERVAL_S,
            )
        self._clear_loop_failure("display_gesture_interval")
        return self._interval_ns(
            interval_s if interval_s is not None else self.poll_interval_s,
            fallback_s=self.poll_interval_s,
            minimum_s=_MIN_REFRESH_INTERVAL_S,
        )

    def open(self) -> "ProactiveMonitorService":
        """Open resources and start the background proactive worker."""

        with self._lifecycle_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._thread is not None:
                if self._stop_event.is_set():
                    # BREAKING: the legacy service silently returned a wedged object here.
                    # Reopen is now rejected until the previous worker fully drains.
                    raise RuntimeError(
                        "Cannot reopen the proactive monitor while the previous worker is still stopping."
                    )
                return self
            self._open_resources_locked()
            startup_stage = "background_lanes"
            try:
                self._open_background_lanes_locked()
                self._stop_event.clear()
                thread = Thread(target=self._run, daemon=True, name="twinr-proactive")
                self._thread = thread
                startup_stage = "worker_thread"
                thread.start()
            except Exception as exc:
                self._stop_event.set()
                self._thread = None
                self._close_background_lanes_locked(timeout_s=0.05)
                self._close_resources_locked()
                start_message = (
                    "Failed to open proactive background lanes."
                    if startup_stage == "background_lanes"
                    else "Failed to start the proactive monitor worker thread."
                )
                self._append_ops_event(
                    event="proactive_monitor_start_failed",
                    level="error",
                    message=start_message,
                    data={
                        "stage": startup_stage,
                        "error": self._exception_line(exc),
                    },
                )
                raise
            self._append_ops_event(
                event="proactive_monitor_started",
                message="Proactive monitor started.",
                data={
                    "poll_interval_s": self.poll_interval_s,
                    "worker_native_id": thread.native_id,
                },
            )
            self._emit("proactive_monitor=started")
            return self

    def close(self) -> None:
        """Request worker shutdown and close monitor resources."""

        thread_to_join: Thread | None = None
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None and not self._resources_open and not self._background_lanes_open:
                return
            self._stop_event.set()
            if thread is current_thread():
                self._append_ops_event(
                    event="proactive_monitor_stop_requested",
                    message="Proactive monitor stop was requested from the worker thread.",
                    data={},
                )
                self._emit("proactive_monitor=stopping")
                return
            thread_to_join = thread

        worker_stopped = True
        if thread_to_join is not None:
            worker_stopped = self._safe_worker_join(thread_to_join)
            if not worker_stopped:
                self._append_ops_event(
                    event="proactive_monitor_stop_timeout",
                    level="error",
                    message="Proactive monitor worker did not stop within the shutdown budget.",
                    data={"join_timeout_s": self._close_join_timeout_s},
                )
                self._emit("proactive_monitor=stop_timeout")

        with self._lifecycle_lock:
            self._close_background_lanes_locked(timeout_s=min(self._close_join_timeout_s, 0.25))
            self._close_resources_locked()
            if worker_stopped and self._thread is not None and not self._thread.is_alive():
                self._thread = None

    def __enter__(self) -> "ProactiveMonitorService":
        """Enter the monitor context by starting the service."""

        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the monitor context by stopping the service."""

        del exc_type, exc, tb
        self.close()

    def _run(self) -> None:
        """Run the proactive tick loop until stopped."""

        next_tick_at_ns = 0
        next_attention_refresh_at_ns = 0
        next_gesture_refresh_at_ns = 0
        next_heartbeat_at_ns = 0
        try:
            while not self._stop_event.is_set():
                did_work = False
                loop_now_ns = self._now_ns()

                if self.heartbeat is not None and loop_now_ns >= next_heartbeat_at_ns:
                    try:
                        self.heartbeat()
                    except Exception as exc:
                        self._note_loop_failure(
                            subsystem="heartbeat",
                            message="Proactive monitor heartbeat failed.",
                            exc=exc,
                        )
                    else:
                        self._clear_loop_failure("heartbeat")
                    next_heartbeat_at_ns = loop_now_ns + self._interval_ns(
                        self.heartbeat_interval_s,
                        fallback_s=10.0,
                        minimum_s=1.0,
                    )

                attention_due = loop_now_ns >= next_attention_refresh_at_ns
                gesture_due = loop_now_ns >= next_gesture_refresh_at_ns
                shared_cycle_armed = False
                refresh_anchor_ns: int | None = None

                if attention_due or gesture_due:
                    attention_interval_ns = self._resolve_attention_interval_ns()
                    gesture_interval_ns = self._resolve_gesture_interval_ns()
                    try:
                        self.coordinator._open_display_perception_cycle(  # pylint: disable=protected-access
                            attention_due=attention_due,
                            gesture_due=gesture_due,
                        )
                    except Exception as exc:
                        self._note_loop_failure(
                            subsystem="display_cycle_open",
                            message="Opening the display perception cycle failed.",
                            exc=exc,
                        )
                        cycle_delay_ns = self._failure_delay_ns("display_cycle_open")
                        now_ns = self._now_ns()
                        if attention_due:
                            next_attention_refresh_at_ns = now_ns + max(
                                attention_interval_ns,
                                cycle_delay_ns,
                            )
                        if gesture_due:
                            next_gesture_refresh_at_ns = now_ns + max(
                                gesture_interval_ns,
                                cycle_delay_ns,
                            )
                    else:
                        self._clear_loop_failure("display_cycle_open")
                        shared_cycle_armed = getattr(
                            self.coordinator,
                            "_display_perception_cycle",
                            None,
                        ) is not None
                        refresh_anchor_ns = loop_now_ns if shared_cycle_armed else None
                        try:
                            if attention_due:
                                try:
                                    if self.coordinator.refresh_display_attention():
                                        did_work = True
                                except Exception as exc:
                                    self._note_loop_failure(
                                        subsystem="display_attention_refresh",
                                        message="Display attention refresh failed.",
                                        exc=exc,
                                    )
                                    attention_delay_ns = self._failure_delay_ns(
                                        "display_attention_refresh"
                                    )
                                    next_attention_refresh_at_ns = self._now_ns() + max(
                                        attention_interval_ns,
                                        attention_delay_ns,
                                    )
                                else:
                                    self._clear_loop_failure("display_attention_refresh")
                                    self._record_memory_probe(self._display_attention_memory_probe)
                                    attention_base_ns = (
                                        self._now_ns()
                                        if refresh_anchor_ns is None
                                        else refresh_anchor_ns
                                    )
                                    next_attention_refresh_at_ns = (
                                        attention_base_ns + attention_interval_ns
                                    )

                            if gesture_due:
                                try:
                                    if self.coordinator.refresh_display_gesture_emoji():
                                        did_work = True
                                except Exception as exc:
                                    self._note_loop_failure(
                                        subsystem="display_gesture_refresh",
                                        message="Display gesture refresh failed.",
                                        exc=exc,
                                    )
                                    gesture_delay_ns = self._failure_delay_ns(
                                        "display_gesture_refresh"
                                    )
                                    next_gesture_refresh_at_ns = self._now_ns() + max(
                                        gesture_interval_ns,
                                        gesture_delay_ns,
                                    )
                                else:
                                    self._clear_loop_failure("display_gesture_refresh")
                                    self._record_memory_probe(self._display_gesture_memory_probe)
                                    gesture_base_ns = (
                                        self._now_ns()
                                        if refresh_anchor_ns is None
                                        else refresh_anchor_ns
                                    )
                                    next_gesture_refresh_at_ns = (
                                        gesture_base_ns + gesture_interval_ns
                                    )
                        finally:
                            try:
                                self.coordinator._close_display_perception_cycle()  # pylint: disable=protected-access
                            except Exception as exc:
                                self._note_loop_failure(
                                    subsystem="display_cycle_close",
                                    message="Closing the display perception cycle failed.",
                                    exc=exc,
                                )
                                cycle_delay_ns = self._failure_delay_ns("display_cycle_close")
                                now_ns = self._now_ns()
                                if attention_due:
                                    next_attention_refresh_at_ns = max(
                                        next_attention_refresh_at_ns,
                                        now_ns + cycle_delay_ns,
                                    )
                                if gesture_due:
                                    next_gesture_refresh_at_ns = max(
                                        next_gesture_refresh_at_ns,
                                        now_ns + cycle_delay_ns,
                                    )
                            else:
                                self._clear_loop_failure("display_cycle_close")

                if self._now_ns() >= next_tick_at_ns:
                    try:
                        self.coordinator.tick()
                    except Exception as exc:
                        self._note_loop_failure(
                            subsystem="tick",
                            message="Proactive monitor tick failed.",
                            exc=exc,
                        )
                        tick_delay_ns = self._failure_delay_ns("tick")
                        next_tick_at_ns = self._now_ns() + max(
                            self._interval_ns(
                                self.poll_interval_s,
                                fallback_s=self.poll_interval_s,
                                minimum_s=0.2,
                            ),
                            tick_delay_ns,
                        )
                    else:
                        did_work = True
                        self._clear_loop_failure("tick")
                        self._record_memory_probe(self._tick_memory_probe)
                        next_tick_at_ns = self._now_ns() + self._interval_ns(
                            self.poll_interval_s,
                            fallback_s=self.poll_interval_s,
                            minimum_s=0.2,
                        )

                next_due_ns = min(
                    next_tick_at_ns,
                    next_attention_refresh_at_ns,
                    next_gesture_refresh_at_ns,
                    next_heartbeat_at_ns if self.heartbeat is not None else next_tick_at_ns,
                )
                now_ns = self._now_ns()
                wait_ns = max(0, next_due_ns - now_ns)
                if did_work:
                    wait_ns = min(wait_ns, int(_MIN_POST_WORK_WAIT_S * _NS_PER_S))
                else:
                    wait_ns = min(wait_ns, int(_MAX_IDLE_WAIT_S * _NS_PER_S))
                if self._stop_event.wait(wait_ns / _NS_PER_S):
                    return
        except Exception as exc:
            self._note_loop_failure(
                subsystem="worker_crash",
                message="The proactive monitor worker crashed unexpectedly.",
                exc=exc,
            )
        finally:
            with self._lifecycle_lock:
                if self._thread is current_thread():
                    self._thread = None
                self._close_background_lanes_locked(
                    timeout_s=min(self._close_join_timeout_s, 0.25)
                )
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_stopped",
                    message="Proactive monitor stopped.",
                    data={},
                )
                self._emit("proactive_monitor=stopped")


__all__ = ["ProactiveMonitorService"]
