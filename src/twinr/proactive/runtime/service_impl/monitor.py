"""Monitor lifecycle wrapper for the proactive runtime coordinator.

Purpose: own resource open/close sequencing and the bounded background worker
thread around an already-configured ``ProactiveCoordinator``.

Invariants: lifecycle idempotency, shutdown behavior, and worker error events
must remain identical to the legacy service implementation.
"""

from __future__ import annotations

from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable, cast
import time

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


class ProactiveMonitorService:
    """Run the proactive coordinator in a bounded background worker."""

    def __init__(
        self,
        coordinator: ProactiveCoordinator,
        *,
        poll_interval_s: float,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize one monitor service around a configured coordinator."""

        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.emit = emit or (lambda _line: None)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lifecycle_lock = Lock()
        self._resources_open = False
        self._close_join_timeout_s = _DEFAULT_CLOSE_JOIN_TIMEOUT_S

    def _emit(self, line: str) -> None:
        """Emit one service-local telemetry line safely."""

        _safe_emit(self.emit, line)

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
                    "error": _exception_text(exc),
                },
            )

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
                data={"error": _exception_text(exc)},
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

    def open(self) -> "ProactiveMonitorService":
        """Open resources and start the background proactive worker."""

        with self._lifecycle_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._thread is not None:
                return self
            self._open_resources_locked()
            self.coordinator.open_background_lanes()
            self._stop_event.clear()
            thread = Thread(target=self._run, daemon=True, name="twinr-proactive")
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                self._thread = None
                self.coordinator.close_background_lanes(timeout_s=0.05)
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_start_failed",
                    level="error",
                    message="Failed to start the proactive monitor worker thread.",
                    data={"error": _exception_text(exc)},
                )
                raise
            self._append_ops_event(
                event="proactive_monitor_started",
                message="Proactive monitor started.",
                data={"poll_interval_s": self.poll_interval_s},
            )
            self._emit("proactive_monitor=started")
            return self

    def close(self) -> None:
        """Request worker shutdown and close monitor resources."""

        thread_to_join: Thread | None = None
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None and not self._resources_open:
                return
            self._stop_event.set()
            self.coordinator.close_background_lanes(timeout_s=min(self._close_join_timeout_s, 0.25))
            self._close_resources_locked()
            if thread is current_thread():
                self._append_ops_event(
                    event="proactive_monitor_stop_requested",
                    message="Proactive monitor stop was requested from the worker thread.",
                    data={},
                )
                self._emit("proactive_monitor=stopping")
                return
            thread_to_join = thread
        if thread_to_join is not None:
            thread_to_join.join(timeout=self._close_join_timeout_s)
            if thread_to_join.is_alive():
                self._append_ops_event(
                    event="proactive_monitor_stop_timeout",
                    level="error",
                    message="Proactive monitor worker did not stop within the shutdown budget.",
                    data={"join_timeout_s": self._close_join_timeout_s},
                )
                self._emit("proactive_monitor=stop_timeout")

    def __enter__(self) -> "ProactiveMonitorService":
        """Enter the monitor context by starting the service."""

        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the monitor context by stopping the service."""

        del exc_type, exc, tb
        self.close()

    def _run(self) -> None:
        """Run the proactive tick loop until stopped."""

        next_tick_at = 0.0
        next_attention_refresh_at = 0.0
        next_gesture_refresh_at = 0.0
        try:
            while not self._stop_event.is_set():
                did_work = False
                loop_now = time.monotonic()
                attention_due = loop_now >= next_attention_refresh_at
                gesture_due = loop_now >= next_gesture_refresh_at
                shared_cycle_armed = False
                if attention_due or gesture_due:
                    self.coordinator._open_display_perception_cycle(  # pylint: disable=protected-access
                        attention_due=attention_due,
                        gesture_due=gesture_due,
                    )
                    shared_cycle_armed = getattr(
                        self.coordinator,
                        "_display_perception_cycle",
                        None,
                    ) is not None
                refresh_anchor_s = loop_now if shared_cycle_armed else None
                try:
                    if attention_due:
                        try:
                            if self.coordinator.refresh_display_attention():
                                did_work = True
                        except Exception as exc:
                            error_text = _exception_text(exc)
                            self._emit(f"proactive_error={error_text}")
                            self._append_ops_event(
                                event="proactive_error",
                                level="error",
                                message="Display attention refresh failed.",
                                data={"error": error_text},
                            )
                        interval_s = resolve_display_attention_refresh_interval(self.coordinator.config)
                        attention_base_s = (
                            time.monotonic() if refresh_anchor_s is None else refresh_anchor_s
                        )
                        next_attention_refresh_at = (
                            attention_base_s + interval_s
                            if interval_s is not None
                            else attention_base_s + self.poll_interval_s
                        )
                    if gesture_due:
                        try:
                            if self.coordinator.refresh_display_gesture_emoji():
                                did_work = True
                        except Exception as exc:
                            error_text = _exception_text(exc)
                            self._emit(f"proactive_error={error_text}")
                            self._append_ops_event(
                                event="proactive_error",
                                level="error",
                                message="Display gesture refresh failed.",
                                data={"error": error_text},
                            )
                        interval_s = resolve_display_gesture_refresh_interval(self.coordinator.config)
                        gesture_base_s = (
                            time.monotonic() if refresh_anchor_s is None else refresh_anchor_s
                        )
                        next_gesture_refresh_at = (
                            gesture_base_s + interval_s
                            if interval_s is not None
                            else gesture_base_s + self.poll_interval_s
                        )
                finally:
                    if attention_due or gesture_due:
                        self.coordinator._close_display_perception_cycle()  # pylint: disable=protected-access
                now = time.monotonic()
                if now >= next_tick_at:
                    try:
                        self.coordinator.tick()
                        did_work = True
                    except Exception as exc:
                        error_text = _exception_text(exc)
                        self._emit(f"proactive_error={error_text}")
                        self._append_ops_event(
                            event="proactive_error",
                            level="error",
                            message="Proactive monitor tick failed.",
                            data={"error": error_text},
                        )
                    next_tick_at = time.monotonic() + self.poll_interval_s
                wait_s = max(0.02, min(0.05, max(0.0, next_tick_at - time.monotonic())))
                if did_work:
                    wait_s = 0.02
                if self._stop_event.wait(wait_s):
                    return
        finally:
            with self._lifecycle_lock:
                if self._thread is current_thread():
                    self._thread = None
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_stopped",
                    message="Proactive monitor stopped.",
                    data={},
                )
                self._emit("proactive_monitor=stopped")


__all__ = ["ProactiveMonitorService"]
