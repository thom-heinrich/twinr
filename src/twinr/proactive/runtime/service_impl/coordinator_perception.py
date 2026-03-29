# CHANGELOG: 2026-03-29
# BUG-1: Fixed concurrent double-loads where attention and gesture could both call
#        observe_perception_stream() and break the "one shared snapshot per display cycle" contract.
# BUG-2: Fixed stale-cycle leakage where an in-flight observation from a previous cycle could outlive
#        close/open and be returned after the cycle had already been invalidated.
# BUG-3: Fixed same-cycle failure re-entry where multiple consumers could retrigger the same failing
#        observation path.
# SEC-1: No practically exploitable security issue was identified in this file alone; no security code change required.
# IMP-1: Added explicit cycle-scoped synchronization and failure memoization for modern multi-threaded/free-threaded Python runtimes.
# IMP-2: Added cycle identity/timing introspection so the shared perception snapshot now behaves like a tracked public context.

"""Shared display-refresh perception-cycle helpers for the proactive coordinator.

Host contract
-------------
The host object is expected to provide:
- self.vision_observer
- self._observe_vision_with_method(callable)
- self._last_attention_vision_refresh_mode
- self._last_gesture_vision_refresh_mode

The shared snapshot returned within a cycle is intentionally reused by reference.
Consumers must treat it as read-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from threading import Condition
from time import monotonic_ns
from typing import Any, Literal, Protocol


_DISPLAY_PERCEPTION_CYCLE_IDS = count(1)

_MODE_SHARED = "perception_stream_shared"
_MODE_MISSING = "perception_stream_missing"

_STATUS_UNLOADED = "unloaded"
_STATUS_LOADING = "loading"
_STATUS_READY = "ready"
_STATUS_FAILED = "failed"
_STATUS_CLOSED = "closed"

ConsumerName = Literal["attention", "gesture"]


class _PerceptionMixinHost(Protocol):
    vision_observer: Any | None
    _last_attention_vision_refresh_mode: str
    _last_gesture_vision_refresh_mode: str

    def _observe_vision_with_method(self, method: Callable[[], Any]) -> Any: ...


@dataclass(slots=True)
class _DisplayPerceptionCycleState:
    """Store one lazily loaded shared display-refresh perception snapshot."""

    observe_fn: Callable[[], Any]
    cycle_id: int = field(default_factory=lambda: next(_DISPLAY_PERCEPTION_CYCLE_IDS))
    opened_monotonic_ns: int = field(default_factory=monotonic_ns)
    condition: Condition = field(default_factory=Condition, repr=False)
    snapshot: Any | None = None
    exception: BaseException | None = None
    loaded_monotonic_ns: int | None = None
    load_duration_ns: int | None = None
    status: str = _STATUS_UNLOADED
    closed: bool = False

    def close(self) -> None:
        """Invalidate the cycle and wake any waiters."""

        with self.condition:
            self.closed = True
            self.snapshot = None
            self.status = _STATUS_CLOSED
            self.condition.notify_all()

    def get_or_load(self, loader: Callable[[], Any]) -> Any | None:
        """Return the shared snapshot, loading it once at most for this cycle."""

        started_ns: int | None = None

        with self.condition:
            while True:
                if self.closed or self.status == _STATUS_CLOSED:
                    return None
                if self.status == _STATUS_READY:
                    return self.snapshot
                if self.status == _STATUS_FAILED:
                    exception = self.exception
                    if exception is None:
                        raise RuntimeError(
                            "display perception cycle entered failed state without an exception"
                        )
                    raise exception
                if self.status == _STATUS_UNLOADED:
                    self.status = _STATUS_LOADING
                    started_ns = monotonic_ns()
                    break
                self.condition.wait()

        try:
            snapshot = loader()
        except BaseException as exc:
            finished_ns = monotonic_ns()
            with self.condition:
                if not self.closed and self.status != _STATUS_CLOSED:
                    self.exception = exc
                    self.loaded_monotonic_ns = finished_ns
                    self.load_duration_ns = (
                        None if started_ns is None else finished_ns - started_ns
                    )
                    self.status = _STATUS_FAILED
                self.condition.notify_all()
            raise

        finished_ns = monotonic_ns()
        with self.condition:
            if self.closed or self.status == _STATUS_CLOSED:
                self.condition.notify_all()
                return None
            self.snapshot = snapshot
            self.loaded_monotonic_ns = finished_ns
            self.load_duration_ns = (
                None if started_ns is None else finished_ns - started_ns
            )
            self.status = _STATUS_READY
            self.condition.notify_all()
            return self.snapshot


class ProactiveCoordinatorPerceptionMixin:
    """Reuse one combined perception snapshot across one display refresh cycle."""

    def _open_display_perception_cycle(
        self: _PerceptionMixinHost,
        *,
        attention_due: bool,
        gesture_due: bool,
    ) -> None:
        """Arm one shared display-refresh cycle when both lanes are due."""

        previous_state = getattr(self, "_display_perception_cycle", None)
        if isinstance(previous_state, _DisplayPerceptionCycleState):
            previous_state.close()

        self._display_perception_cycle = None
        if not (attention_due and gesture_due):
            return

        vision_observer = getattr(self, "vision_observer", None)
        if vision_observer is None:
            return

        observe_perception_stream = getattr(vision_observer, "observe_perception_stream", None)
        if not callable(observe_perception_stream):
            return

        self._display_perception_cycle = _DisplayPerceptionCycleState(
            observe_fn=observe_perception_stream,
        )

    def _close_display_perception_cycle(self: _PerceptionMixinHost) -> None:
        """Drop any shared display-refresh cycle state after one monitor iteration."""

        state = getattr(self, "_display_perception_cycle", None)
        if isinstance(state, _DisplayPerceptionCycleState):
            state.close()
        self._display_perception_cycle = None

    def _shared_display_perception_snapshot(
        self: _PerceptionMixinHost,
        *,
        consumer: ConsumerName | str,
    ) -> Any | None:
        """Return one shared combined perception snapshot when the cycle armed it."""

        state = getattr(self, "_display_perception_cycle", None)
        if not isinstance(state, _DisplayPerceptionCycleState):
            return None

        try:
            snapshot = state.get_or_load(
                lambda: self._observe_vision_with_method(state.observe_fn),
            )
        except BaseException:
            # Preserve the historical two-mode surface for drop-in compatibility.
            self._set_display_perception_cycle_mode(consumer=consumer, mode=_MODE_MISSING)
            raise

        mode = _MODE_SHARED if snapshot is not None else _MODE_MISSING
        self._set_display_perception_cycle_mode(consumer=consumer, mode=mode)
        return snapshot

    def _current_display_perception_cycle_info(
        self: _PerceptionMixinHost,
    ) -> dict[str, Any] | None:
        """Return additive, non-breaking introspection data for the active cycle."""

        state = getattr(self, "_display_perception_cycle", None)
        if not isinstance(state, _DisplayPerceptionCycleState):
            return None

        with state.condition:
            return {
                "cycle_id": state.cycle_id,
                "opened_monotonic_ns": state.opened_monotonic_ns,
                "loaded_monotonic_ns": state.loaded_monotonic_ns,
                "load_duration_ns": state.load_duration_ns,
                "status": state.status,
                "closed": state.closed,
                "has_snapshot": state.snapshot is not None,
                "has_exception": state.exception is not None,
            }

    def _set_display_perception_cycle_mode(
        self: _PerceptionMixinHost,
        *,
        consumer: ConsumerName | str,
        mode: str,
    ) -> None:
        """Update lane-local refresh mode bookkeeping."""

        if consumer == "attention":
            self._last_attention_vision_refresh_mode = mode
        elif consumer == "gesture":
            self._last_gesture_vision_refresh_mode = mode


__all__ = ["ProactiveCoordinatorPerceptionMixin"]