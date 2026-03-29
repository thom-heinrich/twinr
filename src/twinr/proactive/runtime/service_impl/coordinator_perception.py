"""Shared display-refresh perception-cycle helpers for the proactive coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _DisplayPerceptionCycleState:
    """Store one lazily loaded shared display-refresh perception snapshot."""

    observe_fn: Callable[[], Any]
    snapshot: Any | None = None
    loaded: bool = False


class ProactiveCoordinatorPerceptionMixin:
    """Reuse one combined perception snapshot across one display refresh cycle."""

    def _open_display_perception_cycle(
        self,
        *,
        attention_due: bool,
        gesture_due: bool,
    ) -> None:
        """Arm one shared display-refresh cycle when both lanes are due."""

        self._display_perception_cycle = None
        if not (attention_due and gesture_due):
            return
        if self.vision_observer is None:
            return
        observe_perception_stream = getattr(self.vision_observer, "observe_perception_stream", None)
        if not callable(observe_perception_stream):
            return
        self._display_perception_cycle = _DisplayPerceptionCycleState(
            observe_fn=observe_perception_stream,
        )

    def _close_display_perception_cycle(self) -> None:
        """Drop any shared display-refresh cycle state after one monitor iteration."""

        self._display_perception_cycle = None

    def _shared_display_perception_snapshot(
        self,
        *,
        consumer: str,
    ):
        """Return one shared combined perception snapshot when the cycle armed it."""

        state = getattr(self, "_display_perception_cycle", None)
        if not isinstance(state, _DisplayPerceptionCycleState):
            return None
        if not state.loaded:
            state.snapshot = self._observe_vision_with_method(state.observe_fn)
            state.loaded = True
        mode = "perception_stream_shared" if state.snapshot is not None else "perception_stream_missing"
        if consumer == "attention":
            self._last_attention_vision_refresh_mode = mode
        elif consumer == "gesture":
            self._last_gesture_vision_refresh_mode = mode
        return state.snapshot


__all__ = ["ProactiveCoordinatorPerceptionMixin"]
