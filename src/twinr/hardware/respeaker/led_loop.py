"""Drive ReSpeaker LED feedback from the shared Twinr runtime snapshot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore, RuntimeSnapshotStoreError
from twinr.hardware.respeaker.led_controller import ReSpeakerLedController
from twinr.hardware.respeaker.led_profiles import resolve_respeaker_led_profile


_DEFAULT_POLL_INTERVAL_S = 0.08


def _default_emit(line: str) -> None:
    """Print one bounded telemetry line."""

    print(line, flush=True)


def _never_stop() -> bool:
    """Return the default stop signal for standalone LED loops."""

    return False


@dataclass(slots=True)
class ReSpeakerLedLoop:
    """Continuously render calm XVF3800 LED feedback from runtime state."""

    config: TwinrConfig
    controller: ReSpeakerLedController
    snapshot_store: RuntimeSnapshotStore
    emit: Callable[[str], None] = _default_emit
    sleep: Callable[[float], object] = time.sleep
    monotonic_clock: Callable[[], float] = time.monotonic
    stop_requested: Callable[[], bool] = _never_stop
    _last_snapshot_error: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "ReSpeakerLedLoop":
        """Build one runtime LED loop from configuration."""

        return cls(
            config=config,
            controller=ReSpeakerLedController(emit=emit or _default_emit),
            snapshot_store=RuntimeSnapshotStore(config.runtime_state_path),
            emit=emit or _default_emit,
        )

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the LED loop until stopped or a duration elapses."""

        started_at = self.monotonic_clock()
        try:
            while True:
                if self.stop_requested():
                    return 0
                if duration_s is not None and (self.monotonic_clock() - started_at) >= duration_s:
                    return 0
                profile = self._current_profile()
                self.controller.render(profile, at_monotonic_s=self.monotonic_clock())
                self.sleep(_DEFAULT_POLL_INTERVAL_S)
        finally:
            try:
                self.controller.off()
            except Exception as exc:
                self.emit(f"respeaker_led_shutdown_failed={type(exc).__name__}")

    def _current_profile(self):
        try:
            snapshot = self.snapshot_store.load()
        except RuntimeSnapshotStoreError as exc:
            reason = type(exc).__name__
            if reason != self._last_snapshot_error:
                self._last_snapshot_error = reason
                self.emit(f"respeaker_led_snapshot_failed={reason}")
            return resolve_respeaker_led_profile(runtime_status="error", error_message=reason)
        self._last_snapshot_error = None
        return resolve_respeaker_led_profile(
            runtime_status=getattr(snapshot, "status", None),
            error_message=getattr(snapshot, "error_message", None),
        )


__all__ = [
    "ReSpeakerLedLoop",
]
