"""Compatibility wrapper for the realtime hardware workflow loop.

Import ``TwinrRealtimeHardwareLoop`` from this module. The implementation now
lives in focused sibling modules so bootstrap, session orchestration, capture,
and turn execution can evolve independently without changing callers.
"""

##REFACTOR: 2026-03-27##
from __future__ import annotations

from pathlib import Path

from twinr.agent.workflows.realtime_runner_impl import TwinrRealtimeHardwareLoopImpl
from twinr.agent.workflows.startup_boot_sound import start_startup_boot_sound
from twinr.integrations import build_smart_home_hub_adapter


class TwinrRealtimeHardwareLoop(TwinrRealtimeHardwareLoopImpl):
    """Preserve the legacy module import path and patch surfaces."""

    def _start_startup_boot_sound(self) -> None:
        start_startup_boot_sound(
            config=self.config,
            playback_coordinator=self.playback_coordinator,
            emit=self.emit,
        )

    def _build_managed_smart_home_adapter(self):
        return build_smart_home_hub_adapter(Path(self.config.project_root))


__all__ = [
    "TwinrRealtimeHardwareLoop",
    "build_smart_home_hub_adapter",
    "start_startup_boot_sound",
]
