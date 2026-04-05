"""Compatibility wrapper for realtime background delivery helpers."""


##REFACTOR: 2026-03-27##
from __future__ import annotations

from twinr.agent.workflows.realtime_runtime.background_impl import TwinrRealtimeBackgroundMixinImpl


class TwinrRealtimeBackgroundMixin(TwinrRealtimeBackgroundMixinImpl):
    """Preserve the legacy mixin import path for background delivery."""


__all__ = ["TwinrRealtimeBackgroundMixin"]
