"""Compose the refactored realtime workflow loop from focused mixins."""

# mypy: ignore-errors

from __future__ import annotations

from twinr.agent.workflows.realtime_runner_impl.bootstrap import TwinrRealtimeBootstrapMixin
from twinr.agent.workflows.realtime_runner_impl.session import TwinrRealtimeSessionMixin
from twinr.agent.workflows.realtime_runner_impl.turn_capture import TwinrRealtimeTurnCaptureMixin
from twinr.agent.workflows.realtime_runner_impl.turn_execution import TwinrRealtimeTurnExecutionMixin
from twinr.agent.workflows.realtime_runtime.background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.realtime_runtime.support import TwinrRealtimeSupportMixin
from twinr.agent.workflows.realtime_runner_tools import TwinrRealtimeToolDelegatesMixin


class TwinrRealtimeHardwareLoopImpl(
    TwinrRealtimeTurnExecutionMixin,
    TwinrRealtimeTurnCaptureMixin,
    TwinrRealtimeSessionMixin,
    TwinrRealtimeBootstrapMixin,
    TwinrRealtimeBackgroundMixin,
    TwinrRealtimeToolDelegatesMixin,
    TwinrRealtimeSupportMixin,
):
    """Coordinate realtime sessions, voice activation, and background delivery."""
