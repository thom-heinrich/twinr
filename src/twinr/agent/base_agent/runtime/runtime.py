"""Compose the concrete Twinr runtime from focused runtime mixins."""

from __future__ import annotations

from twinr.agent.base_agent.runtime.automation import TwinrRuntimeAutomationMixin
from twinr.agent.base_agent.runtime.base import TwinrRuntimeBase
from twinr.agent.base_agent.runtime.context import TwinrRuntimeContextMixin
from twinr.agent.base_agent.runtime.discovery import TwinrRuntimeDiscoveryMixin
from twinr.agent.base_agent.runtime.flow import TwinrRuntimeFlowMixin
from twinr.agent.base_agent.runtime.memory import TwinrRuntimeMemoryMixin
from twinr.agent.base_agent.runtime.snapshot import TwinrRuntimeSnapshotMixin


class TwinrRuntime(
    TwinrRuntimeFlowMixin,
    TwinrRuntimeContextMixin,
    TwinrRuntimeDiscoveryMixin,
    TwinrRuntimeMemoryMixin,
    TwinrRuntimeAutomationMixin,
    TwinrRuntimeSnapshotMixin,
    TwinrRuntimeBase,
):
    """Expose the complete runtime API used by workflows and tools."""

    pass


__all__ = ["TwinrRuntime"]
