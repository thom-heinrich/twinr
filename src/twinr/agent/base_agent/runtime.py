from __future__ import annotations

from twinr.agent.base_agent.runtime_automation import TwinrRuntimeAutomationMixin
from twinr.agent.base_agent.runtime_base import TwinrRuntimeBase
from twinr.agent.base_agent.runtime_context import TwinrRuntimeContextMixin
from twinr.agent.base_agent.runtime_flow import TwinrRuntimeFlowMixin
from twinr.agent.base_agent.runtime_memory import TwinrRuntimeMemoryMixin
from twinr.agent.base_agent.runtime_snapshot import TwinrRuntimeSnapshotMixin


class TwinrRuntime(
    TwinrRuntimeFlowMixin,
    TwinrRuntimeContextMixin,
    TwinrRuntimeMemoryMixin,
    TwinrRuntimeAutomationMixin,
    TwinrRuntimeSnapshotMixin,
    TwinrRuntimeBase,
):
    pass


__all__ = ["TwinrRuntime"]
