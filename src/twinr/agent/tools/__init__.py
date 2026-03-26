"""Expose the public Twinr tool package surface.

This package is the stable import layer for tool-lane prompting helpers,
tool-schema builders, and runtime tool orchestration used by workflows,
orchestrators, and provider adapters. Concrete handler implementations remain
internal to ``twinr.agent.tools.handlers`` and are intentionally not
re-exported here.

Runtime-loop symbols stay lazily loaded so provider adapters can import the
schema builders without pulling the full runtime stack back into provider
session initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from twinr.agent.tools.prompting.instructions import (
    COMPACT_TOOL_AGENT_INSTRUCTIONS,
    DEFAULT_TOOL_AGENT_INSTRUCTIONS,
    FIRST_WORD_AGENT_INSTRUCTIONS,
    build_compact_tool_agent_instructions,
    build_first_word_instructions,
    build_local_route_first_word_instructions,
    build_specialist_tool_agent_instructions,
    build_supervisor_decision_instructions,
    build_supervisor_tool_agent_instructions,
    build_tool_agent_instructions,
    tool_agent_time_context,
)
from twinr.agent.tools.schemas.contracts import (
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_realtime_tool_schemas,
)

if TYPE_CHECKING:
    from twinr.agent.tools.runtime.dual_lane_loop import DualLaneToolLoop
    from twinr.agent.tools.runtime.executor import RealtimeToolExecutor
    from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta
    from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop
    from twinr.agent.tools.runtime.registry import bind_realtime_tool_handlers, realtime_tool_names

_LAZY_RUNTIME_EXPORTS: dict[str, tuple[str, str]] = {
    "DualLaneToolLoop": ("twinr.agent.tools.runtime.dual_lane_loop", "DualLaneToolLoop"),
    "RealtimeToolExecutor": ("twinr.agent.tools.runtime.executor", "RealtimeToolExecutor"),
    "SpeechLaneDelta": ("twinr.agent.tools.runtime.speech_lane", "SpeechLaneDelta"),
    "StreamingToolLoopResult": ("twinr.agent.tools.runtime.streaming_loop", "StreamingToolLoopResult"),
    "ToolCallingStreamingLoop": ("twinr.agent.tools.runtime.streaming_loop", "ToolCallingStreamingLoop"),
    "bind_realtime_tool_handlers": ("twinr.agent.tools.runtime.registry", "bind_realtime_tool_handlers"),
    "realtime_tool_names": ("twinr.agent.tools.runtime.registry", "realtime_tool_names"),
}

__all__ = [
    "COMPACT_TOOL_AGENT_INSTRUCTIONS",
    "DEFAULT_TOOL_AGENT_INSTRUCTIONS",
    "FIRST_WORD_AGENT_INSTRUCTIONS",
    "RealtimeToolExecutor",
    "StreamingToolLoopResult",
    "ToolCallingStreamingLoop",
    "build_agent_tool_schemas",
    "build_first_word_instructions",
    "build_local_route_first_word_instructions",
    "build_compact_agent_tool_schemas",
    "build_compact_tool_agent_instructions",
    "build_specialist_tool_agent_instructions",
    "build_supervisor_decision_instructions",
    "build_supervisor_tool_agent_instructions",
    "build_tool_agent_instructions",
    "DualLaneToolLoop",
    "SpeechLaneDelta",
    "bind_realtime_tool_handlers",
    "build_realtime_tool_schemas",
    "realtime_tool_names",
    "tool_agent_time_context",
]


def __getattr__(name: str) -> Any:
    """Load runtime-layer exports lazily to avoid provider/runtime cycles."""

    try:
        module_name, attribute_name = _LAZY_RUNTIME_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in normal module introspection."""

    return sorted(set(globals()) | set(__all__))
