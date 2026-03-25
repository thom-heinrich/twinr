"""Expose the public Twinr tool package surface.

This package is the stable import layer for tool-lane prompting helpers,
tool-schema builders, and runtime tool orchestration used by workflows,
orchestrators, and provider adapters. Concrete handler implementations remain
internal to ``twinr.agent.tools.handlers`` and are intentionally not
re-exported here.
"""

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
from twinr.agent.tools.runtime.dual_lane_loop import DualLaneToolLoop
from twinr.agent.tools.runtime.executor import RealtimeToolExecutor
from twinr.agent.tools.runtime.registry import bind_realtime_tool_handlers, realtime_tool_names
from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop
from twinr.agent.tools.schemas.contracts import (
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_realtime_tool_schemas,
)

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
