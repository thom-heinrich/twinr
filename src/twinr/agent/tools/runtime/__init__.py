"""Expose the runtime tool orchestration surface for Twinr live turns.

This package bundles the canonical realtime tool binding registry, the thin
dispatcher that adapts workflow owners to handler callables, and the
single-lane and dual-lane tool loops used by the runner and streaming
workflows. Import from this package or from ``twinr.agent.tools`` instead of
reaching into private module helpers from callers.
"""

from .dual_lane_loop import DualLaneToolLoop, SpeechLaneDelta
from .executor import RealtimeToolExecutor
from .registry import RealtimeToolBindingError, bind_realtime_tool_handlers, realtime_tool_names
from .streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop

__all__ = [
    "DualLaneToolLoop",
    "RealtimeToolBindingError",
    "RealtimeToolExecutor",
    "SpeechLaneDelta",
    "StreamingToolLoopResult",
    "ToolCallingStreamingLoop",
    "bind_realtime_tool_handlers",
    "realtime_tool_names",
]
