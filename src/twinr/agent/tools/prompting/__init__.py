"""Prompting instructions for Twinr tool-capable agents."""

from .instructions import (
    COMPACT_TOOL_AGENT_INSTRUCTIONS,
    DEFAULT_TOOL_AGENT_INSTRUCTIONS,
    FIRST_WORD_AGENT_INSTRUCTIONS,
    SPECIALIST_TOOL_AGENT_INSTRUCTIONS,
    SUPERVISOR_DECISION_AGENT_INSTRUCTIONS,
    SUPERVISOR_TOOL_AGENT_INSTRUCTIONS,
    build_compact_tool_agent_instructions,
    build_first_word_instructions,
    build_local_route_first_word_instructions,
    build_specialist_tool_agent_instructions,
    build_supervisor_decision_instructions,
    build_supervisor_tool_agent_instructions,
    build_tool_agent_instructions,
    tool_agent_time_context,
)

__all__ = [
    "COMPACT_TOOL_AGENT_INSTRUCTIONS",
    "DEFAULT_TOOL_AGENT_INSTRUCTIONS",
    "FIRST_WORD_AGENT_INSTRUCTIONS",
    "SPECIALIST_TOOL_AGENT_INSTRUCTIONS",
    "SUPERVISOR_DECISION_AGENT_INSTRUCTIONS",
    "SUPERVISOR_TOOL_AGENT_INSTRUCTIONS",
    "build_compact_tool_agent_instructions",
    "build_first_word_instructions",
    "build_local_route_first_word_instructions",
    "build_specialist_tool_agent_instructions",
    "build_supervisor_decision_instructions",
    "build_supervisor_tool_agent_instructions",
    "build_tool_agent_instructions",
    "tool_agent_time_context",
]
