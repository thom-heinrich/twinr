"""Build canonical and adapted JSON schemas for Twinr agent tools.

##REFACTOR: 2026-03-27##

This module remains the stable public import path for Twinr's schema builders.
The production implementation now lives in ``contracts_impl`` so the canonical
schema surface can stay modular without requiring any caller changes.
"""

from __future__ import annotations

from typing import Any, Iterable

from .contracts_impl.main import (
    build_agent_tool_schemas as _build_agent_tool_schemas_impl,
    build_compact_agent_tool_schemas as _build_compact_agent_tool_schemas_impl,
    build_realtime_tool_schemas as _build_realtime_tool_schemas_impl,
)

__all__ = [
    "build_agent_tool_schemas",
    "build_compact_agent_tool_schemas",
    "build_realtime_tool_schemas",
]


def build_agent_tool_schemas(
    tool_names: Iterable[str] | str | bytes | bytearray | None,
) -> list[dict[str, Any]]:
    """Build the canonical JSON schemas for the requested tool names."""

    return _build_agent_tool_schemas_impl(tool_names)


def build_compact_agent_tool_schemas(
    tool_names: Iterable[str] | str | bytes | bytearray | None,
) -> list[dict[str, Any]]:
    """Build compact tool schemas with shortened descriptions."""

    return _build_compact_agent_tool_schemas_impl(tool_names)


def build_realtime_tool_schemas(
    tool_names: Iterable[str] | str | bytes | bytearray | None,
) -> list[dict[str, Any]]:
    """Build realtime-safe tool schemas for providers with reduced support."""

    return _build_realtime_tool_schemas_impl(tool_names)
