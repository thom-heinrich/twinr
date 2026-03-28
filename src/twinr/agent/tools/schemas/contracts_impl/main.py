"""Compose Twinr tool schema families into canonical and adapted surfaces."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .automations import TOOL_BUILDERS as AUTOMATION_TOOL_BUILDERS
from .context import SchemaBuildContext, build_schema_context
from .general import EPILOGUE_TOOL_BUILDERS, PRIMARY_TOOL_BUILDERS
from .self_coding import TOOL_BUILDERS as SELF_CODING_TOOL_BUILDERS
from .shared import normalize_tool_names
from .smarthome import TOOL_BUILDERS as SMART_HOME_TOOL_BUILDERS
from .variants import build_realtime_tool_schema, compact_tool_schema

ToolBuilder = Callable[[SchemaBuildContext], dict[str, Any]]


def _load_family_builders() -> tuple[tuple[str, ToolBuilder], ...]:
    from .memory import TOOL_BUILDERS as MEMORY_TOOL_BUILDERS
    from .runtime_state import TOOL_BUILDERS as RUNTIME_STATE_TOOL_BUILDERS

    return (
        PRIMARY_TOOL_BUILDERS
        + AUTOMATION_TOOL_BUILDERS
        + SMART_HOME_TOOL_BUILDERS
        + SELF_CODING_TOOL_BUILDERS
        + MEMORY_TOOL_BUILDERS
        + RUNTIME_STATE_TOOL_BUILDERS
        + EPILOGUE_TOOL_BUILDERS
    )


def build_agent_tool_schemas(tool_names: Iterable[str] | str | bytes | bytearray | None) -> list[dict[str, Any]]:
    normalized_tool_names = normalize_tool_names(tool_names)
    available = set(normalized_tool_names)
    context = build_schema_context()
    return [builder(context) for name, builder in _load_family_builders() if name in available]


def build_compact_agent_tool_schemas(
    tool_names: Iterable[str] | str | bytes | bytearray | None,
) -> list[dict[str, Any]]:
    return [compact_tool_schema(schema) for schema in build_agent_tool_schemas(tool_names)]


def build_realtime_tool_schemas(
    tool_names: Iterable[str] | str | bytes | bytearray | None,
) -> list[dict[str, Any]]:
    return [build_realtime_tool_schema(schema) for schema in build_agent_tool_schemas(tool_names)]
