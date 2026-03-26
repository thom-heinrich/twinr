"""Normalize direct runtime-local handoff payloads for the streaming fast lane."""

from __future__ import annotations

from typing import Any

from twinr.agent.base_agent.contracts import (
    normalize_supervisor_decision_runtime_tool_arguments,
    normalize_supervisor_decision_runtime_tool_name,
)

from .loop_support import strip_text


def decision_runtime_local_tool_call(
    decision: Any | None,
) -> tuple[str, dict[str, object]] | None:
    """Extract one direct runtime-local tool call from a supervisor decision."""

    if decision is None:
        return None
    tool_name = normalize_supervisor_decision_runtime_tool_name(
        getattr(decision, "runtime_tool_name", None)
    )
    if tool_name is None:
        return None
    arguments = normalize_supervisor_decision_runtime_tool_arguments(
        getattr(decision, "runtime_tool_arguments", None)
    )
    return tool_name, dict(arguments or {})


def runtime_local_tool_reply_text(output: object) -> str:
    """Extract one short direct spoken reply from a runtime-local tool output."""

    if not isinstance(output, dict):
        return ""
    for key in ("spoken_text", "answer_text", "summary", "message"):
        text = strip_text(output.get(key))
        if text:
            return text
    return ""
