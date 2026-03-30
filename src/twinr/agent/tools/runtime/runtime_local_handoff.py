"""Normalize direct runtime-local handoff payloads for the streaming fast lane."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinr.agent.base_agent.contracts import (
    normalize_supervisor_decision_runtime_tool_arguments,
    normalize_supervisor_decision_runtime_tool_name,
)

from .loop_support import strip_text


def _whole_positive_minutes(value: object | None) -> bool:
    """Return whether the value encodes a positive whole-minute duration."""

    if value is None or value == "" or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return value.is_integer() and value > 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            return int(stripped) > 0
        except ValueError:
            return False
    return False


def _runtime_local_tool_arguments_complete(
    tool_name: str,
    arguments: Mapping[str, object],
) -> bool:
    """Return whether the shortcut payload is complete enough for direct execution."""

    if tool_name != "manage_voice_quiet_mode":
        return True
    action = str(arguments.get("action", "")).strip().lower()
    if action not in {"set", "clear", "status"}:
        return False
    if action != "set":
        return True
    return _whole_positive_minutes(arguments.get("duration_minutes"))


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
    normalized_arguments = dict(arguments or {})
    if not _runtime_local_tool_arguments_complete(tool_name, normalized_arguments):
        return None
    return tool_name, normalized_arguments


def has_executable_runtime_local_tool_call(decision: Any | None) -> bool:
    """Return whether the decision carries one complete direct runtime-local tool call."""

    try:
        return decision_runtime_local_tool_call(decision) is not None
    except (TypeError, ValueError):
        return False


def runtime_local_tool_reply_text(output: object) -> str:
    """Extract one short direct spoken reply from a runtime-local tool output."""

    if not isinstance(output, dict):
        return ""
    for key in ("spoken_text", "answer_text", "summary", "message"):
        text = strip_text(output.get(key))
        if text:
            return text
    return ""
