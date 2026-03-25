"""Own dual-lane handoff schema, normalization, and merge helpers."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .loop_support import strip_text


logger = logging.getLogger(__name__)
_ALLOWED_HANDOFF_KINDS = frozenset({"general", "search", "memory", "automation"})


_HANDOFF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "handoff_specialist_worker",
    "description": (
        "Use the slower specialist worker when the answer needs fresh web research, more synthesis, "
        "or a deeper multi-step tool pass than the fast supervisor should handle directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["general", "search", "memory", "automation"],
                "description": (
                    "Short handoff category that best describes why the specialist is needed. "
                    "Use search only for external/web research. "
                    "Use automation for the user's own smart-home inventory, room/device state, recent in-home smart-home events, and automation work."
                ),
            },
            "goal": {
                "type": "string",
                "description": "Short description of what the specialist should achieve for this turn.",
            },
            "spoken_ack": {
                "type": ["string", "null"],
                "description": (
                    "Optional short user-facing acknowledgement in the configured language that can be spoken "
                    "immediately before the specialist work starts. Use null when no immediate progress line "
                    "should be spoken."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "Optional rewritten task for the specialist worker. Omit to reuse the original user prompt.",
            },
            "allow_web_search": {
                "type": "boolean",
                "description": (
                    "Set true only when the specialist truly needs live web search. "
                    "Keep this false for the user's own smart-home inventory, room/device state, or recent in-home smart-home events."
                ),
            },
            "location_hint": {
                "type": "string",
                "description": (
                    "Optional explicit place already named by the user, for example a city, district, or street. "
                    "Use this for search handoffs when the target location matters."
                ),
            },
            "date_context": {
                "type": "string",
                "description": (
                    "Optional absolute date or local date/time context for search handoffs when the user referred "
                    "to relative dates such as today or tomorrow."
                ),
            },
        },
        "required": ["kind", "goal"],
        "additionalProperties": False,
    },
}


def normalize_decision_action(value: Any) -> str:
    """Normalize the supervisor action into one of Twinr's supported modes."""

    normalized = strip_text(value).lower()
    if normalized in {"direct", "end_conversation"}:
        return normalized
    return "handoff"


def normalize_handoff_kind(value: Any) -> str:
    """Normalize the handoff kind to the supported specialist categories."""

    normalized = strip_text(value).lower()
    if normalized in _ALLOWED_HANDOFF_KINDS:
        return normalized
    return "general"


def normalize_spoken_ack(arguments: dict[str, Any]) -> str:
    """Return the spoken acknowledgement exactly as provided by the supervisor."""

    return strip_text(arguments.get("spoken_ack"))


def normalize_handoff_arguments(
    arguments: dict[str, Any],
    *,
    fallback_prompt: str,
) -> dict[str, Any]:
    """Normalize a supervisor handoff payload into the canonical shape."""

    normalized: dict[str, Any] = {
        "kind": normalize_handoff_kind(arguments.get("kind")),
        "goal": strip_text(arguments.get("goal")) or fallback_prompt,
        "spoken_ack": normalize_spoken_ack(arguments),
    }
    prompt = strip_text(arguments.get("prompt"))
    if prompt:
        normalized["prompt"] = prompt
    if "allow_web_search" in arguments:
        normalized["allow_web_search"] = arguments.get("allow_web_search")
    location_hint = strip_text(arguments.get("location_hint"))
    if location_hint:
        normalized["location_hint"] = location_hint
    date_context = strip_text(arguments.get("date_context"))
    if date_context:
        normalized["date_context"] = date_context
    return normalized


def handoff_allow_web_search(arguments: dict[str, Any], default: bool | None) -> bool | None:
    """Resolve the handoff web-search flag from booleans or common string forms."""

    if "allow_web_search" not in arguments:
        return default
    value = arguments.get("allow_web_search")
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = strip_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def specialist_handoff_context(arguments: dict[str, Any]) -> str:
    """Build a short instruction suffix that explains the handoff intent."""

    goal = strip_text(arguments.get("goal"))
    kind = normalize_handoff_kind(arguments.get("kind"))
    prompt = strip_text(arguments.get("prompt"))
    context_parts = [f"Specialist handoff kind: {kind}."]
    if goal:
        context_parts.append(f"Specialist goal: {goal}")
    if prompt:
        context_parts.append(f"Specialist rewritten prompt: {prompt}")
    return " ".join(context_parts)


def decision_fallback_handoff(
    decision: Any,
    *,
    prompt: str,
) -> dict[str, Any]:
    """Convert a full-context direct decision into a safe handoff payload."""

    kind = normalize_handoff_kind(getattr(decision, "kind", None))
    if kind == "general":
        kind = "memory"
    return {
        "kind": kind,
        "goal": strip_text(getattr(decision, "goal", None)) or prompt,
        "spoken_ack": strip_text(getattr(decision, "spoken_ack", None)),
        "prompt": strip_text(getattr(decision, "prompt", None)),
        "allow_web_search": getattr(decision, "allow_web_search", None),
        "location_hint": strip_text(getattr(decision, "location_hint", None)),
        "date_context": strip_text(getattr(decision, "date_context", None)),
        "response_id": getattr(decision, "response_id", None),
        "request_id": getattr(decision, "request_id", None),
        "model": getattr(decision, "model", None),
        "token_usage": getattr(decision, "token_usage", None),
    }


def supervisor_decision_has_required_user_reply(decision: Any) -> bool:
    """Return whether the decision satisfies the user-facing reply contract."""

    action = normalize_decision_action(getattr(decision, "action", None))
    if action == "handoff":
        return True
    return bool(strip_text(getattr(decision, "spoken_reply", None)))


def merge_tool_schemas(
    primary: Sequence[dict[str, Any]],
    extra: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Merge tool schemas while de-duplicating by declared tool name."""

    merged: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for schema in (*primary, *extra):
        name = tool_schema_name(schema)
        if name is not None:
            if name in seen_names:
                continue
            seen_names.add(name)
        merged.append(schema)
    return tuple(merged)


def tool_schema_name(schema: dict[str, Any]) -> str | None:
    """Extract a tool name from either accepted schema format."""

    name = schema.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function_schema = schema.get("function")
    if isinstance(function_schema, dict):
        function_name = function_schema.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def merge_handoff_items(
    supervisor_items: Sequence[Any],
    specialist_records: Sequence[Any],
    attribute: str,
) -> list[Any]:
    """Interleave specialist items immediately after their handoff marker."""

    merged: list[Any] = []
    specialist_index = 0
    for item in supervisor_items:
        merged.append(item)
        if getattr(item, "name", None) == "handoff_specialist_worker":
            if specialist_index < len(specialist_records):
                result = getattr(specialist_records[specialist_index], "result", None)
                if result is not None:
                    merged.extend(getattr(result, attribute))
                else:
                    logger.warning(
                        "Specialist record %d has no result payload for %s.",
                        specialist_index,
                        attribute,
                    )
            else:
                logger.warning(
                    "Missing specialist record for supervisor handoff item %d.",
                    specialist_index,
                )
            specialist_index += 1

    if specialist_index < len(specialist_records):
        logger.warning(
            "Found %d specialist record(s) without a matching supervisor handoff item.",
            len(specialist_records) - specialist_index,
        )
        for record in specialist_records[specialist_index:]:
            result = getattr(record, "result", None)
            if result is not None:
                merged.extend(getattr(result, attribute))
    return merged
