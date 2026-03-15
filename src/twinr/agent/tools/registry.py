from __future__ import annotations

from inspect import Parameter, signature  # AUDIT-FIX(#3): Validate handler signatures during binding.
from typing import Any, Callable

_REALTIME_TOOL_BINDINGS: tuple[tuple[str, str], ...] = (
    ("print_receipt", "handle_print_receipt"),
    ("search_live_info", "handle_search_live_info"),
    ("schedule_reminder", "handle_schedule_reminder"),
    ("list_automations", "handle_list_automations"),
    ("create_time_automation", "handle_create_time_automation"),
    ("create_sensor_automation", "handle_create_sensor_automation"),
    ("update_time_automation", "handle_update_time_automation"),
    ("update_sensor_automation", "handle_update_sensor_automation"),
    ("delete_automation", "handle_delete_automation"),
    ("remember_memory", "handle_remember_memory"),
    ("remember_contact", "handle_remember_contact"),
    ("lookup_contact", "handle_lookup_contact"),
    ("get_memory_conflicts", "handle_get_memory_conflicts"),
    ("resolve_memory_conflict", "handle_resolve_memory_conflict"),
    ("remember_preference", "handle_remember_preference"),
    ("remember_plan", "handle_remember_plan"),
    ("update_user_profile", "handle_update_user_profile"),
    ("update_personality", "handle_update_personality"),
    ("update_simple_setting", "handle_update_simple_setting"),
    ("enroll_voice_profile", "handle_enroll_voice_profile"),
    ("get_voice_profile_status", "handle_get_voice_profile_status"),
    ("reset_voice_profile", "handle_reset_voice_profile"),
    ("inspect_camera", "handle_inspect_camera"),
    ("end_conversation", "handle_end_conversation"),
)

RealtimeToolHandler = Callable[[dict[str, Any]], Any]


class RealtimeToolBindingError(RuntimeError):  # AUDIT-FIX(#1): Dedicated error type for deterministic bootstrap failures.
    """Raised when realtime tool bindings cannot be validated or bound safely."""


def _validate_realtime_tool_bindings() -> tuple[str, ...]:
    # AUDIT-FIX(#2): Fail fast on malformed or duplicate registry entries instead of silently shadowing tools.
    seen_tool_names: set[str] = set()
    tool_names: list[str] = []

    for index, binding in enumerate(_REALTIME_TOOL_BINDINGS):
        try:
            tool_name, attribute_name = binding
        except (TypeError, ValueError) as exc:
            raise RealtimeToolBindingError(
                f"Invalid realtime tool binding at index {index}: "
                f"expected (tool_name, attribute_name), got {binding!r}."
            ) from exc

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise RealtimeToolBindingError(
                f"Invalid realtime tool name at index {index}: {tool_name!r}."
            )

        if not isinstance(attribute_name, str) or not attribute_name.strip():
            raise RealtimeToolBindingError(
                f"Invalid handler attribute name for tool {tool_name!r}: {attribute_name!r}."
            )

        if tool_name in seen_tool_names:
            raise RealtimeToolBindingError(
                f"Duplicate realtime tool name {tool_name!r} at index {index}."
            )

        seen_tool_names.add(tool_name)
        tool_names.append(tool_name)

    return tuple(tool_names)


_REALTIME_TOOL_NAMES: tuple[str, ...] = _validate_realtime_tool_bindings()  # AUDIT-FIX(#2): Cache validated names once.


def realtime_tool_names() -> tuple[str, ...]:
    return _REALTIME_TOOL_NAMES


def _handler_accepts_payload(handler: Callable[..., Any]) -> bool:
    # AUDIT-FIX(#3): Reject handlers that cannot be called with the single payload object used by the dispatcher.
    try:
        parameters = tuple(signature(handler).parameters.values())
    except (TypeError, ValueError):
        return True

    required_positional = 0
    positional_capacity = 0
    required_keyword_only = 0
    accepts_varargs = False

    for parameter in parameters:
        if parameter.kind == Parameter.VAR_POSITIONAL:
            accepts_varargs = True
            continue

        if parameter.kind == Parameter.VAR_KEYWORD:
            continue

        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            positional_capacity += 1
            if parameter.default is Parameter.empty:
                required_positional += 1
            continue

        if parameter.kind == Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty:
            required_keyword_only += 1

    if required_keyword_only > 0:
        return False

    if required_positional > 1:
        return False

    if accepts_varargs:
        return True

    return positional_capacity >= 1


def bind_realtime_tool_handlers(handler_owner: object) -> dict[str, RealtimeToolHandler]:
    if handler_owner is None:
        raise RealtimeToolBindingError(
            "Cannot bind realtime tool handlers from None."
        )  # AUDIT-FIX(#1): Fail fast with a clear configuration error instead of an opaque AttributeError.

    bound_handlers: dict[str, RealtimeToolHandler] = {}
    binding_errors: list[str] = []
    owner_type_name = type(handler_owner).__name__

    for tool_name, attribute_name in _REALTIME_TOOL_BINDINGS:
        try:
            handler = getattr(handler_owner, attribute_name)
        except AttributeError:
            binding_errors.append(
                f"{tool_name!r}: missing attribute {attribute_name!r}"
            )  # AUDIT-FIX(#1): Report missing handlers deterministically.
            continue
        except Exception as exc:
            binding_errors.append(
                f"{tool_name!r}: failed to access {attribute_name!r} "
                f"({exc.__class__.__name__}: {exc})"
            )  # AUDIT-FIX(#1): Surface descriptor/property access faults cleanly.
            continue

        if not callable(handler):
            binding_errors.append(
                f"{tool_name!r}: attribute {attribute_name!r} is not callable"
            )  # AUDIT-FIX(#1): Prevent non-callables from entering the runtime tool map.
            continue

        if not _handler_accepts_payload(handler):
            binding_errors.append(
                f"{tool_name!r}: handler {attribute_name!r} cannot accept a single payload argument"
            )  # AUDIT-FIX(#3): Catch signature mismatch before first live invocation.
            continue

        bound_handlers[tool_name] = handler

    if binding_errors:
        raise RealtimeToolBindingError(
            f"Failed to bind realtime tool handlers for {owner_type_name}: "
            + "; ".join(binding_errors)
        )  # AUDIT-FIX(#1): Aggregate all registry defects into one actionable startup-fatal error.

    return bound_handlers