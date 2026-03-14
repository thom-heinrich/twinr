from __future__ import annotations

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


def realtime_tool_names() -> tuple[str, ...]:
    return tuple(name for name, _ in _REALTIME_TOOL_BINDINGS)


def bind_realtime_tool_handlers(handler_owner: object) -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        name: getattr(handler_owner, attribute_name)
        for name, attribute_name in _REALTIME_TOOL_BINDINGS
    }
