"""Dispatch runtime tool calls to the concrete handler modules.

``RealtimeToolExecutor`` turns a workflow owner object into the ``handle_*``
method surface expected by the realtime tool registry. The methods here stay
intentionally thin so handler-side validation, normalization, and side effects
remain in ``twinr.agent.tools.handlers``.
"""

from __future__ import annotations

from typing import Any

from ..handlers import (
    automations,
    browser,
    household_identity,
    intelligence,
    memory,
    output,
    portrait_identity,
    reminders,
    self_coding,
    service_connect,
    settings,
    smarthome,
    user_discovery,
    voice_quiet,
    voice_profile,
    whatsapp,
)
from .observability import observe_realtime_tool_call


class RealtimeToolExecutor:
    """Expose bound realtime tool handlers for a workflow owner.

    The owner is typically a runner or session object that provides the runtime
    methods and state required by the concrete handler modules. Each
    ``handle_*`` method delegates to exactly one handler function and returns
    the JSON-safe payload that the tool loop forwards back to the model.

    Attributes:
        owner: Runtime owner object passed through to handler modules.
    """

    def __init__(self, owner: Any) -> None:
        """Store the runtime owner used by all delegated handlers."""
        self.owner = owner
        self_coding.ensure_self_coding_runtime(self.owner)

    def _invoke(
        self,
        tool_name: str,
        handler,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        """Run one tool handler behind the shared observability seam."""

        return observe_realtime_tool_call(
            self.owner,
            tool_name=tool_name,
            handler=handler,
            arguments=arguments,
        )

    def handle_print_receipt(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("print_receipt", output.handle_print_receipt, arguments)

    def handle_search_live_info(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("search_live_info", output.handle_search_live_info, arguments)

    def handle_browser_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("browser_automation", browser.handle_browser_automation, arguments)

    def handle_connect_service_integration(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("connect_service_integration", service_connect.handle_connect_service_integration, arguments)

    def handle_schedule_reminder(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("schedule_reminder", reminders.handle_schedule_reminder, arguments)

    def handle_list_automations(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("list_automations", automations.handle_list_automations, arguments)

    def handle_create_time_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("create_time_automation", automations.handle_create_time_automation, arguments)

    def handle_create_sensor_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("create_sensor_automation", automations.handle_create_sensor_automation, arguments)

    def handle_update_time_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("update_time_automation", automations.handle_update_time_automation, arguments)

    def handle_update_sensor_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("update_sensor_automation", automations.handle_update_sensor_automation, arguments)

    def handle_delete_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("delete_automation", automations.handle_delete_automation, arguments)

    def handle_list_smart_home_entities(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("list_smart_home_entities", smarthome.handle_list_smart_home_entities, arguments)

    def handle_read_smart_home_state(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("read_smart_home_state", smarthome.handle_read_smart_home_state, arguments)

    def handle_control_smart_home_entities(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("control_smart_home_entities", smarthome.handle_control_smart_home_entities, arguments)

    def handle_read_smart_home_sensor_stream(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("read_smart_home_sensor_stream", smarthome.handle_read_smart_home_sensor_stream, arguments)

    def handle_propose_skill_learning(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("propose_skill_learning", self_coding.handle_propose_skill_learning, arguments)

    def handle_answer_skill_question(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("answer_skill_question", self_coding.handle_answer_skill_question, arguments)

    def handle_confirm_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("confirm_skill_activation", self_coding.handle_confirm_skill_activation, arguments)

    def handle_rollback_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("rollback_skill_activation", self_coding.handle_rollback_skill_activation, arguments)

    def handle_pause_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("pause_skill_activation", self_coding.handle_pause_skill_activation, arguments)

    def handle_reactivate_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("reactivate_skill_activation", self_coding.handle_reactivate_skill_activation, arguments)

    def handle_run_self_coding_skill_scheduled(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("run_self_coding_skill_scheduled", self_coding.handle_run_self_coding_skill_scheduled, arguments)

    def handle_run_self_coding_skill_sensor(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("run_self_coding_skill_sensor", self_coding.handle_run_self_coding_skill_sensor, arguments)

    def handle_remember_memory(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("remember_memory", memory.handle_remember_memory, arguments)

    def handle_remember_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("remember_contact", memory.handle_remember_contact, arguments)

    def handle_lookup_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("lookup_contact", memory.handle_lookup_contact, arguments)

    def handle_send_whatsapp_message(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("send_whatsapp_message", whatsapp.handle_send_whatsapp_message, arguments)

    def handle_get_memory_conflicts(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("get_memory_conflicts", memory.handle_get_memory_conflicts, arguments)

    def handle_resolve_memory_conflict(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("resolve_memory_conflict", memory.handle_resolve_memory_conflict, arguments)

    def handle_remember_preference(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("remember_preference", memory.handle_remember_preference, arguments)

    def handle_remember_plan(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("remember_plan", memory.handle_remember_plan, arguments)

    def handle_update_user_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("update_user_profile", memory.handle_update_user_profile, arguments)

    def handle_update_personality(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("update_personality", memory.handle_update_personality, arguments)

    def handle_manage_user_discovery(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("manage_user_discovery", user_discovery.handle_manage_user_discovery, arguments)

    def handle_configure_world_intelligence(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("configure_world_intelligence", intelligence.handle_configure_world_intelligence, arguments)

    def handle_update_simple_setting(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("update_simple_setting", settings.handle_update_simple_setting, arguments)

    def handle_manage_voice_quiet_mode(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("manage_voice_quiet_mode", voice_quiet.handle_manage_voice_quiet_mode, arguments)

    def handle_enroll_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("enroll_voice_profile", voice_profile.handle_enroll_voice_profile, arguments)

    def handle_get_voice_profile_status(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("get_voice_profile_status", voice_profile.handle_get_voice_profile_status, arguments)

    def handle_reset_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("reset_voice_profile", voice_profile.handle_reset_voice_profile, arguments)

    def handle_enroll_portrait_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("enroll_portrait_identity", portrait_identity.handle_enroll_portrait_identity, arguments)

    def handle_get_portrait_identity_status(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("get_portrait_identity_status", portrait_identity.handle_get_portrait_identity_status, arguments)

    def handle_reset_portrait_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("reset_portrait_identity", portrait_identity.handle_reset_portrait_identity, arguments)

    def handle_manage_household_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("manage_household_identity", household_identity.handle_manage_household_identity, arguments)

    def handle_inspect_camera(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("inspect_camera", output.handle_inspect_camera, arguments)

    def handle_end_conversation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._invoke("end_conversation", output.handle_end_conversation, arguments)
