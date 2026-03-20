"""Delegate realtime tool callbacks to the shared tool executor."""

from __future__ import annotations


class TwinrRealtimeToolDelegatesMixin:
    """Expose the `handle_*` methods expected by realtime tool bindings."""

    def _handle_print_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_print_receipt(arguments)

    def _handle_end_conversation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_end_conversation(arguments)

    def _handle_schedule_reminder_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_schedule_reminder(arguments)

    def _handle_list_automations_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_list_automations(arguments)

    def _handle_create_time_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_create_time_automation(arguments)

    def _handle_create_sensor_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_create_sensor_automation(arguments)

    def _handle_update_time_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_update_time_automation(arguments)

    def _handle_update_sensor_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_update_sensor_automation(arguments)

    def _handle_delete_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_delete_automation(arguments)

    def _handle_propose_skill_learning_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_propose_skill_learning(arguments)

    def _handle_answer_skill_question_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_answer_skill_question(arguments)

    def _handle_confirm_skill_activation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_confirm_skill_activation(arguments)

    def _handle_rollback_skill_activation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_rollback_skill_activation(arguments)

    def _handle_remember_memory_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_remember_memory(arguments)

    def _handle_remember_contact_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_remember_contact(arguments)

    def _handle_lookup_contact_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_lookup_contact(arguments)

    def _handle_get_memory_conflicts_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_get_memory_conflicts(arguments)

    def _handle_resolve_memory_conflict_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_resolve_memory_conflict(arguments)

    def _handle_remember_preference_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_remember_preference(arguments)

    def _handle_remember_plan_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_remember_plan(arguments)

    def _handle_update_user_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_update_user_profile(arguments)

    def _handle_update_personality_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_update_personality(arguments)

    def _handle_configure_world_intelligence_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_configure_world_intelligence(arguments)

    def _handle_update_simple_setting_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_update_simple_setting(arguments)

    def _handle_enroll_voice_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_enroll_voice_profile(arguments)

    def _handle_get_voice_profile_status_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_get_voice_profile_status(arguments)

    def _handle_reset_voice_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_reset_voice_profile(arguments)

    def _handle_enroll_portrait_identity_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_enroll_portrait_identity(arguments)

    def _handle_get_portrait_identity_status_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_get_portrait_identity_status(arguments)

    def _handle_reset_portrait_identity_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_reset_portrait_identity(arguments)

    def _handle_search_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_search_live_info(arguments)

    def _handle_inspect_camera_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.tool_executor.handle_inspect_camera(arguments)
