"""Dispatch runtime tool calls to the concrete handler modules.

``RealtimeToolExecutor`` turns a workflow owner object into the ``handle_*``
method surface expected by the realtime tool registry. The methods here stay
intentionally thin so handler-side validation, normalization, and side effects
remain in ``twinr.agent.tools.handlers``.
"""

from __future__ import annotations

from typing import Any

from ..handlers import automations, household_identity, intelligence, memory, output, portrait_identity, reminders, self_coding, settings, smarthome, voice_profile


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

    def handle_print_receipt(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_print_receipt(self.owner, arguments)

    def handle_search_live_info(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_search_live_info(self.owner, arguments)

    def handle_schedule_reminder(self, arguments: dict[str, object]) -> dict[str, object]:
        return reminders.handle_schedule_reminder(self.owner, arguments)

    def handle_list_automations(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_list_automations(self.owner, arguments)

    def handle_create_time_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_create_time_automation(self.owner, arguments)

    def handle_create_sensor_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_create_sensor_automation(self.owner, arguments)

    def handle_update_time_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_update_time_automation(self.owner, arguments)

    def handle_update_sensor_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_update_sensor_automation(self.owner, arguments)

    def handle_delete_automation(self, arguments: dict[str, object]) -> dict[str, object]:
        return automations.handle_delete_automation(self.owner, arguments)

    def handle_list_smart_home_entities(self, arguments: dict[str, object]) -> dict[str, object]:
        return smarthome.handle_list_smart_home_entities(self.owner, arguments)

    def handle_read_smart_home_state(self, arguments: dict[str, object]) -> dict[str, object]:
        return smarthome.handle_read_smart_home_state(self.owner, arguments)

    def handle_control_smart_home_entities(self, arguments: dict[str, object]) -> dict[str, object]:
        return smarthome.handle_control_smart_home_entities(self.owner, arguments)

    def handle_read_smart_home_sensor_stream(self, arguments: dict[str, object]) -> dict[str, object]:
        return smarthome.handle_read_smart_home_sensor_stream(self.owner, arguments)

    def handle_propose_skill_learning(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_propose_skill_learning(self.owner, arguments)

    def handle_answer_skill_question(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_answer_skill_question(self.owner, arguments)

    def handle_confirm_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_confirm_skill_activation(self.owner, arguments)

    def handle_rollback_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_rollback_skill_activation(self.owner, arguments)

    def handle_pause_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_pause_skill_activation(self.owner, arguments)

    def handle_reactivate_skill_activation(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_reactivate_skill_activation(self.owner, arguments)

    def handle_run_self_coding_skill_scheduled(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_run_self_coding_skill_scheduled(self.owner, arguments)

    def handle_run_self_coding_skill_sensor(self, arguments: dict[str, object]) -> dict[str, object]:
        return self_coding.handle_run_self_coding_skill_sensor(self.owner, arguments)

    def handle_remember_memory(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_memory(self.owner, arguments)

    def handle_remember_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_contact(self.owner, arguments)

    def handle_lookup_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_lookup_contact(self.owner, arguments)

    def handle_get_memory_conflicts(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_get_memory_conflicts(self.owner, arguments)

    def handle_resolve_memory_conflict(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_resolve_memory_conflict(self.owner, arguments)

    def handle_remember_preference(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_preference(self.owner, arguments)

    def handle_remember_plan(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_plan(self.owner, arguments)

    def handle_update_user_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_update_user_profile(self.owner, arguments)

    def handle_update_personality(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_update_personality(self.owner, arguments)

    def handle_configure_world_intelligence(self, arguments: dict[str, object]) -> dict[str, object]:
        return intelligence.handle_configure_world_intelligence(self.owner, arguments)

    def handle_update_simple_setting(self, arguments: dict[str, object]) -> dict[str, object]:
        return settings.handle_update_simple_setting(self.owner, arguments)

    def handle_enroll_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_enroll_voice_profile(self.owner, arguments)

    def handle_get_voice_profile_status(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_get_voice_profile_status(self.owner, arguments)

    def handle_reset_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_reset_voice_profile(self.owner, arguments)

    def handle_enroll_portrait_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return portrait_identity.handle_enroll_portrait_identity(self.owner, arguments)

    def handle_get_portrait_identity_status(self, arguments: dict[str, object]) -> dict[str, object]:
        return portrait_identity.handle_get_portrait_identity_status(self.owner, arguments)

    def handle_reset_portrait_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return portrait_identity.handle_reset_portrait_identity(self.owner, arguments)

    def handle_manage_household_identity(self, arguments: dict[str, object]) -> dict[str, object]:
        return household_identity.handle_manage_household_identity(self.owner, arguments)

    def handle_inspect_camera(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_inspect_camera(self.owner, arguments)

    def handle_end_conversation(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_end_conversation(self.owner, arguments)
