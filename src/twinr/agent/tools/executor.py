from __future__ import annotations

from typing import Any

from . import automations, memory, output, reminders, settings, voice_profile


class RealtimeToolExecutor:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

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

    def handle_remember_memory(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_memory(self.owner, arguments)

    def handle_remember_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_contact(self.owner, arguments)

    def handle_lookup_contact(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_lookup_contact(self.owner, arguments)

    def handle_remember_preference(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_preference(self.owner, arguments)

    def handle_remember_plan(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_remember_plan(self.owner, arguments)

    def handle_update_user_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_update_user_profile(self.owner, arguments)

    def handle_update_personality(self, arguments: dict[str, object]) -> dict[str, object]:
        return memory.handle_update_personality(self.owner, arguments)

    def handle_update_simple_setting(self, arguments: dict[str, object]) -> dict[str, object]:
        return settings.handle_update_simple_setting(self.owner, arguments)

    def handle_enroll_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_enroll_voice_profile(self.owner, arguments)

    def handle_get_voice_profile_status(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_get_voice_profile_status(self.owner, arguments)

    def handle_reset_voice_profile(self, arguments: dict[str, object]) -> dict[str, object]:
        return voice_profile.handle_reset_voice_profile(self.owner, arguments)

    def handle_inspect_camera(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_inspect_camera(self.owner, arguments)

    def handle_end_conversation(self, arguments: dict[str, object]) -> dict[str, object]:
        return output.handle_end_conversation(self.owner, arguments)
