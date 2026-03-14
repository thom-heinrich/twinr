from __future__ import annotations

from typing import Any, Iterable

from twinr.agent.base_agent.simple_settings import (
    spoken_voice_options_context,
    supported_setting_names,
    supported_spoken_voices,
)
from twinr.automations import supported_sensor_trigger_kinds

_CANONICAL_ENGLISH_MEMORY_NOTE = (
    "All semantic text fields must be canonical English, even if the user spoke another language. "
    "Keep names, phone numbers, email addresses, IDs, codes, and exact quoted text verbatim."
)


def build_agent_tool_schemas(tool_names: Iterable[str]) -> list[dict[str, Any]]:
    available = set(tool_names)
    tools: list[dict[str, Any]] = []
    if "print_receipt" in available:
        tools.append(
            {
                "type": "function",
                "name": "print_receipt",
                "description": (
                    "Print short, user-facing content on the thermal receipt printer "
                    "when the user explicitly asks for a printout. "
                    "Use focus_hint to describe what from the recent context should be printed. "
                    "Optionally pass text when exact printable wording is already known."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_hint": {
                            "type": "string",
                            "description": "Short hint describing what from the recent conversation should be printed.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Optional exact text if the printable wording is already known.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "search_live_info" in available:
        tools.append(
            {
                "type": "function",
                "name": "search_live_info",
                "description": (
                    "Look up fresh or externally verifiable web information for the user. "
                    "Use this for broad web research, not only a fixed list of example domains."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact question to research on the web.",
                        },
                        "location_hint": {
                            "type": "string",
                            "description": "Optional location such as a city or district relevant to the search.",
                        },
                        "date_context": {
                            "type": "string",
                            "description": "Optional absolute date or time context if the user referred to relative dates.",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            }
        )
    if "schedule_reminder" in available:
        tools.append(
            {
                "type": "function",
                "name": "schedule_reminder",
                "description": (
                    "Schedule a future reminder or timer when the user asks to be reminded later or to set a timer. "
                    "Always send due_at as an absolute ISO 8601 local datetime with timezone offset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "due_at": {
                            "type": "string",
                            "description": "Absolute local due time in ISO 8601 format, for example 2026-03-14T12:00:00+01:00.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short summary of what Twinr should remind the user about.",
                        },
                        "details": {
                            "type": "string",
                            "description": "Optional extra detail to include when the reminder is spoken.",
                        },
                        "kind": {
                            "type": "string",
                            "description": "Short type such as reminder, timer, appointment, medication, task, or alarm.",
                        },
                        "original_request": {
                            "type": "string",
                            "description": "Optional short quote or paraphrase of the user's original reminder request.",
                        },
                    },
                    "required": ["due_at", "summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "list_automations" in available:
        tools.append(
            {
                "type": "function",
                "name": "list_automations",
                "description": (
                    "List the currently configured time-based and sensor-triggered automations so you can answer questions about them "
                    "or choose one to update or delete."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_disabled": {
                            "type": "boolean",
                            "description": "Set true if disabled automations should also be included.",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "create_time_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "create_time_automation",
                "description": (
                    "Create a time-based automation for one-off or recurring actions such as daily weather, "
                    "daily news, or printed headlines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short operator-friendly name for the automation."},
                        "description": {"type": "string", "description": "Optional short description of what the automation does."},
                        "schedule": {
                            "type": "string",
                            "enum": ["once", "daily", "weekly"],
                            "description": "Time schedule type.",
                        },
                        "due_at": {
                            "type": "string",
                            "description": "Absolute ISO 8601 local datetime with timezone offset for once schedules.",
                        },
                        "time_of_day": {
                            "type": "string",
                            "description": "Local time in HH:MM for daily or weekly schedules.",
                        },
                        "weekdays": {
                            "type": "array",
                            "description": "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6.",
                            "items": {"type": "integer"},
                        },
                        "delivery": {
                            "type": "string",
                            "enum": ["spoken", "printed"],
                            "description": "Whether the automation should speak or print when it runs.",
                        },
                        "content_mode": {
                            "type": "string",
                            "enum": ["llm_prompt", "static_text"],
                            "description": "Use llm_prompt for generated content or static_text for fixed wording.",
                        },
                        "content": {"type": "string", "description": "The prompt or static text the automation should use."},
                        "allow_web_search": {
                            "type": "boolean",
                            "description": "Set true when the automation needs fresh live information from the web.",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether the automation should be active immediately.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional short tags for operator organization.",
                        },
                        "timezone_name": {
                            "type": "string",
                            "description": "Optional timezone name. Use the local Twinr timezone unless there is a clear reason not to.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                        },
                    },
                    "required": ["name", "schedule", "delivery", "content_mode", "content"],
                    "additionalProperties": False,
                },
            }
        )
    if "create_sensor_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "create_sensor_automation",
                "description": (
                    "Create an automation triggered by PIR motion, camera visibility/object readings, "
                    "or background microphone/VAD state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short operator-friendly name for the automation."},
                        "description": {"type": "string", "description": "Optional short description of what the automation does."},
                        "trigger_kind": {
                            "type": "string",
                            "enum": list(supported_sensor_trigger_kinds()),
                            "description": "Supported sensor trigger type.",
                        },
                        "hold_seconds": {
                            "type": "number",
                            "description": "Optional required hold duration before firing. Required for quiet/no-motion triggers.",
                        },
                        "cooldown_seconds": {
                            "type": "number",
                            "description": "Optional cooldown after the automation fired.",
                        },
                        "delivery": {
                            "type": "string",
                            "enum": ["spoken", "printed"],
                            "description": "Whether the automation should speak or print when it runs.",
                        },
                        "content_mode": {
                            "type": "string",
                            "enum": ["llm_prompt", "static_text"],
                            "description": "Use llm_prompt for generated content or static_text for fixed wording.",
                        },
                        "content": {"type": "string", "description": "The prompt or static text the automation should use."},
                        "allow_web_search": {
                            "type": "boolean",
                            "description": "Set true when the automation needs fresh live information from the web.",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether the automation should be active immediately.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional short tags for operator organization.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                        },
                    },
                    "required": ["name", "trigger_kind", "delivery", "content_mode", "content"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_time_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_time_automation",
                "description": "Update an existing time-based automation. Use list_automations first if you need to identify it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "automation_ref": {"type": "string", "description": "Automation id or a clear automation name."},
                        "name": {"type": "string", "description": "Optional new automation name."},
                        "description": {"type": "string", "description": "Optional new description."},
                        "schedule": {
                            "type": "string",
                            "enum": ["once", "daily", "weekly"],
                            "description": "Optional new time schedule type.",
                        },
                        "due_at": {
                            "type": "string",
                            "description": "Absolute ISO 8601 local datetime with timezone offset for once schedules.",
                        },
                        "time_of_day": {
                            "type": "string",
                            "description": "Local time in HH:MM for daily or weekly schedules.",
                        },
                        "weekdays": {
                            "type": "array",
                            "description": "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6.",
                            "items": {"type": "integer"},
                        },
                        "delivery": {
                            "type": "string",
                            "enum": ["spoken", "printed"],
                            "description": "Optional new delivery mode.",
                        },
                        "content_mode": {
                            "type": "string",
                            "enum": ["llm_prompt", "static_text"],
                            "description": "Optional new content mode.",
                        },
                        "content": {"type": "string", "description": "Optional new prompt or static text."},
                        "allow_web_search": {
                            "type": "boolean",
                            "description": "Optional new live-search flag for llm_prompt content.",
                        },
                        "enabled": {"type": "boolean", "description": "Optional enabled toggle."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional full replacement tag list.",
                        },
                        "timezone_name": {"type": "string", "description": "Optional new timezone name."},
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                        },
                    },
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_sensor_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_sensor_automation",
                "description": (
                    "Update an existing supported sensor-triggered automation. Use list_automations first if you need to identify it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "automation_ref": {"type": "string", "description": "Automation id or a clear automation name."},
                        "name": {"type": "string", "description": "Optional new automation name."},
                        "description": {"type": "string", "description": "Optional new description."},
                        "trigger_kind": {
                            "type": "string",
                            "enum": list(supported_sensor_trigger_kinds()),
                            "description": "Optional new supported sensor trigger type.",
                        },
                        "hold_seconds": {"type": "number", "description": "Optional hold duration before firing."},
                        "cooldown_seconds": {"type": "number", "description": "Optional cooldown after the automation fired."},
                        "delivery": {
                            "type": "string",
                            "enum": ["spoken", "printed"],
                            "description": "Optional new delivery mode.",
                        },
                        "content_mode": {
                            "type": "string",
                            "enum": ["llm_prompt", "static_text"],
                            "description": "Optional new content mode.",
                        },
                        "content": {"type": "string", "description": "Optional new prompt or static text."},
                        "allow_web_search": {
                            "type": "boolean",
                            "description": "Optional new live-search flag for llm_prompt content.",
                        },
                        "enabled": {"type": "boolean", "description": "Optional enabled toggle."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional full replacement tag list.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                        },
                    },
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "delete_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "delete_automation",
                "description": "Delete an existing scheduled automation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "automation_ref": {"type": "string", "description": "Automation id or a clear automation name."},
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the deletion when extra confirmation is needed.",
                        },
                    },
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_memory" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_memory",
                "description": (
                    "Store an important memory for future turns when the user explicitly asks you to remember something. "
                    "Use only for clear remember/save requests, not for ordinary conversation. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": "Short type such as appointment, contact, reminder, preference, fact, or task.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short factual summary of what should be remembered.",
                        },
                        "details": {
                            "type": "string",
                            "description": "Optional extra detail that helps later recall.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed.",
                        },
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_contact" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_contact",
                "description": (
                    "Store or refine a remembered contact in Twinr's structured graph memory. "
                    "Use this when the user explicitly wants Twinr to remember a person with a phone number, email, relation, or role. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "given_name": {"type": "string", "description": "First name or main short name of the contact."},
                        "family_name": {"type": "string", "description": "Optional family name if known."},
                        "phone": {"type": "string", "description": "Optional phone number if the user gave one."},
                        "email": {"type": "string", "description": "Optional email if the user gave one."},
                        "role": {
                            "type": "string",
                            "description": "Optional role such as physiotherapist, daughter, neighbor, or friend.",
                        },
                        "relation": {
                            "type": "string",
                            "description": "Optional relationship wording such as daughter, family, or helper.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional short detail that helps future disambiguation.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed.",
                        },
                    },
                    "required": ["given_name"],
                    "additionalProperties": False,
                },
            }
        )
    if "lookup_contact" in available:
        tools.append(
            {
                "type": "function",
                "name": "lookup_contact",
                "description": (
                    "Look up a remembered contact and return the stored phone number or email, or ask for clarification when multiple matches exist."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name or short name of the contact to look up."},
                        "family_name": {"type": "string", "description": "Optional family name if the user gave one."},
                        "role": {
                            "type": "string",
                            "description": "Optional role such as physiotherapist, daughter, or neighbor.",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_preference" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_preference",
                "description": (
                    "Store a stable personal preference in Twinr's structured graph memory, such as a preferred brand, favorite shop, or disliked food. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Short category such as brand, store, food, drink, activity, music, or thing.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The preferred or disliked thing, place, or brand.",
                        },
                        "for_product": {"type": "string", "description": "Optional product context, for example coffee."},
                        "sentiment": {
                            "type": "string",
                            "description": "Use prefer, like, dislike, or usually_buy_at.",
                        },
                        "details": {"type": "string", "description": "Optional short detail for later recall."},
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed.",
                        },
                    },
                    "required": ["category", "value"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_plan" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_plan",
                "description": (
                    "Store a short future intention or plan in Twinr's structured graph memory, such as wanting to go for a walk today. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Short plan summary."},
                        "when": {
                            "type": "string",
                            "description": "Optional time wording such as today, tomorrow, 2026-03-14, or next Monday.",
                        },
                        "details": {"type": "string", "description": "Optional short detail for later recall."},
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed.",
                        },
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_user_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_user_profile",
                "description": (
                    "Update stable user profile or preference context for future turns when the user explicitly asks you to remember it. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Short category such as preferred_name, location, preference, contact, or routine.",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Short, durable instruction or fact to store in the user profile.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent profile change when extra confirmation is needed.",
                        },
                    },
                    "required": ["category", "instruction"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_personality" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_personality",
                "description": (
                    "Update how Twinr should speak or behave in future turns when the user explicitly asks for a behavior change. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Short category such as response_style, humor, language, verbosity, or greeting_style.",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Short future-behavior instruction to store in Twinr personality context.",
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent behavior change when extra confirmation is needed.",
                        },
                    },
                    "required": ["category", "instruction"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_simple_setting" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_simple_setting",
                "description": (
                    "Adjust one of Twinr's small bounded runtime settings after an explicit user request. "
                    "Use memory_capacity when the user wants Twinr to remember more or less recent conversation. "
                    "Use spoken_voice when the user wants a different voice. "
                    "Use speech_speed when the user wants Twinr to speak slower or faster."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "setting": {
                            "type": "string",
                            "enum": list(supported_setting_names()),
                            "description": (
                                "Supported setting. memory_capacity changes how much recent conversation Twinr keeps. "
                                f"spoken_voice changes the spoken voice and must be set to one of: {', '.join(supported_spoken_voices())}. "
                                "It may also use a short descriptive style request such as male, female, neutral, warm, soft, deep, bright, or firm; Twinr will map that to the closest supported voice. "
                                "speech_speed changes the overall speaking speed for both normal TTS and realtime speech. "
                                "speech_pause_ms changes how long Twinr waits for a short pause before stopping recording. "
                                "follow_up_timeout_s changes how long the hands-free follow-up listening window stays open."
                            ),
                        },
                        "action": {
                            "type": "string",
                            "enum": ["increase", "decrease", "set"],
                            "description": "Use increase/decrease for relative requests and set when the user gave a concrete value.",
                        },
                        "value": {
                            "anyOf": [{"type": "number"}, {"type": "string"}],
                            "description": (
                                "Optional concrete value for action=set. "
                                "For memory_capacity use levels 1 to 4. "
                                f"For spoken_voice use one of: {spoken_voice_options_context()}. "
                                "Descriptive values such as male, female, neutral, warm, soft, deep, bright, or firm are also accepted and will be mapped to the closest supported voice. "
                                "For speech_speed use a factor between 0.75 and 1.15. "
                                "For speech_pause_ms use milliseconds. "
                                "For follow_up_timeout_s use seconds."
                            ),
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed this persistent setting change when extra confirmation is needed.",
                        },
                    },
                    "required": ["setting", "action"],
                    "additionalProperties": False,
                },
            }
        )
    if "enroll_voice_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "enroll_voice_profile",
                "description": (
                    "Create or refresh the local Twinr voice profile from the current spoken turn. "
                    "Use only when the user explicitly asks Twinr to learn or update their voice profile."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed a replacement when extra confirmation is needed.",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "get_voice_profile_status" in available:
        tools.append(
            {
                "type": "function",
                "name": "get_voice_profile_status",
                "description": "Read the local voice-profile status and current live speaker signal.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "reset_voice_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "reset_voice_profile",
                "description": "Delete the local Twinr voice profile when the user explicitly asks to remove it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": {
                            "type": "boolean",
                            "description": "Set true only after the user clearly confirmed the reset when extra confirmation is needed.",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "end_conversation" in available:
        tools.append(
            {
                "type": "function",
                "name": "end_conversation",
                "description": (
                    "End the current follow-up listening loop when the user clearly indicates they are done for now, "
                    "for example by saying thanks, stop, pause, bye, or tschuss."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Optional short note describing why the conversation should end.",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "inspect_camera" in available:
        tools.append(
            {
                "type": "function",
                "name": "inspect_camera",
                "description": (
                    "Inspect the current live camera view when the user asks you to look at them, "
                    "an object, a document, or something they are showing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact user request about what should be inspected in the camera view.",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


def build_realtime_tool_schemas(tool_names: Iterable[str]) -> list[dict[str, Any]]:
    return build_agent_tool_schemas(tool_names)
