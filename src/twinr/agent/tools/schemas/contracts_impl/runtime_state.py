"""Runtime-state, settings, identity, and world-intelligence schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import array_property, boolean_property, number_property, simple_setting_rules, string_property


def build_configure_world_intelligence_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "configure_world_intelligence",
        "description": (
            "Manage Twinr's ongoing RSS or Atom sources for calm place and world awareness. "
            "Use this to list, subscribe, discover, deactivate, or force-refresh feed subscriptions. "
            "Do not use it for ordinary one-off live questions; use it only for installer setup, explicit source changes, "
            "or occasional recalibration of Twinr's ongoing world-intelligence sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": string_property(
                    "Which world-intelligence action to run.",
                    enum=["list", "subscribe", "discover", "deactivate", "refresh_now"],
                ),
                "query": string_property(
                    "Optional discovery query that asks the live web backend to find source pages exposing RSS or Atom feeds.",
                    min_length=1,
                ),
                "label": string_property(
                    "Optional short label such as Hamburg local politics or Germany energy policy.",
                    min_length=1,
                ),
                "location_hint": string_property(
                    "Optional city, district, or region that should guide feed discovery or labeling.",
                    min_length=1,
                ),
                "region": string_property(
                    "Optional region name that should be stored with the subscription and later world signals.",
                    min_length=1,
                ),
                "topics": array_property(
                    "Optional recurring topics this subscription should cover.",
                    string_property("Topic label.", min_length=1),
                    unique_items=True,
                ),
                "feed_urls": array_property(
                    "Optional explicit RSS or Atom feed URLs to subscribe or deactivate.",
                    string_property("Feed URL.", min_length=1),
                    unique_items=True,
                ),
                "subscription_refs": array_property(
                    "Optional subscription ids to deactivate.",
                    string_property("Subscription id.", min_length=1),
                    unique_items=True,
                ),
                "scope": string_property(
                    "Optional world-awareness scope for new subscriptions.",
                    enum=["local", "regional", "national", "global", "topic"],
                ),
                "priority": number_property(
                    "Optional salience/priority weight for the subscription between 0 and 1.",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "refresh_interval_hours": number_property(
                    "Optional refresh cadence in hours. Keep this calm and infrequent; not less than 24.",
                    minimum=24.0,
                ),
                "auto_subscribe": boolean_property(
                    "For discover, set true if discovered feed URLs should be persisted immediately."
                ),
                "refresh_after_change": boolean_property(
                    "Set true if Twinr should refresh the subscribed feeds immediately after the change."
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent source change when extra confirmation is needed."
                ),
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }


def build_update_simple_setting_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_simple_setting",
        "description": (
            "Adjust one of Twinr's small bounded runtime settings after an explicit user request. "
            "Use memory_capacity when the user wants Twinr to remember more or less recent conversation. "
            "Use spoken_voice when the user wants a different voice and resolve descriptive requests to a supported Twinr voice name before calling the tool. "
            "Use speech_speed when the user wants Twinr to speak slower or faster."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "setting": string_property(
                    "Supported setting. memory_capacity changes how much recent conversation Twinr keeps. "
                    f"spoken_voice changes the spoken voice and must be set to one of: {', '.join(context.spoken_voices)}. "
                    "speech_speed changes the overall speaking speed for both normal TTS and realtime speech. "
                    "speech_pause_ms changes how long Twinr waits for a short pause before stopping recording. "
                    "follow_up_timeout_s changes how long the hands-free follow-up listening window stays open.",
                    enum=context.setting_names,
                ),
                "action": string_property(
                    "Use increase/decrease for relative requests and set when the user gave a concrete value.",
                    enum=["increase", "decrease", "set"],
                ),
                "value": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string", "minLength": 1},
                    ],
                    "description": (
                        "Optional concrete value for action=set. "
                        "For memory_capacity use levels 1 to 4. "
                        f"For spoken_voice pass one supported voice name from this catalog: {context.spoken_voice_catalog}. "
                        "Do not pass a free-form description. "
                        "For speech_speed use a factor between 0.75 and 1.15. "
                        "For speech_pause_ms use milliseconds. "
                        "For follow_up_timeout_s use seconds."
                    ),
                },
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent setting change when extra confirmation is needed."
                ),
            },
            "allOf": simple_setting_rules(context.spoken_voices),
            "required": ["setting", "action"],
            "additionalProperties": False,
        },
    }


def build_manage_voice_quiet_mode_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "manage_voice_quiet_mode",
        "description": (
            "Manage Twinr's temporary voice quiet window. "
            "Use this when the user wants Twinr to stay quiet for a bounded time so TV, radio, or room speech does not reopen the transcript-first wake or automatic follow-up path. "
            "Use action status when the user asks whether Twinr is currently quiet or still listening. "
            "If the user wants quiet but did not give a clear duration yet, ask a short follow-up question instead of pretending the quiet window is active. "
            "This is temporary runtime state, not a persistent setting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": string_property(
                    "Choose set to start a temporary quiet window, clear to end it early, or status to inspect the current window.",
                    enum=["set", "clear", "status"],
                ),
                "duration_minutes": number_property(
                    "Required for action=set. Bounded quiet duration in whole minutes.",
                    minimum=1,
                    maximum=720,
                    integer=True,
                ),
                "reason": string_property(
                    "Optional short reason such as TV news, radio, or background audio.",
                    min_length=1,
                ),
            },
            "allOf": [
                {
                    "if": {"properties": {"action": {"const": "set"}}, "required": ["action"]},
                    "then": {"required": ["duration_minutes"]},
                }
            ],
            "required": ["action"],
            "additionalProperties": False,
        },
    }


def build_enroll_voice_profile_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "enroll_voice_profile",
        "description": (
            "Create or refresh the local Twinr voice profile from the current spoken turn. "
            "Use only when the user explicitly asks Twinr to learn or update their voice profile."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed a replacement when extra confirmation is needed."
                )
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_get_voice_profile_status_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
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


def build_reset_voice_profile_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "reset_voice_profile",
        "description": "Delete the local Twinr voice profile when the user explicitly asks to remove it.",
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the reset when extra confirmation is needed."
                )
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_enroll_portrait_identity_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "enroll_portrait_identity",
        "description": (
            "Capture the current live camera view and add it to Twinr's local on-device face profile. "
            "Use only when the user explicitly asks Twinr to remember, learn, refresh, or update their face."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "display_name": string_property(
                    "Optional friendly display name for the saved local face profile when the user explicitly wants it.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this local face-profile save when extra confirmation is needed."
                ),
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_get_portrait_identity_status_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "get_portrait_identity_status",
        "description": (
            "Read Twinr's local on-device face-profile status, including how many portrait references are saved "
            "and whether the current live camera view matches it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this local face-profile status lookup when extra confirmation is needed."
                )
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_reset_portrait_identity_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "reset_portrait_identity",
        "description": "Delete Twinr's local on-device face profile when the user explicitly asks to remove it.",
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the local face-profile reset when extra confirmation is needed."
                )
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_manage_household_identity_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "manage_household_identity",
        "description": (
            "Manage Twinr's shared local household identity state across face, voice, live matching, "
            "and explicit confirm or deny feedback. Use this tool for local household identity enrollment, "
            "identity status, or when the user confirms that Twinr recognized the right or wrong person."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": string_property(
                    "Choose one supported local household identity action.",
                    enum=("status", "enroll_face", "enroll_voice", "confirm_identity", "deny_identity"),
                ),
                "user_id": string_property(
                    "Optional stable local member identifier when the user explicitly names which enrolled household member should be updated or confirmed.",
                    min_length=1,
                ),
                "display_name": string_property(
                    "Optional friendly household member name when the user explicitly provides it.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed a persistent local identity enrollment when extra confirmation is needed."
                ),
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }


TOOL_BUILDERS = (
    ("configure_world_intelligence", build_configure_world_intelligence_schema),
    ("update_simple_setting", build_update_simple_setting_schema),
    ("manage_voice_quiet_mode", build_manage_voice_quiet_mode_schema),
    ("enroll_voice_profile", build_enroll_voice_profile_schema),
    ("get_voice_profile_status", build_get_voice_profile_status_schema),
    ("reset_voice_profile", build_reset_voice_profile_schema),
    ("enroll_portrait_identity", build_enroll_portrait_identity_schema),
    ("get_portrait_identity_status", build_get_portrait_identity_status_schema),
    ("reset_portrait_identity", build_reset_portrait_identity_schema),
    ("manage_household_identity", build_manage_household_identity_schema),
)
