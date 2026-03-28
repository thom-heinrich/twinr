"""Memory, discovery, and contact-related tool schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import CANONICAL_ENGLISH_MEMORY_NOTE, array_property, boolean_property, number_property, string_property


def build_remember_memory_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_memory",
        "description": (
            "Store an important memory for future turns when the user explicitly asks you to remember something. "
            "Use only for clear remember/save requests, not for ordinary conversation. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": string_property(
                    "Short type such as appointment, contact, reminder, preference, fact, or task.",
                    min_length=1,
                ),
                "summary": string_property(
                    "Short factual summary of what should be remembered.",
                    min_length=1,
                ),
                "details": string_property(
                    "Optional extra detail that helps later recall.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                ),
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    }


def build_remember_contact_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_contact",
        "description": (
            "Store or refine a remembered contact in Twinr's structured graph memory. "
            "Use this when the user explicitly wants Twinr to remember a person with a phone number, email, relation, or role. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "given_name": string_property(
                    "First name or main short name of the contact.",
                    min_length=1,
                ),
                "family_name": string_property(
                    "Optional family name if known.",
                    min_length=1,
                ),
                "phone": string_property(
                    "Optional phone number if the user gave one.",
                    min_length=1,
                ),
                "email": string_property(
                    "Optional email if the user gave one.",
                    min_length=1,
                    string_format="email",
                ),
                "role": string_property(
                    "Optional role such as physiotherapist, daughter, neighbor, or friend.",
                    min_length=1,
                ),
                "relation": string_property(
                    "Optional relationship wording such as daughter, family, or helper.",
                    min_length=1,
                ),
                "notes": string_property(
                    "Optional short detail that helps future disambiguation.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                ),
            },
            "required": ["given_name"],
            "additionalProperties": False,
        },
    }


def build_lookup_contact_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup_contact",
        "description": (
            "Look up a remembered contact and return the stored phone number or email. "
            "Use this for exact contact details instead of relying on hidden memory context, and ask for clarification when multiple matches exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": string_property(
                    "Name or short name of the contact to look up.",
                    min_length=1,
                ),
                "family_name": string_property(
                    "Optional family name if the user gave one.",
                    min_length=1,
                ),
                "role": string_property(
                    "Optional role such as physiotherapist, daughter, or neighbor.",
                    min_length=1,
                ),
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def build_send_whatsapp_message_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "send_whatsapp_message",
        "description": (
            "Send a WhatsApp message to one remembered contact after an explicit final confirmation. "
            "Use this when the user asks Twinr to write or send a WhatsApp message to someone already remembered in contact memory. "
            "Resolve the recipient through remembered-contact lookup instead of inventing or guessing a number. "
            "If the user has not told you the exact message text yet, call the tool anyway so it can ask the bounded follow-up question. "
            "If the contact is ambiguous or has multiple phone numbers, the tool returns a clarification question. "
            "If the tool returns confirmation_required, ask that exact confirmation question and call it again with confirmed=true only after the user clearly says yes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": string_property(
                    "Name or short first name of the remembered contact who should receive the WhatsApp message.",
                    min_length=1,
                ),
                "family_name": string_property(
                    "Optional family name when it helps choose the right remembered contact.",
                    min_length=1,
                ),
                "role": string_property(
                    "Optional role or relation such as daughter, neighbor, caregiver, or physiotherapist.",
                    min_length=1,
                ),
                "contact_label": string_property(
                    "Optional exact remembered contact label copied verbatim from a prior clarification option when disambiguating an ambiguous recipient.",
                    min_length=1,
                ),
                "phone_last4": string_property(
                    "Optional last digits of the chosen remembered phone number when the contact has multiple phone numbers and the tool asked for a suffix choice.",
                    min_length=2,
                ),
                "message": string_property(
                    "The exact WhatsApp message text that Twinr should send.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this exact WhatsApp send."
                ),
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def build_get_memory_conflicts_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "get_memory_conflicts",
        "description": (
            "Inspect open long-term memory conflicts when the user asks what Twinr is unsure about, "
            "or when you need the current conflict option IDs before resolving one. "
            "Use this tool for exact conflict inspection instead of answering from hidden conflict summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": string_property(
                    "Optional short query describing the current topic, such as Corinna, physiotherapist, "
                    "phone number, spouse, or coffee brand.",
                    min_length=1,
                ),
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_resolve_memory_conflict_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "resolve_memory_conflict",
        "description": (
            "Resolve one open long-term memory conflict after the user clearly identified which stored option is correct. "
            "Use the slot_key and selected_memory_id from conflict context or from get_memory_conflicts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot_key": string_property(
                    "The conflict slot key to resolve, for example contact:person:corinna_maier:phone.",
                    min_length=1,
                ),
                "selected_memory_id": string_property(
                    "The chosen memory_id from the conflict options that should become active.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the persistent memory correction when extra confirmation is needed."
                ),
            },
            "required": ["slot_key", "selected_memory_id"],
            "additionalProperties": False,
        },
    }


def build_remember_preference_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_preference",
        "description": (
            "Store a stable personal preference in Twinr's structured graph memory, such as a preferred brand, favorite shop, or disliked food. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": string_property(
                    "Short category such as brand, store, food, drink, activity, music, or thing.",
                    min_length=1,
                ),
                "value": string_property(
                    "The preferred or disliked thing, place, or brand.",
                    min_length=1,
                ),
                "for_product": string_property(
                    "Optional product context, for example coffee.",
                    min_length=1,
                ),
                "sentiment": string_property(
                    "Use prefer, like, dislike, or usually_buy_at.",
                    min_length=1,
                ),
                "details": string_property(
                    "Optional short detail for later recall.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                ),
            },
            "required": ["category", "value"],
            "additionalProperties": False,
        },
    }


def build_remember_plan_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_plan",
        "description": (
            "Store a short future intention or plan in Twinr's structured graph memory, such as wanting to go for a walk today. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": string_property("Short plan summary.", min_length=1),
                "when": string_property(
                    "Optional time wording such as today, tomorrow, 2026-03-14, or next Monday.",
                    min_length=1,
                ),
                "details": string_property(
                    "Optional short detail for later recall.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                ),
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    }


def build_update_user_profile_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_user_profile",
        "description": (
            "Update stable user profile or preference context for future turns when the user explicitly asks you to remember it. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": string_property(
                    "Short category such as preferred_name, location, preference, contact, or routine.",
                    min_length=1,
                ),
                "instruction": string_property(
                    "Short, durable instruction or fact to store in the user profile.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent profile change when extra confirmation is needed."
                ),
            },
            "required": ["category", "instruction"],
            "additionalProperties": False,
        },
    }


def build_update_personality_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_personality",
        "description": (
            "Update how Twinr should speak or behave in future turns when the user explicitly asks for a behavior change. "
            f"{CANONICAL_ENGLISH_MEMORY_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": string_property(
                    "Short category such as response_style, humor, language, verbosity, or greeting_style.",
                    min_length=1,
                ),
                "instruction": string_property(
                    "Short future-behavior instruction to store in Twinr personality context.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent behavior change when extra confirmation is needed."
                ),
            },
            "required": ["category", "instruction"],
            "additionalProperties": False,
        },
    }


def _build_discovery_learned_fact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "storage": string_property(
                "Use user_profile for stable personal facts and personality for how Twinr should address or behave toward the user.",
                enum=["user_profile", "personality"],
            ),
            "text": string_property(
                "Short durable fact or behavior preference in canonical English.",
                min_length=1,
            ),
        },
        "required": ["storage", "text"],
        "additionalProperties": False,
    }


def _build_discovery_memory_route_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "route_kind": string_property(
                "Structured discovery route kind.",
                enum=["user_profile", "personality", "contact", "preference", "plan", "durable_memory"],
            ),
            "text": string_property("Managed-context fact text.", min_length=1),
            "category": string_property("Preference category.", min_length=1),
            "given_name": string_property("Contact given name.", min_length=1),
            "family_name": string_property("Contact family name.", min_length=1),
            "phone": string_property("Contact phone number.", min_length=1),
            "email": string_property("Contact email address.", min_length=1),
            "role": string_property("Contact role.", min_length=1),
            "relation": string_property("Contact relation to the user.", min_length=1),
            "notes": string_property("Short contact notes.", min_length=1),
            "value": string_property("Preference value.", min_length=1),
            "sentiment": string_property("Preference sentiment.", enum=["prefer", "like", "dislike", "avoid"]),
            "for_product": string_property("Optional preference scope or product.", min_length=1),
            "summary": string_property("Plan or durable-memory summary.", min_length=1),
            "when_text": string_property("Optional natural-language timing text for a plan.", min_length=1),
            "details": string_property("Optional additional details.", min_length=1),
            "kind": string_property("Durable-memory kind label.", min_length=1),
        },
        "required": ["route_kind"],
        "additionalProperties": False,
    }


def build_manage_user_discovery_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "manage_user_discovery",
        "description": (
            "Manage Twinr's guided get-to-know-you flow across the initial setup and later short lifelong-learning follow-ups. "
            "Use this when the user wants to start or continue the setup, offers to tell Twinr something about themselves, freely volunteers stable profile details that should enter the bounded discovery flow, answers an active get-to-know-you question, asks to pause or skip a topic, or says not now to a visible discovery invitation. "
            "When the user answered, include compact learned_facts in canonical English as durable summaries, not raw transcript quotes, and use one learned_fact or memory_route per distinct learned detail. "
            "Direct first-person profile statements, wish-form self-statements about preferred name or form of address, or direct profile corrections from an identified speaker already count as approval for discovery saves or mutations. "
            "If the user explicitly corrects or deletes a previously learned detail, use review_profile and then replace_fact or delete_fact in the same turn when needed, even if discovery setup is still active."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": string_property(
                    "Discovery action.",
                    enum=[
                        "start_or_resume",
                        "answer",
                        "skip_topic",
                        "pause_session",
                        "snooze",
                        "status",
                        "review_profile",
                        "replace_fact",
                        "delete_fact",
                    ],
                ),
                "topic_id": string_property(
                    "Optional predefined topic such as basics, companion_style, social, interests, hobbies, routines, pets, no_goes, or health.",
                    min_length=1,
                ),
                "fact_id": string_property(
                    "Required for replace_fact and delete_fact. Use the exact fact_id from a prior review_profile result and omit it for other actions.",
                    min_length=1,
                ),
                "learned_facts": array_property(
                    "Optional compact durable facts learned from the user's answer. Use canonical English, not raw transcript quotes.",
                    _build_discovery_learned_fact_schema(),
                    max_items=8,
                ),
                "memory_routes": array_property(
                    "Optional structured durable routes learned from the user's answer or used as replacements. Use canonical English semantic text, create one route per distinct learned detail, and only fill fields relevant to that route_kind.",
                    _build_discovery_memory_route_schema(),
                    max_items=8,
                ),
                "topic_complete": boolean_property(
                    "Set true only when the current topic is sufficiently covered for now and Twinr may move on or wrap up."
                ),
                "permission_granted": boolean_property(
                    "For sensitive topics such as health, set true only after the user clearly agreed to continue on that topic, or false when the user declined."
                ),
                "snooze_days": number_property(
                    "Optional whole-number snooze length in days for not-now responses.",
                    minimum=1,
                    maximum=14,
                    integer=True,
                ),
                "confirmed": boolean_property(
                    "Set true when the user already gave the needed approval for this persistent save, including direct first-person stable profile statements, preferred-name or address-preference self-statements, or direct corrections from an identified speaker; leave false only when extra speaker confirmation is still required."
                ),
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }


TOOL_BUILDERS = (
    ("remember_memory", build_remember_memory_schema),
    ("remember_contact", build_remember_contact_schema),
    ("lookup_contact", build_lookup_contact_schema),
    ("send_whatsapp_message", build_send_whatsapp_message_schema),
    ("get_memory_conflicts", build_get_memory_conflicts_schema),
    ("resolve_memory_conflict", build_resolve_memory_conflict_schema),
    ("remember_preference", build_remember_preference_schema),
    ("remember_plan", build_remember_plan_schema),
    ("update_user_profile", build_update_user_profile_schema),
    ("update_personality", build_update_personality_schema),
    ("manage_user_discovery", build_manage_user_discovery_schema),
)
