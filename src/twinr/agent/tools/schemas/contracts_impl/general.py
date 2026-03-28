"""General-purpose tool schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import (
    array_property,
    boolean_property,
    iso8601_datetime_property,
    number_property,
    string_property,
)


def build_print_receipt_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "print_receipt",
        "description": (
            "Print short, user-facing content on the thermal receipt printer "
            "when the user explicitly asks for a printout. "
            "Use focus_hint to describe what from the recent context should be printed. "
            "When the user asks for exact wording, quoted text, or a literal string to print, "
            "you must pass that exact printable wording in text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus_hint": string_property(
                    "Short hint describing what from the recent conversation should be printed.",
                    min_length=1,
                ),
                "text": string_property(
                    "Exact printable wording. Required when the user asked to print exact text, "
                    "quoted text, or a literal string.",
                    min_length=1,
                ),
            },
            "anyOf": [{"required": ["focus_hint"]}, {"required": ["text"]}],
            "required": [],
            "additionalProperties": False,
        },
    }


def build_search_live_info_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "search_live_info",
        "description": (
            "Look up fresh or externally verifiable web information for the user. "
            "Use this for broad web research, not only a fixed list of example domains. "
            "Do not use it for page interaction, booking flows, forms, checkout/cart state, or social profile/post/story checks that may require opening the live site; those belong to browser_automation or a permission question for deeper site checking. "
            "Do not use it for the user's own smart-home inventory, room/device state, or recent in-home smart-home events; those belong to the smart-home tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": string_property(
                    "The exact question to research on the web.",
                    min_length=1,
                ),
                "location_hint": string_property(
                    "Optional location such as a city or district relevant to the search.",
                    min_length=1,
                ),
                "date_context": string_property(
                    "Optional absolute date or time context if the user referred to relative dates.",
                    min_length=1,
                ),
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    }


def build_browser_automation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "browser_automation",
        "description": (
            "Use bounded live browser automation for a specific website when the task requires page interaction, "
            "multi-step navigation, form filling, or verifying live page state that generic web research cannot answer reliably. "
            "Prefer search_live_info for broad web research or fresh questions that do not require site interaction. "
            "If this would only be a deeper follow-up after ordinary web research, prefer a short model-authored offer to try a different method that may take a little longer but can inspect the site more directly before starting the browser run, unless the user already explicitly asked for site interaction. "
            "If ordinary web research said the exact detail could not be verified or that no current evidence was found, treat that as unresolved rather than as a final answer when a specific site check could still clarify it. "
            "A short follow-up assent to an already proposed deeper site check counts as explicit approval for that browser run. "
            "A freshness-sensitive question about a place, business, organization, or event does not by itself count as explicit browser authorization. "
            "Do not use this tool just to add optional extra checking after ordinary web research already answered the real question sufficiently."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": string_property(
                    "Short task the browser run should complete on the site, for example verify opening hours or extract a result after interaction.",
                    min_length=1,
                ),
                "start_url": string_property(
                    "Optional exact URL to open first when the user already named a specific page or site entry point.",
                    min_length=1,
                ),
                "allowed_domains": array_property(
                    "Narrow host allowlist for this run. Keep it specific to the site you need.",
                    string_property(
                        "One allowed host name for this browser run.",
                        min_length=1,
                    ),
                    min_items=1,
                    max_items=16,
                    unique_items=True,
                ),
                "max_steps": number_property(
                    "Optional upper bound on browser actions for this run. Keep it small unless the page flow clearly needs more steps.",
                    minimum=1,
                    maximum=32,
                    integer=True,
                ),
                "max_runtime_s": number_property(
                    "Optional upper bound on total runtime in seconds.",
                    minimum=0.1,
                    maximum=900.0,
                ),
                "capture_screenshot": boolean_property(
                    "Set true when a screenshot artifact should be kept for verification or operator review."
                ),
                "capture_html": boolean_property(
                    "Set true when an HTML snapshot should be kept for later inspection."
                ),
            },
            "required": ["goal", "allowed_domains"],
            "additionalProperties": False,
        },
    }


def build_connect_service_integration_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "connect_service_integration",
        "description": (
            "Start a bounded service-connect or pairing flow for a named external service. "
            "Use this when the user asks Twinr to connect, link, pair, or set up a service such as WhatsApp. "
            "The follow-up status or QR appears on Twinr's right info panel. "
            "If the user did not clearly name the target service, ask a short follow-up question instead of guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": string_property(
                    "The exact service name the user wants to connect, for example whatsapp.",
                    min_length=1,
                ),
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    }


def build_schedule_reminder_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "schedule_reminder",
        "description": (
            "Schedule a future reminder or timer when the user asks to be reminded later or to set a timer. "
            "Always send due_at as an absolute ISO 8601 local datetime with timezone offset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "due_at": iso8601_datetime_property(
                    "Absolute local due time in ISO 8601 format, for example 2026-03-14T12:00:00+01:00."
                ),
                "summary": string_property(
                    "Short summary of what Twinr should remind the user about.",
                    min_length=1,
                ),
                "details": string_property(
                    "Optional extra detail to include when the reminder is spoken.",
                    min_length=1,
                ),
                "kind": string_property(
                    "Short type such as reminder, timer, appointment, medication, task, or alarm.",
                    min_length=1,
                ),
                "original_request": string_property(
                    "Optional short quote or paraphrase of the user's original reminder request.",
                    min_length=1,
                ),
            },
            "required": ["due_at", "summary"],
            "additionalProperties": False,
        },
    }


def build_end_conversation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "end_conversation",
        "description": (
            "End the current follow-up listening loop when the user clearly indicates they are done for now, "
            "for example by saying thanks, stop, pause, bye, or tschuss."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": string_property(
                    "Optional short note describing why the conversation should end.",
                    min_length=1,
                ),
                "spoken_reply": string_property(
                    "Short goodbye that Twinr should say immediately while ending the conversation.",
                    min_length=1,
                ),
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_inspect_camera_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "inspect_camera",
        "description": (
            "Inspect the current live camera view when the user asks you to look at them, "
            "an object, a document, or something they are showing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": string_property(
                    "The exact user request about what should be inspected in the camera view.",
                    min_length=1,
                )
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    }


TOOL_BUILDERS = (
    ("print_receipt", build_print_receipt_schema),
    ("search_live_info", build_search_live_info_schema),
    ("browser_automation", build_browser_automation_schema),
    ("connect_service_integration", build_connect_service_integration_schema),
    ("schedule_reminder", build_schedule_reminder_schema),
    ("end_conversation", build_end_conversation_schema),
    ("inspect_camera", build_inspect_camera_schema),
)

PRIMARY_TOOL_BUILDERS = TOOL_BUILDERS[:5]
EPILOGUE_TOOL_BUILDERS = TOOL_BUILDERS[5:]
