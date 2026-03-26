"""Resolve bounded WhatsApp draft follow-ups with a dedicated model step.

Open WhatsApp send flows should stay structured across turns without relying on
free-form chat history alone. This helper asks the tool-capable model for the
next ``send_whatsapp_message`` call inside a tightly bounded context, then
executes that tool call locally and returns the resulting structured reply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    ConversationLike,
    ToolCallingAgentProvider,
)

from .pending_actions import PendingTextChannelAction


@dataclass(frozen=True, slots=True)
class PendingWhatsAppFollowUpResult:
    """Describe one provider-guided WhatsApp draft follow-up turn."""

    reply_text: str
    tool_results: tuple[AgentToolResult, ...] = ()
    clear_pending: bool = False
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    provider_model: str | None = None


def resolve_pending_whatsapp_follow_up(
    pending_action: PendingTextChannelAction,
    user_text: str,
    *,
    provider: ToolCallingAgentProvider,
    conversation: ConversationLike,
    instructions: str,
    tool_schemas: tuple[dict[str, object], ...],
    allow_web_search: bool | None,
    execute_tool,
) -> PendingWhatsAppFollowUpResult | None:
    """Continue one active ``send_whatsapp_message`` draft with a bounded model call.

    Returns ``None`` when the pending action is unrelated to WhatsApp sending or
    when the draft state is not one of the bounded follow-up states handled
    here.
    """

    if pending_action.tool_name != "send_whatsapp_message":
        return None
    if pending_action.state not in {
        "awaiting_message",
        "awaiting_confirmation",
        "awaiting_contact_clarification",
        "awaiting_phone_clarification",
    }:
        return None

    response = provider.start_turn_streaming(
        _pending_follow_up_prompt(pending_action, user_text),
        conversation=conversation,
        instructions=instructions,
        tool_schemas=tool_schemas,
        allow_web_search=allow_web_search,
        on_text_delta=None,
    )
    tool_call = _extract_single_send_whatsapp_tool_call(response.tool_calls)
    output = execute_tool(dict(tool_call.arguments))
    tool_result = AgentToolResult(
        call_id=tool_call.call_id,
        name=tool_call.name,
        output=output,
        serialized_output=json.dumps(output, ensure_ascii=False, sort_keys=True),
    )
    return PendingWhatsAppFollowUpResult(
        reply_text=_reply_text_from_tool_output(output, pending_action=pending_action),
        tool_results=(tool_result,),
        provider_response_id=_optional_text(getattr(response, "response_id", None)),
        provider_request_id=_optional_text(getattr(response, "request_id", None)),
        provider_model=_optional_text(getattr(response, "model", None)),
    )


def _pending_follow_up_prompt(
    pending_action: PendingTextChannelAction,
    user_text: str,
) -> str:
    stripped = str(user_text or "").strip()
    stored_fields = _stored_recipient_fields_text(pending_action)
    if pending_action.state == "awaiting_message":
        return (
            "Continue the active send_whatsapp_message draft. "
            "The user is answering what the WhatsApp should say. "
            "Call send_whatsapp_message now with the stored recipient fields and "
            "the user's latest text as the literal message body unless the user "
            "explicitly changes the recipient. "
            f"Stored recipient fields: {stored_fields}.\n"
            f"Latest user reply:\n{stripped}"
        )
    if pending_action.state == "awaiting_confirmation":
        return (
            "Continue the active send_whatsapp_message draft. "
            "The user is answering the final send confirmation question. "
            "Call send_whatsapp_message now with the stored recipient/message "
            "fields and set confirmed according to the user's latest reply. "
            f"Stored draft fields: {stored_fields}.\n"
            f"Latest user reply:\n{stripped}"
        )
    if pending_action.state == "awaiting_contact_clarification":
        return (
            "Continue the active send_whatsapp_message draft. "
            "The user is clarifying which remembered contact should receive the "
            "message. Match the latest reply against the stored clarification "
            "options and call send_whatsapp_message with contact_label copied "
            "exactly from the matching stored option label when one option fits. "
            "Do not paraphrase or decompose the chosen option into looser fields "
            "unless no exact option matches. "
            f"Stored draft fields: {stored_fields}.\n"
            f"Latest user reply:\n{stripped}"
        )
    if pending_action.state == "awaiting_phone_clarification":
        return (
            "Continue the active send_whatsapp_message draft. "
            "The user is clarifying which stored phone number should be used. "
            f"Call send_whatsapp_message with the needed phone_last4 choice. Stored draft fields: {stored_fields}.\n"
            f"Latest user reply:\n{stripped}"
        )
    return stripped


def _stored_recipient_fields_text(pending_action: PendingTextChannelAction) -> str:
    fields = [f"name={pending_action.name}"]
    if pending_action.family_name:
        fields.append(f"family_name={pending_action.family_name}")
    if pending_action.lookup_role:
        fields.append(f"role={pending_action.lookup_role}")
    if pending_action.contact_label:
        fields.append(f"contact_label={pending_action.contact_label}")
    if pending_action.phone_last4:
        fields.append(f"phone_last4={pending_action.phone_last4}")
    if pending_action.message_preview:
        fields.append(f"message={pending_action.message_preview}")
    return ", ".join(fields)


def _extract_single_send_whatsapp_tool_call(tool_calls: object) -> AgentToolCall:
    if tool_calls is None:
        normalized: tuple[object, ...] = ()
    else:
        normalized = tuple(cast(list[object] | tuple[object, ...], tool_calls))
    if len(normalized) != 1:
        raise RuntimeError("pending WhatsApp follow-up requires exactly one tool call")
    tool_call = normalized[0]
    if not isinstance(tool_call, AgentToolCall):
        raise RuntimeError("pending WhatsApp follow-up returned a malformed tool call")
    if tool_call.name != "send_whatsapp_message":
        raise RuntimeError("pending WhatsApp follow-up must call send_whatsapp_message")
    if not isinstance(tool_call.arguments, dict):
        raise RuntimeError("pending WhatsApp follow-up tool arguments must be a JSON object")
    return tool_call


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _reply_text_from_tool_output(
    output: dict[str, object],
    *,
    pending_action: PendingTextChannelAction,
) -> str:
    question = str(output.get("question", "") or "").strip()
    if question:
        return question
    message = str(output.get("message", "") or "").strip()
    if message:
        return message
    status = str(output.get("status", "") or "").strip()
    recipient_label = (
        str(output.get("recipient_label", "") or "").strip()
        or pending_action.recipient_label
        or pending_action.name
    )
    if status == "sent":
        return f"Ich habe {recipient_label} geschrieben."
    if status == "delivery_failed":
        return "Die WhatsApp konnte gerade nicht gesendet werden."
    if status == "not_found":
        return "Ich konnte diesen Kontakt gerade nicht finden."
    if status == "contact_missing_phone":
        return f"Für {recipient_label} ist keine Telefonnummer gespeichert."
    if status == "contact_missing_whatsapp_address":
        return f"Für {recipient_label} brauche ich eine vollständige Telefonnummer mit + am Anfang."
    return "Ich konnte die WhatsApp gerade nicht weiterführen."


__all__ = [
    "PendingWhatsAppFollowUpResult",
    "resolve_pending_whatsapp_follow_up",
]
