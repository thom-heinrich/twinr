"""Handle bounded WhatsApp-send tool calls for remembered contacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinr.channels.whatsapp.config import normalize_whatsapp_jid
from twinr.channels.whatsapp.outbound import dispatch_whatsapp_outbound_message

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import ArgumentValidationError, optional_bool

_MAX_NAME_LENGTH = 120
_MAX_MESSAGE_LENGTH = 4096
_MIN_PHONE_SUFFIX_DIGITS = 2


def _ensure_arguments_mapping(arguments: dict[str, object]) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise ArgumentValidationError("arguments must be a JSON object")
    return dict(arguments)


def _optional_text(arguments: Mapping[str, object], key: str, *, max_length: int) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length:
        raise ArgumentValidationError(f"{key} is too long")
    return text


def _required_text(arguments: Mapping[str, object], key: str, *, max_length: int) -> str:
    text = _optional_text(arguments, key, max_length=max_length)
    if text is None:
        raise ArgumentValidationError(f"{key} is required")
    return text


def _phone_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return None
    if len(digits) < _MIN_PHONE_SUFFIX_DIGITS:
        return None
    return digits


def _masked_phone_suffix(phone: str) -> str:
    digits = "".join(character for character in str(phone) if character.isdigit())
    if not digits:
        return "unknown"
    suffix = digits[-4:] if len(digits) >= 4 else digits
    return f"...{suffix}"


def _masked_phone_suffixes(phones: object) -> list[str]:
    if not isinstance(phones, (list, tuple)):
        return []
    normalized: list[str] = []
    for item in phones or ():
        masked = _masked_phone_suffix(str(item))
        if masked not in normalized:
            normalized.append(masked)
    return normalized


def _lookup_contact(
    owner: Any,
    *,
    name: str,
    family_name: str | None,
    role: str | None,
    contact_label: str | None,
):
    runtime = getattr(owner, "runtime", None)
    if runtime is None or not hasattr(runtime, "lookup_contact"):
        raise RuntimeError("send_whatsapp_message requires owner.runtime.lookup_contact")
    payload: dict[str, object] = {
        "name": name,
        "family_name": family_name,
        "role": role,
    }
    if contact_label is not None:
        payload["contact_label"] = contact_label
    return runtime.lookup_contact(**payload)


def _select_phone(phones: list[str], *, phone_suffix: str | None) -> str | None:
    if not phones:
        return None
    if len(phones) == 1:
        return phones[0]
    if phone_suffix is None:
        return None
    matches = [phone for phone in phones if "".join(character for character in phone if character.isdigit()).endswith(phone_suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _dispatch_outbound_message(
    owner: Any,
    *,
    chat_jid: str,
    text: str,
    recipient_label: str,
    reply_to_message_id: str | None = None,
):
    """Deliver through an owner-provided direct channel dispatcher when present."""

    owner_dispatch = getattr(owner, "dispatch_whatsapp_outbound_message", None)
    if callable(owner_dispatch):
        return owner_dispatch(
            chat_jid=chat_jid,
            text=text,
            recipient_label=recipient_label,
            reply_to_message_id=reply_to_message_id,
        )
    return dispatch_whatsapp_outbound_message(
        owner.config,
        chat_jid=chat_jid,
        text=text,
        recipient_label=recipient_label,
        reply_to_message_id=reply_to_message_id,
    )


def handle_send_whatsapp_message(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Resolve a remembered contact, ask for final confirmation, then send."""

    arguments = _ensure_arguments_mapping(arguments)
    name = _required_text(arguments, "name", max_length=_MAX_NAME_LENGTH)
    family_name = _optional_text(arguments, "family_name", max_length=_MAX_NAME_LENGTH)
    role = _optional_text(arguments, "role", max_length=_MAX_NAME_LENGTH)
    if role is None:
        # Accept the legacy clarification field name so in-flight pending drafts
        # and older prompts still resolve the intended remembered contact.
        role = _optional_text(arguments, "lookup_role", max_length=_MAX_NAME_LENGTH)
    contact_label = _optional_text(arguments, "contact_label", max_length=_MAX_NAME_LENGTH)
    message = _optional_text(arguments, "message", max_length=_MAX_MESSAGE_LENGTH)
    confirmed = optional_bool(arguments, "confirmed", default=False) is True
    phone_suffix = _phone_suffix(_optional_text(arguments, "phone_last4", max_length=16))

    lookup = _lookup_contact(
        owner,
        name=name,
        family_name=family_name,
        role=role,
        contact_label=contact_label,
    )
    status = str(getattr(lookup, "status", "") or "").strip()
    emit_best_effort(owner, "whatsapp_send_tool_call=true")

    if status == "not_found":
        record_event_best_effort(
            owner,
            "whatsapp_send_contact_missing",
            "Twinr could not find a remembered WhatsApp recipient.",
            {"name": name},
        )
        return {
            "status": "not_found",
            "name": name,
            "message": "I could not find that remembered contact.",
        }

    if status == "needs_clarification":
        options = []
        for option in getattr(lookup, "options", ()) or ():
            options.append(
                {
                    "label": str(getattr(option, "label", "") or "").strip(),
                    "role": str(getattr(option, "role", "") or "").strip() or None,
                    "phone_last4_options": _masked_phone_suffixes(getattr(option, "phones", ())),
                }
            )
        record_event_best_effort(
            owner,
            "whatsapp_send_clarification_needed",
            "Twinr needs contact clarification before sending a WhatsApp message.",
            {"name": name, "option_count": len(options)},
        )
        return {
            "status": "needs_clarification",
            "question": str(getattr(lookup, "question", "") or "").strip() or f"Which {name} should I message?",
            "name": name,
            "family_name": family_name,
            "lookup_role": role,
            "contact_label": contact_label,
            "message_preview": message,
            "options": options,
        }

    match = getattr(lookup, "match", None)
    if match is None:
        raise RuntimeError("send_whatsapp_message lookup returned no contact match")

    recipient_label = str(getattr(match, "label", "") or "").strip() or name
    recipient_role = str(getattr(match, "role", "") or "").strip() or None
    resolved_contact_label = contact_label or recipient_label
    phones = [str(phone).strip() for phone in getattr(match, "phones", ()) or () if str(phone).strip()]
    masked_selected_phone = ""
    if not phones:
        return {
            "status": "contact_missing_phone",
            "recipient_label": recipient_label,
            "message": "That remembered contact has no phone number yet.",
        }

    selected_phone = _select_phone(phones, phone_suffix=phone_suffix)
    if selected_phone is None and len(phones) > 1:
        return {
            "status": "needs_phone_clarification",
            "question": f"Which phone number of {recipient_label} should I use for WhatsApp?",
            "name": name,
            "family_name": family_name,
            "lookup_role": role,
            "contact_label": resolved_contact_label,
            "recipient_label": recipient_label,
            "message_preview": message,
            "options": _masked_phone_suffixes(phones),
        }

    if selected_phone is None:
        selected_phone = phones[0]
    masked_selected_phone = _masked_phone_suffix(selected_phone)

    if not selected_phone.startswith("+"):
        return {
            "status": "contact_missing_whatsapp_address",
            "recipient_label": recipient_label,
            "message": (
                "That remembered contact needs a full international phone number "
                "starting with + before I can send a WhatsApp message."
            ),
        }

    if message is None:
        return {
            "status": "message_required",
            "question": f"What should the WhatsApp to {recipient_label} say?",
            "name": name,
            "family_name": family_name,
            "lookup_role": role,
            "contact_label": resolved_contact_label,
            "recipient_label": recipient_label,
        }

    confirmation_question = f"Should I send this WhatsApp to {recipient_label}: {message}"
    if not confirmed:
        response = {
            "status": "confirmation_required",
            "requires_confirmation": True,
            "question": confirmation_question,
            "name": name,
            "family_name": family_name,
            "lookup_role": role,
            "contact_label": resolved_contact_label,
            "recipient_label": recipient_label,
            "role": recipient_role,
            "message_preview": message,
        }
        if len(phones) > 1:
            response["phone_last4"] = masked_selected_phone
        return response

    result = _dispatch_outbound_message(
        owner,
        chat_jid=normalize_whatsapp_jid(selected_phone),
        text=message,
        recipient_label=recipient_label,
    )
    emit_best_effort(owner, f"whatsapp_send_status={result.status}")
    record_event_best_effort(
        owner,
        "whatsapp_send_completed",
        "Twinr completed a bounded WhatsApp send attempt.",
        {
            "recipient_label": recipient_label,
            "status": result.status,
            "ok": result.ok,
            "has_message_id": bool(result.message_id),
            "error_code": result.error_code,
        },
    )
    if result.ok:
        return {
            "status": "sent",
            "recipient_label": recipient_label,
            "phone_last4": masked_selected_phone,
            "message_id": result.message_id,
        }
    return {
        "status": "delivery_failed",
        "recipient_label": recipient_label,
        "phone_last4": masked_selected_phone,
        "error_code": result.error_code,
        "message": result.error or "WhatsApp delivery failed.",
    }
