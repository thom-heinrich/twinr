"""Track bounded multi-turn text-channel actions that need a follow-up reply.

This keeps external text channels like WhatsApp from depending on the model to
reconstruct an unfinished structured action purely from free-form chat history.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from tempfile import mkstemp
from typing import Any

from twinr.agent.base_agent.contracts import AgentToolResult

LOGGER = logging.getLogger(__name__)
_STATE_SCHEMA = "pending_text_channel_actions_v1"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
    file_descriptor, temp_name = mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _load_json(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


@dataclass(frozen=True, slots=True)
class PendingTextChannelContactOption:
    """Store one bounded remembered-contact clarification candidate."""

    label: str
    role: str | None = None
    phone_last4_options: tuple[str, ...] = ()

    def summary(self) -> str:
        """Return one compact summary string for prompts and debugging."""

        parts = [f"label={self.label}"]
        if self.role:
            parts.append(f"role={self.role}")
        if self.phone_last4_options:
            parts.append("phone_last4_options=" + ", ".join(self.phone_last4_options))
        return "; ".join(parts)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PendingTextChannelContactOption":
        """Hydrate one persisted clarification option."""

        label = str(payload.get("label", "") or "").strip()
        if not label:
            raise ValueError("PendingTextChannelContactOption.label is required")
        role = PendingTextChannelActionStore._optional_text(payload.get("role"))
        raw_phone_last4_options = payload.get("phone_last4_options", ())
        if not isinstance(raw_phone_last4_options, Sequence) or isinstance(raw_phone_last4_options, (str, bytes)):
            raw_phone_last4_options = ()
        phone_last4_options = tuple(
            str(item).strip()
            for item in raw_phone_last4_options
            if str(item).strip()
        )
        return cls(
            label=label,
            role=role,
            phone_last4_options=phone_last4_options,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one clarification option to a JSON-safe payload."""

        payload: dict[str, object] = {
            "label": self.label,
        }
        if self.role:
            payload["role"] = self.role
        if self.phone_last4_options:
            payload["phone_last4_options"] = list(self.phone_last4_options)
        return payload


@dataclass(frozen=True, slots=True)
class PendingTextChannelAction:
    """Describe one active structured follow-up awaiting the next user message."""

    tool_name: str
    state: str
    name: str
    family_name: str | None = None
    lookup_role: str | None = None
    contact_label: str | None = None
    recipient_label: str | None = None
    phone_last4: str | None = None
    message_preview: str | None = None
    question: str | None = None
    clarification_options: tuple[str, ...] = ()
    clarification_candidates: tuple[PendingTextChannelContactOption, ...] = ()

    def system_message(self) -> str:
        """Return a compact machine-facing reminder for the next provider turn."""

        label = self.recipient_label or self.name
        reuse_fields = [
            f"name={self.name}",
        ]
        if self.family_name:
            reuse_fields.append(f"family_name={self.family_name}")
        if self.lookup_role:
            reuse_fields.append(f"role={self.lookup_role}")
        if self.contact_label:
            reuse_fields.append(f"contact_label={self.contact_label}")
        if self.state == "awaiting_message":
            return (
                "Pending structured action: send_whatsapp_message. "
                f"A remembered-contact WhatsApp draft to {label} is active. "
                "The user's next message should be treated as the exact WhatsApp body for that draft "
                "unless the user clearly cancels or changes the recipient. "
                f"Reuse these fields: {', '.join(reuse_fields)}. "
                "Do not send phone_last4 unless the tool explicitly asked for a phone-number suffix."
            )
        if self.state == "awaiting_confirmation":
            confirmation_fields = list(reuse_fields)
            if self.message_preview:
                confirmation_fields.append(f"message={self.message_preview}")
            if self.phone_last4:
                confirmation_fields.append(f"phone_last4={self.phone_last4}")
            return (
                "Pending structured action: send_whatsapp_message. "
                f"A WhatsApp draft to {label} with message {self.message_preview or ''} is awaiting explicit final confirmation. "
                "Interpret the user's next reply as the answer to that confirmation question. "
                f"Reuse these fields: {', '.join(confirmation_fields)}. "
                "If the user clearly confirms, call send_whatsapp_message again with confirmed=true."
            )
        if self.state == "awaiting_phone_clarification":
            clarification_fields = list(reuse_fields)
            if self.message_preview:
                clarification_fields.append(f"message={self.message_preview}")
            return (
                "Pending structured action: send_whatsapp_message. "
                f"A WhatsApp draft to {label} is waiting for a phone-number suffix choice. "
                "Interpret the user's next reply as that clarification when possible. "
                f"Reuse these fields: {', '.join(clarification_fields)}."
            )
        if self.state == "awaiting_contact_clarification":
            contact_fields = []
            if self.family_name:
                contact_fields.append(f"family_name={self.family_name}")
            if self.lookup_role:
                contact_fields.append(f"role={self.lookup_role}")
            if self.contact_label:
                contact_fields.append(f"contact_label={self.contact_label}")
            if self.message_preview:
                contact_fields.append(f"message={self.message_preview}")
            question_text = f" Pending question: {self.question}." if self.question else ""
            options_text = ""
            if self.clarification_options:
                options_text = (
                    " Available contact options: "
                    + " | ".join(self.clarification_options)
                    + "."
                )
            return (
                "Pending structured action: send_whatsapp_message. "
                f"A WhatsApp draft is waiting for contact clarification for name {self.name}. "
                "Interpret the user's next reply as that contact clarification when possible. "
                "Choose the matching contact and continue the same WhatsApp draft by copying the exact stored contact_label when one option matches. "
                "Only fall back to family_name, role, or phone_last4 when the exact stored option label is unavailable. "
                f"Reuse these fields: name={self.name}"
                + (f", {', '.join(contact_fields)}." if contact_fields else ".")
                + question_text
                + options_text
            )
        return (
            "Pending structured action: send_whatsapp_message. "
            "Continue the active WhatsApp send flow instead of treating the turn as unrelated small talk."
        )

    def rewrite_prompt(self, user_text: str) -> str:
        """Return a stronger prompt wrapper for the next provider turn."""

        stripped = str(user_text).strip()
        if self.state == "awaiting_message":
            return (
                "The user is answering the pending question about what the WhatsApp message should say. "
                "Treat the following text as the literal WhatsApp body unless the user clearly cancels or changes the request:\n"
                f"{stripped}"
            )
        if self.state == "awaiting_confirmation":
            return (
                "The user is answering the pending final confirmation question for a WhatsApp send. "
                "Interpret the following reply as the confirmation answer for the stored draft:\n"
                f"{stripped}"
            )
        if self.state == "awaiting_phone_clarification":
            return (
                "The user is answering a pending clarification question for a WhatsApp send. "
                "Interpret the following reply as the clarification for that draft:\n"
                f"{stripped}"
            )
        if self.state == "awaiting_contact_clarification":
            options_text = ""
            if self.clarification_options:
                options_text = (
                    "\nStored contact options:\n- "
                    + "\n- ".join(self.clarification_options)
                )
            return (
                "The user is answering a pending contact-clarification question for a WhatsApp send. "
                "Match the reply against the stored options and continue the same send_whatsapp_message draft. "
                "Copy contact_label exactly from the matching stored option instead of paraphrasing it. "
                "Use family_name, role, or phone_last4 only when no exact stored option label is available."
                + (f"\nPending question: {self.question}" if self.question else "")
                + options_text
                + "\n"
                + stripped
            )
        return stripped

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PendingTextChannelAction":
        """Hydrate one persisted pending action."""

        tool_name = str(payload.get("tool_name", "") or "").strip()
        state = str(payload.get("state", "") or "").strip()
        name = str(payload.get("name", "") or "").strip()
        if not tool_name:
            raise ValueError("PendingTextChannelAction.tool_name is required")
        if not state:
            raise ValueError("PendingTextChannelAction.state is required")
        if not name:
            raise ValueError("PendingTextChannelAction.name is required")
        clarification_candidates_payload = payload.get("clarification_candidates", ())
        if not isinstance(clarification_candidates_payload, Sequence) or isinstance(
            clarification_candidates_payload,
            (str, bytes),
        ):
            clarification_candidates_payload = ()
        clarification_candidates = tuple(
            PendingTextChannelContactOption.from_dict(item)
            for item in clarification_candidates_payload
            if isinstance(item, Mapping)
        )
        raw_clarification_options = payload.get("clarification_options", ())
        if not isinstance(raw_clarification_options, Sequence) or isinstance(raw_clarification_options, (str, bytes)):
            raw_clarification_options = ()
        return cls(
            tool_name=tool_name,
            state=state,
            name=name,
            family_name=PendingTextChannelActionStore._optional_text(payload.get("family_name")),
            lookup_role=PendingTextChannelActionStore._optional_text(payload.get("lookup_role")),
            contact_label=PendingTextChannelActionStore._optional_text(payload.get("contact_label")),
            recipient_label=PendingTextChannelActionStore._optional_text(payload.get("recipient_label")),
            phone_last4=PendingTextChannelActionStore._optional_text(payload.get("phone_last4")),
            message_preview=PendingTextChannelActionStore._optional_text(payload.get("message_preview")),
            question=PendingTextChannelActionStore._optional_text(payload.get("question")),
            clarification_options=tuple(
                str(item).strip()
                for item in raw_clarification_options
                if str(item).strip()
            ),
            clarification_candidates=clarification_candidates,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one pending action to a JSON-safe payload."""

        payload: dict[str, object] = {
            "tool_name": self.tool_name,
            "state": self.state,
            "name": self.name,
        }
        if self.family_name:
            payload["family_name"] = self.family_name
        if self.lookup_role:
            payload["lookup_role"] = self.lookup_role
        if self.contact_label:
            payload["contact_label"] = self.contact_label
        if self.recipient_label:
            payload["recipient_label"] = self.recipient_label
        if self.phone_last4:
            payload["phone_last4"] = self.phone_last4
        if self.message_preview:
            payload["message_preview"] = self.message_preview
        if self.question:
            payload["question"] = self.question
        if self.clarification_options:
            payload["clarification_options"] = list(self.clarification_options)
        if self.clarification_candidates:
            payload["clarification_candidates"] = [
                candidate.to_dict() for candidate in self.clarification_candidates
            ]
        return payload


class PendingTextChannelActionStore:
    """Keep one active structured follow-up per external channel conversation."""

    def __init__(self, *, path: Path | None = None) -> None:
        self._lock = RLock()
        self._path = path.resolve(strict=False) if path is not None else None
        self._by_conversation: dict[str, PendingTextChannelAction] = self._load_initial_state()

    def get(self, conversation_id: str) -> PendingTextChannelAction | None:
        with self._lock:
            return self._by_conversation.get(str(conversation_id).strip())

    def clear(self, conversation_id: str) -> None:
        normalized = str(conversation_id).strip()
        if not normalized:
            return
        with self._lock:
            self._by_conversation.pop(normalized, None)
            self._persist_locked()

    def update_from_tool_results(self, conversation_id: str, tool_results: Sequence[AgentToolResult]) -> None:
        """Refresh pending state from the latest structured tool outputs."""

        normalized = str(conversation_id).strip()
        if not normalized:
            return

        with self._lock:
            current = self._by_conversation.get(normalized)
            pending = self._pending_action_from_tool_results(tool_results, current=current)
            if pending is None:
                self._by_conversation.pop(normalized, None)
            else:
                self._by_conversation[normalized] = pending
            self._persist_locked()

    def _load_initial_state(self) -> dict[str, PendingTextChannelAction]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            payload = _load_json(self._path)
        except Exception:
            LOGGER.exception("Failed to read pending text-channel action store at %s.", self._path)
            return {}
        if str(payload.get("schema", "") or "").strip() != _STATE_SCHEMA:
            LOGGER.warning("Ignoring pending text-channel action store with unknown schema at %s.", self._path)
            return {}
        raw_entries = payload.get("pending_by_conversation", {})
        if not isinstance(raw_entries, Mapping):
            LOGGER.warning("Ignoring malformed pending text-channel action store at %s.", self._path)
            return {}
        loaded: dict[str, PendingTextChannelAction] = {}
        for raw_conversation_id, raw_action in raw_entries.items():
            conversation_id = str(raw_conversation_id or "").strip()
            if not conversation_id or not isinstance(raw_action, Mapping):
                continue
            try:
                loaded[conversation_id] = PendingTextChannelAction.from_dict(raw_action)
            except Exception:
                LOGGER.exception(
                    "Ignoring malformed pending text-channel action for conversation %s in %s.",
                    conversation_id,
                    self._path,
                )
        return loaded

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        payload: dict[str, object] = {
            "schema": _STATE_SCHEMA,
            "pending_by_conversation": {
                conversation_id: action.to_dict()
                for conversation_id, action in sorted(self._by_conversation.items())
            },
        }
        _atomic_write_json(self._path, payload)

    @staticmethod
    def _pending_action_from_tool_results(
        tool_results: Sequence[AgentToolResult],
        *,
        current: PendingTextChannelAction | None,
    ) -> PendingTextChannelAction | None:
        for tool_result in reversed(tuple(tool_results)):
            if str(getattr(tool_result, "name", "") or "").strip() != "send_whatsapp_message":
                continue
            output = getattr(tool_result, "output", None)
            if not isinstance(output, Mapping):
                continue
            pending = PendingTextChannelActionStore._pending_action_from_output(output)
            if pending is not None:
                return pending
            if PendingTextChannelActionStore._should_preserve_current_pending(output):
                return current
            return None
        return current

    @staticmethod
    def _should_preserve_current_pending(output: Mapping[str, object]) -> bool:
        status = str(output.get("status", "") or "").strip()
        error_code = str(output.get("error_code", "") or "").strip()
        return status == "error" and error_code in {
            "tool_execution_failed",
            "unsupported_tool",
            "unserializable_tool_output",
        }

    @staticmethod
    def _pending_action_from_output(output: Mapping[str, object]) -> PendingTextChannelAction | None:
        status = str(output.get("status", "") or "").strip()
        if status not in {
            "message_required",
            "confirmation_required",
            "needs_phone_clarification",
            "needs_clarification",
        }:
            return None
        name = str(output.get("name", "") or "").strip()
        if not name:
            return None
        family_name = PendingTextChannelActionStore._optional_text(output.get("family_name"))
        lookup_role = PendingTextChannelActionStore._optional_text(output.get("lookup_role"))
        contact_label = PendingTextChannelActionStore._optional_text(output.get("contact_label"))
        recipient_label = PendingTextChannelActionStore._optional_text(output.get("recipient_label"))
        phone_last4 = PendingTextChannelActionStore._optional_text(output.get("phone_last4"))
        message_preview = PendingTextChannelActionStore._optional_text(output.get("message_preview"))
        question = PendingTextChannelActionStore._optional_text(output.get("question"))
        clarification_candidates = PendingTextChannelActionStore._clarification_candidates(output.get("options"))
        clarification_options = tuple(
            candidate.summary() for candidate in clarification_candidates
        )
        state = {
            "message_required": "awaiting_message",
            "confirmation_required": "awaiting_confirmation",
            "needs_phone_clarification": "awaiting_phone_clarification",
            "needs_clarification": "awaiting_contact_clarification",
        }[status]
        return PendingTextChannelAction(
            tool_name="send_whatsapp_message",
            state=state,
            name=name,
            family_name=family_name,
            lookup_role=lookup_role,
            contact_label=contact_label,
            recipient_label=recipient_label,
            phone_last4=phone_last4,
            message_preview=message_preview,
            question=question,
            clarification_options=clarification_options,
            clarification_candidates=clarification_candidates,
        )

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _clarification_candidates(value: Any) -> tuple[PendingTextChannelContactOption, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return ()
        normalized: list[PendingTextChannelContactOption] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            label = PendingTextChannelActionStore._optional_text(item.get("label"))
            if not label:
                continue
            role = PendingTextChannelActionStore._optional_text(item.get("role"))
            phone_last4_options = item.get("phone_last4_options")
            suffixes: tuple[str, ...] = ()
            if isinstance(phone_last4_options, Sequence) and not isinstance(
                phone_last4_options,
                (str, bytes, bytearray),
            ):
                suffixes = tuple(
                    suffix
                    for suffix in (
                        PendingTextChannelActionStore._optional_text(option)
                        for option in phone_last4_options
                    )
                    if suffix
                )
            candidate = PendingTextChannelContactOption(
                label=label,
                role=role,
                phone_last4_options=suffixes,
            )
            if candidate not in normalized:
                normalized.append(candidate)
        return tuple(normalized)


__all__ = [
    "PendingTextChannelAction",
    "PendingTextChannelContactOption",
    "PendingTextChannelActionStore",
]
