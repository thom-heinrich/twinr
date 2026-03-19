"""Define the shared contract for Twinr's external text channels.

These dataclasses normalize external channel payloads into a provider-agnostic
shape so WhatsApp, Signal, Telegram, and future phone or SMS transports can
share the same turn-execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _require_text(value: str, *, field_name: str) -> str:
    """Return a stripped non-empty string for the given contract field."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _optional_text(value: str | None, *, field_name: str) -> str | None:
    """Return a stripped optional string or ``None`` when blank."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string when provided")
    cleaned = value.strip()
    return cleaned or None


def _freeze_metadata(metadata: Mapping[str, str] | None) -> Mapping[str, str]:
    """Copy metadata into a read-only mapping with stringified values."""

    if metadata is None:
        return MappingProxyType({})
    normalized = {str(key): str(value) for key, value in dict(metadata).items()}
    return MappingProxyType(normalized)


class ChannelTransportError(RuntimeError):
    """Raise when a channel worker or transport cannot deliver a message."""

    pass


@dataclass(frozen=True, slots=True)
class ChannelInboundMessage:
    """Represent one normalized inbound text message from an external channel."""

    channel: str
    message_id: str
    conversation_id: str
    sender_id: str
    text: str
    sender_display_name: str | None = None
    received_at: str | None = None
    is_group: bool = False
    is_from_self: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _require_text(self.channel, field_name="channel"))
        object.__setattr__(self, "message_id", _require_text(self.message_id, field_name="message_id"))
        object.__setattr__(self, "conversation_id", _require_text(self.conversation_id, field_name="conversation_id"))
        object.__setattr__(self, "sender_id", _require_text(self.sender_id, field_name="sender_id"))
        object.__setattr__(self, "text", _require_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "sender_display_name",
            _optional_text(self.sender_display_name, field_name="sender_display_name"),
        )
        object.__setattr__(self, "received_at", _optional_text(self.received_at, field_name="received_at"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ChannelOutboundMessage:
    """Represent one normalized outbound text reply for an external channel."""

    channel: str
    conversation_id: str
    recipient_id: str
    text: str
    reply_to_message_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _require_text(self.channel, field_name="channel"))
        object.__setattr__(self, "conversation_id", _require_text(self.conversation_id, field_name="conversation_id"))
        object.__setattr__(self, "recipient_id", _require_text(self.recipient_id, field_name="recipient_id"))
        object.__setattr__(self, "text", _require_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "reply_to_message_id",
            _optional_text(self.reply_to_message_id, field_name="reply_to_message_id"),
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
