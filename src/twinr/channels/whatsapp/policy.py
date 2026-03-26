"""Enforce allowlist, self-chat, and outbound-echo policy for WhatsApp."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time

from twinr.channels.contracts import ChannelInboundMessage

from .config import WhatsAppChannelConfig, normalize_whatsapp_digits


@dataclass(frozen=True, slots=True)
class WhatsAppPolicyDecision:
    """Summarize whether one inbound WhatsApp message should reach Twinr."""

    accepted: bool
    reason: str


@dataclass(slots=True)
class _SentMessageCache:
    """Track recently sent message IDs so echoed self-chat replies are ignored."""

    ttl_s: float
    max_entries: int
    _entries: OrderedDict[str, float] = field(init=False, repr=False, default_factory=OrderedDict)

    def remember(self, message_id: str) -> None:
        """Store one outbound message ID with a short-lived expiry."""

        now = time.monotonic()
        self._purge(now)
        self._entries[str(message_id)] = now + self.ttl_s
        self._entries.move_to_end(str(message_id))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def contains(self, message_id: str) -> bool:
        """Return whether the message ID is still considered a local outbound echo."""

        now = time.monotonic()
        self._purge(now)
        expiry = self._entries.get(str(message_id))
        return expiry is not None and expiry >= now

    def _purge(self, now: float) -> None:
        while self._entries:
            oldest_id, expiry = next(iter(self._entries.items()))
            if expiry >= now:
                break
            self._entries.pop(oldest_id, None)


@dataclass(slots=True)
class WhatsAppMessagePolicy:
    """Apply the focused WhatsApp policy for Twinr's internal self-chat tests."""

    config: WhatsAppChannelConfig
    _sent_cache: _SentMessageCache = field(init=False, repr=False)
    _inbound_cache: _SentMessageCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sent_cache = _SentMessageCache(
            ttl_s=self.config.sent_cache_ttl_s,
            max_entries=self.config.sent_cache_max_entries,
        )
        self._inbound_cache = _SentMessageCache(
            ttl_s=self.config.sent_cache_ttl_s,
            max_entries=self.config.sent_cache_max_entries,
        )

    def remember_outbound(self, message_id: str | None) -> None:
        """Remember one outbound WhatsApp message ID for echo filtering."""

        if isinstance(message_id, str) and message_id.strip():
            self._sent_cache.remember(message_id.strip())

    def evaluate(
        self,
        message: ChannelInboundMessage,
        *,
        account_jid: str | None,
    ) -> WhatsAppPolicyDecision:
        """Return whether the inbound message should be processed by Twinr."""

        if self._inbound_cache.contains(message.message_id):
            return WhatsAppPolicyDecision(False, "duplicate_inbound")
        if message.is_group and not self.config.groups_enabled:
            return WhatsAppPolicyDecision(False, "groups_disabled")
        if not self._matches_allowlist(message.sender_id):
            return WhatsAppPolicyDecision(False, "sender_not_allowlisted")
        if message.is_from_self:
            if self._sent_cache.contains(message.message_id):
                return WhatsAppPolicyDecision(False, "outbound_echo")
            if not self.config.self_chat_mode:
                return WhatsAppPolicyDecision(False, "from_self_requires_self_chat_mode")
            if not self._is_self_chat(message.conversation_id, account_jid=account_jid):
                return WhatsAppPolicyDecision(False, "from_self_not_account_chat")
            self._inbound_cache.remember(message.message_id)
            return WhatsAppPolicyDecision(True, "self_chat_inbound")
        self._inbound_cache.remember(message.message_id)
        return WhatsAppPolicyDecision(True, "allowlisted_direct_message")

    def _matches_allowlist(self, sender_id: str) -> bool:
        try:
            return normalize_whatsapp_digits(sender_id) == normalize_whatsapp_digits(self.config.allow_from)
        except (TypeError, ValueError):
            return False

    def _is_self_chat(self, conversation_id: str, *, account_jid: str | None) -> bool:
        if not account_jid:
            return False
        try:
            return normalize_whatsapp_digits(conversation_id) == normalize_whatsapp_digits(account_jid)
        except (TypeError, ValueError):
            return False
