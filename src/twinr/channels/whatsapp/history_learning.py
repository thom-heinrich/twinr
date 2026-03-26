"""Import bounded WhatsApp history into shared Twinr memory.

The importer runs inside the existing WhatsApp channel service, temporarily
reopens the worker in history-sync mode, collects recent direct-message
history, converts user-authored chat segments into neutral shared-memory turns,
and persists them without treating other humans as Twinr's own replies for
personality learning.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Protocol

from twinr.channels.contracts import ChannelInboundMessage
from twinr.integrations import SocialHistoryLearningConfig
from twinr.memory.longterm.runtime.service import LongTermMemoryService

from .config import WhatsAppChannelConfig
from .worker_bridge import (
    WhatsAppWorkerBridge,
    WhatsAppWorkerExitedError,
    WhatsAppWorkerHistoryBatchEvent,
    WhatsAppWorkerHistoryMessage,
    WhatsAppWorkerStatusEvent,
)


class _BridgeProtocol(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def next_event(
        self,
        *,
        timeout_s: float | None = None,
    ) -> ChannelHistoryWorkerEvent | None: ...


ChannelHistoryEvent = WhatsAppWorkerStatusEvent | WhatsAppWorkerHistoryBatchEvent
ChannelHistoryWorkerEvent = ChannelInboundMessage | ChannelHistoryEvent


_OPEN_TIMEOUT_S = 25.0
_SETTLE_TIMEOUT_S = 2.0
_NO_HISTORY_GRACE_S = 4.0
_MAX_SESSION_S = 75.0


def _label_for_message(message: WhatsAppWorkerHistoryMessage) -> str:
    for candidate in (message.sender_label, message.chat_label, message.sender_id):
        if candidate:
            return str(candidate).strip()
    return "contact"


def _utc_datetime_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(max(timestamp_ms, 0) / 1000.0, tz=UTC)


def _isoformat_or_none(value: int | None) -> str | None:
    if value is None:
        return None
    return _utc_datetime_from_ms(value).replace(microsecond=0).isoformat()


def _join_messages(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _contact_reply_text(contact_label: str | None, reply_parts: list[str]) -> str:
    label = (contact_label or "contact").strip() or "contact"
    return f"WhatsApp contact {label} replied: {_join_messages(reply_parts)}"


@dataclass(frozen=True, slots=True)
class ImportedWhatsAppHistoryTurn:
    """Represent one neutral shared-memory turn derived from chat history."""

    transcript: str
    response: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class WhatsAppHistoryImportSummary:
    """Describe one completed WhatsApp history import run."""

    status: str
    detail: str
    imported_messages: int
    imported_turns: int
    imported_chats: int
    oldest_at: str | None = None
    newest_at: str | None = None


def build_import_turns(
    messages: tuple[WhatsAppWorkerHistoryMessage, ...],
    *,
    lookback_key: str,
) -> tuple[ImportedWhatsAppHistoryTurn, ...]:
    """Convert user-authored direct-message history into neutral memory turns."""

    lookback = SocialHistoryLearningConfig(enabled=True, lookback_key=lookback_key).lookback_duration
    cutoff = datetime.now(UTC) - lookback
    per_chat: dict[str, list[WhatsAppWorkerHistoryMessage]] = defaultdict(list)
    for message in messages:
        if message.is_group:
            continue
        if _utc_datetime_from_ms(message.timestamp_ms) < cutoff:
            continue
        per_chat[message.conversation_id].append(message)

    imported: list[ImportedWhatsAppHistoryTurn] = []
    for chat_messages in per_chat.values():
        ordered = sorted(chat_messages, key=lambda item: (item.timestamp_ms, item.message_id))
        pending_user: list[str] = []
        pending_started_at: int | None = None
        pending_reply: list[str] = []
        pending_contact_label: str | None = None
        for message in ordered:
            if message.is_from_self:
                if pending_user and pending_reply:
                    imported.append(
                        ImportedWhatsAppHistoryTurn(
                            transcript=_join_messages(pending_user),
                            response=_contact_reply_text(pending_contact_label, pending_reply),
                            occurred_at=_utc_datetime_from_ms(pending_started_at or message.timestamp_ms),
                        )
                    )
                    pending_user = []
                    pending_reply = []
                    pending_contact_label = None
                    pending_started_at = None
                if pending_started_at is None:
                    pending_started_at = message.timestamp_ms
                pending_user.append(message.text)
                continue
            if not pending_user:
                continue
            pending_contact_label = _label_for_message(message)
            pending_reply.append(message.text)

        if pending_user:
            response = (
                _contact_reply_text(pending_contact_label, pending_reply)
                if pending_reply
                else "No reply was captured after this user-authored WhatsApp message in the selected WhatsApp history window."
            )
            imported.append(
                ImportedWhatsAppHistoryTurn(
                    transcript=_join_messages(pending_user),
                    response=response,
                    occurred_at=_utc_datetime_from_ms(pending_started_at or ordered[-1].timestamp_ms),
                )
            )

    return tuple(turn for turn in imported if turn.transcript and turn.response)


@dataclass(slots=True)
class WhatsAppHistoryLearningService:
    """Collect and import WhatsApp history through a temporary history-sync worker."""

    whatsapp_config: WhatsAppChannelConfig
    memory_service: LongTermMemoryService
    bridge_factory: type[WhatsAppWorkerBridge] = WhatsAppWorkerBridge

    def run_import(self, *, lookback_key: str) -> WhatsAppHistoryImportSummary:
        """Run one bounded history import and persist it into shared memory."""

        bridge: _BridgeProtocol = self.bridge_factory(
            self.whatsapp_config,
            history_sync_enabled=True,
        )
        collected: dict[str, WhatsAppWorkerHistoryMessage] = {}
        oldest_seen_ms: int | None = None
        newest_seen_ms: int | None = None
        opened_at: float | None = None
        last_history_at: float | None = None
        started_at = time.monotonic()
        try:
            bridge.start()
            while True:
                now = time.monotonic()
                if now - started_at >= _MAX_SESSION_S:
                    break
                try:
                    event = bridge.next_event(timeout_s=0.5)
                except WhatsAppWorkerExitedError as exc:
                    raise RuntimeError(f"WhatsApp history worker exited unexpectedly: {exc}") from exc
                if event is None:
                    if opened_at is None and now - started_at >= _OPEN_TIMEOUT_S:
                        raise RuntimeError("WhatsApp history session did not open in time.")
                    if opened_at is not None:
                        if last_history_at is None and now - opened_at >= _NO_HISTORY_GRACE_S:
                            break
                        if last_history_at is not None and now - last_history_at >= _SETTLE_TIMEOUT_S:
                            break
                    continue
                if isinstance(event, WhatsAppWorkerStatusEvent):
                    if event.fatal:
                        raise RuntimeError(event.detail or "WhatsApp worker entered a fatal history-sync state.")
                    if event.connection == "open" and opened_at is None:
                        opened_at = now
                    continue
                if isinstance(event, ChannelInboundMessage):
                    continue
                if not isinstance(event, WhatsAppWorkerHistoryBatchEvent):
                    continue
                last_history_at = now
                for message in event.messages:
                    collected.setdefault(message.message_id, message)
                    oldest_seen_ms = message.timestamp_ms if oldest_seen_ms is None else min(oldest_seen_ms, message.timestamp_ms)
                    newest_seen_ms = message.timestamp_ms if newest_seen_ms is None else max(newest_seen_ms, message.timestamp_ms)
        finally:
            bridge.stop()

        ordered_messages = tuple(sorted(collected.values(), key=lambda item: (item.timestamp_ms, item.message_id)))
        lookback = SocialHistoryLearningConfig(enabled=True, lookback_key=lookback_key).lookback_duration
        cutoff_ms = int((datetime.now(UTC) - lookback).timestamp() * 1000)
        in_window_messages = tuple(
            message
            for message in ordered_messages
            if not message.is_group and message.timestamp_ms >= cutoff_ms
        )
        turns = build_import_turns(ordered_messages, lookback_key=lookback_key)
        for turn in turns:
            self.memory_service.import_external_conversation_turn(
                transcript=turn.transcript,
                response=turn.response,
                source="whatsapp_history",
                modality="text",
                created_at=turn.occurred_at,
            )
        imported_chat_ids = {
            message.conversation_id
            for message in in_window_messages
        }
        status = "partial" if oldest_seen_ms is not None and oldest_seen_ms > cutoff_ms else "completed"
        if not ordered_messages:
            status = "completed"
        detail = (
            "WhatsApp history import completed."
            if status == "completed"
            else "WhatsApp returned only part of the requested history window; Twinr imported the available portion."
        )
        if not ordered_messages:
            detail = "No direct-message history was returned for the selected WhatsApp window."
        return WhatsAppHistoryImportSummary(
            status=status,
            detail=detail,
            imported_messages=len(in_window_messages),
            imported_turns=len(turns),
            imported_chats=len(imported_chat_ids),
            oldest_at=_isoformat_or_none(oldest_seen_ms),
            newest_at=_isoformat_or_none(newest_seen_ms),
        )


__all__ = [
    "ImportedWhatsAppHistoryTurn",
    "WhatsAppHistoryImportSummary",
    "WhatsAppHistoryLearningService",
    "build_import_turns",
]
