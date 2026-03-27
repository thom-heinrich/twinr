"""Persist bounded structured WhatsApp inbound event evidence for debugging.

The WhatsApp channel can fail in transport- or provider-specific ways that do
not show up in the higher-level Twinr runtime snapshot. This module records a
small redacted JSONL audit trail for inbound events so a single repro can show
whether the loop received one or multiple upstream messages, which raw worker
fields they carried, and what Twinr decided to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any

from twinr.channels.contracts import ChannelInboundMessage


LOGGER = logging.getLogger(__name__)

_MAX_LOG_LINES = 256
_MAX_LOG_BYTES = 256 * 1024


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text_length(value: str) -> int:
    return len(value.encode("utf-8"))


@dataclass(slots=True)
class WhatsAppInboundEventLog:
    """Append a bounded JSONL trail for accepted and rejected inbound events."""

    root_dir: Path
    max_lines: int = _MAX_LOG_LINES
    max_bytes: int = _MAX_LOG_BYTES
    _path: Path = field(init=False, repr=False)
    _lock: RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.root_dir / "inbound_events.jsonl"
        self._lock = RLock()

    def record_received(self, message: ChannelInboundMessage) -> None:
        self._append(
            {
                "stage": "received",
                **self._message_summary(message),
            }
        )

    def record_policy_decision(
        self,
        message: ChannelInboundMessage,
        *,
        accepted: bool,
        reason: str,
    ) -> None:
        self._append(
            {
                "stage": "policy_decision",
                "accepted": bool(accepted),
                "reason": str(reason).strip(),
                **self._message_summary(message),
            }
        )

    def record_delivery(
        self,
        message: ChannelInboundMessage,
        *,
        outbound_message_id: str | None,
        reply_text: str,
    ) -> None:
        self._append(
            {
                "stage": "delivery",
                "outbound_message_id": (outbound_message_id or "").strip() or None,
                "reply_text_sha256": _hash_text(reply_text),
                "reply_text_bytes": _bounded_text_length(reply_text),
                **self._message_summary(message),
            }
        )

    def record_delivery_error(self, message: ChannelInboundMessage, *, error: str) -> None:
        self._append(
            {
                "stage": "delivery_error",
                "error": str(error).strip()[:400],
                **self._message_summary(message),
            }
        )

    @staticmethod
    def _message_summary(message: ChannelInboundMessage) -> dict[str, Any]:
        metadata = dict(message.metadata)
        return {
            "recorded_at": _utc_now_iso(),
            "channel": message.channel,
            "message_id": message.message_id,
            "conversation_id": message.conversation_id,
            "sender_id": message.sender_id,
            "received_at": message.received_at,
            "is_from_self": bool(message.is_from_self),
            "is_group": bool(message.is_group),
            "text_sha256": _hash_text(message.text),
            "text_bytes": _bounded_text_length(message.text),
            "worker_upsert_type": metadata.get("worker_upsert_type") or None,
            "worker_request_id": metadata.get("worker_request_id") or None,
            "worker_remote_jid": metadata.get("worker_remote_jid") or None,
            "worker_remote_jid_alt": metadata.get("worker_remote_jid_alt") or None,
            "worker_participant": metadata.get("worker_participant") or None,
            "worker_participant_alt": metadata.get("worker_participant_alt") or None,
            "worker_message_timestamp": metadata.get("worker_message_timestamp") or None,
            "worker_context_stanza_id": metadata.get("worker_context_stanza_id") or None,
            "worker_account_jid": metadata.get("account_jid") or None,
        }

    def _append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                self._trim_if_needed()
            except OSError:
                LOGGER.exception("Failed to append WhatsApp inbound event evidence to %s.", self._path)

    def _trim_if_needed(self) -> None:
        try:
            if not self._path.exists():
                return
            if self._path.stat().st_size <= self.max_bytes:
                return
            lines = self._path.read_text(encoding="utf-8").splitlines()
            kept = lines[-self.max_lines :]
            self._path.write_text(
                "".join(f"{line}\n" for line in kept),
                encoding="utf-8",
            )
        except OSError:
            LOGGER.exception("Failed to trim WhatsApp inbound event evidence at %s.", self._path)
