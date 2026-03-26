"""Coordinate outbound WhatsApp sends through the long-lived channel service.

The active Baileys worker must remain the only owner of the linked-device
session. Voice/runtime turns therefore do not send WhatsApp messages directly;
they enqueue a bounded outbound request on disk and wait for the running
WhatsApp channel loop to claim and deliver it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import mkstemp
import time
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig

from .config import WhatsAppChannelConfig

_DIRECTORY_MODE = 0o777
_FILE_MODE = 0o666
_RESULT_POLL_INTERVAL_S = 0.2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat()


def _parse_iso8601(value: object, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed.astimezone(timezone.utc)


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if field_name == "reply_to_message_id":
        return text
    return text


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, _DIRECTORY_MODE)


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    parent = path.parent.resolve(strict=False)
    _ensure_directory(parent)
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
            os.fchmod(handle.fileno(), _FILE_MODE)
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
        os.chmod(path, _FILE_MODE)
        _fsync_directory(parent)
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
class WhatsAppOutboundRequest:
    """Persist one outbound WhatsApp send request for the channel loop."""

    request_id: str
    created_at: str
    expires_at: str
    chat_jid: str
    text: str
    recipient_label: str | None = None
    reply_to_message_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, field_name="request_id"))
        object.__setattr__(self, "created_at", _required_text(self.created_at, field_name="created_at"))
        object.__setattr__(self, "expires_at", _required_text(self.expires_at, field_name="expires_at"))
        object.__setattr__(self, "chat_jid", _required_text(self.chat_jid, field_name="chat_jid"))
        object.__setattr__(self, "text", _required_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "recipient_label",
            _optional_text(self.recipient_label, field_name="recipient_label"),
        )
        object.__setattr__(
            self,
            "reply_to_message_id",
            _optional_text(self.reply_to_message_id, field_name="reply_to_message_id"),
        )
        _parse_iso8601(self.created_at, field_name="created_at")
        _parse_iso8601(self.expires_at, field_name="expires_at")

    @classmethod
    def create(
        cls,
        *,
        chat_jid: str,
        text: str,
        recipient_label: str | None,
        reply_to_message_id: str | None,
        expires_in_s: float,
    ) -> "WhatsAppOutboundRequest":
        now = _utc_now()
        created_at = now.replace(microsecond=0).isoformat()
        expires_at = (now + timedelta(seconds=max(1.0, float(expires_in_s)))).replace(microsecond=0).isoformat()
        return cls(
            request_id=f"{int(time.time() * 1000)}-{uuid4().hex}",
            created_at=created_at,
            expires_at=expires_at,
            chat_jid=chat_jid,
            text=text,
            recipient_label=recipient_label,
            reply_to_message_id=reply_to_message_id,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WhatsAppOutboundRequest":
        return cls(
            request_id=_required_text(payload.get("request_id"), field_name="request_id"),
            created_at=_required_text(payload.get("created_at"), field_name="created_at"),
            expires_at=_required_text(payload.get("expires_at"), field_name="expires_at"),
            chat_jid=_required_text(payload.get("chat_jid"), field_name="chat_jid"),
            text=_required_text(payload.get("text"), field_name="text"),
            recipient_label=_optional_text(payload.get("recipient_label"), field_name="recipient_label"),
            reply_to_message_id=_optional_text(
                payload.get("reply_to_message_id"),
                field_name="reply_to_message_id",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "chat_jid": self.chat_jid,
            "text": self.text,
        }
        if self.recipient_label:
            payload["recipient_label"] = self.recipient_label
        if self.reply_to_message_id:
            payload["reply_to_message_id"] = self.reply_to_message_id
        return payload

    def is_expired(self, *, now: datetime | None = None) -> bool:
        reference = _utc_now() if now is None else now.astimezone(timezone.utc)
        return _parse_iso8601(self.expires_at, field_name="expires_at") <= reference


@dataclass(frozen=True, slots=True)
class WhatsAppOutboundResult:
    """Persist the final outcome of one outbound WhatsApp request."""

    request_id: str
    status: str
    ok: bool
    completed_at: str
    message_id: str | None = None
    error_code: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, field_name="request_id"))
        object.__setattr__(self, "status", _required_text(self.status, field_name="status"))
        object.__setattr__(self, "completed_at", _required_text(self.completed_at, field_name="completed_at"))
        object.__setattr__(self, "message_id", _optional_text(self.message_id, field_name="message_id"))
        object.__setattr__(self, "error_code", _optional_text(self.error_code, field_name="error_code"))
        object.__setattr__(self, "error", _optional_text(self.error, field_name="error"))
        _parse_iso8601(self.completed_at, field_name="completed_at")

    @classmethod
    def sent(cls, *, request_id: str, message_id: str | None) -> "WhatsAppOutboundResult":
        return cls(
            request_id=request_id,
            status="sent",
            ok=True,
            completed_at=_utc_now_iso(),
            message_id=message_id,
        )

    @classmethod
    def failed(
        cls,
        *,
        request_id: str,
        error_code: str,
        error: str,
    ) -> "WhatsAppOutboundResult":
        return cls(
            request_id=request_id,
            status="failed",
            ok=False,
            completed_at=_utc_now_iso(),
            error_code=error_code,
            error=error,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WhatsAppOutboundResult":
        return cls(
            request_id=_required_text(payload.get("request_id"), field_name="request_id"),
            status=_required_text(payload.get("status"), field_name="status"),
            ok=bool(payload.get("ok", False)),
            completed_at=_required_text(payload.get("completed_at"), field_name="completed_at"),
            message_id=_optional_text(payload.get("message_id"), field_name="message_id"),
            error_code=_optional_text(payload.get("error_code"), field_name="error_code"),
            error=_optional_text(payload.get("error"), field_name="error"),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": self.request_id,
            "status": self.status,
            "ok": self.ok,
            "completed_at": self.completed_at,
        }
        if self.message_id:
            payload["message_id"] = self.message_id
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.error:
            payload["error"] = self.error
        return payload


class WhatsAppOutboundQueue:
    """Store outbound WhatsApp requests/results under one shared queue root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.pending_dir = self.root / "pending"
        self.processing_dir = self.root / "processing"
        self.results_dir = self.root / "results"

    @classmethod
    def from_whatsapp_config(cls, config: WhatsAppChannelConfig) -> "WhatsAppOutboundQueue":
        return cls(config.auth_dir.parent / "outbound")

    @classmethod
    def from_twinr_config(cls, config: TwinrConfig) -> "WhatsAppOutboundQueue":
        return cls.from_whatsapp_config(WhatsAppChannelConfig.from_twinr_config(config))

    def ensure_storage(self) -> None:
        for path in (self.root, self.pending_dir, self.processing_dir, self.results_dir):
            _ensure_directory(path)

    def submit_request(
        self,
        *,
        chat_jid: str,
        text: str,
        recipient_label: str | None,
        reply_to_message_id: str | None = None,
        expires_in_s: float,
    ) -> WhatsAppOutboundRequest:
        self.ensure_storage()
        request = WhatsAppOutboundRequest.create(
            chat_jid=chat_jid,
            text=text,
            recipient_label=recipient_label,
            reply_to_message_id=reply_to_message_id,
            expires_in_s=expires_in_s,
        )
        _atomic_write_json(self.pending_path(request.request_id), request.to_dict())
        return request

    def claim_next_request(self) -> WhatsAppOutboundRequest | None:
        self.ensure_storage()
        for path in sorted(self.pending_dir.glob("*.json")):
            processing_path = self.processing_path(path.stem)
            try:
                os.replace(str(path), str(processing_path))
                _fsync_directory(self.pending_dir)
                _fsync_directory(self.processing_dir)
            except FileNotFoundError:
                continue
            try:
                return WhatsAppOutboundRequest.from_dict(_load_json(processing_path))
            except Exception as exc:
                self.complete_request(
                    WhatsAppOutboundResult.failed(
                        request_id=path.stem,
                        error_code="malformed_request",
                        error=f"Malformed outbound request payload: {exc}",
                    )
                )
        return None

    def complete_request(self, result: WhatsAppOutboundResult) -> None:
        self.ensure_storage()
        _atomic_write_json(self.result_path(result.request_id), result.to_dict())
        processing_path = self.processing_path(result.request_id)
        try:
            processing_path.unlink()
            _fsync_directory(self.processing_dir)
        except FileNotFoundError:
            pass

    def wait_for_result(
        self,
        request_id: str,
        *,
        timeout_s: float,
    ) -> WhatsAppOutboundResult | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        result_path = self.result_path(request_id)
        while True:
            if result_path.is_file():
                return WhatsAppOutboundResult.from_dict(_load_json(result_path))
            if time.monotonic() >= deadline:
                return None
            time.sleep(_RESULT_POLL_INTERVAL_S)

    def fail_stale_processing_requests(self) -> int:
        self.ensure_storage()
        failed = 0
        for path in sorted(self.processing_dir.glob("*.json")):
            request_id = path.stem
            self.complete_request(
                WhatsAppOutboundResult.failed(
                    request_id=request_id,
                    error_code="unknown_after_restart",
                    error=(
                        "The WhatsApp channel restarted before it could confirm whether "
                        "the message was sent. Please retry once if needed."
                    ),
                )
            )
            failed += 1
        return failed

    def pending_path(self, request_id: str) -> Path:
        return self.pending_dir / f"{request_id}.json"

    def processing_path(self, request_id: str) -> Path:
        return self.processing_dir / f"{request_id}.json"

    def result_path(self, request_id: str) -> Path:
        return self.results_dir / f"{request_id}.json"


def dispatch_whatsapp_outbound_message(
    config: TwinrConfig,
    *,
    chat_jid: str,
    text: str,
    recipient_label: str,
    reply_to_message_id: str | None = None,
    wait_timeout_s: float | None = None,
) -> WhatsAppOutboundResult:
    """Queue one outbound WhatsApp send and wait for the channel-loop result."""

    whatsapp_config = WhatsAppChannelConfig.from_twinr_config(config)
    if not (whatsapp_config.auth_dir / "creds.json").is_file():
        return WhatsAppOutboundResult.failed(
            request_id="unpaired",
            error_code="not_paired",
            error="WhatsApp is not paired on this Twinr runtime.",
        )

    queue = WhatsAppOutboundQueue.from_whatsapp_config(whatsapp_config)
    timeout_s = max(5.0, float(wait_timeout_s or (whatsapp_config.send_timeout_s + 5.0)))
    request = queue.submit_request(
        chat_jid=chat_jid,
        text=text,
        recipient_label=recipient_label,
        reply_to_message_id=reply_to_message_id,
        expires_in_s=timeout_s,
    )
    result = queue.wait_for_result(request.request_id, timeout_s=timeout_s)
    if result is None:
        return WhatsAppOutboundResult.failed(
            request_id=request.request_id,
            error_code="channel_timeout",
            error="Timed out while waiting for the running WhatsApp channel to deliver the message.",
        )
    return result


__all__ = [
    "WhatsAppOutboundQueue",
    "WhatsAppOutboundRequest",
    "WhatsAppOutboundResult",
    "dispatch_whatsapp_outbound_message",
]
