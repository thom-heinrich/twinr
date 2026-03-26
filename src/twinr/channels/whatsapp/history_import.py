"""Coordinate bounded WhatsApp history imports through the channel service.

The running WhatsApp channel service remains the only owner of the linked
device session. Portal-triggered history imports therefore flow through this
small on-disk queue and are executed inside that same long-lived service.
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
    return text


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return None
    return None


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
class WhatsAppHistoryImportRequest:
    """Persist one bounded WhatsApp history import request."""

    request_id: str
    created_at: str
    expires_at: str
    source: str
    lookback_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, field_name="request_id"))
        object.__setattr__(self, "created_at", _required_text(self.created_at, field_name="created_at"))
        object.__setattr__(self, "expires_at", _required_text(self.expires_at, field_name="expires_at"))
        object.__setattr__(self, "source", _required_text(self.source, field_name="source"))
        object.__setattr__(self, "lookback_key", _required_text(self.lookback_key, field_name="lookback_key"))
        _parse_iso8601(self.created_at, field_name="created_at")
        _parse_iso8601(self.expires_at, field_name="expires_at")

    @classmethod
    def create(
        cls,
        *,
        source: str,
        lookback_key: str,
        expires_in_s: float,
    ) -> "WhatsAppHistoryImportRequest":
        now = _utc_now()
        created_at = now.replace(microsecond=0).isoformat()
        expires_at = (now + timedelta(seconds=max(30.0, float(expires_in_s)))).replace(microsecond=0).isoformat()
        return cls(
            request_id=f"{int(time.time() * 1000)}-{uuid4().hex}",
            created_at=created_at,
            expires_at=expires_at,
            source=source,
            lookback_key=lookback_key,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WhatsAppHistoryImportRequest":
        return cls(
            request_id=_required_text(payload.get("request_id"), field_name="request_id"),
            created_at=_required_text(payload.get("created_at"), field_name="created_at"),
            expires_at=_required_text(payload.get("expires_at"), field_name="expires_at"),
            source=_required_text(payload.get("source"), field_name="source"),
            lookback_key=_required_text(payload.get("lookback_key"), field_name="lookback_key"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "source": self.source,
            "lookback_key": self.lookback_key,
        }

    def is_expired(self, *, now: datetime | None = None) -> bool:
        reference = _utc_now() if now is None else now.astimezone(timezone.utc)
        return _parse_iso8601(self.expires_at, field_name="expires_at") <= reference


@dataclass(frozen=True, slots=True)
class WhatsAppHistoryImportResult:
    """Persist the final outcome of one history import request."""

    request_id: str
    status: str
    ok: bool
    completed_at: str
    detail: str | None = None
    error_code: str | None = None
    error: str | None = None
    imported_messages: int | None = None
    imported_turns: int | None = None
    imported_chats: int | None = None
    oldest_at: str | None = None
    newest_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, field_name="request_id"))
        object.__setattr__(self, "status", _required_text(self.status, field_name="status"))
        object.__setattr__(self, "completed_at", _required_text(self.completed_at, field_name="completed_at"))
        object.__setattr__(self, "detail", _optional_text(self.detail, field_name="detail"))
        object.__setattr__(self, "error_code", _optional_text(self.error_code, field_name="error_code"))
        object.__setattr__(self, "error", _optional_text(self.error, field_name="error"))
        object.__setattr__(self, "imported_messages", _optional_int(self.imported_messages))
        object.__setattr__(self, "imported_turns", _optional_int(self.imported_turns))
        object.__setattr__(self, "imported_chats", _optional_int(self.imported_chats))
        object.__setattr__(self, "oldest_at", _optional_text(self.oldest_at, field_name="oldest_at"))
        object.__setattr__(self, "newest_at", _optional_text(self.newest_at, field_name="newest_at"))
        _parse_iso8601(self.completed_at, field_name="completed_at")

    @classmethod
    def completed(
        cls,
        *,
        request_id: str,
        status: str,
        detail: str,
        imported_messages: int,
        imported_turns: int,
        imported_chats: int,
        oldest_at: str | None,
        newest_at: str | None,
    ) -> "WhatsAppHistoryImportResult":
        return cls(
            request_id=request_id,
            status=status,
            ok=status in {"completed", "partial"},
            completed_at=_utc_now_iso(),
            detail=detail,
            imported_messages=imported_messages,
            imported_turns=imported_turns,
            imported_chats=imported_chats,
            oldest_at=oldest_at,
            newest_at=newest_at,
        )

    @classmethod
    def failed(
        cls,
        *,
        request_id: str,
        error_code: str,
        error: str,
    ) -> "WhatsAppHistoryImportResult":
        return cls(
            request_id=request_id,
            status="failed",
            ok=False,
            completed_at=_utc_now_iso(),
            error_code=error_code,
            error=error,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WhatsAppHistoryImportResult":
        return cls(
            request_id=_required_text(payload.get("request_id"), field_name="request_id"),
            status=_required_text(payload.get("status"), field_name="status"),
            ok=bool(payload.get("ok", False)),
            completed_at=_required_text(payload.get("completed_at"), field_name="completed_at"),
            detail=_optional_text(payload.get("detail"), field_name="detail"),
            error_code=_optional_text(payload.get("error_code"), field_name="error_code"),
            error=_optional_text(payload.get("error"), field_name="error"),
            imported_messages=_optional_int(payload.get("imported_messages")),
            imported_turns=_optional_int(payload.get("imported_turns")),
            imported_chats=_optional_int(payload.get("imported_chats")),
            oldest_at=_optional_text(payload.get("oldest_at"), field_name="oldest_at"),
            newest_at=_optional_text(payload.get("newest_at"), field_name="newest_at"),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": self.request_id,
            "status": self.status,
            "ok": self.ok,
            "completed_at": self.completed_at,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.error:
            payload["error"] = self.error
        if self.imported_messages is not None:
            payload["imported_messages"] = self.imported_messages
        if self.imported_turns is not None:
            payload["imported_turns"] = self.imported_turns
        if self.imported_chats is not None:
            payload["imported_chats"] = self.imported_chats
        if self.oldest_at:
            payload["oldest_at"] = self.oldest_at
        if self.newest_at:
            payload["newest_at"] = self.newest_at
        return payload


class WhatsAppHistoryImportQueue:
    """Persist bounded history import requests for the WhatsApp channel loop."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)
        self.pending_dir = self.root / "pending"
        self.processing_dir = self.root / "processing"
        self.results_dir = self.root / "results"
        for path in (self.pending_dir, self.processing_dir, self.results_dir):
            _ensure_directory(path)

    @classmethod
    def from_whatsapp_config(cls, config: WhatsAppChannelConfig) -> "WhatsAppHistoryImportQueue":
        return cls(config.auth_dir.parent / "history_import")

    @classmethod
    def from_twinr_config(cls, config: TwinrConfig) -> "WhatsAppHistoryImportQueue":
        return cls.from_whatsapp_config(WhatsAppChannelConfig.from_twinr_config(config))

    def submit_request(
        self,
        *,
        source: str,
        lookback_key: str,
        expires_in_s: float = 180.0,
    ) -> WhatsAppHistoryImportRequest:
        request = WhatsAppHistoryImportRequest.create(
            source=source,
            lookback_key=lookback_key,
            expires_in_s=expires_in_s,
        )
        _atomic_write_json(self.pending_dir / f"{request.request_id}.json", request.to_dict())
        return request

    def claim_next_request(self) -> WhatsAppHistoryImportRequest | None:
        for path in sorted(self.pending_dir.glob("*.json")):
            processing_path = self.processing_dir / path.name
            try:
                os.replace(path, processing_path)
            except FileNotFoundError:
                continue
            payload = _load_json(processing_path)
            return WhatsAppHistoryImportRequest.from_dict(payload)
        return None

    def complete_request(self, result: WhatsAppHistoryImportResult) -> None:
        processing_path = self.processing_dir / f"{result.request_id}.json"
        if processing_path.exists():
            processing_path.unlink()
        _atomic_write_json(self.results_dir / f"{result.request_id}.json", result.to_dict())

    def wait_for_result(self, request_id: str, *, timeout_s: float) -> WhatsAppHistoryImportResult | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        result_path = self.results_dir / f"{request_id}.json"
        while time.monotonic() <= deadline:
            if result_path.exists():
                return WhatsAppHistoryImportResult.from_dict(_load_json(result_path))
            time.sleep(_RESULT_POLL_INTERVAL_S)
        return None

    def fail_stale_processing_requests(self) -> int:
        failed = 0
        for path in sorted(self.processing_dir.glob("*.json")):
            try:
                request = WhatsAppHistoryImportRequest.from_dict(_load_json(path))
            except Exception:
                request_id = path.stem
            else:
                request_id = request.request_id
            path.unlink(missing_ok=True)
            self.complete_request(
                WhatsAppHistoryImportResult.failed(
                    request_id=request_id,
                    error_code="unknown_after_restart",
                    error="The WhatsApp history import was interrupted while the channel service restarted.",
                )
            )
            failed += 1
        return failed


__all__ = [
    "WhatsAppHistoryImportQueue",
    "WhatsAppHistoryImportRequest",
    "WhatsAppHistoryImportResult",
]
