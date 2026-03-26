"""Generic file-backed pairing state for Twinr channel onboarding flows.

This module is runtime-neutral on purpose. The web dashboard, voice-triggered
service-connect flows, and future operator/runtime surfaces should all share
the same bounded pairing snapshot contract instead of keeping onboarding state
inside one UI package.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Callable

from twinr.web.support.store import read_text_file, write_text_file


def _now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _normalize_channel_id(raw_value: str) -> str:
    """Normalize one channel identifier for file-backed onboarding state."""

    text = str(raw_value or "").strip().lower()
    if not text:
        raise ValueError("Channel id is required.")
    normalized = "".join(character for character in text if character.isalnum() or character in {"_", "-"})
    if normalized != text or not normalized:
        raise ValueError(f"Unsupported channel id: {raw_value!r}")
    return normalized


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    """Coerce one stored bool-like value defensively."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: object) -> int | None:
    """Coerce one stored integer-like value."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: object) -> str | None:
    """Coerce one stored scalar-like value to text."""

    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


@dataclass(frozen=True, slots=True)
class ChannelPairingSnapshot:
    """Persist one channel-onboarding pairing status snapshot."""

    channel_id: str
    phase: str = "idle"
    summary: str = "Not started"
    detail: str = "No pairing window has been started yet."
    running: bool = False
    qr_needed: bool = False
    qr_svg: str | None = None
    qr_image_data_url: str | None = None
    paired: bool = False
    fatal: bool = False
    auth_repair_needed: bool = False
    worker_ready: bool = False
    account_id: str | None = None
    last_worker_detail: str | None = None
    last_reconnect_reason: str | None = None
    reconnect_in_ms: int | None = None
    status_code: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot for the file-backed onboarding store."""

        return {
            "channel_id": self.channel_id,
            "phase": self.phase,
            "summary": self.summary,
            "detail": self.detail,
            "running": self.running,
            "qr_needed": self.qr_needed,
            "qr_svg": self.qr_svg,
            "qr_image_data_url": self.qr_image_data_url,
            "paired": self.paired,
            "fatal": self.fatal,
            "auth_repair_needed": self.auth_repair_needed,
            "worker_ready": self.worker_ready,
            "account_id": self.account_id,
            "last_worker_detail": self.last_worker_detail,
            "last_reconnect_reason": self.last_reconnect_reason,
            "reconnect_in_ms": self.reconnect_in_ms,
            "status_code": self.status_code,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def initial(cls, channel_id: str) -> "ChannelPairingSnapshot":
        """Return the initial persisted state for one channel."""

        return cls(channel_id=_normalize_channel_id(channel_id))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, channel_id: str) -> "ChannelPairingSnapshot":
        """Hydrate one snapshot from a persisted JSON-like mapping."""

        normalized_channel_id = _normalize_channel_id(channel_id)
        if not isinstance(payload, dict):
            return cls.initial(normalized_channel_id)

        return cls(
            channel_id=normalized_channel_id,
            phase=_coerce_text(payload.get("phase")) or "idle",
            summary=_coerce_text(payload.get("summary")) or "Not started",
            detail=_coerce_text(payload.get("detail")) or "No pairing window has been started yet.",
            running=_coerce_bool(payload.get("running"), default=False),
            qr_needed=_coerce_bool(payload.get("qr_needed"), default=False),
            qr_svg=_coerce_text(payload.get("qr_svg")),
            qr_image_data_url=_coerce_text(payload.get("qr_image_data_url")),
            paired=_coerce_bool(payload.get("paired"), default=False),
            fatal=_coerce_bool(payload.get("fatal"), default=False),
            auth_repair_needed=_coerce_bool(payload.get("auth_repair_needed"), default=False),
            worker_ready=_coerce_bool(payload.get("worker_ready"), default=False),
            account_id=_coerce_text(payload.get("account_id")),
            last_worker_detail=_coerce_text(payload.get("last_worker_detail")),
            last_reconnect_reason=_coerce_text(payload.get("last_reconnect_reason")),
            reconnect_in_ms=_coerce_int(payload.get("reconnect_in_ms")),
            status_code=_coerce_int(payload.get("status_code")),
            exit_code=_coerce_int(payload.get("exit_code")),
            started_at=_coerce_text(payload.get("started_at")),
            updated_at=_coerce_text(payload.get("updated_at")),
            finished_at=_coerce_text(payload.get("finished_at")),
        )


@dataclass(slots=True)
class FileBackedChannelOnboardingStore:
    """Persist one channel-onboarding snapshot under the Twinr project root."""

    path: Path
    channel_id: str

    @classmethod
    def from_project_root(cls, project_root: str | Path, *, channel_id: str) -> "FileBackedChannelOnboardingStore":
        """Build the rooted onboarding store for one channel."""

        normalized_channel_id = _normalize_channel_id(channel_id)
        root = Path(project_root).expanduser().resolve(strict=False)
        return cls(
            path=root / "state" / "channel_onboarding" / f"{normalized_channel_id}.json",
            channel_id=normalized_channel_id,
        )

    def load(self) -> ChannelPairingSnapshot:
        """Load the latest snapshot or the initial state when missing."""

        raw_text = read_text_file(self.path)
        if not raw_text.strip():
            return ChannelPairingSnapshot.initial(self.channel_id)
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return ChannelPairingSnapshot(
                channel_id=self.channel_id,
                phase="failed",
                summary="Status store broken",
                detail=f"The stored onboarding status at {self.path} could not be read.",
                fatal=True,
                auth_repair_needed=False,
                updated_at=_now_iso(),
                finished_at=_now_iso(),
            )
        return ChannelPairingSnapshot.from_dict(payload, channel_id=self.channel_id)

    def save(self, snapshot: ChannelPairingSnapshot) -> ChannelPairingSnapshot:
        """Persist one snapshot and return the normalized stored value."""

        stored = ChannelPairingSnapshot.from_dict(snapshot.to_dict(), channel_id=self.channel_id)
        write_text_file(
            self.path,
            json.dumps(stored.to_dict(), ensure_ascii=True, indent=2, sort_keys=True),
        )
        return stored


@dataclass(slots=True)
class _ChannelPairingHandle:
    """Track one in-process pairing thread."""

    thread: threading.Thread


class InProcessChannelPairingRegistry:
    """Coordinate one bounded pairing job per channel inside the current process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles: dict[str, _ChannelPairingHandle] = {}

    def is_running(self, channel_id: str) -> bool:
        """Return whether one pairing job is currently alive for ``channel_id``."""

        normalized_channel_id = _normalize_channel_id(channel_id)
        with self._lock:
            handle = self._handles.get(normalized_channel_id)
            if handle is None:
                return False
            if handle.thread.is_alive():
                return True
            self._handles.pop(normalized_channel_id, None)
            return False

    def start(self, channel_id: str, target: Callable[[], None]) -> bool:
        """Start one pairing job when the channel is currently idle."""

        normalized_channel_id = _normalize_channel_id(channel_id)
        with self._lock:
            existing = self._handles.get(normalized_channel_id)
            if existing is not None and existing.thread.is_alive():
                return False

            thread = threading.Thread(
                target=self._run_job,
                args=(normalized_channel_id, target),
                name=f"channel-onboarding-{normalized_channel_id}",
                daemon=True,
            )
            self._handles[normalized_channel_id] = _ChannelPairingHandle(thread=thread)
            thread.start()
            return True

    def _run_job(self, channel_id: str, target: Callable[[], None]) -> None:
        try:
            target()
        finally:
            with self._lock:
                self._handles.pop(channel_id, None)


__all__ = [
    "ChannelPairingSnapshot",
    "FileBackedChannelOnboardingStore",
    "InProcessChannelPairingRegistry",
    "_now_iso",
]
