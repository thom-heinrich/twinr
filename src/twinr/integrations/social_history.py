"""Normalize social-history learning consent and import state.

This module keeps the user-approved social-history configuration independent
from the portal presentation and the WhatsApp runtime import path. The
integration store remains the single persisted source of truth for the opt-in,
selected source, lookback window, and latest import outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Final

from twinr.integrations.store import ManagedIntegrationConfig


SOCIAL_HISTORY_LEARNING_INTEGRATION_ID: Final[str] = "social_history_learning"
SOCIAL_HISTORY_SOURCE_WHATSAPP: Final[str] = "whatsapp"
SOCIAL_HISTORY_ALLOWED_SOURCES: Final[tuple[str, ...]] = (SOCIAL_HISTORY_SOURCE_WHATSAPP,)
SOCIAL_HISTORY_LOOKBACK_OPTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("4w", "Last 4 weeks"),
    ("3m", "Last 3 months"),
    ("6m", "Last 6 months"),
    ("1y", "Last 1 year"),
)
_LOOKBACK_DURATIONS: Final[dict[str, timedelta]] = {
    "4w": timedelta(weeks=4),
    "3m": timedelta(days=90),
    "6m": timedelta(days=183),
    "1y": timedelta(days=365),
}
_ALLOWED_IMPORT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "disabled",
        "idle",
        "queued",
        "running",
        "completed",
        "partial",
        "failed",
    }
)


def _normalize_text(value: object, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _normalize_source(value: object) -> str:
    candidate = _normalize_text(value, default=SOCIAL_HISTORY_SOURCE_WHATSAPP).casefold()
    return candidate if candidate in SOCIAL_HISTORY_ALLOWED_SOURCES else SOCIAL_HISTORY_SOURCE_WHATSAPP


def _normalize_lookback_key(value: object) -> str:
    candidate = _normalize_text(value, default="3m").casefold()
    return candidate if candidate in _LOOKBACK_DURATIONS else "3m"


def _normalize_iso8601(value: object | None) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _normalize_non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return default
    else:
        return default
    return max(parsed, 0)


def _normalize_import_status(value: object, *, enabled: bool) -> str:
    candidate = _normalize_text(value, default="idle" if enabled else "disabled").casefold()
    if candidate in _ALLOWED_IMPORT_STATUSES:
        return candidate
    return "idle" if enabled else "disabled"


@dataclass(frozen=True, slots=True)
class SocialHistoryLearningConfig:
    """Represent one normalized social-history learning consent record."""

    enabled: bool = False
    source: str = SOCIAL_HISTORY_SOURCE_WHATSAPP
    lookback_key: str = "3m"
    last_import_status: str = "disabled"
    last_import_request_id: str | None = None
    last_import_started_at: str | None = None
    last_import_finished_at: str | None = None
    last_import_error: str | None = None
    last_import_detail: str | None = None
    last_import_messages: int = 0
    last_import_turns: int = 0
    last_import_chats: int = 0
    last_import_oldest_at: str | None = None
    last_import_newest_at: str | None = None

    @classmethod
    def from_record(cls, record: ManagedIntegrationConfig | None) -> "SocialHistoryLearningConfig":
        """Build one normalized config view from a managed integration record."""

        settings = {} if record is None else dict(getattr(record, "settings", {}) or {})
        enabled = bool(getattr(record, "enabled", False))
        normalized = cls(
            enabled=enabled,
            source=_normalize_source(settings.get("source")),
            lookback_key=_normalize_lookback_key(settings.get("lookback_key")),
            last_import_status=_normalize_import_status(settings.get("last_import_status"), enabled=enabled),
            last_import_request_id=_normalize_text(settings.get("last_import_request_id")) or None,
            last_import_started_at=_normalize_iso8601(settings.get("last_import_started_at")),
            last_import_finished_at=_normalize_iso8601(settings.get("last_import_finished_at")),
            last_import_error=_normalize_text(settings.get("last_import_error")) or None,
            last_import_detail=_normalize_text(settings.get("last_import_detail")) or None,
            last_import_messages=_normalize_non_negative_int(settings.get("last_import_messages")),
            last_import_turns=_normalize_non_negative_int(settings.get("last_import_turns")),
            last_import_chats=_normalize_non_negative_int(settings.get("last_import_chats")),
            last_import_oldest_at=_normalize_iso8601(settings.get("last_import_oldest_at")),
            last_import_newest_at=_normalize_iso8601(settings.get("last_import_newest_at")),
        )
        if not normalized.enabled and normalized.last_import_status not in {"completed", "partial"}:
            return replace(normalized, last_import_status="disabled")
        return normalized

    @property
    def lookback_duration(self) -> timedelta:
        """Return the selected bounded lookback window as a timedelta."""

        return _LOOKBACK_DURATIONS[self.lookback_key]

    @property
    def lookback_label(self) -> str:
        """Return the operator-facing label for the selected lookback window."""

        return dict(SOCIAL_HISTORY_LOOKBACK_OPTIONS)[self.lookback_key]

    def to_record(self) -> ManagedIntegrationConfig:
        """Serialize the normalized config back into one managed record."""

        settings = {
            "source": self.source,
            "lookback_key": self.lookback_key,
            "last_import_status": self.last_import_status,
            "last_import_messages": str(self.last_import_messages),
            "last_import_turns": str(self.last_import_turns),
            "last_import_chats": str(self.last_import_chats),
        }
        optional_settings = {
            "last_import_request_id": self.last_import_request_id,
            "last_import_started_at": self.last_import_started_at,
            "last_import_finished_at": self.last_import_finished_at,
            "last_import_error": self.last_import_error,
            "last_import_detail": self.last_import_detail,
            "last_import_oldest_at": self.last_import_oldest_at,
            "last_import_newest_at": self.last_import_newest_at,
        }
        for key, value in optional_settings.items():
            if value:
                settings[key] = value
        return ManagedIntegrationConfig(
            integration_id=SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
            enabled=self.enabled,
            settings=settings,
        )


def social_history_record_with_import_state(
    record: ManagedIntegrationConfig | None,
    *,
    status: str,
    request_id: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    detail: str | None = None,
    error: str | None = None,
    messages: int | None = None,
    turns: int | None = None,
    chats: int | None = None,
    oldest_at: str | None = None,
    newest_at: str | None = None,
) -> ManagedIntegrationConfig:
    """Return one updated record with normalized import-status metadata."""

    current = SocialHistoryLearningConfig.from_record(record)
    normalized_status = _normalize_import_status(status, enabled=current.enabled)
    updated = SocialHistoryLearningConfig(
        enabled=current.enabled,
        source=current.source,
        lookback_key=current.lookback_key,
        last_import_status=normalized_status,
        last_import_request_id=request_id or current.last_import_request_id,
        last_import_started_at=_normalize_iso8601(started_at) or current.last_import_started_at,
        last_import_finished_at=_normalize_iso8601(finished_at) or current.last_import_finished_at,
        last_import_error=_normalize_text(error) or (None if normalized_status in {"queued", "running", "completed", "partial"} else current.last_import_error),
        last_import_detail=_normalize_text(detail) or current.last_import_detail,
        last_import_messages=current.last_import_messages if messages is None else _normalize_non_negative_int(messages),
        last_import_turns=current.last_import_turns if turns is None else _normalize_non_negative_int(turns),
        last_import_chats=current.last_import_chats if chats is None else _normalize_non_negative_int(chats),
        last_import_oldest_at=_normalize_iso8601(oldest_at) or current.last_import_oldest_at,
        last_import_newest_at=_normalize_iso8601(newest_at) or current.last_import_newest_at,
    )
    if normalized_status in {"queued", "running"}:
        updated = SocialHistoryLearningConfig(
            enabled=updated.enabled,
            source=updated.source,
            lookback_key=updated.lookback_key,
            last_import_status=updated.last_import_status,
            last_import_request_id=updated.last_import_request_id,
            last_import_started_at=updated.last_import_started_at,
            last_import_finished_at=None,
            last_import_error=None,
            last_import_detail=updated.last_import_detail,
            last_import_messages=updated.last_import_messages,
            last_import_turns=updated.last_import_turns,
            last_import_chats=updated.last_import_chats,
            last_import_oldest_at=updated.last_import_oldest_at,
            last_import_newest_at=updated.last_import_newest_at,
        )
    return updated.to_record()
